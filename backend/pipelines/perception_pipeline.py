# backend/pipelines/perception_pipeline.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE:
#   Orchestrates the full WorkspaceVision inference pipeline.
#   One call to analyze() runs the complete chain:
#   image → validate/resize → detect → segment → caption → scene graph
#
# THIS IS THE ONLY FILE the FastAPI endpoints need to call.
# All model logic stays in models/, all helpers stay in utils/.
#
# PIPELINE FLOW:
#   1. prepare_image()      → validate + resize input
#   2. Detector.detect()    → find objects with Grounding DINO
#   3. Segmenter.segment()  → trace pixel masks with SAM2
#   4. draw_segmentation_masks() → composite masks onto image for frontend
#   5. Captioner.caption()  → describe each object + full scene with BLIP
#   6. SceneGraphBuilder()  → build structured spatial representation
#   7. Return AnalysisResult → single object with everything inside
#
# DESIGN DECISIONS:
#   - Segmentation is OPTIONAL (skip_segmentation=True saves ~300ms)
#   - annotated_image is ALWAYS returned — boxes only when masks skipped
#   - Each step is timed individually for MLflow logging
#   - Errors in one step don't crash others — graceful degradation
# ─────────────────────────────────────────────────────────────────────────────

import io
import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Union
from pathlib import Path

from PIL import Image

# ── Internal imports ──────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_settings
from models.model_loader import get_models
from models.detector import Detector, Detection
from models.segmenter import Segmenter, SegmentationResult
from models.captioner import Captioner, CaptionResult
from utils.image_utils import (
    prepare_image,
    image_to_base64,            # ← ADDED: encode annotated image for API
    draw_segmentation_masks,    # ← ADDED: composite SAM2 masks onto image
    draw_detection_boxes,       # ← ADDED: boxes-only fallback (video / skip_seg)
)
from utils.scene_graph import SceneGraphBuilder, SceneGraph

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Result data structure
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AnalysisResult:
    """
    Complete output of one WorkspaceVision pipeline run.

    Everything the API, chatbot, and frontend need is in here.

    Attributes:
        scene_graph          : Structured spatial map of all detected objects
        detections           : Raw detection list (label, confidence, box)
        seg_results          : Segmentation masks (empty if skip_segmentation=True)
        caption_results      : Per-object captions
        scene_caption        : Full-image scene description from BLIP
        annotated_image_b64  : Base64 JPEG — original image with SAM2 masks drawn on
        image_meta           : original_size, final_size, was_resized
        timings              : Per-step latency in seconds (for MLflow)
        total_time           : Total pipeline wall time in seconds
        error                : Error message if pipeline partially failed (None = success)
    """
    scene_graph         : Optional[SceneGraph]      = None
    detections          : List[Detection]            = field(default_factory=list)
    seg_results         : List[SegmentationResult]   = field(default_factory=list)
    caption_results     : List[CaptionResult]        = field(default_factory=list)
    scene_caption       : str                        = ""
    annotated_image_b64 : str                        = ""   # ← ADDED
    image_meta          : dict                       = field(default_factory=dict)
    timings             : dict                       = field(default_factory=dict)
    total_time          : float                      = 0.0
    error               : Optional[str]              = None

    def to_dict(self) -> dict:
        """
        Serialises the full result to a plain dict for API responses.
        Raw mask arrays are excluded (too large for JSON).
        annotated_image is included as a base64 JPEG string.
        """
        return {
            "scene_graph"      : self.scene_graph.to_dict() if self.scene_graph else {},
            "scene_caption"    : self.scene_caption,
            "detections"       : [d.to_dict() for d in self.detections],
            "caption_results"  : [c.to_dict() for c in self.caption_results],
            "annotated_image"  : self.annotated_image_b64,   # ← ADDED
            "image_meta"       : self.image_meta,
            "timings"          : {k: round(v, 3) for k, v in self.timings.items()},
            "total_time"       : round(self.total_time, 3),
            "error"            : self.error,
        }

    def get_labels(self) -> List[str]:
        """Returns a flat list of all detected object labels."""
        return [d.label for d in self.detections]

    def get_object_count(self) -> int:
        """Returns total number of detected objects."""
        return len(self.detections)


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline
# ══════════════════════════════════════════════════════════════════════════════

class PerceptionPipeline:
    """
    Orchestrates the full WorkspaceVision inference pipeline.

    Usage:
        pipeline = PerceptionPipeline()
        result   = pipeline.analyze(image_bytes)

        print(result.scene_caption)
        print(result.scene_graph.to_text())
        for d in result.detections:
            print(d.label, d.confidence)
    """

    def __init__(self):
        """
        Initialises pipeline by pulling already-loaded models from registry.
        Does NOT reload models — just holds references.
        Model loading happens once at backend startup via get_models().
        """
        settings = get_settings()
        registry = get_models()

        self.detector  = Detector(registry)
        self.segmenter = Segmenter(registry)
        self.captioner = Captioner(registry)
        self.settings  = settings

        logger.info("[PerceptionPipeline] Initialised.")

    # ──────────────────────────────────────────────────────────────────────────
    def analyze(
        self,
        source               : Union[bytes, io.BytesIO, str, Path, Image.Image],
        labels               : Optional[List[str]] = None,
        confidence_threshold : float = 0.3,
        max_detections       : int   = 10,
        skip_segmentation    : bool  = False,
    ) -> AnalysisResult:
        """
        Runs the full perception pipeline on an image.

        Args:
            source               : Image as bytes, BytesIO, file path, or PIL Image
            labels               : Object labels to detect (None = workspace defaults)
            confidence_threshold : Minimum detection confidence (default 0.3)
            max_detections       : Max objects to detect (default 10)
            skip_segmentation    : Skip SAM2 step to save ~300ms (default False)

        Returns:
            AnalysisResult with scene_graph, detections, captions,
            annotated_image (base64), and timings
        """

        pipeline_start = time.time()
        timings        = {}
        result         = AnalysisResult()

        try:

            # ── Step 1: Prepare image ──────────────────────────────────────
            t0 = time.time()

            if isinstance(source, Image.Image):
                # Already a PIL image — just validate and resize
                from utils.image_utils import validate_image, resize_image
                is_valid, msg = validate_image(source)
                if not is_valid:
                    raise ValueError(f"Image validation failed: {msg}")
                image      = resize_image(source, max_size=self.settings.MAX_IMAGE_SIZE)
                image_meta = {
                    "original_size" : source.size,
                    "final_size"    : image.size,
                    "was_resized"   : source.size != image.size,
                }
            else:
                image, image_meta = prepare_image(
                    source,
                    max_size=self.settings.MAX_IMAGE_SIZE
                )

            timings["prepare_image"] = time.time() - t0
            result.image_meta        = image_meta
            logger.info(f"[Pipeline] Image prepared: {image.size}")

            # ── Step 2: Detect objects ─────────────────────────────────────
            t0 = time.time()

            if labels:
                detections = self.detector.detect(
                    image                = image,
                    labels               = labels,
                    confidence_threshold = confidence_threshold,
                    max_detections       = max_detections,
                )
            else:
                detections = self.detector.detect_workspace_objects(
                    image                = image,
                    confidence_threshold = confidence_threshold,
                    max_detections       = max_detections,
                )

            timings["detection"] = time.time() - t0
            result.detections    = detections
            logger.info(f"[Pipeline] Detected {len(detections)} objects.")

            if not detections:
                # No objects found — encode clean image, return early
                result.annotated_image_b64 = image_to_base64(image, format="JPEG")
                result.scene_graph         = SceneGraphBuilder(image.size).build([], [], [])
                result.total_time          = time.time() - pipeline_start
                result.timings             = timings
                return result

            # ── Step 3: Segment objects (optional) ────────────────────────
            t0          = time.time()
            seg_results = []

            if not skip_segmentation:
                try:
                    seg_results = self.segmenter.segment_detections(image, detections)
                    logger.info(f"[Pipeline] Segmented {len(seg_results)} objects.")
                except Exception as e:
                    # Segmentation failure is non-fatal — log and continue
                    logger.warning(f"[Pipeline] Segmentation failed (non-fatal): {e}")
                    result.error = f"Segmentation warning: {e}"

            timings["segmentation"] = time.time() - t0
            result.seg_results      = seg_results

            # ── Step 3b: Draw masks onto image → annotated_image ──────────
            # This is done immediately after segmentation so the annotated
            # image is available for the API response regardless of whether
            # captioning or scene graph succeed.
            t0 = time.time()

            if seg_results:
                # Masks available → composite SAM2 masks + labels onto image
                annotated_pil = draw_segmentation_masks(
                    image        = image,
                    seg_results  = seg_results,
                    alpha        = 0.45,
                    draw_contour = True,
                )
                logger.info("[Pipeline] Segmentation masks drawn onto annotated image.")
            else:
                # No masks (skipped or failed) → draw bounding boxes instead
                annotated_pil = draw_detection_boxes(image, detections)
                logger.info("[Pipeline] No masks — drew bounding boxes as fallback.")

            # Encode to base64 JPEG for JSON transport
            result.annotated_image_b64 = image_to_base64(annotated_pil, format="JPEG")
            timings["draw_masks"]      = time.time() - t0

            # ── Step 4: Caption objects + scene ───────────────────────────
            t0 = time.time()

            try:
                caption_results = self.captioner.caption_detections(image, detections)
                scene_caption   = self.captioner.caption_scene(image)
            except Exception as e:
                logger.warning(f"[Pipeline] Captioning failed (non-fatal): {e}")
                caption_results = []
                scene_caption   = ""
                result.error    = f"Captioning warning: {e}"

            timings["captioning"]    = time.time() - t0
            result.caption_results   = caption_results
            result.scene_caption     = scene_caption
            logger.info(f"[Pipeline] Scene caption: {scene_caption[:60]}...")

            # ── Step 5: Build scene graph ──────────────────────────────────
            t0      = time.time()
            builder = SceneGraphBuilder(image_size=image.size)
            graph   = builder.build(
                detections      = detections,
                seg_results     = seg_results,
                caption_results = caption_results,
                scene_caption   = scene_caption,
            )

            timings["scene_graph"] = time.time() - t0
            result.scene_graph     = graph
            logger.info(f"[Pipeline] Scene graph built: {graph.object_count} objects.")

        except Exception as e:
            # Fatal error — log it, return partial result with error message
            logger.error(f"[Pipeline] Fatal error: {e}", exc_info=True)
            result.error = str(e)

        finally:
            # Always record total time regardless of success or failure
            result.total_time = time.time() - pipeline_start
            result.timings    = timings

        return result
