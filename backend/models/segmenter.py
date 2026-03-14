# backend/models/segmenter.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE:
#   Wraps SAM2 into a clean segment() interface.
#   Takes a PIL image + bounding boxes (from detector.py) and returns
#   precise binary pixel masks — one mask per detected object.
#
# HOW SAM2 WORKS:
#   SAM2 (Segment Anything Model v2) takes "prompts" to guide segmentation.
#   We use BOUNDING BOX prompts — the boxes from Grounding DINO tell SAM2
#   exactly WHERE to segment, so it traces the precise object outline.
#
#   Input  : PIL image + list of boxes [[x1,y1,x2,y2], ...]
#   Output : list of binary numpy masks, one per box
#            mask[y][x] = True  → this pixel belongs to the object
#            mask[y][x] = False → this pixel is background
#
# PIPELINE POSITION:
#   detector.py → [boxes] → segmenter.py → [masks] → captioner.py
# ─────────────────────────────────────────────────────────────────────────────

import logging
import numpy as np
import torch
from PIL import Image
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


# ── Segmentation result data structure ───────────────────────────────────────

@dataclass
class SegmentationResult:
    """
    Represents the segmentation mask for one detected object.

    Attributes:
        label      : Object class name inherited from Detection e.g. "chair"
        confidence : Confidence score inherited from Detection
        box        : Bounding box [x1, y1, x2, y2] in pixels
        mask       : Binary numpy array, shape (H, W), dtype bool
                     True = object pixel, False = background pixel
        mask_area  : Number of pixels belonging to this object
    """
    label      : str
    confidence : float
    box        : List[float]          = field(default_factory=list)
    mask       : Optional[np.ndarray] = field(default=None, repr=False)
    mask_area  : int                  = 0

    def to_dict(self) -> dict:
        """
        Converts to plain dict for API responses.
        Mask is NOT included — it's a large array, sent separately if needed.
        """
        return {
            "label"      : self.label,
            "confidence" : round(self.confidence, 4),
            "box"        : [round(v, 2) for v in self.box],
            "mask_area"  : self.mask_area,
        }

    def get_mask_rle(self) -> dict:
        """
        Encodes the binary mask as Run-Length Encoding (RLE) for compact
        API transfer. RLE stores runs of True/False values instead of
        storing every single pixel — much smaller payload.

        Returns:
            dict with 'counts' (list of run lengths) and 'size' [H, W]
        """
        if self.mask is None:
            return {}

        # Flatten mask to 1D, compute run lengths
        flat   = self.mask.flatten().astype(np.uint8)
        runs   = np.diff(np.concatenate([[0], flat, [0]]))
        starts = np.where(runs > 0)[0]
        ends   = np.where(runs < 0)[0]
        counts = (ends - starts).tolist()

        return {
            "counts" : counts,
            "size"   : list(self.mask.shape),  # [height, width]
        }


# ── Segmenter class ───────────────────────────────────────────────────────────

class Segmenter:
    """
    SAM2-based precise object segmenter.

    Usage:
        from models.segmenter import Segmenter
        from models.model_loader import get_models

        registry  = get_models()
        segmenter = Segmenter(registry)

        results = segmenter.segment_detections(image, detections)

        for r in results:
            print(r.label, r.mask_area)
    """

    def __init__(self, registry):
        """
        Accepts the ModelRegistry singleton — does NOT reload any models.
        Just holds a reference to the already-loaded SAM2 predictor.

        Args:
            registry: ModelRegistry instance from get_models()
        """
        self.predictor = registry.sam2_predictor
        self.device    = registry.device

        logger.info("[Segmenter] Initialized with SAM2.")

    # ──────────────────────────────────────────────────────────────────────────
    def segment(                          # ← 4 spaces — method inside class
        self,                             # ← 8 spaces — continuation of def
        image : Image.Image,              # ← 8 spaces
        boxes : List[List[float]],        # ← 8 spaces
    ) -> List[np.ndarray]:                # ← 8 spaces
        """
        Runs SAM2 on an image with bounding box prompts.
        set_image() is called ONCE (encodes image features).
        predict() is called ONCE PER BOX — SAM2 does not support
        reliable batched multi-box prediction in a single call.

        Args:
            image : PIL.Image — the workspace image
            boxes : List of [x1, y1, x2, y2] boxes in PIXEL coordinates

        Returns:
            List of binary numpy arrays (dtype=bool), one per box.
        """

        if not boxes:                     # ← 8 spaces — method body
            logger.warning("[Segmenter] No boxes provided — returning empty list.")
            return []                     # ← 12 spaces — INSIDE the if block
                                          #   (was at 8 spaces before = always returned!)

        # Convert PIL → numpy RGB — SAM2 requires numpy input
        image_np = np.array(image.convert("RGB"))   # ← 8 spaces — method body

        # Autocast dtype — float16 on GPU, bfloat16 on CPU
        autocast_dtype = (                # ← 8 spaces
            torch.float16 if self.device.type == "cuda" else torch.bfloat16
        )

        # Encode image ONCE — this is the expensive step (~200ms)
        # All per-box predict() calls reuse these features for free
        with torch.no_grad():             # ← 8 spaces
            with torch.autocast(device_type=self.device.type, dtype=autocast_dtype):
                self.predictor.set_image(image_np)

        masks_list = []                   # ← 8 spaces

        # Call predict() ONCE PER BOX — batching multiple boxes in one
        # predict() call produces noisy scattered masks across the image
        for box in boxes:                 # ← 8 spaces
            box_np = np.array(box, dtype=np.float32)   # shape (4,)

            with torch.no_grad():         # ← 12 spaces — inside for loop
                with torch.autocast(device_type=self.device.type, dtype=autocast_dtype):
                    masks, scores, _ = self.predictor.predict(
                        point_coords     = None,
                        point_labels     = None,
                        box              = box_np,   # single box shape (4,)
                        multimask_output = False,    # one mask, not 3 candidates
                    )

            # masks shape: (1, H, W) or (1, 1, H, W) depending on SAM2 version
            # Squeeze all leading singleton dims to get (H, W)
            binary_mask = np.squeeze(masks).astype(bool)   # ← 12 spaces
            masks_list.append(binary_mask)            # ← 12 spaces

            logger.debug(                 # ← 12 spaces
                f"[Segmenter] Box {box} → mask area: {binary_mask.sum()} px"
            )

        logger.info(f"[Segmenter] Segmented {len(masks_list)} objects (one-by-one).")
        return masks_list                 # ← 8 spaces — OUTSIDE the for loop

    # ──────────────────────────────────────────────────────────────────────────
    def segment_detections(
        self,
        image      : Image.Image,
        detections : list,           # List[Detection] from detector.py
    ) -> List[SegmentationResult]:
        """
            Takes Detection objects from detector.py and returns SegmentationResult
            objects with SAM2 masks attached.

        IMPORTANT — COORDINATE FORMAT:
        Grounding DINO post_process_grounded_object_detection() returns boxes
        ALREADY in pixel coordinates [x1, y1, x2, y2].
        Do NOT multiply by image dimensions — they are already pixels.

        Args:
            image      : PIL.Image — the workspace image
            detections : List of Detection objects from Detector.detect()

        Returns:
            List of SegmentationResult — one per detection, order preserved.
        """

        if not detections:
            logger.warning("[Segmenter] No detections provided.")
            return []

        # Pass boxes DIRECTLY to SAM2 — already in pixel coordinates
        # DO NOT multiply by img_w / img_h — that was the source of
        # the horizontal banding bug (values like 92,250px on 750px image)
        pixel_boxes = []
        for d in detections:
            x1, y1, x2, y2 = d.box
            pixel_boxes.append([
                float(x1),
                float(y1),
                float(x2),
                float(y2),
            ])

        # Run SAM2 with correct pixel-coordinate boxes
        masks = self.segment(image, pixel_boxes)

        # Combine Detection metadata + SAM2 masks into SegmentationResult
        results = []
        for detection, mask in zip(detections, masks):
            results.append(SegmentationResult(
                label      = detection.label,
                confidence = detection.confidence,
                box        = detection.box,
                mask       = mask,
                mask_area  = int(mask.sum()),
            ))

        logger.info(f"[Segmenter] Built {len(results)} SegmentationResults.")
        return results


    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def mask_to_rgba(
        image : Image.Image,
        mask  : np.ndarray,
        color : tuple = (0, 255, 0),
        alpha : int   = 120,
    ) -> Image.Image:
        """
        Overlays a segmentation mask on the original image as a coloured
        semi-transparent highlight. Used for visualisation only.

        Args:
            image : Original PIL.Image (RGB)
            mask  : Binary numpy mask (H, W), dtype bool
            color : RGB tuple for the highlight colour (default: green)
            alpha : Transparency 0=invisible, 255=solid (default: 120)

        Returns:
            PIL.Image with the mask area highlighted in the given colour.
        """
        # Create a blank RGBA overlay the same size as the image
        overlay    = Image.new("RGBA", image.size, (0, 0, 0, 0))
        overlay_np = np.array(overlay)

        # Paint the mask region with the chosen colour + alpha
        overlay_np[mask] = [color[0], color[1], color[2], alpha]

        # Composite: paste coloured overlay onto the original image
        base   = image.convert("RGBA")
        result = Image.alpha_composite(base, Image.fromarray(overlay_np))

        return result.convert("RGB")
