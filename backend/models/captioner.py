# backend/models/captioner.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE:
#   Wraps BLIP-1 or BLIP-2 into a clean caption() interface.
#   Takes a PIL image and returns a natural language description.
#   Can caption the FULL image or individual CROPPED objects.
#   Automatically adapts prompting strategy based on blip_version from registry.
#
# BLIP-2 (Salesforce/blip2-opt-2.7b):
#   Frozen ViT + Q-Former bridge + frozen OPT-2.7B LLM.
#   More grounded than BLIP-1, fewer hallucinations.
#   Best with VQA-style prompts: "Question: ... Answer:"
#   Outputs: pixel_values + input_ids + attention_mask
#   Always uses float16.
#
# BLIP-1 (Salesforce/blip-image-captioning-large):
#   ViT encoder + text decoder (simpler architecture).
#   Lighter (~1.5GB VRAM), works on any system.
#   Best with sentence completion prompts: "a workspace with"
#   Outputs: pixel_values + input_ids
#   Uses config dtype (float16 on GPU, float32 on CPU).
#
# DTYPE HANDLING (both versions):
#   pixel_values  → model dtype (float16 or float32)
#   input_ids     → int64  (never cast to float)
#   attention_mask→ int64  (never cast to float)
#
# PIPELINE POSITION:
#   detector.py → segmenter.py → [crops] → captioner.py → captions
# ─────────────────────────────────────────────────────────────────────────────

import logging
import torch
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


# ── Caption result data structure ─────────────────────────────────────────────

@dataclass
class CaptionResult:
    """
    Represents a generated caption for one image or image region.

    Attributes:
        caption    : Generated text description
        label      : Object label if captioning a crop (optional)
        confidence : Detection confidence inherited from Detection (optional)
        box        : Bounding box [x1,y1,x2,y2] if captioning a crop (optional)
    """
    caption    : str
    label      : Optional[str]   = None
    confidence : Optional[float] = None
    box        : List[float]     = field(default_factory=list)

    def to_dict(self) -> dict:
        """Converts to plain dict for API responses."""
        return {
            "caption"    : self.caption,
            "label"      : self.label,
            "confidence" : round(self.confidence, 4) if self.confidence else None,
            "box"        : [round(v, 2) for v in self.box],
        }


# ── Captioner class ───────────────────────────────────────────────────────────

class Captioner:
    """
    BLIP-1 / BLIP-2 image captioner (auto-detected from registry).

    Usage:
        from models.captioner import Captioner
        from models.model_loader import get_models

        registry  = get_models()
        captioner = Captioner(registry)

        # Caption full image
        caption = captioner.caption_image(image)
        print(caption)  # "a person standing in a workshop with tools"

        # Caption individual detected objects
        results = captioner.caption_detections(image, detections)
        for r in results:
            print(r.label, "→", r.caption)
    """

    def __init__(self, registry):
        """
        Accepts the ModelRegistry singleton — does NOT reload any models.
        Reads blip_version from registry to adapt prompting strategy.

        Args:
            registry: ModelRegistry instance from get_models()
        """
        self.model        = registry.blip_model
        self.processor    = registry.blip_processor
        self.device       = registry.device
        self.blip_version = registry.blip_version   # "blip1" or "blip2"
        # BLIP-2 always float16; BLIP-1 uses whatever dtype the registry has
        self.dtype = torch.float16 if self.blip_version == "blip2" else registry.dtype

        logger.info(f"[Captioner] Initialized with {self.blip_version.upper()}.")

    # ──────────────────────────────────────────────────────────────────────────
    def caption_image(
        self,
        image          : Image.Image,
        prompt         : Optional[str] = None,
        max_new_tokens : int = 50,
    ) -> str:
        """
        Generates a caption for a full image using BLIP-2.

        Two modes:
          - prompt=None  → unconditional: BLIP-2 generates a free description
          - prompt="..." → conditional : use VQA-style prompt for best results
                           e.g. "Question: what is on the workbench? Answer:"

        Args:
            image          : PIL.Image to caption
            prompt         : Optional text prompt (None = free captioning)
            max_new_tokens : Max tokens to generate (default 50)

        Returns:
            Caption string e.g. "a workbench with tools and a lamp"
        """

        # ── Preprocess image (+ optional prompt) ──────────────────────────
        # Blip2Processor resizes image, normalizes pixels, tokenizes prompt.
        # Returns dict with pixel_values, and optionally input_ids + attention_mask
        if prompt:
            inputs = self.processor(
                images      = image,
                text        = prompt,
                return_tensors = "pt",
            )
        else:
            inputs = self.processor(
                images         = image,
                return_tensors = "pt",
            )

        # ── Move tensors to GPU with correct dtypes ────────────────────────
        # pixel_values  → float16 (image features, must match model dtype)
        # input_ids     → int64  (token indices, must NOT be cast to float)
        # attention_mask→ int64  (binary mask, must NOT be cast to float)
        inputs = {
            k: (v.to(self.device, dtype=torch.float16)   # float tensors → float16
                if v.dtype in (torch.float32, torch.float64)
                else v.to(self.device))                   # int tensors → keep dtype
            for k, v in inputs.items()
        }

        # ── Generate caption ───────────────────────────────────────────────
        # BLIP-2 decoder runs autoregressively via OPT-2.7B
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens = max_new_tokens,
                do_sample      = False,    # greedy decoding — deterministic
            )

        # ── Decode token IDs → text ────────────────────────────────────────
        # skip_special_tokens removes <pad>, </s> etc.
        # BLIP-2 with OPT sometimes echoes the prompt — strip it
        caption = self.processor.decode(
            output_ids[0],
            skip_special_tokens = True,
        ).strip()

        # Strip echoed prompt prefix if BLIP-2 repeated it in output
        if prompt and caption.lower().startswith(prompt.lower()):
            caption = caption[len(prompt):].strip()

        logger.debug(f"[Captioner] Caption: {caption}")
        return caption

    # ──────────────────────────────────────────────────────────────────────────
    def caption_crop(
        self,
        image          : Image.Image,
        box            : List[float],
        padding        : int = 10,
        max_new_tokens : int = 40,
    ) -> str:
        """
        Crops a region from the image and captions just that region.

        Args:
            image          : Full PIL.Image
            box            : [x1, y1, x2, y2] in PIXEL coordinates
                            (detector outputs absolute pixel values, not 0-1)
            padding        : Extra pixels around box for context (default 10)
            max_new_tokens : Max caption length (default 40)
        """
        img_w, img_h = image.size

        # Boxes are already in pixel coordinates — do NOT multiply by img dimensions
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])

        # Add padding — clamp strictly to image boundaries
        x1 = max(0,     x1 - padding)
        y1 = max(0,     y1 - padding)
        x2 = min(img_w, x2 + padding)
        y2 = min(img_h, y2 + padding)

        # Safety guard — degenerate box fallback
        if x2 <= x1 or y2 <= y1:
            logger.warning(
                f"[Captioner] Degenerate crop box {box} → captioning full image."
            )
            return self.caption_image(image, max_new_tokens=max_new_tokens)

        crop = image.crop((x1, y1, x2, y2)).convert("RGB")
        return self.caption_image(crop, max_new_tokens=max_new_tokens)

    # ──────────────────────────────────────────────────────────────────────────
    def caption_detections(
        self,
        image          : Image.Image,
        detections     : list,            # List[Detection] from detector.py
        max_new_tokens : int = 40,
    ) -> List[CaptionResult]:
        """
        Captions each detected object individually by cropping its box.

        Args:
            image          : Full PIL.Image
            detections     : List of Detection objects from Detector.detect()
            max_new_tokens : Max words per caption (default 40)

        Returns:
            List of CaptionResult — one per detection, preserving order.
        """

        if not detections:
            logger.warning("[Captioner] No detections to caption.")
            return []

        results = []
        for detection in detections:
            caption = self.caption_crop(
                image          = image,
                box            = detection.box,
                max_new_tokens = max_new_tokens,
            )
            results.append(CaptionResult(
                caption    = caption,
                label      = detection.label,
                confidence = detection.confidence,
                box        = detection.box,
            ))
            logger.debug(f"[Captioner] {detection.label} → {caption}")

        logger.info(f"[Captioner] Captioned {len(results)} objects.")
        return results

    # ──────────────────────────────────────────────────────────────────────────
    def caption_scene(
        self,
        image          : Image.Image,
        max_new_tokens : int = 60,
    ) -> str:
        """
        Generates a high-level scene description of the entire workspace.

        Prompt strategy differs by model:
          BLIP-2: VQA-style "Question: ... Answer:" activates OPT decoder properly
          BLIP-1: Sentence completion "a workspace with" triggers decoder continuation

        Args:
            image          : Full PIL.Image of the workspace
            max_new_tokens : Max words in scene description (default 60)

        Returns:
            Scene description string
        """
        if self.blip_version == "blip2":
            # VQA-style prompt — BLIP-2 OPT works best with Q&A format
            prompt = "Question: Describe what you see in this workspace. Answer:"
        else:
            # Sentence completion — BLIP-1 continues the phrase
            prompt = "a workspace with"

        return self.caption_image(
            image          = image,
            prompt         = prompt,
            max_new_tokens = max_new_tokens,
        )
