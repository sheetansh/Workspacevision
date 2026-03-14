# backend/models/detector.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE:
#   Wraps Grounding DINO into a clean detect() interface.
#   Takes an image + a text prompt and returns all detected objects
#   as a list of dicts with: label, confidence score, bounding box.
#
# HOW GROUNDING DINO WORKS:
#   Unlike classic detectors (YOLO, Faster-RCNN) that detect fixed categories,
#   Grounding DINO is "open vocabulary" — you describe what to find in plain
#   English. It matches your text to image regions using CLIP-style alignment.
#
#   Input  : PIL image + prompt string e.g. "chair . desk . monitor ."
#   Output : bounding boxes + labels + confidence scores
#
# NOTE ON PROMPT FORMAT:
#   Grounding DINO expects objects separated by " . " with a trailing dot.
#   Example: "chair . desk . monitor ." — NOT "chair, desk, monitor"
#   This file handles that formatting automatically.
# ─────────────────────────────────────────────────────────────────────────────

import logging
import torch
from PIL import Image
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


# ── Detection result data structure ──────────────────────────────────────────

@dataclass
class Detection:
    """
    Represents a single detected object in an image.

    Attributes:
        label      : Object class name e.g. "hammer"
        confidence : Score between 0.0 and 1.0 (higher = more certain)
        box        : Bounding box [x_min, y_min, x_max, y_max] in pixels
                     Origin (0,0) is top-left corner of the image.
    """
    label      : str
    confidence : float
    box        : List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Converts Detection to a plain dict — used for API responses."""
        return {
            "label"      : self.label,
            "confidence" : round(self.confidence, 4),
            "box"        : [round(v, 2) for v in self.box],
        }


# ── Default workspace label vocabulary ───────────────────────────────────────
# Used by detect_workspace_objects() when no custom labels are provided.
# Covers common workshop + office + general workspace objects.
# Keep this list focused — too many labels dilutes DINO's attention.

# ── Workspace-type label presets ───────────────────────────────────────────
# Each preset is tuned for a specific workspace type.
# Users select a preset in the frontend sidebar or pass custom labels.
# Keep each list focused — too many labels dilutes DINO's attention.

WORKSPACE_PRESETS = {
    "General Workspace": [
        "person", "chair", "table", "desk", "monitor", "laptop", "keyboard",
        "mouse", "phone", "tablet", "lamp", "shelf", "cabinet", "box",
        "bottle", "cup", "window", "door", "notepad", "pen", "headphones",
        "glasses", "plant", "watch", "camera", "hard drive", "poster",
    ],
    "DIY / Workshop": [
        "person", "hammer", "pliers", "wrench", "screwdriver", "saw",
        "clamp", "chisel", "measuring tape", "level", "drill", "power drill",
        "grinder", "sander", "workbench", "pegboard", "toolbox", "ladder",
        "safety glasses", "gloves", "nail", "screw", "bolt", "wood",
        "paint brush", "paint can", "hatchet", "wire cutter", "jigsaw",
        "nail gun", "impact driver", "angle grinder", "tin snips", "ruler",
        "knife", "apron", "cordless drill", "sandpaper", "vise", "file",
        "tape", "3D printer", "fan", "metal sheet", "spray gun",
    ],
    "Construction Site": [
        "person", "hard hat", "safety vest", "helmet", "crane", "excavator",
        "bulldozer", "scaffold", "ladder", "wheelbarrow", "cement mixer",
        "brick", "pipe", "beam", "rebar", "shovel", "pickaxe",
        "drill", "saw", "measuring tape", "level", "cone", "barrier",
        "table saw", "plywood", "trowel", "cement", "concrete", "bucket",
        "solar panel", "welding helmet", "welder", "overalls", "steel beam",
        "wood plank", "gloves", "safety glasses",
    ],
    "Electronics / Electrical Lab": [
        "person", "circuit board", "soldering iron", "multimeter",
        "oscilloscope", "wire", "cable", "resistor", "capacitor",
        "breadboard", "power supply", "battery", "heat gun",
        "magnifying glass", "pliers", "screwdriver", "tweezers",
        "monitor", "laptop", "desk", "lamp", "fan", "transformer",
        "connector", "inductor", "heatsink", "microscope", "clamp",
        "soldering station", "LED", "chip", "fuse", "diode",
        "transistor", "relay", "switch", "voltage regulator", "potentiometer",
    ],
    "Office": [
        "person", "chair", "desk", "monitor", "laptop", "keyboard", "mouse",
        "phone", "headphones", "printer", "scanner", "book", "notebook",
        "pen", "stapler", "whiteboard", "projector", "lamp", "plant",
        "coffee cup", "water bottle", "backpack", "cable", "drawer",
        "filing cabinet", "shelf", "paper", "folder", "sticky note", "clock",
    ],
}

# Default preset used when no labels are provided
DEFAULT_WORKSPACE_LABELS = WORKSPACE_PRESETS["General Workspace"]


# ── Detector class ────────────────────────────────────────────────────────────

class Detector:
    """
    Grounding DINO-based open-vocabulary object detector.

    Usage:
        from models.detector import Detector
        from models.model_loader import get_models

        registry = get_models()
        detector = Detector(registry)
        detections = detector.detect(image, labels=["hammer", "drill", "pliers"])
    """

    def __init__(self, registry):
        """
        Accepts the ModelRegistry singleton — does NOT reload any models.
        Just holds references to the already-loaded DINO model + processor.

        Args:
            registry: ModelRegistry instance from get_models()
        """
        self.model     = registry.dino_model
        self.processor = registry.dino_processor
        self.device    = registry.device
        self.dtype     = registry.dtype

        logger.info("[Detector] Initialized with Grounding DINO.")

    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _build_prompt(labels: List[str]) -> str:
        """
        Converts a list of label strings into Grounding DINO prompt format.

        Grounding DINO requires objects separated by ' . ' with trailing dot.
        Handles deduplication and lowercasing automatically.

        Example:
            ["Hammer", "Drill", "Pliers"] → "hammer . drill . pliers ."

        Args:
            labels: List of object names to detect

        Returns:
            Formatted prompt string
        """
        # Lowercase, strip whitespace, remove duplicates preserving order
        seen   = set()
        unique = []
        for label in labels:
            clean = label.strip().lower()
            if clean and clean not in seen:
                seen.add(clean)
                unique.append(clean)

        # Join with ' . ' separator and add trailing dot (DINO requirement)
        return " . ".join(unique) + " ."

    # ──────────────────────────────────────────────────────────────────────────
    def detect(
        self,
        image                : Image.Image,
        labels               : List[str],
        confidence_threshold : float = 0.3,
        max_detections       : int   = 10,
    ) -> List[Detection]:
        """
        Runs Grounding DINO on an image and returns detected objects.

        Steps:
          1. Format labels into DINO prompt string
          2. Preprocess image + text with AutoProcessor
          3. Run model forward pass (no gradient = faster, less VRAM)
          4. Post-process raw logits → boxes + scores + labels
          5. Filter by confidence threshold and max detections
          6. Convert boxes from normalized [0,1] to pixel coordinates

        Args:
            image                : PIL.Image — the workspace image to analyze
            labels               : List of object names to look for
            confidence_threshold : Minimum score to keep a detection (0.0–1.0)
            max_detections       : Cap on total detections returned

        Returns:
            List of Detection objects sorted by confidence (highest first)
        """

        # ── Step 1: Build the text prompt ─────────────────────────────────
        prompt = self._build_prompt(labels)
        logger.debug(f"[Detector] Prompt: {prompt}")

        # ── Step 2: Preprocess inputs ──────────────────────────────────────
        # AutoProcessor resizes image, normalizes pixels, tokenizes text
        # return_tensors="pt" → PyTorch tensors ready for the model
        inputs = self.processor(
            images         = image,
            text           = prompt,
            return_tensors = "pt",
        )

        # Move all input tensors to the same device as the model (GPU/CPU)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # ── Step 3: Forward pass ───────────────────────────────────────────
        # torch.no_grad() disables gradient tracking → saves VRAM + speeds up
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                outputs = self.model(**inputs)

        # ── Step 4: Post-process raw model outputs ─────────────────────────
        # Converts raw logits → [x1, y1, x2, y2] in pixel coordinates
        target_size = torch.tensor([[image.height, image.width]]).to(self.device)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold  = confidence_threshold,
            text_threshold = confidence_threshold,
            target_sizes   = target_size,
        )[0]  # [0] because we process one image at a time

        # ── Step 5: Build Detection list ───────────────────────────────────
        detections = []

        boxes      = results["boxes"].cpu().tolist()   # [[x1,y1,x2,y2], ...]
        scores     = results["scores"].cpu().tolist()  # [0.87, 0.65, ...]
        labels_out = results["labels"]                 # ["hammer", "drill", ...]

        for box, score, label in zip(boxes, scores, labels_out):
            detections.append(Detection(
                label      = label,
                confidence = score,
                box        = box,   # already in pixel coords after post_process
            ))

        # ── Step 6: Sort by confidence, cap at max_detections ──────────────
        detections.sort(key=lambda d: d.confidence, reverse=True)
        detections = detections[:max_detections]

        logger.info(f"[Detector] Found {len(detections)} objects.")
        return detections

    # ──────────────────────────────────────────────────────────────────────────
    def detect_workspace_objects(
        self,
        image                : Image.Image,
        confidence_threshold : float = 0.3,
        max_detections       : int   = 10,
    ) -> List[Detection]:
        """
        Convenience method — detects common workspace objects automatically.
        Uses DEFAULT_WORKSPACE_LABELS — no need to pass a label list.

        Called by perception_pipeline.py when the user provides no custom labels.

        Args:
            image                : PIL.Image to analyze
            confidence_threshold : Minimum confidence score (default 0.3)
            max_detections       : Max objects to return (default 10)

        Returns:
            List of Detection objects
        """
        return self.detect(
            image                = image,
            labels               = DEFAULT_WORKSPACE_LABELS,  # ← clean list, no string concat bug
            confidence_threshold = confidence_threshold,
            max_detections       = max_detections,
        )
