# backend/utils/image_utils.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE:
#   All image preprocessing utilities for WorkspaceVision.
#   Every image entering the pipeline passes through here FIRST.
#   Handles: loading, validation, resizing, format conversion, base64 encoding.
#
# WHY THIS FILE EXISTS:
#   AI models are strict about input format. DINO, SAM2, and BLIP each expect:
#   - RGB colour mode (not RGBA, grayscale, or palette)
#   - Specific size limits (large images waste VRAM and slow inference)
#   - PIL.Image objects (not file paths or raw bytes)
#   This file enforces all of that in ONE place so model files stay clean.
#
# USAGE:
#   from utils.image_utils import load_image, resize_image, image_to_base64
# ─────────────────────────────────────────────────────────────────────────────

import io
import base64
import logging
from pathlib import Path
from typing import Tuple, Optional, Union, List

import cv2                                      # ← ADDED: needed for mask drawing
import numpy as np
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Maximum image dimension in pixels before resizing kicks in
# 1024px balances detection quality vs VRAM usage on GTX 1080 Ti
MAX_IMAGE_SIZE = 1024

# Minimum image dimension — images smaller than this are too small to analyse
MIN_IMAGE_SIZE = 64

# Supported input file extensions
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

# JPEG quality for saving/encoding (0-100, 95 = near-lossless)
JPEG_QUALITY = 95

# ── Segmentation mask colour palette — RGB tuples ─────────────────────────────
# 10 distinct colours that are visually separable on most workspace images.
# Cycles when more than 10 objects are present.
MASK_COLOURS = [
    (255,  56,  56),   # red
    (255, 157,  51),   # orange
    ( 76, 175,  80),   # green
    ( 33, 150, 243),   # blue
    (156,  39, 176),   # purple
    (255, 235,  59),   # yellow
    (  0, 188, 212),   # cyan
    (255,  87,  34),   # deep orange
    (121,  85,  72),   # brown
    (  0, 150, 136),   # teal
]


# ══════════════════════════════════════════════════════════════════════════════
# Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_image(source: Union[str, Path, bytes, io.BytesIO]) -> Image.Image:
    """
    Loads an image from multiple source types and returns a clean RGB PIL Image.

    Accepted source types:
        - str / Path  : file path on disk e.g. "workspace.jpg"
        - bytes       : raw image bytes e.g. from an HTTP upload
        - io.BytesIO  : byte stream e.g. from FastAPI UploadFile

    Always returns RGB — converts RGBA, palette, grayscale automatically.

    Args:
        source: File path, raw bytes, or BytesIO stream

    Returns:
        PIL.Image in RGB mode

    Raises:
        ValueError : If the source type is unsupported or file not found
        ValueError : If the file cannot be decoded as an image
    """

    try:
        if isinstance(source, (str, Path)):
            # ── Load from file path ────────────────────────────────────────
            path = Path(source)
            if not path.exists():
                raise ValueError(f"Image file not found: {path}")
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ValueError(
                    f"Unsupported file type: {path.suffix}. "
                    f"Supported: {SUPPORTED_EXTENSIONS}"
                )
            image = Image.open(path)

        elif isinstance(source, bytes):
            # ── Load from raw bytes ────────────────────────────────────────
            image = Image.open(io.BytesIO(source))

        elif isinstance(source, io.BytesIO):
            # ── Load from BytesIO stream ───────────────────────────────────
            source.seek(0)  # rewind to start — stream may have been read already
            image = Image.open(source)

        else:
            raise ValueError(
                f"Unsupported source type: {type(source)}. "
                f"Expected str, Path, bytes, or BytesIO."
            )

        # Convert to RGB — models don't handle RGBA, palette, or grayscale
        # RGBA → RGB drops alpha channel (transparency becomes white)
        # P (palette) → RGB expands indexed colours
        # L (grayscale) → RGB duplicates channel to 3 channels
        image = image.convert("RGB")

        logger.debug(f"[image_utils] Loaded image: {image.size} {image.mode}")
        return image

    except UnidentifiedImageError:
        raise ValueError("File could not be decoded as an image. Is it corrupted?")


# ══════════════════════════════════════════════════════════════════════════════
# Validation
# ══════════════════════════════════════════════════════════════════════════════

def validate_image(image: Image.Image) -> Tuple[bool, str]:
    """
    Checks if an image meets minimum quality requirements for analysis.

    Checks performed:
        1. Mode is RGB
        2. Both dimensions are at least MIN_IMAGE_SIZE (64px)
        3. Image is not completely black or white (blank image check)

    Args:
        image: PIL.Image to validate

    Returns:
        Tuple (is_valid: bool, message: str)
            is_valid=True  → image is safe to process
            is_valid=False → message explains what's wrong
    """

    # Check 1: colour mode
    if image.mode != "RGB":
        return False, f"Image mode must be RGB, got: {image.mode}"

    # Check 2: minimum size
    w, h = image.size
    if w < MIN_IMAGE_SIZE or h < MIN_IMAGE_SIZE:
        return False, (
            f"Image too small: {w}x{h}px. "
            f"Minimum: {MIN_IMAGE_SIZE}x{MIN_IMAGE_SIZE}px."
        )

    # Check 3: blank image detection
    # Convert to numpy, compute std deviation — blank images have std ≈ 0
    img_array = np.array(image)
    if img_array.std() < 1.0:
        return False, "Image appears to be blank (uniform colour)."

    return True, "OK"


# ══════════════════════════════════════════════════════════════════════════════
# Resizing
# ══════════════════════════════════════════════════════════════════════════════

def resize_image(
    image    : Image.Image,
    max_size : int = MAX_IMAGE_SIZE,
    min_size : int = MIN_IMAGE_SIZE,
) -> Image.Image:
    """
    Resizes image so its longest side is at most max_size pixels.
    Preserves aspect ratio — never distorts the image.
    Does nothing if the image is already within limits.

    Example:
        1920x1080 image with max_size=1024
        → scale factor = 1024/1920 = 0.533
        → output: 1024x576

    Args:
        image    : PIL.Image to resize
        max_size : Max pixels on longest side (default 1024)
        min_size : Minimum pixels on shortest side (default 64)

    Returns:
        Resized PIL.Image (or original if already within limits)
    """

    w, h    = image.size
    longest = max(w, h)

    # No resize needed — image already fits within max_size
    if longest <= max_size:
        return image

    # Calculate scale factor to bring longest side to max_size
    scale = max_size / longest
    new_w = max(min_size, int(w * scale))
    new_h = max(min_size, int(h * scale))

    # LANCZOS = highest quality downsampling filter (anti-aliased)
    resized = image.resize((new_w, new_h), Image.LANCZOS)

    logger.debug(f"[image_utils] Resized: {w}x{h} → {new_w}x{new_h}")
    return resized


def resize_to_square(
    image : Image.Image,
    size  : int = 384,
) -> Image.Image:
    """
    Resizes image to an exact square by padding with black borders.
    Used when a model requires fixed square input (e.g. some BLIP variants).

    Preserves aspect ratio by padding — no cropping or distortion.

    Args:
        image : PIL.Image to resize
        size  : Target square side length in pixels (default 384)

    Returns:
        Square PIL.Image with black padding on shorter sides
    """

    # Resize keeping aspect ratio so longest side = size
    image.thumbnail((size, size), Image.LANCZOS)

    # Create black square canvas
    canvas = Image.new("RGB", (size, size), (0, 0, 0))

    # Paste resized image centred on canvas
    offset_x = (size - image.width)  // 2
    offset_y = (size - image.height) // 2
    canvas.paste(image, (offset_x, offset_y))

    return canvas


# ══════════════════════════════════════════════════════════════════════════════
# Format Conversion
# ══════════════════════════════════════════════════════════════════════════════

def image_to_bytes(
    image   : Image.Image,
    format  : str = "JPEG",
    quality : int = JPEG_QUALITY,
) -> bytes:
    """
    Converts a PIL Image to raw bytes for HTTP responses or file saving.

    Args:
        image   : PIL.Image to convert
        format  : Output format — "JPEG", "PNG", or "WEBP" (default "JPEG")
        quality : Compression quality 0-100, only for JPEG/WEBP (default 95)

    Returns:
        Raw image bytes
    """
    buffer = io.BytesIO()

    if format.upper() == "PNG":
        # PNG is lossless — no quality setting
        image.save(buffer, format="PNG")
    else:
        image.save(buffer, format=format, quality=quality)

    return buffer.getvalue()


def image_to_base64(
    image   : Image.Image,
    format  : str = "JPEG",
    quality : int = JPEG_QUALITY,
) -> str:
    """
    Encodes a PIL Image as a base64 string for JSON API responses.

    Base64 encodes binary image data as ASCII text so it can be embedded
    directly in JSON without needing a separate file transfer.

    Usage in API response:
        {"image": "data:image/jpeg;base64,/9j/4AAQ..."}

    Args:
        image   : PIL.Image to encode
        format  : Image format for encoding (default "JPEG")
        quality : JPEG quality (default 95)

    Returns:
        Base64 encoded string (without data URI prefix)
    """
    raw_bytes = image_to_bytes(image, format=format, quality=quality)
    return base64.b64encode(raw_bytes).decode("utf-8")


def base64_to_image(b64_string: str) -> Image.Image:
    """
    Decodes a base64 string back to a PIL Image.
    Inverse of image_to_base64().

    Handles both raw base64 and data URI format:
        "data:image/jpeg;base64,/9j/4AAQ..."  ← strips prefix automatically
        "/9j/4AAQ..."                          ← raw base64, used directly

    Args:
        b64_string: Base64 encoded image string

    Returns:
        PIL.Image in RGB mode
    """
    # Strip data URI prefix if present: "data:image/jpeg;base64,"
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]

    raw_bytes = base64.b64decode(b64_string)
    return load_image(raw_bytes)


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """
    Converts PIL Image to numpy array (H, W, 3) uint8.
    Used when passing images to SAM2 which expects numpy input.

    Args:
        image: PIL.Image in RGB mode

    Returns:
        numpy array shape (H, W, 3), dtype uint8, values 0-255
    """
    return np.array(image.convert("RGB"))


def numpy_to_pil(array: np.ndarray) -> Image.Image:
    """
    Converts numpy array back to PIL Image.
    Inverse of pil_to_numpy().

    Args:
        array: numpy array shape (H, W, 3), dtype uint8

    Returns:
        PIL.Image in RGB mode
    """
    return Image.fromarray(array.astype(np.uint8)).convert("RGB")


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline Helper
# ══════════════════════════════════════════════════════════════════════════════

def prepare_image(
    source   : Union[str, Path, bytes, io.BytesIO],
    max_size : int = MAX_IMAGE_SIZE,
) -> Tuple[Image.Image, dict]:
    """
    Full preparation pipeline for any image entering WorkspaceVision.
    Combines load → validate → resize into one single call.

    This is the ONLY function the perception pipeline needs to call.
    Everything else in this file is called internally by this function.

    Args:
        source   : File path, bytes, or BytesIO stream
        max_size : Max dimension before resizing (default 1024)

    Returns:
        Tuple:
            - image : Ready-to-use PIL.Image (RGB, resized, validated)
            - meta  : Dict with original_size, final_size, was_resized

    Raises:
        ValueError: If image fails validation
    """

    # Step 1: Load from whatever source type was given
    image         = load_image(source)
    original_size = image.size  # (width, height) before resize

    # Step 2: Validate — raise immediately if image is unusable
    is_valid, message = validate_image(image)
    if not is_valid:
        raise ValueError(f"[image_utils] Image validation failed: {message}")

    # Step 3: Resize if needed
    image      = resize_image(image, max_size=max_size)
    final_size = image.size

    # Step 4: Return image + metadata for logging / API response
    meta = {
        "original_size" : original_size,
        "final_size"    : final_size,
        "was_resized"   : original_size != final_size,
    }

    logger.info(
        f"[image_utils] Prepared image: "
        f"{original_size} → {final_size} "
        f"({'resized' if meta['was_resized'] else 'no resize'})"
    )

    return image, meta


# ══════════════════════════════════════════════════════════════════════════════
# Segmentation Visualisation                        ← ADDED SECTION
# ══════════════════════════════════════════════════════════════════════════════

def draw_segmentation_masks(
    image        : Image.Image,
    seg_results  : list,           # List[SegmentationResult] from segmenter.py
    alpha        : float = 0.45,   # mask fill opacity — 0=invisible, 1=solid
    draw_contour : bool  = True,   # draw solid border around each mask
    draw_boxes   : bool  = True,   # draw bounding boxes around each object
) -> Image.Image:
    """
    Overlays all SAM2 segmentation masks onto a PIL image.
    Each object gets a unique colour from MASK_COLOURS.
    Returns a NEW image — the original is never modified.

    Args:
        image        : PIL.Image RGB — the workspace image (already resized)
        seg_results  : List of SegmentationResult objects from segmenter.py
                       Each must have .mask (HxW bool ndarray), .label, .confidence
        alpha        : Colour fill transparency (default 0.45 = 45% opaque)
        draw_contour : Whether to draw a solid coloured border (default True)

    Returns:
        PIL.Image RGB with all masks blended in
    """

    # Work in float32 numpy — PIL cannot blend per-pixel efficiently
    output = np.array(image.convert("RGB"), dtype=np.float32)
    h, w   = output.shape[:2]

    for i, seg in enumerate(seg_results):

        # Support both dataclass objects (SegmentationResult) and plain dicts
        if isinstance(seg, dict):
            mask  = seg.get("mask")
            label = seg.get("label",      "?")
            score = seg.get("confidence", 0.0)
            box   = seg.get("box")
        else:
        # SegmentationResult dataclass — access attributes directly
            mask  = getattr(seg, "mask",       None)
            label = getattr(seg, "label",      "?")
            score = getattr(seg, "confidence", 0.0)
            box   = getattr(seg, "box",        None)

        # Skip if mask is absent
        if mask is None:
            logger.warning(f"[draw_segmentation_masks] No mask for '{label}', skipping.")
            continue

        # Ensure binary uint8 (SAM2 returns bool arrays)
        mask = np.array(mask, dtype=np.uint8)
        mask = (mask > 0).astype(np.uint8)

        # Skip completely empty masks — nothing to draw
        if mask.sum() == 0:
            continue

        # Resize mask if SAM2 resolution differs from display image resolution
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        colour    = MASK_COLOURS[i % len(MASK_COLOURS)]   # RGB tuple
        mask_bool = mask.astype(bool)

        # ── Semi-transparent colour fill ──────────────────────────────────
        # Linear blend: output = (1-alpha)*original + alpha*colour
        output[mask_bool] = (
            (1.0 - alpha) * output[mask_bool]
            + alpha * np.array(colour, dtype=np.float32)
        )

        # ── Solid contour border ──────────────────────────────────────────
        if draw_contour:
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            # cv2.drawContours expects BGR — convert from RGB
            bgr          = (colour[2], colour[1], colour[0])
            output_uint8 = np.clip(output, 0, 255).astype(np.uint8)
            cv2.drawContours(output_uint8, contours, -1, bgr, thickness=2)
            output = output_uint8.astype(np.float32)

        # ── Label text at mask centroid ───────────────────────────────────
        ys, xs = np.where(mask_bool)
        if len(xs) > 0:
            cx = int(xs.mean())
            cy = int(ys.mean())
            text         = f"{label} {score:.0%}"
            output_uint8 = np.clip(output, 0, 255).astype(np.uint8)
            # White text with dark outline for readability on any background
            cv2.putText(
                output_uint8, text,
                (max(0, cx - 30), max(15, cy)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 0), 3, cv2.LINE_AA,      # dark outline
            )
            cv2.putText(
                output_uint8, text,
                (max(0, cx - 30), max(15, cy)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA, # white fill
            )
            output = output_uint8.astype(np.float32)

        # ── Bounding box rectangle ─────────────────────────────────────
        if draw_boxes and box and len(box) == 4:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            bgr          = (colour[2], colour[1], colour[0])
            output_uint8 = np.clip(output, 0, 255).astype(np.uint8)
            cv2.rectangle(output_uint8, (x1, y1), (x2, y2), bgr, thickness=2)
            output = output_uint8.astype(np.float32)

    logger.info(f"[draw_segmentation_masks] Drew {len(seg_results)} masks.")
    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8))


# ══════════════════════════════════════════════════════════════════════════════
# Detection-only Visualisation (no SAM2 masks)
# ══════════════════════════════════════════════════════════════════════════════

def draw_detection_boxes(
    image      : Image.Image,
    detections : list,          # List[Detection] from detector.py
) -> Image.Image:
    """
    Draws bounding boxes + labels on a PIL image using detection results.
    Used when segmentation is skipped (e.g. video frames).

    Args:
        image      : PIL.Image RGB
        detections : List of Detection objects (must have .box, .label, .confidence)

    Returns:
        PIL.Image RGB with boxes and labels drawn
    """
    output = np.array(image.convert("RGB"), dtype=np.uint8).copy()

    for i, det in enumerate(detections):
        if isinstance(det, dict):
            box   = det.get("box", [])
            label = det.get("label", "?")
            score = det.get("confidence", 0.0)
        else:
            box   = getattr(det, "box", [])
            label = getattr(det, "label", "?")
            score = getattr(det, "confidence", 0.0)

        if not box or len(box) != 4:
            continue

        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        colour = MASK_COLOURS[i % len(MASK_COLOURS)]
        bgr    = (colour[2], colour[1], colour[0])

        # Draw box
        cv2.rectangle(output, (x1, y1), (x2, y2), bgr, thickness=2)

        # Draw label with background
        text = f"{label} {score:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(output, (x1, y1 - th - 8), (x1 + tw + 4, y1), bgr, -1)
        cv2.putText(
            output, text, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 1, cv2.LINE_AA,
        )

    logger.info(f"[draw_detection_boxes] Drew {len(detections)} boxes.")
    return Image.fromarray(output)
