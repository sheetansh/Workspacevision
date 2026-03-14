# backend/models/model_loader.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE:
#   Central model registry for WorkspaceVision.
#   Loads Grounding DINO, SAM2, and BLIP (v1 or v2) ONCE when the backend starts.
#   All other modules call get_models() to reuse the loaded models.
#   BLIP version is auto-detected from BLIP_MODEL in .env:
#     "blip2" in model ID → BLIP-2 path   (e.g. Salesforce/blip2-opt-2.7b)
#     otherwise           → BLIP-1 path   (e.g. Salesforce/blip-image-captioning-large)
# ─────────────────────────────────────────────────────────────────────────────

import sys
import logging
import torch
from pathlib import Path

from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)

# SAM2 — load only SAM2ImagePredictor (from_pretrained handles everything)
try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    logging.warning("[WorkspaceVision] SAM2 not installed.")

# Add backend folder to path so config.py is importable
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_settings

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# ModelRegistry
# ══════════════════════════════════════════════════════════════════════════════

class ModelRegistry:
    """
    Holds all 3 loaded models + processors as attributes.
    Call load_all() once at startup. Then use attributes directly.
    """

    def __init__(self):
        self.dino_model      = None
        self.dino_processor  = None
        self.sam2_predictor  = None
        self.blip_model      = None
        self.blip_processor  = None
        self.blip_version    = None   # "blip1" or "blip2" — set during loading
        self.device          = None
        self.dtype           = None
        self._loaded         = False

    # ──────────────────────────────────────────────────────────────────────────
    def load_all(self):
        """
        Loads all three models in sequence.
        Called ONCE at backend startup via get_models() singleton.
        Fails loudly if any model cannot load.
        """
        settings = get_settings()

        # Resolve device and dtype from config
        self.device = settings.get_torch_device()
        self.dtype  = settings.get_torch_dtype()

        logger.info(f"[WorkspaceVision] Loading on: {self.device} | {self.dtype}")
        print(f"\n{'='*55}")
        print(f"  WorkspaceVision — Loading AI Models")
        print(f"  Device : {self.device}")
        print(f"  Dtype  : {self.dtype}")
        print(f"{'='*55}")

        self._load_grounding_dino(settings)
        self._load_sam2(settings)
        self._load_blip(settings)

        self._loaded = True
        print(f"\n{'='*55}")
        print(f"  All models loaded successfully!")
        print(f"{'='*55}\n")
        logger.info("[WorkspaceVision] All models loaded.")

    # ──────────────────────────────────────────────────────────────────────────
    def _load_grounding_dino(self, settings):
        """
        Loads Grounding DINO open-vocabulary detector from HuggingFace.
        Uses AutoProcessor + AutoModelForZeroShotObjectDetection.
        """
        model_id = settings.GROUNDING_DINO_MODEL
        print(f"\n  [1/3] Loading Grounding DINO: {model_id}")
        logger.info(f"Loading Grounding DINO: {model_id}")

        try:
            self.dino_processor = AutoProcessor.from_pretrained(model_id)
            self.dino_model = (
                AutoModelForZeroShotObjectDetection
                .from_pretrained(model_id)
                .to(self.device)
                .to(self.dtype)
                .eval()
            )
            print(f"  Grounding DINO loaded.")
            logger.info("Grounding DINO loaded.")

        except Exception as e:
            logger.error(f"Failed to load Grounding DINO: {e}")
            raise RuntimeError(f"[WorkspaceVision] DINO load failed: {e}") from e

    # ──────────────────────────────────────────────────────────────────────────
    def _load_sam2(self, settings):
        """
        Loads SAM2 image predictor via HuggingFace from_pretrained().
        from_pretrained() handles config + checkpoint download automatically.
        """
        model_id = settings.SAM2_MODEL
        print(f"\n  [2/3] Loading SAM2: {model_id}")
        logger.info(f"Loading SAM2: {model_id}")

        if not SAM2_AVAILABLE:
            raise RuntimeError(
                "[WorkspaceVision] SAM2 not installed. "
                "Run: pip install git+https://github.com/facebookresearch/sam2.git"
            )

        try:
            self.sam2_predictor = SAM2ImagePredictor.from_pretrained(
                model_id,
                device=str(self.device),
            )
            print(f"  SAM2 loaded.")
            logger.info("SAM2 loaded.")

        except Exception as e:
            logger.error(f"Failed to load SAM2: {e}")
            raise RuntimeError(f"[WorkspaceVision] SAM2 load failed: {e}") from e

    # ──────────────────────────────────────────────────────────────────────────
    def _load_blip(self, settings):
        """
        Loads either BLIP-1 or BLIP-2 based on the model ID in .env.

        SWITCHING BETWEEN VERSIONS:
          .env BLIP_MODEL=Salesforce/blip2-opt-2.7b       → loads BLIP-2
          .env BLIP_MODEL=Salesforce/blip-image-captioning-large → loads BLIP-1

        Detection logic: if "blip2" is in the model ID → BLIP-2 path.
        Otherwise → BLIP-1 path.

        BLIP-2 NOTES:
          - Manual processor construction avoids processor_config.json crash
            (num_query_tokens key unknown in transformers 4.41.2)
          - low_cpu_mem_usage=True prevents RAM spike during float32→float16
          - ~6 GB VRAM (float16), needs 32GB system RAM to load safely

        BLIP-1 NOTES:
          - Standard from_pretrained() works out of the box
          - ~1.5 GB VRAM (float16), loads on any system
          - Less accurate on complex scenes, more hallucinations

        VRAM budget on GTX 1080 Ti (11GB):
          DINO:       ~1.0 GB
          SAM2-small: ~1.5 GB
          BLIP-1:     ~1.5 GB  (float16)  → Total ~4.0 GB
          BLIP-2:     ~6.0 GB  (float16)  → Total ~8.5 GB
        """
        model_id = settings.BLIP_MODEL
        is_blip2 = "blip2" in model_id.lower()

        version_str = "BLIP-2" if is_blip2 else "BLIP-1"
        self.blip_version = "blip2" if is_blip2 else "blip1"

        print(f"\n  [3/3] Loading {version_str} captioner: {model_id}")
        logger.info(f"Loading {version_str} captioner: {model_id}")

        try:
            if is_blip2:
                self._load_blip2_model(model_id)
            else:
                self._load_blip1_model(model_id)

            print(f"  {version_str} captioner loaded.")
            logger.info(f"{version_str} captioner loaded.")

        except Exception as e:
            logger.error(f"Failed to load {version_str}: {e}")
            raise RuntimeError(f"[WorkspaceVision] {version_str} load failed: {e}") from e

    # ──────────────────────────────────────────────────────────────────────────
    def _load_blip2_model(self, model_id: str):
        """
        BLIP-2 loading path (e.g. Salesforce/blip2-opt-2.7b).

        Manual processor construction bypasses processor_config.json crash.
        low_cpu_mem_usage=True avoids the ~14GB RAM spike that previously
        crashed the system — loads weights shard-by-shard instead of all at once.
        """
        from transformers import (
            Blip2Processor,
            Blip2ForConditionalGeneration,
            BlipImageProcessor,
            GPT2Tokenizer,
        )

        # Build processor manually — avoids processor_config.json crash
        image_processor     = BlipImageProcessor.from_pretrained(model_id)
        tokenizer           = GPT2Tokenizer.from_pretrained(
            model_id,
            use_fast = False,   # avoids tokenizer.json Rust parser error
        )
        self.blip_processor = Blip2Processor(
            image_processor = image_processor,
            tokenizer       = tokenizer,
        )

        # Load model — low_cpu_mem_usage prevents RAM spike during dtype conversion
        self.blip_model = (
            Blip2ForConditionalGeneration
            .from_pretrained(
                model_id,
                torch_dtype        = torch.float16,
                low_cpu_mem_usage  = True,
            )
            .to(self.device)
            .eval()
        )

    # ──────────────────────────────────────────────────────────────────────────
    def _load_blip1_model(self, model_id: str):
        """
        BLIP-1 loading path (e.g. Salesforce/blip-image-captioning-large).

        Standard from_pretrained() — no manual processor construction needed.
        Much lighter: ~1.5 GB VRAM vs ~6 GB for BLIP-2.
        """
        from transformers import BlipProcessor, BlipForConditionalGeneration

        self.blip_processor = BlipProcessor.from_pretrained(model_id)
        self.blip_model = (
            BlipForConditionalGeneration
            .from_pretrained(
                model_id,
                torch_dtype = self.dtype,
            )
            .to(self.device)
            .eval()
        )

    # ──────────────────────────────────────────────────────────────────────────
    def is_ready(self) -> bool:
        """Returns True only after all 3 models loaded successfully."""
        return self._loaded

    def get_device_info(self) -> dict:
        """Returns GPU name and VRAM usage. Used by /api/v1/models/status."""
        info = {
            "device"        : str(self.device),
            "gpu_name"      : None,
            "vram_used_gb"  : None,
            "vram_total_gb" : None,
        }
        if torch.cuda.is_available():
            info["gpu_name"]      = torch.cuda.get_device_name(0)
            info["vram_used_gb"]  = round(torch.cuda.memory_allocated(0) / 1e9, 2)
            info["vram_total_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 2
            )
        return info


# ══════════════════════════════════════════════════════════════════════════════
# Singleton management
# ══════════════════════════════════════════════════════════════════════════════

_model_registry: ModelRegistry | None = None


def get_models() -> ModelRegistry:
    """
    Returns the singleton ModelRegistry.
    Loads all models on the FIRST call only.
    All subsequent calls return the already-loaded registry instantly.
    """
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry()
        _model_registry.load_all()
    return _model_registry


def reset_models():
    """Resets the singleton. Used in testing only."""
    global _model_registry
    _model_registry = None
    logger.info("[WorkspaceVision] Model registry reset.")
