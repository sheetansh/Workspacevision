# backend/config.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE: Central configuration for the entire backend.
#          Every other backend module imports settings FROM HERE.
#          This means if you ever change a setting, you change it in ONE place.
#
# HOW IT WORKS:
#   - pydantic-settings reads your .env file automatically
#   - Each variable defined below maps to one line in your .env file
#   - If a .env variable is missing, the default value is used instead
#   - Type annotations (str, int, float) make sure the values are the right type
#
# WHAT IS PYDANTIC?
#   Pydantic is a Python library that validates data. Think of it as a
#   "strict form": if you say a field must be an integer and someone gives
#   you a string, Pydantic throws a clear error instead of silently failing.
#
# USAGE IN OTHER FILES:
#   from config import settings
#   print(settings.OPENAI_API_KEY)
#   print(settings.DEVICE)
# ─────────────────────────────────────────────────────────────────────────────

import torch                               # PyTorch: used to detect GPU availability
from pydantic_settings import BaseSettings # Reads .env file and validates all values
from pydantic import Field                 # Field() lets us add default values + descriptions
from pathlib import Path                   # Path: cross-platform file path handling (/ works on Windows too)
from functools import lru_cache            # lru_cache: caches the result of a function so it only runs once


class Settings(BaseSettings):
    """
    Settings class: All configuration lives here as class attributes.
    pydantic_settings automatically reads matching variable names from .env file.
    
    Example: OPENAI_API_KEY=sk-xxx in .env → settings.OPENAI_API_KEY = "sk-xxx"
    """

    # ── OpenAI Settings ───────────────────────────────────────────────────────
    OPENAI_API_KEY: str = Field(
        default="",
        description="Your OpenAI API key. Get it from https://platform.openai.com/api-keys"
    )
    OPENAI_CHAT_MODEL: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model for chatbot responses"
    )
    OPENAI_EMBEDDING_MODEL: str = Field(
        default="text-embedding-3-small",
        description="OpenAI model for converting text to vectors (used in RAG)"
    )

    # ── Backend Server Settings ───────────────────────────────────────────────
    BACKEND_HOST: str = Field(
        default="0.0.0.0",   # 0.0.0.0 means: accept connections from any machine on the network
        description="Host address the FastAPI server binds to"
    )
    BACKEND_PORT: int = Field(
        default=8000,
        description="Port the FastAPI server listens on"
    )
    BACKEND_URL: str = Field(
        default="http://localhost:8000",
        description="URL the Streamlit frontend uses to reach this backend"
    )

    # ── Model Identifiers (HuggingFace model IDs) ─────────────────────────────
    GROUNDING_DINO_MODEL: str = Field(
        default="IDEA-Research/grounding-dino-tiny",
        description="HuggingFace model ID for Grounding DINO (object detector)"
    )
    SAM2_MODEL: str = Field(
        default="facebook/sam2-hiera-small",
        description="HuggingFace model ID for SAM2 (segmentation model)"
    )
    BLIP_MODEL: str = Field(
        default     = "Salesforce/blip2-opt-2.7b",
        description = "HuggingFace model ID for BLIP captioner. "
                      "Use 'blip2-opt-2.7b' for BLIP-2 or 'blip-image-captioning-large' for BLIP-1"
    )

    # ── Hardware Settings ─────────────────────────────────────────────────────
    DEVICE: str = Field(
        default="auto",
        description="'cuda' for GPU, 'cpu' for CPU, 'auto' to detect automatically"
    )
    MODEL_PRECISION: str = Field(
        default="float16",
        description="'float16' = faster/less VRAM, 'float32' = slower/more accurate"
    )

    # ── ChromaDB Settings ─────────────────────────────────────────────────────
    CHROMA_DB_PATH: str = Field(
        default="./backend/knowledge/chroma_store",
        description="Folder path where ChromaDB stores embeddings on disk"
    )

    # ── MLflow Settings ───────────────────────────────────────────────────────
    MLFLOW_TRACKING_URI: str = Field(
        default="./mlops/tracking",
        description="Folder where MLflow stores experiment logs"
    )
    MLFLOW_EXPERIMENT_NAME: str = Field(
        default="workspacevision_inference",
        description="MLflow experiment name (groups all runs together)"
    )

    # ── Application Behaviour Settings ────────────────────────────────────────
    MAX_IMAGE_SIZE: int = Field(
        default=1024,
        description="Max image dimension in pixels before resizing"
    )
    VIDEO_FPS_PROCESS: int = Field(
        default=1,
        description="Number of video frames per second to analyze"
    )
    MAX_DETECTIONS: int = Field(
        default=10,
        description="Maximum number of objects to detect per image"
    )
    DETECTION_CONFIDENCE_THRESHOLD: float = Field(
        default=0.3,
        description="Minimum confidence score to accept a detection (0.0 to 1.0)"
    )
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging verbosity: DEBUG | INFO | WARNING | ERROR"
    )

    # ── Pydantic Settings Configuration ───────────────────────────────────────
    class Config:
        """
        This inner class tells pydantic_settings WHERE to find the .env file.
        env_file=".env" means: look for a file named ".env" in the working directory.
        case_sensitive=False means: OPENAI_API_KEY and openai_api_key both work.
        extra="ignore" means: if .env has extra variables we don't define here,
                               ignore them instead of raising an error.
        """
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


    def get_torch_device(self) -> torch.device:
        """
        Resolves the correct PyTorch device based on DEVICE setting.
        
        Returns a torch.device object:
          - torch.device("cuda") → use NVIDIA GPU
          - torch.device("cpu")  → use CPU only
        
        If DEVICE="auto":
          - Checks if CUDA (NVIDIA GPU) is available using PyTorch
          - Falls back to CPU if no GPU found
        
        WHY torch.device()?
          All PyTorch models and tensors must be on the SAME device.
          When you load a model with .to(device), it moves to that device.
          This function ensures every module uses the same device consistently.
        """
        if self.DEVICE == "auto":
            # torch.cuda.is_available() returns True if an NVIDIA GPU with CUDA is found
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            # Use whatever the user explicitly set in .env
            device_str = self.DEVICE

        device = torch.device(device_str)

        # Print which device was selected (visible in backend startup logs)
        print(f"[WorkspaceVision] 🖥️  Device resolved: {device}")
        if device_str == "cuda":
            # torch.cuda.get_device_name(0) returns the GPU name, e.g. "NVIDIA GeForce RTX 3080 Ti"
            print(f"[WorkspaceVision] 🎮  GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("[WorkspaceVision] ⚠️  No GPU detected — running on CPU (slower)")

        return device


    def get_torch_dtype(self):
        """
            Returns the correct PyTorch data type for model weights.
    
            torch.float16 = 16-bit floats → uses ~50% less GPU memory, slightly less precise
            torch.float32 = 32-bit floats → uses more GPU memory, full precision
    
            NOTE: We check DEVICE setting directly here instead of calling
            get_torch_device() to avoid triggering the device print message twice.
            CPU does not benefit from float16 so we always return float32 for CPU.
        """
    # Resolve device string directly (no print side effects)
        if self.DEVICE == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_str = self.DEVICE

    # CPU always uses float32 — float16 gives no benefit on CPU
        if device_str == "cpu":
            return torch.float32

    # GPU: use float16 or float32 based on MODEL_PRECISION setting in .env
        return torch.float16 if self.MODEL_PRECISION == "float16" else torch.float32


# ── Singleton settings instance ───────────────────────────────────────────────
# @lru_cache(maxsize=1) means: the first time get_settings() is called,
# it creates the Settings object. Every call after that returns the SAME object.
# This prevents re-reading the .env file on every import — efficient and consistent.

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns the singleton Settings instance.
    Use this function everywhere instead of creating Settings() directly.
    
    Usage in any backend file:
        from config import get_settings
        settings = get_settings()
        print(settings.OPENAI_API_KEY)
    """
    return Settings()


# ── Convenience: module-level settings object ─────────────────────────────────
# This allows a simpler import style:
#   from config import settings          ← use this everywhere
# Instead of:
#   from config import get_settings; settings = get_settings()

settings = get_settings()
