# backend/Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# GPU-enabled backend image for WorkspaceVision.
# Uses NVIDIA CUDA base with PyTorch pre-installed + all ML dependencies.
#
# Build:   docker compose build backend
# Run:     docker compose up backend
# ─────────────────────────────────────────────────────────────────────────────

FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# ── System dependencies ──────────────────────────────────────────────────────
# libgl1 + libglib2.0 needed by OpenCV
# git needed for SAM2 install from GitHub
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────────────────────
WORKDIR /app

# ── Install SAM2 from GitHub (must come before requirements) ─────────────────
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/sam2.git

# ── Python dependencies ──────────────────────────────────────────────────────
# Copy requirements first for Docker layer caching —
# code changes won't re-install all packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy backend code ────────────────────────────────────────────────────────
COPY . .

# ── HuggingFace cache directory ──────────────────────────────────────────────
# Models are downloaded on first run and cached in this volume
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

# ── Expose port ──────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ─────────────────────────────────────────────────────────────
# start-period=120s gives models time to download on first run
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Start command ────────────────────────────────────────────────────────────
# No --reload in Docker (no file watching needed in containers)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
