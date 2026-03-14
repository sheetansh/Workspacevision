# backend/main.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE:
#   FastAPI application entry point for WorkspaceVision.
#   Registers all API routers, handles startup/shutdown events,
#   and exposes all HTTP endpoints the frontend calls.
#
# STARTUP SEQUENCE:
#   1. Load config from .env
#   2. Load all AI models (DINO + SAM2 + BLIP) — once, on startup
#   3. Initialise knowledge base (ChromaDB)
#   4. Register all routers
#   5. Server ready to accept requests
#
# ENDPOINT SUMMARY:
#   GET  /health                    → liveness check
#   GET  /api/v1/models/status      → GPU + model status
#   POST /api/v1/analyze/image      → full pipeline (detect+segment+caption)
#   POST /api/v1/analyze/detect     → detection only (fast)
#   POST /api/v1/chat               → chatbot question answering
#   POST /api/v1/knowledge/ingest   → upload PDF manual
#   GET  /api/v1/knowledge/stats    → ChromaDB statistics
#
# RUN:
#   uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
# ─────────────────────────────────────────────────────────────────────────────


import io
import logging
import time
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Optional

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from config import get_settings
from models.model_loader import get_models
from pipelines.perception_pipeline import PerceptionPipeline
from knowledge.knowledge_base import get_knowledge_base
from chatbot.chatbot import get_chatbot, ChatMessage
from mlops.tracker import get_tracker
from schemas.api_models import (
    HealthResponse,
    ModelStatusResponse,
    AnalysisResponse,
    DetectionOnlyResponse,
    ChatRequest,
    ChatResponse,
    KnowledgeIngestResponse,
    KnowledgeStatsResponse,
)
from pipelines.video_pipeline import VideoPipeline

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()


# ══════════════════════════════════════════════════════════════════════════════
# Application lifespan — startup + shutdown
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs ONCE at startup before accepting requests, and once at shutdown.

    WHY LIFESPAN INSTEAD OF @app.on_event("startup")?
      lifespan() is the modern FastAPI pattern (v0.93+).
      on_event is deprecated. lifespan uses async context manager:
        code before yield → startup
        code after yield  → shutdown
    """

    # ── STARTUP ───────────────────────────────────────────────────────────
    logger.info("="*55)
    logger.info("  WorkspaceVision Backend — Starting Up")
    logger.info("="*55)

    # Step 1: Load all AI models — blocks until all 3 are loaded
    logger.info("[Startup] Loading AI models...")
    get_models()
    logger.info("[Startup] All models loaded.")

    # Step 2: Initialise knowledge base (ChromaDB)
    logger.info("[Startup] Initialising knowledge base...")
    kb = get_knowledge_base()
    kb.ingest_all_manuals()   # ingests any new PDFs in manuals/ folder
    stats = kb.get_stats()
    logger.info(f"[Startup] Knowledge base ready. {stats['total_chunks']} chunks.")

    # Step 3: Initialise pipeline + chatbot singletons
    logger.info("[Startup] Warming up pipeline and chatbot...")
    app.state.pipeline = PerceptionPipeline()
    app.state.chatbot  = get_chatbot()

    # Step 4: Initialise MLflow tracker
    logger.info("[Startup] Initialising MLflow tracker...")
    app.state.tracker  = get_tracker()
    logger.info("[Startup] Ready to serve requests.")
    logger.info("="*55)

    yield   # ← server runs here, handling requests

    # ── SHUTDOWN ──────────────────────────────────────────────────────────
    logger.info("[Shutdown] WorkspaceVision shutting down.")


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI app
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title       = "WorkspaceVision API",
    description = "AI-powered workspace analysis: object detection, segmentation, captioning, and RAG chatbot.",
    version     = "1.0.0",
    lifespan    = lifespan,
)

# ── CORS middleware ────────────────────────────────────────────────────────────
# Allows the Streamlit frontend (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # restrict to frontend URL in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# Dependencies
# ══════════════════════════════════════════════════════════════════════════════

def get_pipeline() -> PerceptionPipeline:
    """FastAPI dependency — returns the singleton pipeline from app state."""
    return app.state.pipeline


def get_chatbot_dep():
    """FastAPI dependency — returns the singleton chatbot from app state."""
    return app.state.chatbot


# ══════════════════════════════════════════════════════════════════════════════
# Health check
# ══════════════════════════════════════════════════════════════════════════════

@app.get(
    "/health",
    response_model = HealthResponse,
    tags           = ["Health"],
    summary        = "Server liveness check",
)
async def health_check():
    """
    Returns 200 OK if the server is running.
    Used by Docker health checks and load balancers.
    """
    return HealthResponse(
        status  = "ok",
        version = app.version,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Model status
# ══════════════════════════════════════════════════════════════════════════════

@app.get(
    "/api/v1/models/status",
    response_model = ModelStatusResponse,
    tags           = ["Models"],
    summary        = "GPU and model load status",
)
async def model_status():
    """
    Returns loaded model names, GPU name, and current VRAM usage.
    Used by the Streamlit sidebar to show system status.
    """
    registry    = get_models()
    device_info = registry.get_device_info()
    kb_stats    = get_knowledge_base().get_stats()

    return ModelStatusResponse(
        models_loaded     = registry.is_ready(),
        device            = device_info["device"],
        gpu_name          = device_info["gpu_name"],
        vram_used_gb      = device_info["vram_used_gb"],
        vram_total_gb     = device_info["vram_total_gb"],
        knowledge_chunks  = kb_stats["total_chunks"],
    )


# ══════════════════════════════════════════════════════════════════════════════
# Image analysis — full pipeline
# ══════════════════════════════════════════════════════════════════════════════

@app.post(
    "/api/v1/analyze/image",
    response_model = AnalysisResponse,
    tags           = ["Analysis"],
    summary        = "Full pipeline: detect + segment + caption + scene graph",
)
async def analyze_image(
    file                 : UploadFile = File(..., description="Workspace image (JPEG/PNG)"),
    confidence_threshold : float      = Form(default=0.3,  description="Min detection confidence 0.0–1.0"),
    max_detections       : int        = Form(default=10,   description="Max objects to detect"),
    skip_segmentation    : bool       = Form(default=False,description="Skip SAM2 to save ~300ms"),
    labels               : Optional[str] = Form(default=None, description="Comma-separated object labels. Leave empty for workspace defaults"),
    pipeline             : PerceptionPipeline = Depends(get_pipeline),
):
    """
    Runs the full WorkspaceVision pipeline on an uploaded image.

    Steps: validate → detect (DINO) → segment (SAM2) → caption (BLIP) → scene graph

    Returns the complete AnalysisResult including scene graph text for RAG.
    """

    # ── Validate file type ─────────────────────────────────────────────────
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(
            status_code = 415,
            detail      = f"Unsupported file type: {file.content_type}. Use JPEG, PNG, or WebP."
        )

    # ── Read image bytes ───────────────────────────────────────────────────
    image_bytes = await file.read()

    # ── Parse optional labels ──────────────────────────────────────────────
    # Labels come as comma-separated string from multipart form
    # e.g. "chair,desk,monitor" → ["chair", "desk", "monitor"]
    label_list = None
    if labels:
        label_list = [l.strip() for l in labels.split(",") if l.strip()]

    # ── Run pipeline ───────────────────────────────────────────────────────
    t0     = time.time()
    result = pipeline.analyze(
        source               = image_bytes,
        labels               = label_list,
        confidence_threshold = confidence_threshold,
        max_detections       = max_detections,
        skip_segmentation    = skip_segmentation,
    )
    logger.info(f"[/analyze/image] Done in {time.time()-t0:.2f}s | {result.get_object_count()} objects")

    # ── Log to MLflow ──────────────────────────────────────────────────────
    try:
        tracker = app.state.tracker
        with tracker.run_context(run_name="image_analysis"):
            tracker.log_inference(result, source="api")
    except Exception as e:
        logger.warning(f"[MLflow] Tracking failed (non-fatal): {e}")

    # ── Raise on fatal error ───────────────────────────────────────────────
    if result.error and result.get_object_count() == 0:
        raise HTTPException(status_code=500, detail=result.error)

    return result.to_dict()


# ══════════════════════════════════════════════════════════════════════════════
# Detection only — fast endpoint
# ══════════════════════════════════════════════════════════════════════════════

@app.post(
    "/api/v1/analyze/detect",
    response_model = DetectionOnlyResponse,
    tags           = ["Analysis"],
    summary        = "Detection only — fast, no segmentation or captioning",
)
async def detect_only(
    file                 : UploadFile = File(...),
    confidence_threshold : float      = Form(default=0.3),
    max_detections       : int        = Form(default=10),
    labels               : Optional[str] = Form(default=None),
    pipeline             : PerceptionPipeline = Depends(get_pipeline),
):
    """
    Runs only Grounding DINO — returns bounding boxes and labels.
    ~4x faster than full pipeline. Use for real-time or preview use cases.
    """

    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Use JPEG, PNG, or WebP.")

    image_bytes = await file.read()
    label_list  = [l.strip() for l in labels.split(",")] if labels else None

    # skip_segmentation=True + we also skip captioning via pipeline flag
    result = pipeline.analyze(
        source               = image_bytes,
        labels               = label_list,
        confidence_threshold = confidence_threshold,
        max_detections       = max_detections,
        skip_segmentation    = True,
    )

    return DetectionOnlyResponse(
        detections = [d.to_dict() for d in result.detections],
        image_meta = result.image_meta,
        timing     = result.timings.get("detection", 0.0),
    )

@app.post(
    "/api/v1/analyze/video",
    tags    = ["Analysis"],
    summary = "Frame-by-frame video analysis",
)
async def analyze_video(
    file                 : UploadFile = File(...),
    frame_limit          : int        = Form(default=10),
    confidence_threshold : float      = Form(default=0.3),
    labels               : str        = Form(default=""),
):
    """
    Processes a video file frame by frame.
    Returns per-frame detection results and a scene summary.
    """
    allowed = {"video/mp4", "video/avi", "video/quicktime", "video/x-msvideo"}
    if file.content_type not in allowed:
        raise HTTPException(
            status_code = 415,
            detail      = f"Unsupported video type: {file.content_type}"
        )

    video_bytes   = await file.read()
    video_pipeline = VideoPipeline()

    # Parse comma-separated labels (empty string = use defaults)
    label_list = [l.strip() for l in labels.split(",") if l.strip()] if labels else None

    result = video_pipeline.analyze_video_bytes(
        video_bytes          = video_bytes,
        filename             = file.filename,
        frame_limit          = frame_limit,
        confidence_threshold = confidence_threshold,
        skip_segmentation    = True,   # always skip for video speed
        labels               = label_list,
    )

    return result.to_dict()

# ══════════════════════════════════════════════════════════════════════════════
# Chatbot
# ══════════════════════════════════════════════════════════════════════════════

@app.post(
    "/api/v1/chat",
    response_model = ChatResponse,
    tags           = ["Chatbot"],
    summary        = "Ask a question about the workspace",
)
async def chat(
    request : ChatRequest,
    chatbot = Depends(get_chatbot_dep),
):
    """
    Answers a user question about their workspace.

    Fuses:
      - Scene graph text (from a previous /analyze/image call)
      - Retrieved knowledge chunks (from ChromaDB)
      - Conversation history (for multi-turn context)

    scene_text and history are optional — the chatbot degrades gracefully.
    """

    # Convert dict history to ChatMessage objects
    history = [
        ChatMessage(role=m["role"], content=m["content"])
        for m in (request.history or [])
    ]

    response = chatbot.chat(
        question       = request.question,
        scene_text     = request.scene_text,
        history        = history,
        focused_object = request.focused_object,
    )

    # ── Log to MLflow ──────────────────────────────────────────────────────
    try:
        tracker = app.state.tracker
        with tracker.run_context(run_name="chat"):
            tracker.log_chat(request.question, response, source="api")
    except Exception as e:
        logger.warning(f"[MLflow] Chat tracking failed (non-fatal): {e}")

    if response.error and not response.answer:
        raise HTTPException(status_code=500, detail=response.error)

    return ChatResponse(**response.to_dict())


# ══════════════════════════════════════════════════════════════════════════════
# Knowledge base
# ══════════════════════════════════════════════════════════════════════════════

@app.post(
    "/api/v1/knowledge/ingest",
    response_model = KnowledgeIngestResponse,
    tags           = ["Knowledge"],
    summary        = "Upload a PDF manual to the knowledge base",
)
async def ingest_pdf(
    file : UploadFile = File(..., description="PDF tool manual to ingest"),
):
    """
    Uploads and ingests a PDF file into ChromaDB.
    Splits into chunks, embeds with OpenAI, and stores permanently.
    Safe to re-upload the same file — duplicates are skipped automatically.
    """

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files are accepted.")

    # Save PDF to manuals directory
    manuals_dir = Path(__file__).parent / "knowledge" / "manuals"
    manuals_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = manuals_dir / file.filename
    content  = await file.read()
    pdf_path.write_bytes(content)

    # Ingest into ChromaDB
    kb     = get_knowledge_base()
    count  = kb.ingest_pdf(pdf_path)

    return KnowledgeIngestResponse(
        filename    = file.filename,
        chunks_added= count,
        message     = f"Ingested {count} new chunks from {file.filename}."
                      if count > 0 else f"{file.filename} already ingested — no new chunks added.",
    )


@app.get(
    "/api/v1/knowledge/stats",
    response_model = KnowledgeStatsResponse,
    tags           = ["Knowledge"],
    summary        = "ChromaDB knowledge base statistics",
)
async def knowledge_stats():
    """Returns total chunks stored, collection name, and persist path."""
    stats = get_knowledge_base().get_stats()
    return KnowledgeStatsResponse(**stats)
