# backend/schemas/api_models.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE:
#   Pydantic request/response models for all FastAPI endpoints.
#   Defines the exact JSON shape every endpoint sends and receives.
# ─────────────────────────────────────────────────────────────────────────────

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status  : str
    version : str


# ── Model status ──────────────────────────────────────────────────────────────

class ModelStatusResponse(BaseModel):
    models_loaded    : bool
    device           : str
    gpu_name         : Optional[str]  = None
    vram_used_gb     : Optional[float]= None
    vram_total_gb    : Optional[float]= None
    knowledge_chunks : int            = 0


# ── Analysis ──────────────────────────────────────────────────────────────────

class DetectionItem(BaseModel):
    label      : str
    confidence : float
    box        : List[float]


class CaptionItem(BaseModel):
    label      : Optional[str]  = None
    confidence : Optional[float]= None
    box        : List[float]    = []
    caption    : str


class SceneObjectItem(BaseModel):
    id         : int
    label      : str
    confidence : float
    box        : List[float]
    caption    : str            = ""
    mask_area  : int            = 0
    center     : tuple          = ()
    region     : str            = ""


class RelationshipItem(BaseModel):
    subject    : str
    relation   : str
    target     : str
    sentence   : str


class SceneGraphResponse(BaseModel):
    object_count  : int
    summary       : str
    image_size    : tuple
    objects       : List[SceneObjectItem]   = []
    relationships : List[RelationshipItem]  = []


class AnalysisResponse(BaseModel):
    scene_graph     : Dict[str, Any]  = {}
    scene_caption   : str             = ""
    detections      : List[Dict]      = []
    caption_results : List[Dict]      = []
    image_meta      : Dict[str, Any]  = {}
    timings         : Dict[str, float]= {}
    total_time      : float           = 0.0
    error           : Optional[str]   = None
    annotated_image : str             = ""


class DetectionOnlyResponse(BaseModel):
    detections : List[Dict]
    image_meta : Dict[str, Any]
    timing     : float


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatMessageSchema(BaseModel):
    role    : str   = Field(..., description="'user' or 'assistant'")
    content : str


class ChatRequest(BaseModel):
    question       : str              = Field(..., description="User's question")
    scene_text     : Optional[str]    = Field(None, description="Scene graph text from /analyze/image")
    history        : Optional[List[Dict[str, str]]] = Field(default=[], description="Previous chat turns")
    focused_object : Optional[str]    = Field(None, description="Object label to focus retrieval on")


class ChatResponse(BaseModel):
    answer      : str
    sources     : List[Dict]      = []
    scene_used  : bool            = False
    rag_used    : bool            = False
    token_usage : Dict[str, int]  = {}
    error       : Optional[str]   = None


# ── Knowledge ─────────────────────────────────────────────────────────────────

class KnowledgeIngestResponse(BaseModel):
    filename     : str
    chunks_added : int
    message      : str


class KnowledgeStatsResponse(BaseModel):
    total_chunks    : int
    collection_name : str
    persist_path    : str
    manuals_dir     : str
