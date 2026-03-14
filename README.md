# WorkspaceVision

An AI-powered workspace analysis application for **DIY enthusiasts and industrial workers**. WorkspaceVision lets users upload images or videos of any workspace, automatically detects and segments every object in the scene, generates natural-language captions for each one, and then lets the user ask questions about any detected object — answered by an LLM grounded in real safety manuals and tool documentation.

---

## 1. Idea & Inspiration

The project originated from the intersection of two interests: Computer Vision applied to practical, real-world workspaces, and the emerging capability of Vision-Language Models (VLMs) to describe and reason about physical environments.

**Core idea:**
- A camera or image upload captures a workspace (workshop, construction site, electronics lab, office)
- The system identifies every object, outlines it, and generates a description
- The user can then select any object and ask questions like *"How do I use this safely?"* or *"What is the correct voltage range for this tool?"*
- The system answers using an LLM grounded in pre-loaded OSHA manuals and tool guides — not just generic knowledge

**Inspiration:**
- [Harvard Edge AI Book](https://github.com/harvard-edge/cs249r_book) — Computer Vision Foundations, Deep Learning Applications, Object Recognition and Detection
- The "Awesome Vision-Language Models" survey — overview of VLMs across detection, segmentation, and captioning tasks
- MS COCO, ADE20K, Open Images — standard academic benchmarks that shaped model selection
- Classic encoder-decoder image captioning (PyTorch + MS COCO) — conceptual reference for the captioning component

**Target users:** DIY hobbyists working in home workshops, and industrial workers in training scenarios where instant, contextual tool guidance can improve safety.

---

## 2. Planning & Design

### Requirements

Before writing any code, three MVP constraints were set:
1. Keep the dataset minimal — ~20 images, ~15 videos, ~10 PDF manuals is sufficient to demonstrate the pipeline
2. Use pre-trained models only — no custom training required for the MVP
3. Split the system cleanly into four independent layers so each can be developed and tested in isolation

### Architecture — 4 Layers

```
┌──────────────────────────────────────────────────────────────┐
│  Layer 1: UI & Input                                         │
│  Streamlit (3 pages: Image Analysis, Video Analysis, Chat)   │
└────────────────────────┬─────────────────────────────────────┘
                         │ HTTP (REST)
┌────────────────────────▼─────────────────────────────────────┐
│  Layer 2: Perception Pipeline  (FastAPI backend)             │
│  Grounding DINO → SAM2 → BLIP-2 → Scene Graph               │
└────────────────────────┬─────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────┐
│  Layer 3: Knowledge & Chatbot                                │
│  ChromaDB (vector store) + LangChain + GPT-4o-mini (RAG)    │
└────────────────────────┬─────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────┐
│  Layer 4: MLOps & Infrastructure                             │
│  MLflow (tracking) · Docker · docker-compose                 │
└──────────────────────────────────────────────────────────────┘
```

### Model Selection Rationale

| Task | Model chosen | Why |
|---|---|---|
| Object Detection | **Grounding DINO** | Open-vocabulary — detects any object by text prompt, not just COCO classes |
| Segmentation | **SAM2** (Meta) | State-of-the-art mask quality; supports video tracking via mask propagation |
| Captioning | **BLIP-2** (Salesforce) | Combines a vision encoder with a frozen LLM; strong zero-shot captioning |
| RAG / Q&A | **GPT-4o-mini** via LangChain | Affordable, fast, strong instruction-following for safety-critical answers |
| Vector Store | **ChromaDB** | Lightweight, local, no external service required for MVP |

### Deployment topology

The system was designed for a split-machine setup — GPU-heavy inference runs on a desktop, while a laptop connects via LAN to the Streamlit UI:

```
┌──────────────────┐   LAN (HTTP)   ┌──────────────────────────────┐
│    LAPTOP        │ ◄────────────► │         DESKTOP (GPU)        │
│ Browser / UI     │                │ FastAPI  :8000               │
│ Streamlit :8501  │                │ Grounding DINO, SAM2, BLIP-2 │
│ No GPU needed    │                │ LangChain, ChromaDB, MLflow  │
└──────────────────┘                │ RTX 1080 Ti                  │
                                    └──────────────────────────────┘
```

---

## 3. Implementation — Step by Step

Development was structured into 8 sequential batches, each building on the previous:

### Batch 1 — Environment & Configuration
Set up the project skeleton: `config.py` (Pydantic settings from `.env`), `schemas/api_models.py` (all request/response contracts), and `.env.example` documenting every required variable. This established the data contracts before any model code was written.

### Batch 2 — AI Model Layer
Built the four model wrappers independently:
- `model_loader.py` — singleton registry that loads all three models once at startup and keeps them in GPU memory
- `detector.py` — wraps Grounding DINO; accepts an image + label list, returns bounding boxes, labels, and confidence scores
- `segmenter.py` — wraps SAM2; accepts box prompts per detected object, returns pixel-level masks
- `captioner.py` — wraps BLIP-2; generates a scene-level caption and individual per-object crop captions

### Batch 3 — Perception Pipeline
Assembled the three models into a single orchestrated pipeline:
- `image_utils.py` — resizing, normalization, base64 encoding, bounding box / mask overlay drawing
- `scene_graph.py` — in-memory store of all detected objects (id, label, bounding box, mask, caption, spatial relationships)
- `perception_pipeline.py` — orchestrates the full sequence: validate → detect → segment → caption → build scene graph → annotate image

### Batch 4 — Knowledge Base & Chatbot
Built the RAG (Retrieval-Augmented Generation) layer:
- `knowledge_base.py` — uses LangChain's `PyPDFLoader` to parse PDFs, `RecursiveCharacterTextSplitter` to chunk into ~500-character segments with overlap, and OpenAI embeddings to vectorize; stores everything in ChromaDB
- `retriever.py` — embeds the user's question and runs a similarity search in ChromaDB, returning the top-3 most relevant manual chunks
- `chatbot.py` — builds a LangChain prompt fusing the detected object's metadata, retrieved manual context, conversation history, and user question; calls GPT-4o-mini and returns the answer

### Batch 5 — FastAPI Backend
Exposed all functionality as a REST API in `main.py`:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/api/v1/models/status` | GPU + model load status |
| POST | `/api/v1/analyze/image` | Full pipeline: detect + segment + caption |
| POST | `/api/v1/analyze/detect` | Detection only (4× faster, no segmentation) |
| POST | `/api/v1/analyze/video` | Frame-by-frame video analysis |
| POST | `/api/v1/chat` | RAG chatbot Q&A |
| POST | `/api/v1/knowledge/ingest` | Upload a new PDF manual |
| GET | `/api/v1/knowledge/stats` | ChromaDB collection statistics |

All models are loaded once at startup via FastAPI's `lifespan` context manager — no cold-start per request.

### Batch 6 — MLOps & Video Pipeline
- `mlops/tracker.py` — wraps MLflow; logs inference latency, object counts, model versions, and chat query/response metadata per request
- `pipelines/video_pipeline.py` — extracts frames with OpenCV at a configurable FPS (default 2 FPS); runs the full perception pipeline on frame 0 to initialise SAM2's video predictor, then propagates masks across subsequent frames without re-running Grounding DINO on every frame

### Batch 7 — Streamlit Frontend
Three pages, each communicating with the FastAPI backend via `api_client.py`:
- **Image Analysis** — upload an image, view the annotated result with bounding boxes and masks, browse the object list panel
- **Video Analysis** — upload a video, view per-frame detections and a frame timeline
- **Chatbot** — select a detected object from the scene graph, type a question, receive a grounded answer from the RAG pipeline

### Batch 8 — Containerisation
- `Dockerfile` for the backend (CUDA-enabled base image, model downloads at build time)
- `docker-compose.yml` coordinating backend + frontend services with shared volumes for the knowledge base and MLflow tracking store

---

## 4. How It Works — User Flows

### Image Analysis
```
Upload image
  → Grounding DINO detects objects (bounding boxes + labels)
  → SAM2 generates pixel masks for each object
  → BLIP-2 captions the full scene + each object crop
  → Scene graph built and stored
  → Annotated image returned to UI
```

### Asking About an Object
```
Select object from list → type question
  → Chatbot retrieves object metadata from scene graph
  → ChromaDB similarity search finds relevant manual chunks
  → LangChain prompt fuses: object label + caption + manual context + question
  → GPT-4o-mini generates a grounded, safety-aware answer
```

### Video Analysis
```
Upload video
  → OpenCV extracts frames at 2 FPS
  → Frame 0: full pipeline (detect + segment + caption)
  → Frames 1–N: SAM2 mask propagation (tracking, no re-detection)
  → Annotated video assembled and returned
```

### Adding a New Manual
```
Upload PDF
  → PyPDFLoader parses pages
  → Text split into ~500-char overlapping chunks
  → OpenAI Embeddings vectorize each chunk
  → Stored in ChromaDB (duplicates automatically skipped)
```

---

## 5. Technology Stack

| Category | Technology |
|---|---|
| Language | Python 3.10+ |
| Backend framework | FastAPI 0.115 + Uvicorn |
| Frontend | Streamlit |
| Object detection | Grounding DINO (HuggingFace `transformers`) |
| Segmentation | SAM2 (Meta, `sam2` package) |
| Captioning | BLIP-2 (Salesforce, HuggingFace) |
| LLM | OpenAI GPT-4o-mini via LangChain |
| Vector store | ChromaDB |
| RAG framework | LangChain 0.2 |
| PDF parsing | LangChain PyPDFLoader + pypdf |
| Experiment tracking | MLflow 2.13 |
| Image processing | OpenCV 4.9, Pillow 10 |
| Containerisation | Docker + docker-compose |
| GPU runtime | PyTorch (CUDA), managed via conda |

---

## 6. Setup & Running

### Prerequisites
- Conda environment with PyTorch (CUDA) installed
- OpenAI API key

### Installation
```bash
conda activate workspacevision
pip install git+https://github.com/facebookresearch/sam2.git
pip install -r backend/requirements.txt
```

### Configuration
Copy `.env.example` to `.env` and fill in:
```
OPENAI_API_KEY=your_key_here
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
```

### Run (local)
```bash
# Backend (desktop with GPU)
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Frontend (same machine or laptop on LAN)
streamlit run frontend/app.py
```

### Run (Docker)
```bash
docker compose up --build   # first run — builds images and downloads models
docker compose up           # subsequent runs
docker compose down         # stop
docker compose logs -f backend  # live backend logs
```

### LAN access from a second machine
Set `BACKEND_URL=http://<desktop-ip>:8000` in `.env`, then open `http://<desktop-ip>:8501` in a browser on any machine on the same network. If the connection is blocked, allow the ports through Windows Firewall:
```bash
netsh advfirewall firewall add rule name="WorkspaceVision Backend" dir=in action=allow protocol=TCP localport=8000
netsh advfirewall firewall add rule name="WorkspaceVision Frontend" dir=in action=allow protocol=TCP localport=8501
```

---

## 7. Sources & Resources

| Resource | Role in project |
|---|---|
| [Harvard Edge AI Book — cs249r](https://github.com/harvard-edge/cs249r_book) | Primary inspiration; CV foundations, deep learning applications, object detection theory |
| [Grounding DINO — IDEA Research](https://github.com/IDEA-Research/GroundingDINO) | Open-vocabulary object detector; model weights via HuggingFace |
| [SAM2 — Meta AI Research](https://github.com/facebookresearch/sam2) | Segmentation model + video mask propagation |
| [BLIP-2 — Salesforce](https://huggingface.co/Salesforce/blip2-opt-2.7b) | Vision-language captioning model |
| [LangChain documentation](https://python.langchain.com) | RAG pipeline, PDF loading, prompt templates, ChromaDB integration |
| [ChromaDB](https://www.trychroma.com/) | Local vector database for knowledge chunk storage and retrieval |
| [OpenAI GPT-4o-mini](https://platform.openai.com/docs) | LLM backbone for the chatbot |
| [MLflow](https://mlflow.org/) | Experiment and inference tracking |
| [FastAPI](https://fastapi.tiangolo.com/) | Backend REST API framework |
| [Streamlit](https://streamlit.io/) | Rapid frontend UI development |
| OSHA Hand & Power Tools Safety Guide | Pre-loaded knowledge base manual |
| OSHA Electrical Safety Standards | Pre-loaded knowledge base manual |
| OSHA Construction Safety | Pre-loaded knowledge base manual |
| OSHA Machine Guarding | Pre-loaded knowledge base manual |
| OSHA Woodworking Safety | Pre-loaded knowledge base manual |
| SparkFun & Adafruit Multimeter Guides | Pre-loaded knowledge base manual |
| NIOSH Hand Tools Ergonomics Guide | Pre-loaded knowledge base manual |
| Soldering Safety Guide | Pre-loaded knowledge base manual |
| MS COCO / ADE20K / Open Images | Reference datasets used for model selection research |
| "Awesome Vision-Language Models" survey | Background reading on VLM landscape |

---

## 8 Files and Folders
