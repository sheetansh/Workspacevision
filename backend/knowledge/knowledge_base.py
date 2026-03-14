# backend/knowledge/knowledge_base.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE:
#   Manages the ChromaDB vector store for WorkspaceVision's RAG system.
#   Ingests PDF tool manuals → splits into chunks → embeds → stores in ChromaDB.
#   The retriever.py queries this store at chatbot runtime.
#
# HOW RAG WORKS HERE:
#   1. PDF manuals are split into ~500 token text chunks
#   2. Each chunk is embedded (converted to a vector) via OpenAI
#   3. Vectors are stored in ChromaDB on disk (persists between restarts)
#   4. At query time, the user question is embedded and the closest
#      chunk vectors are retrieved — these become the chatbot's context
#
# WHAT IS CHROMADB?
#   ChromaDB is a local vector database. Think of it as a database where
#   instead of searching by exact keywords, you search by MEANING.
#   "ergonomic chair setup" finds chunks about "posture" and "seating"
#   even if those exact words aren't in the query.
#
# FILE LOCATIONS:
#   PDFs   → backend/knowledge/manuals/*.pdf
#   ChromaDB store → ./backend/knowledge/chroma_store (from config.py)
# ─────────────────────────────────────────────────────────────────────────────


import logging
import hashlib
from pathlib import Path
from typing import List, Optional
import sys

# ChromaDB — local vector database
import chromadb
from chromadb.config import Settings as ChromaSettings

# LangChain utilities for PDF loading and text splitting
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Add backend to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_settings


logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# KnowledgeBase
# ══════════════════════════════════════════════════════════════════════════════

class KnowledgeBase:
    """
    Manages the ChromaDB vector store for tool manual knowledge.

    Responsibilities:
      - Create and persist the ChromaDB collection on disk
      - Ingest PDF manuals (split → embed → store)
      - Expose the LangChain Chroma retriever for querying
      - Track which PDFs are already ingested (avoid duplicates)

    Usage:
        kb = KnowledgeBase()
        kb.ingest_all_manuals()          # one-time setup
        retriever = kb.get_retriever()   # used by retriever.py
    """

    # Name of the ChromaDB collection — all tool manual chunks go here
    COLLECTION_NAME = "workspace_knowledge"

    # Manuals folder relative to this file
    MANUALS_DIR = Path(__file__).parent / "manuals"


    def __init__(self):
        """
        Initialises ChromaDB client and LangChain Chroma wrapper.
        Creates the collection if it doesn't exist yet.
        Does NOT ingest any documents — call ingest_all_manuals() for that.
        """
        self.settings = get_settings()

        # ── ChromaDB persist path ──────────────────────────────────────────
        self.chroma_path = Path(self.settings.CHROMA_DB_PATH)
        self.chroma_path.mkdir(parents=True, exist_ok=True)

        # ── OpenAI embeddings ──────────────────────────────────────────────
        # text-embedding-3-small: fast, cheap, 1536-dim vectors
        # Each chunk of text gets converted to one of these vectors
        self.embeddings = OpenAIEmbeddings(
            model      = self.settings.OPENAI_EMBEDDING_MODEL,
            api_key    = self.settings.OPENAI_API_KEY,
        )

        # ── LangChain Chroma wrapper ───────────────────────────────────────
        # This wraps the raw ChromaDB client with LangChain's interface
        # persist_directory tells it WHERE to save/load the vector store
        self.vectorstore = Chroma(
            collection_name    = self.COLLECTION_NAME,
            embedding_function = self.embeddings,
            persist_directory  = str(self.chroma_path),
        )

        logger.info(
            f"[KnowledgeBase] Initialised. "
            f"Store: {self.chroma_path} | "
            f"Collection: {self.COLLECTION_NAME}"
        )


    # ──────────────────────────────────────────────────────────────────────────
    def ingest_pdf(
        self,
        pdf_path     : Path,
        chunk_size   : int = 500,
        chunk_overlap: int = 50,
    ) -> int:
        """
        Ingests a single PDF file into ChromaDB.

        Steps:
          1. Load PDF → list of LangChain Document objects (one per page)
          2. Split documents into ~500 token chunks with 50 token overlap
          3. Generate a unique ID per chunk using MD5 hash of content
          4. Skip chunks already in the store (idempotent — safe to re-run)
          5. Embed + store new chunks in ChromaDB

        Why overlap?
          50 token overlap means consecutive chunks share 50 tokens.
          This prevents losing context that spans a chunk boundary.
          e.g. "...do not use near water. Always wear [CHUNK BREAK] gloves..."
          With overlap, both chunks contain the safety instruction.

        Args:
            pdf_path      : Path to the PDF file
            chunk_size    : Max tokens per chunk (default 500)
            chunk_overlap : Overlap between consecutive chunks (default 50)

        Returns:
            Number of NEW chunks added to the store (0 if already ingested)
        """

        if not pdf_path.exists():
            logger.error(f"[KnowledgeBase] PDF not found: {pdf_path}")
            return 0

        logger.info(f"[KnowledgeBase] Ingesting: {pdf_path.name}")

        # ── Step 1: Load PDF pages ─────────────────────────────────────────
        # PyPDFLoader extracts text from each page as a Document object
        # Document has .page_content (text) and .metadata (source, page num)
        loader    = PyPDFLoader(str(pdf_path))
        documents = loader.load()
        logger.info(f"[KnowledgeBase] Loaded {len(documents)} pages from {pdf_path.name}")

        # ── Step 2: Split into chunks ──────────────────────────────────────
        # RecursiveCharacterTextSplitter tries to split at:
        #   paragraphs → sentences → words → characters (in that order)
        # This preserves semantic units better than fixed-size splits
        splitter = RecursiveCharacterTextSplitter(
            chunk_size    = chunk_size,
            chunk_overlap = chunk_overlap,
            separators    = ["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        logger.info(f"[KnowledgeBase] Split into {len(chunks)} chunks.")

        # ── Step 3: Add source filename to metadata ────────────────────────
        for chunk in chunks:
            chunk.metadata["source_file"] = pdf_path.name
            chunk.metadata["manual_name"] = pdf_path.stem.replace("_", " ").title()

        # ── Step 4: Generate unique IDs + skip duplicates ──────────────────
        # ID = MD5 hash of (source_file + page_content)
        # This makes ingestion idempotent — re-running won't duplicate chunks
        existing_ids = set(self.vectorstore.get()["ids"])

        new_chunks = []
        new_ids    = []
        seen_ids   = set()

        for chunk in chunks:
            chunk_id = hashlib.md5(
                f"{chunk.metadata['source_file']}:{chunk.page_content}".encode()
            ).hexdigest()

            if chunk_id not in existing_ids and chunk_id not in seen_ids:
                new_chunks.append(chunk)
                new_ids.append(chunk_id)
                seen_ids.add(chunk_id)

        if not new_chunks:
            logger.info(f"[KnowledgeBase] {pdf_path.name} already ingested — skipping.")
            return 0

        # ── Step 5: Embed + store new chunks ──────────────────────────────
        # add_documents() calls OpenAI embeddings API and stores in ChromaDB
        self.vectorstore.add_documents(documents=new_chunks, ids=new_ids)

        logger.info(
            f"[KnowledgeBase] Added {len(new_chunks)} new chunks "
            f"from {pdf_path.name}."
        )
        return len(new_chunks)


    # ──────────────────────────────────────────────────────────────────────────
    def ingest_all_manuals(self) -> dict:
        """
        Ingests all PDF files found in the manuals/ directory.
        Safe to call multiple times — skips already-ingested files.

        Returns:
            Dict mapping filename → chunks added
            e.g. {"drill_manual.pdf": 42, "hammer_manual.pdf": 0}
        """

        if not self.MANUALS_DIR.exists():
            logger.warning(
                f"[KnowledgeBase] Manuals directory not found: {self.MANUALS_DIR}. "
                f"Create it and drop PDF files there."
            )
            self.MANUALS_DIR.mkdir(parents=True, exist_ok=True)
            return {}

        pdf_files = list(self.MANUALS_DIR.glob("*.pdf"))

        if not pdf_files:
            logger.warning(
                f"[KnowledgeBase] No PDF files found in {self.MANUALS_DIR}. "
                f"Drop tool manuals there to enable RAG."
            )
            return {}

        results = {}
        for pdf_path in sorted(pdf_files):
            count = self.ingest_pdf(pdf_path)
            results[pdf_path.name] = count

        total_new = sum(results.values())
        logger.info(
            f"[KnowledgeBase] Ingestion complete. "
            f"{total_new} new chunks across {len(pdf_files)} files."
        )
        return results


    # ──────────────────────────────────────────────────────────────────────────
    def ingest_text(
        self,
        text        : str,
        source_name : str,
        metadata    : Optional[dict] = None,
        chunk_size  : int = 500,
        chunk_overlap: int = 50,
    ) -> int:
        """
        Ingests a plain text string directly — no PDF needed.
        Useful for ingesting scene graph text or dynamic workspace notes.

        Args:
            text         : Raw text to ingest
            source_name  : Label for this text e.g. "scene_analysis_001"
            metadata     : Extra metadata dict to attach to each chunk
            chunk_size   : Max tokens per chunk (default 500)
            chunk_overlap: Overlap between chunks (default 50)

        Returns:
            Number of new chunks added
        """
        from langchain.schema import Document

        splitter = RecursiveCharacterTextSplitter(
            chunk_size    = chunk_size,
            chunk_overlap = chunk_overlap,
        )

        base_meta = {"source_file": source_name, "manual_name": source_name}
        if metadata:
            base_meta.update(metadata)

        chunks = splitter.create_documents(
            texts    = [text],
            metadatas= [base_meta],
        )

        existing_ids = set(self.vectorstore.get()["ids"])
        new_chunks, new_ids = [], []

        for chunk in chunks:
            chunk_id = hashlib.md5(
                f"{source_name}:{chunk.page_content}".encode()
            ).hexdigest()
            if chunk_id not in existing_ids:
                new_chunks.append(chunk)
                new_ids.append(chunk_id)

        if new_chunks:
            self.vectorstore.add_documents(documents=new_chunks, ids=new_ids)

        logger.info(f"[KnowledgeBase] Ingested {len(new_chunks)} chunks from '{source_name}'.")
        return len(new_chunks)


    # ──────────────────────────────────────────────────────────────────────────
    def get_retriever(
        self,
        k                    : int   = 4,
        score_threshold      : float = 0.3,
    ):
        """
        Returns a LangChain retriever for semantic search over the store.
        This is what retriever.py calls.

        Args:
            k                : Number of chunks to retrieve per query (default 4)
            score_threshold  : Minimum similarity score to return a result

        Returns:
            LangChain VectorStoreRetriever
        """
        return self.vectorstore.as_retriever(
            search_type   = "similarity",   # ← just top-k, no broken score filter
            search_kwargs = {"k": k},
)

    # ──────────────────────────────────────────────────────────────────────────
    def get_stats(self) -> dict:
        """
        Returns statistics about the current ChromaDB store.
        Used by the /api/v1/models/status endpoint.

        Returns:
            Dict with total_chunks, collection_name, persist_path
        """
        data = self.vectorstore.get()
        return {
            "total_chunks"    : len(data["ids"]),
            "collection_name" : self.COLLECTION_NAME,
            "persist_path"    : str(self.chroma_path),
            "manuals_dir"     : str(self.MANUALS_DIR),
        }


    # ──────────────────────────────────────────────────────────────────────────
    def clear(self):
        """
        Deletes ALL chunks from the store.
        Used in testing only — irreversible in production.
        """
        self.vectorstore.delete_collection()
        # Reinitialise empty collection
        self.vectorstore = Chroma(
            collection_name    = self.COLLECTION_NAME,
            embedding_function = self.embeddings,
            persist_directory  = str(self.chroma_path),
        )
        logger.warning("[KnowledgeBase] Store cleared.")


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_kb_instance: Optional[KnowledgeBase] = None


def get_knowledge_base() -> KnowledgeBase:
    """
    Returns the singleton KnowledgeBase instance.
    Creates it on first call, reuses on all subsequent calls.
    """
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = KnowledgeBase()
    return _kb_instance
