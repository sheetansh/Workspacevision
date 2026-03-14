# backend/knowledge/retriever.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE:
#   Semantic search layer over the ChromaDB vector store.
#   Takes a plain text query and returns the most relevant knowledge chunks.
#   These chunks become the "context" in the RAG chatbot prompt.
#
# HOW RAG RETRIEVAL WORKS:
#   1. User question → embed with OpenAI → query vector
#   2. ChromaDB compares query vector against all stored chunk vectors
#   3. Returns top-k most similar chunks (by cosine similarity)
#   4. Chunks are passed to the chatbot as grounding context
#
# WHY A SEPARATE RETRIEVER FILE?
#   knowledge_base.py manages STORAGE (ingest, persist, clear).
#   retriever.py manages QUERYING (search, rank, format).
#   Clean separation means the chatbot only imports retriever.py — it
#   never touches ChromaDB directly.
#
# USAGE:
#   from knowledge.retriever import Retriever
#   retriever = Retriever()
#   results = retriever.retrieve("how to use an angle grinder safely")
# ─────────────────────────────────────────────────────────────────────────────


import logging
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from knowledge.knowledge_base import get_knowledge_base


logger = logging.getLogger(__name__)


# ── Retrieved chunk data structure ────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """
    Represents one retrieved knowledge chunk from ChromaDB.

    Attributes:
        content     : The raw text of this chunk
        source_file : PDF filename this chunk came from e.g. "drill_manual.pdf"
        manual_name : Human-readable name e.g. "Drill Manual"
        page        : Page number in the source PDF (0-indexed)
        score       : Similarity score 0.0–1.0 (higher = more relevant)
    """
    content     : str
    source_file : str   = ""
    manual_name : str   = ""
    page        : int   = 0
    score       : float = 0.0


    def to_dict(self) -> dict:
        return {
            "content"     : self.content,
            "source_file" : self.source_file,
            "manual_name" : self.manual_name,
            "page"        : self.page,
            "score"       : round(self.score, 4),
        }


    def to_context_string(self) -> str:
        """
        Formats chunk as a labelled context block for the chatbot prompt.

        Example output:
            [Source: Drill Manual, Page 3]
            Always wear safety goggles when operating the drill...
        """
        source_label = self.manual_name or self.source_file or "Knowledge Base"
        page_label   = f", Page {self.page}" if self.page else ""
        return f"[Source: {source_label}{page_label}]\n{self.content}"


# ── Retriever class ───────────────────────────────────────────────────────────

class Retriever:
    """
    Semantic search over the WorkspaceVision knowledge base.

    Usage:
        retriever = Retriever()

        # Basic retrieval
        chunks = retriever.retrieve("angle grinder safety precautions")

        # Retrieval with scene context injected
        chunks = retriever.retrieve(
            query        = "how should I position my monitor?",
            scene_context= "workspace with monitor, chair, desk, keyboard"
        )

        # Format as text block for chatbot prompt
        context_text = retriever.format_context(chunks)
    """

    def __init__(
        self,
        k               : int   = 4,
        score_threshold : float = 0.3,
    ):
        """
        Initialises the retriever using the singleton KnowledgeBase.

        Args:
            k               : Number of chunks to retrieve per query (default 4)
            score_threshold : Minimum similarity score (default 0.3)
                              0.0 = return anything, 1.0 = exact match only
                              0.3 is a good balance for workspace queries
        """
        self.kb              = get_knowledge_base()
        self.k               = k
        self.score_threshold = score_threshold

        logger.info(
            f"[Retriever] Initialised. "
            f"k={k}, score_threshold={score_threshold}"
        )


    # ──────────────────────────────────────────────────────────────────────────
    def retrieve(
        self,
        query         : str,
        scene_context : Optional[str] = None,
        k             : Optional[int] = None,
    ) -> List[RetrievedChunk]:
        """
        Retrieves the most relevant knowledge chunks for a query.

        If scene_context is provided, it is prepended to the query before
        embedding. This steers the vector search toward chunks relevant
        to BOTH the question AND the current workspace scene.

        Example:
            query="how to avoid wrist strain"
            scene_context="workspace with keyboard, mouse, desk"
            → combined: "workspace with keyboard, mouse, desk. how to avoid wrist strain"
            → retrieves ergonomic keyboard/mouse chunks, not generic wrist content

        Args:
            query         : User's question or search string
            scene_context : Optional scene description to contextualise the query
            k             : Override default number of results (optional)

        Returns:
            List of RetrievedChunk sorted by relevance (highest score first)
        """

        if not query.strip():
            logger.warning("[Retriever] Empty query — returning no results.")
            return []

        # ── Build enriched query ───────────────────────────────────────────
        if scene_context:
            # Prepend scene context — improves retrieval relevance significantly
            enriched_query = f"{scene_context.strip()}. {query.strip()}"
        else:
            enriched_query = query.strip()

        logger.debug(f"[Retriever] Query: {enriched_query[:100]}...")

        # ── Get retriever from knowledge base ──────────────────────────────
        top_k      = k or self.k
        lc_retriever = self.kb.get_retriever(
            k               = top_k,
            score_threshold = self.score_threshold,
        )

        # ── Run semantic search ────────────────────────────────────────────
        try:
            docs = lc_retriever.invoke(enriched_query)
        except Exception as e:
            logger.error(f"[Retriever] Search failed: {e}")
            return []

        # ── Convert LangChain Documents → RetrievedChunk objects ──────────
        chunks = []
        for doc in docs:
            meta = doc.metadata or {}
            chunk = RetrievedChunk(
                content     = doc.page_content,
                source_file = meta.get("source_file", ""),
                manual_name = meta.get("manual_name", ""),
                page        = meta.get("page", 0),
                score       = meta.get("score", 0.0),
            )
            chunks.append(chunk)

        logger.info(f"[Retriever] Retrieved {len(chunks)} chunks for query.")
        return chunks


    # ──────────────────────────────────────────────────────────────────────────
    def retrieve_for_object(
        self,
        object_label  : str,
        user_question : str,
        scene_context : Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """
        Retrieves knowledge relevant to a specific detected object + question.

        Builds a focused query by combining the object label with the user
        question — improves precision when the user asks about a specific item.

        Example:
            object_label="angle grinder", user_question="is it safe to use?"
            → query: "angle grinder is it safe to use?"

        Args:
            object_label  : Detected object label e.g. "angle grinder"
            user_question : User's question about that object
            scene_context : Optional full scene description

        Returns:
            List of RetrievedChunk relevant to the object + question
        """
        focused_query = f"{object_label} {user_question}".strip()
        return self.retrieve(
            query         = focused_query,
            scene_context = scene_context,
        )


    # ──────────────────────────────────────────────────────────────────────────
    def format_context(
        self,
        chunks        : List[RetrievedChunk],
        max_chars     : int = 2000,
    ) -> str:
        """
        Formats retrieved chunks into a single context block for the chatbot.

        Concatenates chunk context strings with separators.
        Truncates total length to max_chars to stay within LLM token limits.

        Args:
            chunks    : List of RetrievedChunk from retrieve()
            max_chars : Maximum total characters in output (default 2000)

        Returns:
            Formatted context string ready to inject into chatbot prompt.
            Returns empty string if no chunks provided.
        """

        if not chunks:
            return ""

        sections   = [chunk.to_context_string() for chunk in chunks]
        full_text  = "\n\n---\n\n".join(sections)

        # Truncate if over limit — add ellipsis to signal truncation
        if len(full_text) > max_chars:
            full_text = full_text[:max_chars] + "\n\n[... context truncated ...]"

        return full_text


    # ──────────────────────────────────────────────────────────────────────────
    def get_stats(self) -> dict:
        """Returns knowledge base stats — passed through from KnowledgeBase."""
        return self.kb.get_stats()
