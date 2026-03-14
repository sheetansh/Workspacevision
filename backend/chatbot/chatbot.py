# backend/chatbot/chatbot.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE:
#   LangChain-powered chatbot that answers questions about a workspace.
#   Fuses THREE sources of context into every answer:
#     1. Scene graph  → what objects are detected + their spatial positions
#     2. RAG context  → relevant chunks from tool manuals / knowledge base
#     3. Chat history → previous turns so the bot remembers the conversation
#
# HOW IT WORKS:
#   User question
#     ↓
#   Retriever.retrieve(question, scene_context)  → relevant knowledge chunks
#     ↓
#   Build prompt: system + scene_graph + rag_context + history + question
#     ↓
#   GPT-4o-mini → answer
#     ↓
#   Return ChatResponse with answer + sources used
#
# DESIGN:
#   - Stateless per request: chat history passed in by the caller
#   - Graceful degradation: works with no RAG context, no scene graph
#   - Source attribution: response includes which manuals were used
# ─────────────────────────────────────────────────────────────────────────────


import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import sys

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_settings
from knowledge.retriever import Retriever, RetrievedChunk


logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ChatMessage:
    """
    Represents one message in a conversation turn.

    Attributes:
        role    : "user" or "assistant"
        content : Message text
    """
    role    : str
    content : str


    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class ChatResponse:
    """
    Complete response from one chatbot turn.

    Attributes:
        answer       : The generated answer text
        sources      : List of knowledge chunks used as context
        scene_used   : True if scene graph was injected into the prompt
        rag_used     : True if RAG context was retrieved and used
        token_usage  : Dict with prompt_tokens, completion_tokens, total_tokens
        error        : Error message if something failed (None = success)
    """
    answer      : str
    sources     : List[RetrievedChunk]  = field(default_factory=list)
    scene_used  : bool                  = False
    rag_used    : bool                  = False
    token_usage : Dict[str, int]        = field(default_factory=dict)
    error       : Optional[str]         = None


    def to_dict(self) -> dict:
        return {
            "answer"      : self.answer,
            "sources"     : [s.to_dict() for s in self.sources],
            "scene_used"  : self.scene_used,
            "rag_used"    : self.rag_used,
            "token_usage" : self.token_usage,
            "error"       : self.error,
        }


# ── WorkspaceVision system prompt ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are WorkspaceVision AI, an intelligent assistant that analyzes 
workspaces and answers questions about workspace objects, ergonomics, tool safety, 
and workspace optimization.

You have access to:
1. A scene analysis of the current workspace (detected objects and their positions)
2. Relevant knowledge from tool manuals and ergonomic guidelines
3. The conversation history with the user

Guidelines:
- Be specific and practical — reference actual detected objects when relevant
- If the scene shows a specific object the user asks about, describe its position
- Cite knowledge sources when giving safety or technical advice
- If you don't have enough information, say so clearly — don't guess
- Keep answers concise but complete (2-4 sentences for simple questions)
- For safety-critical questions, always emphasize the most important precaution first"""


# ── Chatbot class ─────────────────────────────────────────────────────────────

class WorkspaceVisionChatbot:
    """
    LangChain chatbot that answers workspace questions using scene + knowledge.

    Usage:
        chatbot = WorkspaceVisionChatbot()

        # With scene graph from perception pipeline
        response = chatbot.chat(
            question     = "Is my monitor at the right height?",
            scene_text   = analysis_result.scene_graph.to_text(),
            history      = [],
        )
        print(response.answer)

        # Follow-up question — pass history to maintain context
        history = [
            ChatMessage(role="user",      content="Is my monitor at the right height?"),
            ChatMessage(role="assistant", content=response.answer),
        ]
        response2 = chatbot.chat(
            question = "What about my chair?",
            scene_text = analysis_result.scene_graph.to_text(),
            history    = history,
        )
    """

    def __init__(self):
        """
        Initialises the LLM and retriever.
        Does NOT load any ML models — only sets up LangChain + OpenAI client.
        """
        self.settings  = get_settings()
        self.retriever = Retriever(k=4, score_threshold=0.2)

        # ChatOpenAI wraps the OpenAI chat completions API
        # temperature=0.3: low randomness → consistent, factual answers
        # max_tokens=512: enough for detailed answers without runaway responses
        self.llm = ChatOpenAI(
            model       = self.settings.OPENAI_CHAT_MODEL,
            api_key     = self.settings.OPENAI_API_KEY,
            temperature = 0.3,
            max_tokens  = 512,
        )

        logger.info(
            f"[Chatbot] Initialised. "
            f"Model: {self.settings.OPENAI_CHAT_MODEL}"
        )


    # ──────────────────────────────────────────────────────────────────────────
    def chat(
        self,
        question      : str,
        scene_text    : Optional[str] = None,
        history       : Optional[List[ChatMessage]] = None,
        focused_object: Optional[str] = None,
    ) -> ChatResponse:
        """
        Generates a response to a user question about their workspace.

        Args:
            question       : User's question e.g. "Is my monitor too high?"
            scene_text     : scene_graph.to_text() from the perception pipeline
                             If None, chatbot answers from knowledge only
            history        : Previous ChatMessage turns for context continuity
            focused_object : Specific detected object label to focus retrieval on
                             e.g. "monitor" — improves RAG precision

        Returns:
            ChatResponse with answer, sources, and metadata
        """

        if not question.strip():
            return ChatResponse(
                answer="Please ask a question about your workspace.",
                error ="Empty question provided."
            )

        history = history or []

        try:
            # ── Step 1: Retrieve relevant knowledge ───────────────────────
            if focused_object:
                chunks = self.retriever.retrieve_for_object(
                    object_label  = focused_object,
                    user_question = question,
                    scene_context = scene_text,
                )
            else:
                chunks = self.retriever.retrieve(
                    query         = question,
                    scene_context = scene_text,
                )

            rag_context = self.retriever.format_context(chunks, max_chars=2000)
            rag_used    = bool(rag_context)

            # ── Step 2: Build system message with all context ─────────────
            system_content = SYSTEM_PROMPT

            # Inject scene graph if available
            if scene_text:
                system_content += f"\n\n--- CURRENT WORKSPACE SCENE ---\n{scene_text}"

            # Inject RAG context if retrieved
            if rag_context:
                system_content += f"\n\n--- RELEVANT KNOWLEDGE ---\n{rag_context}"

            # ── Step 3: Build message list ─────────────────────────────────
            # LangChain message format:
            #   SystemMessage  → sets AI role + injects all context
            #   HumanMessage   → user turn
            #   AIMessage      → assistant turn (from history)
            messages = [SystemMessage(content=system_content)]

            # Add conversation history — alternate human/AI turns
            for msg in history:
                if msg.role == "user":
                    messages.append(HumanMessage(content=msg.content))
                elif msg.role == "assistant":
                    messages.append(AIMessage(content=msg.content))

            # Add current question as final human message
            messages.append(HumanMessage(content=question))

            # ── Step 4: Call LLM ───────────────────────────────────────────
            logger.debug(f"[Chatbot] Sending {len(messages)} messages to LLM.")
            llm_response = self.llm.invoke(messages)

            # ── Step 5: Extract answer + token usage ──────────────────────
            answer = llm_response.content.strip()

            # Extract token usage from response metadata if available
            token_usage = {}
            if hasattr(llm_response, "response_metadata"):
                usage = llm_response.response_metadata.get("token_usage", {})
                token_usage = {
                    "prompt_tokens"     : usage.get("prompt_tokens", 0),
                    "completion_tokens" : usage.get("completion_tokens", 0),
                    "total_tokens"      : usage.get("total_tokens", 0),
                }

            logger.info(
                f"[Chatbot] Answer generated. "
                f"Tokens: {token_usage.get('total_tokens', '?')}"
            )

            return ChatResponse(
                answer      = answer,
                sources     = chunks,
                scene_used  = bool(scene_text),
                rag_used    = rag_used,
                token_usage = token_usage,
            )

        except Exception as e:
            logger.error(f"[Chatbot] Error: {e}", exc_info=True)
            return ChatResponse(
                answer = "I encountered an error processing your question. Please try again.",
                error  = str(e),
            )


    # ──────────────────────────────────────────────────────────────────────────
    def chat_with_analysis(
        self,
        question        : str,
        analysis_result : Any,   # AnalysisResult from perception_pipeline.py
        history         : Optional[List[ChatMessage]] = None,
        focused_object  : Optional[str] = None,
    ) -> ChatResponse:
        """
        Convenience method — accepts AnalysisResult directly from the pipeline.
        Extracts scene_text automatically so callers don't have to.

        Args:
            question        : User's question
            analysis_result : AnalysisResult from PerceptionPipeline.analyze()
            history         : Conversation history
            focused_object  : Object label to focus retrieval on

        Returns:
            ChatResponse
        """
        scene_text = None
        if analysis_result and analysis_result.scene_graph:
            scene_text = analysis_result.scene_graph.to_text()

        return self.chat(
            question       = question,
            scene_text     = scene_text,
            history        = history,
            focused_object = focused_object,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_chatbot_instance: Optional[WorkspaceVisionChatbot] = None


def get_chatbot() -> WorkspaceVisionChatbot:
    """
    Returns the singleton chatbot instance.
    Creates on first call, reuses on all subsequent calls.
    """
    global _chatbot_instance
    if _chatbot_instance is None:
        _chatbot_instance = WorkspaceVisionChatbot()
    return _chatbot_instance
