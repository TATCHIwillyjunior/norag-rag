import json
import os
import re
import time
import uuid
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field

from rag_core import BASE_DIR, LLM_BASE_URL, LLM_MODEL, RagError
from tracing import Span, Trace, compute_cost, record_trace

SESSIONS_FILE = BASE_DIR / "data" / "conversation_sessions.json"

REWRITE_SYSTEM_PROMPT = """You are a query rewriting assistant for a technical support RAG system.
Given the conversation history and the user's latest follow-up question, rewrite the question into a clear, standalone query that captures all necessary context from previous turns.
Do not answer the question. Only return the standalone query.
If the question is already fully self-contained, return it as is.
Return ONLY plain text, without quotes or explanations."""

SUGGEST_SYSTEM_PROMPT = """You are an assistant suggesting next questions in a technical RAG app.
Given the question and the answer grounded in the documents, suggest exactly 2 or 3 relevant, interesting follow-up questions that could be answered from this technical documentation corpus.
Return ONLY a JSON array of strings:
["Question 1?", "Question 2?", "Question 3?"]
No other text, no markdown backticks."""


class ConversationSession:
    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or uuid.uuid4().hex[:10]
        self.turns: list[dict[str, Any]] = []

    def add_turn(self, user_question: str, rewritten_query: str, answer: str, references: list[str], trace_id: str):
        self.turns.append(
            {
                "timestamp": time.time(),
                "user_question": user_question,
                "rewritten_query": rewritten_query,
                "answer": answer,
                "references": references,
                "trace_id": trace_id,
            }
        )

    def get_history_text(self, max_turns: int = 3) -> str:
        recent = self.turns[-max_turns:]
        if not recent:
            return ""
        lines = []
        for t in recent:
            lines.append(f"User: {t['user_question']}")
            lines.append(f"Assistant: {t['answer'][:300]}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_count": len(self.turns),
            "turns": self.turns,
        }


# In-memory session cache
_SESSIONS: dict[str, ConversationSession] = {}


def get_or_create_session(session_id: str | None = None) -> ConversationSession:
    if session_id and session_id in _SESSIONS:
        return _SESSIONS[session_id]
    new_sess = ConversationSession(session_id)
    _SESSIONS[new_sess.session_id] = new_sess
    return new_sess


def clear_session(session_id: str) -> bool:
    if session_id in _SESSIONS:
        del _SESSIONS[session_id]
        return True
    return False


def get_llm_client() -> tuple[OpenAI, str]:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RagError(500, "not_configured", "GOOGLE_API_KEY is missing in the .env file.")
    client = OpenAI(base_url=LLM_BASE_URL, api_key=api_key, timeout=30, max_retries=1)
    return client, LLM_MODEL


def rewrite_query(session: ConversationSession, raw_question: str, trace: Trace | None = None) -> tuple[str, int, int]:
    history = session.get_history_text()
    if not history:
        # First turn: question is already standalone
        return raw_question, 0, 0

    client, model = get_llm_client()
    user_prompt = f"Conversation history:\n{history}\n\nLatest follow-up question: {raw_question}\n\nStandalone query:"

    span_cm = trace.span("conversation.rewrite") if trace else None
    span = span_cm.__enter__() if span_cm else None

    tin, tout = 0, 0
    rewritten = raw_question
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        if completion.usage:
            tin = completion.usage.prompt_tokens
            tout = completion.usage.completion_tokens
        res = (completion.choices[0].message.content or "").strip()
        if res and len(res) > 3:
            rewritten = res.strip('"\n ')
        if span:
            span.set_attribute("original_query", raw_question)
            span.set_attribute("rewritten_query", rewritten)
            span.set_attribute("tokens_in", tin)
            span.set_attribute("tokens_out", tout)
            span.set_attribute("cost_usd", compute_cost(tin, tout))
            span.finish("ok")
    except Exception as e:
        if span:
            span.set_attribute("error", str(e))
            span.finish("error")
        rewritten = raw_question
    finally:
        if span_cm and trace:
            trace.current_span_id = span.parent_span_id if span else None

    return rewritten, tin, tout


def suggest_next_questions(standalone_query: str, answer: str, trace: Trace | None = None) -> list[str]:
    client, model = get_llm_client()
    user_prompt = f"Question asked: {standalone_query}\nAnswer provided: {answer[:400]}\n\nSuggested follow-ups:"

    span_cm = trace.span("conversation.suggest") if trace else None
    span = span_cm.__enter__() if span_cm else None

    tin, tout = 0, 0
    suggestions = []
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SUGGEST_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        if completion.usage:
            tin = completion.usage.prompt_tokens
            tout = completion.usage.completion_tokens
        text = (completion.choices[0].message.content or "").strip()
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            arr = json.loads(match.group(0))
            if isinstance(arr, list):
                suggestions = [str(x).strip() for x in arr if str(x).strip()][:3]
        if span:
            span.set_attribute("suggestions", suggestions)
            span.set_attribute("tokens_in", tin)
            span.set_attribute("tokens_out", tout)
            span.set_attribute("cost_usd", compute_cost(tin, tout))
            span.finish("ok")
    except Exception as e:
        if span:
            span.set_attribute("error", str(e))
            span.finish("error")
        # Default fallback suggestions
        suggestions = ["Can you elaborate on this?", "Are there any exceptions or limits?"]
    finally:
        if span_cm and trace:
            trace.current_span_id = span.parent_span_id if span else None

    return suggestions
