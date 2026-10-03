from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A / Baseline Agent.

    Requirements:
    - Within-session memory only
    - No persistent `User.md`
    - Should forget long-term facts across new threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return the agent response and token accounting."""
        if not self.force_offline and self.langchain_agent is not None:
            try:
                response = self.langchain_agent.invoke(message)
                resp_text = getattr(response, "content", str(response))
                session = self.sessions.setdefault(thread_id, SessionState())
                turn_prompt = (
                    sum(estimate_tokens(m["content"]) for m in session.messages)
                    + estimate_tokens(message)
                )
                turn_tokens = estimate_tokens(resp_text)
                session.prompt_tokens_processed += turn_prompt
                session.token_usage += turn_tokens
                session.messages.append({"role": "user", "content": message})
                session.messages.append({"role": "assistant", "content": resp_text})
                return {
                    "response": resp_text,
                    "tokens": turn_tokens,
                    "prompt_tokens": turn_prompt,
                }
            except Exception:
                pass
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str | None = None) -> int:
        """Return cumulative agent token count for one thread or all threads."""
        if thread_id is None:
            return sum(s.token_usage for s in self.sessions.values())
        session = self.sessions.get(thread_id)
        return session.token_usage if session else 0

    def prompt_token_usage(self, thread_id: str | None = None) -> int:
        """Estimate cumulative prompt context kept processing for one thread or all threads."""
        if thread_id is None:
            return sum(s.prompt_tokens_processed for s in self.sessions.values())
        session = self.sessions.get(thread_id)
        return session.prompt_tokens_processed if session else 0

    def compaction_count(self, thread_id: str | None = None) -> int:
        """Baseline has no compact memory."""
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Implement deterministic offline behavior for Baseline Agent."""
        session = self.sessions.setdefault(thread_id, SessionState())

        # Prompt context load for this turn: all prior messages in thread + current message
        history_tokens = sum(estimate_tokens(m["content"]) for m in session.messages)
        current_msg_tokens = estimate_tokens(message)
        turn_prompt_tokens = history_tokens + current_msg_tokens
        session.prompt_tokens_processed += turn_prompt_tokens

        session.messages.append({"role": "user", "content": message})

        # Deterministic offline response:
        # Baseline only knows messages inside the current thread_id.
        # When asked recall questions in a new thread, it has no prior messages in this session.
        msg_lower = message.lower()
        if "?" in message or any(q in msg_lower for q in ["nhắc lại", "là gì", "ở đâu", "ai không"]):
            prior_declarations = [
                m["content"] for m in session.messages[:-1] if m["role"] == "user"
            ]
            if prior_declarations:
                reply_text = f"Trong phiên trò chuyện này, bạn đã nhắc đến: {prior_declarations[-1]}."
            else:
                reply_text = "Xin lỗi, tôi không có thông tin về bạn trong phiên làm việc này."
        else:
            reply_text = "Tôi đã ghi nhận thông tin của bạn trong phiên trò chuyện này."

        turn_tokens = estimate_tokens(reply_text)
        session.token_usage += turn_tokens
        session.messages.append({"role": "assistant", "content": reply_text})

        return {
            "response": reply_text,
            "tokens": turn_tokens,
            "prompt_tokens": turn_prompt_tokens,
        }

    def _maybe_build_langchain_agent(self):
        """Optionally wire live LLM agent when configured."""
        if self.force_offline:
            return None
        if not self.config.model.api_key and self.config.model.provider not in ("ollama", "custom"):
            return None
        try:
            return build_chat_model(self.config.model)
        except Exception:
            return None
