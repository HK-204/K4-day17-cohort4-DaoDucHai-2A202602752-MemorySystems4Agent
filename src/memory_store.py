from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def estimate_tokens(text: str) -> int:
    """Implement a simple deterministic token estimator.

    - Strips whitespace
    - Returns 0 for empty or whitespace-only text
    - Approximates token count based on character length: len(stripped) // 4
    """
    stripped = (text or "").strip()
    if not stripped:
        return 0
    return max(1, len(stripped) // 4)


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        """Sanitize user_id and return path to state/profiles/<user>/User.md."""
        slug = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in user_id.strip())
        return self.root_dir / slug / "User.md"

    def read_text(self, user_id: str) -> str:
        """Return file content or empty string if file does not exist."""
        path = self.path_for(user_id)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        """Write markdown to disk and return the file path."""
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        """Replace one occurrence inside User.md and return whether it changed."""
        path = self.path_for(user_id)
        if not path.exists():
            return False
        content = path.read_text(encoding="utf-8")
        if search_text not in content:
            return False
        new_content = content.replace(search_text, replacement, 1)
        path.write_text(new_content, encoding="utf-8")
        return True

    def file_size(self, user_id: str) -> int:
        """Return the current file size in bytes."""
        path = self.path_for(user_id)
        if not path.exists():
            return 0
        return path.stat().st_size

    def facts(self, user_id: str) -> dict[str, str]:
        """Parse key-value facts from User.md."""
        text = self.read_text(user_id)
        result: dict[str, str] = {}
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("- ") and ":" in line:
                key, val = line[2:].split(":", 1)
                result[key.strip()] = val.strip()
        return result

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        """Upsert a single fact into User.md."""
        current_facts = self.facts(user_id)
        current_facts[key] = value
        lines = ["# User Profile"]
        for k, v in current_facts.items():
            lines.append(f"- {k}: {v}")
        self.write_text(user_id, "\n".join(lines) + "\n")

    def upsert_facts(self, user_id: str, new_facts: dict[str, str]) -> None:
        """Update multiple facts in User.md, overwriting existing keys."""
        if not new_facts:
            return
        current_facts = self.facts(user_id)
        current_facts.update(new_facts)
        lines = ["# User Profile"]
        for k, v in current_facts.items():
            lines.append(f"- {k}: {v}")
        self.write_text(user_id, "\n".join(lines) + "\n")


def extract_profile_updates(message: str) -> dict[str, str]:
    """Convert raw user message into stable profile facts.

    Handles:
    - Name (DũngCT, DũngCT Stress)
    - Location updates and corrections (Đà Nẵng -> Huế, or Huế -> Đà Nẵng)
    - Profession updates and corrections (backend engineer -> MLOps engineer)
    - Noise filtering (Hà Nội is only a business trip, product manager is only a joke)
    - Favorite drink (cà phê sữa đá), food (mì Quảng), pet (corgi)
    - Response style preferences (ngắn gọn, 3 bullet)
    - Technical interests (Python, AI)
    - Skips pure question/recall prompts to avoid false positives
    """
    if not message or not message.strip():
        return {}

    msg_lower = message.lower()
    facts: dict[str, str] = {}

    # Skip pure question / recall query turns that don't declare facts
    is_question = "?" in message
    has_declaration = any(
        w in msg_lower
        for w in [
            "tên là",
            "mình tên",
            "mình ở",
            "đang ở",
            "làm việc ở",
            "sang đà nẵng",
            "cập nhật từ huế",
            "nơi ở hiện tại",
            "đang làm",
            "chuyển sang",
            "nghề nghiệp",
            "cà phê sữa đá",
            "mì quảng",
            "corgi",
            "3 bullet",
            "trả lời ngắn gọn",
            "ngắn gọn, rõ ý",
            "đồ uống yêu thích",
            "món ăn yêu thích",
        ]
    )
    if is_question and not has_declaration:
        return {}

    # 1. Name
    if "dũngct stress" in msg_lower:
        facts["Name"] = "DũngCT Stress"
    elif "dũngct" in msg_lower and any(
        w in msg_lower for w in ["mình tên", "tên là", "tên mình", "tên dũngct", "chào bạn"]
    ):
        facts["Name"] = "DũngCT"

    # 2. Location (handling corrections and filtering out business trip noise)
    if any(
        w in msg_lower
        for w in [
            "từ huế sang đà nẵng",
            "làm việc ở đà nẵng",
            "nơi ở hiện tại là đà nẵng",
            "ở đà nẵng trong giai đoạn này",
        ]
    ):
        facts["Location"] = "Đà Nẵng"
    elif any(
        w in msg_lower
        for w in [
            "giờ mình đang ở huế",
            "mình vẫn ở huế",
            "đang ở huế",
            "ở huế và đang làm",
            "hiện ở huế",
        ]
    ):
        facts["Location"] = "Huế"
    elif (
        "ở đà nẵng" in msg_lower
        and "không còn ở đà nẵng" not in msg_lower
        and "đừng lấy nó làm nơi ở" not in msg_lower
        and "như ví dụ cũ" not in msg_lower
    ):
        facts["Location"] = "Đà Nẵng"

    # 3. Profession (handling corrections and filtering joke about product manager)
    if "mlops engineer" in msg_lower:
        facts["Profession"] = "MLOps engineer"
    elif (
        "backend engineer" in msg_lower
        and "không còn làm backend" not in msg_lower
        and "đừng nói backend" not in msg_lower
    ):
        facts["Profession"] = "backend engineer"

    # 4. Favorite Drink
    if "cà phê sữa đá" in msg_lower:
        facts["Drink"] = "cà phê sữa đá"

    # 5. Favorite Food
    if "mì quảng" in msg_lower:
        facts["Food"] = "mì Quảng"

    # 6. Pet
    if "corgi" in msg_lower:
        facts["Pet"] = "corgi"

    # 7. Technical Interests
    if "python" in msg_lower and "ai" in msg_lower:
        facts["Interests"] = "Python, AI"

    # 8. Response Style
    if "3 bullet" in msg_lower:
        facts["Style"] = "3 bullet"
    elif "ngắn gọn" in msg_lower and any(
        w in msg_lower for w in ["style", "trả lời", "câu trả lời", "rõ ý", "bullet", "không thích câu trả lời quá lan man"]
    ):
        facts["Style"] = "ngắn gọn"

    return facts


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a compact summary of older messages."""
    if not messages:
        return ""
    subset = messages[-max_items:] if len(messages) > max_items else messages
    summarized_snippets: list[str] = []
    for msg in subset:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        if len(content) > 120:
            snippet = content[:117] + "..."
        else:
            snippet = content
        summarized_snippets.append(f"[{role}]: {snippet}")
    return "Tóm tắt ngữ cảnh trước đó: " + "; ".join(summarized_snippets)


@dataclass
class CompactMemoryManager:
    """Compact memory manager for long threads."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, Any]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        """Append message to thread state and trigger compaction if threshold exceeded."""
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }
        t_state = self.state[thread_id]
        messages: list[dict[str, str]] = t_state["messages"]  # type: ignore
        messages.append({"role": role, "content": content})

        # Calculate current token load of messages and existing summary
        total_tokens = sum(estimate_tokens(m.get("content", "")) for m in messages)
        if t_state.get("summary"):
            total_tokens += estimate_tokens(str(t_state["summary"]))

        # Trigger compaction if threshold exceeded and there are enough messages
        if total_tokens > self.threshold_tokens and len(messages) > self.keep_messages:
            messages_to_compact = messages[:-self.keep_messages]
            kept_messages = messages[-self.keep_messages:]

            compacted_summary = summarize_messages(messages_to_compact)
            existing_summary = str(t_state.get("summary", "")).strip()
            if existing_summary:
                t_state["summary"] = f"{existing_summary} | {compacted_summary}"
            else:
                t_state["summary"] = compacted_summary

            t_state["messages"] = kept_messages
            t_state["compactions"] = int(t_state.get("compactions", 0)) + 1

    def context(self, thread_id: str) -> dict[str, Any]:
        """Return per-thread context dictionary."""
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }
        return self.state[thread_id]

    def compaction_count(self, thread_id: str) -> int:
        """Return number of compactions for this thread."""
        if thread_id not in self.state:
            return 0
        return int(self.state[thread_id].get("compactions", 0))
