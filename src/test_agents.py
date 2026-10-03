from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import CompactMemoryManager, UserProfileStore
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated config for tests with all 7 required fields."""
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "profiles").mkdir(parents=True, exist_ok=True)

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    return LabConfig(
        base_dir=tmp_path,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=80,  # nhỏ để compact kích hoạt sớm
        compact_keep_messages=2,
        model=ProviderConfig(provider="openai", model_name="stub", temperature=0.0),
        judge_model=ProviderConfig(provider="openai", model_name="stub", temperature=0.0),
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify `User.md` can be created, updated, and edited."""
    store = UserProfileStore(tmp_path / "profiles")

    # Initial state should be empty string
    assert store.read_text("dungct") == ""
    assert store.file_size("dungct") == 0

    # Write initial profile
    initial_content = "# User Profile\n- Name: DũngCT\n- Location: Đà Nẵng\n"
    file_path = store.write_text("dungct", initial_content)
    assert file_path.exists()
    assert store.file_size("dungct") > 0

    # Verify read
    content = store.read_text("dungct")
    assert "DũngCT" in content
    assert "Đà Nẵng" in content

    # Edit location (correction from Đà Nẵng to Huế)
    edit_success = store.edit_text("dungct", "Đà Nẵng", "Huế")
    assert edit_success is True

    # Verify edit
    updated_content = store.read_text("dungct")
    assert "Huế" in updated_content
    assert "Đà Nẵng" not in updated_content

    # Editing non-existent text returns False
    assert store.edit_text("dungct", "Hà Nội", "Sài Gòn") is False


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction."""
    cfg = make_config(tmp_path)
    cm = CompactMemoryManager(
        threshold_tokens=cfg.compact_threshold_tokens,
        keep_messages=cfg.compact_keep_messages,
    )

    thread_id = "thread_compact_test"

    # Append long messages to exceed threshold
    for i in range(8):
        cm.append(
            thread_id,
            "user",
            f"Lượt hội thoại {i}: đây là một tin nhắn đủ dài để vượt qua ngưỡng compact token nhỏ trong kiểm thử.",
        )

    # Check compaction count is strictly positive
    assert cm.compaction_count(thread_id) > 0

    # Verify thread state kept messages monotonicity
    ctx = cm.context(thread_id)
    assert len(ctx["messages"]) == cfg.compact_keep_messages
    assert len(str(ctx.get("summary", "")).strip()) > 0


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify advanced remembers across sessions and baseline does not."""
    cfg = make_config(tmp_path)
    adv = AdvancedAgent(config=cfg, force_offline=True)
    base = BaselineAgent(config=cfg, force_offline=True)

    user_id = "dungct"
    first_thread = "session_01"
    second_thread = "session_02"

    # Thread 1: provide facts
    msg = "Chào bạn, mình tên là DũngCT. Đồ uống yêu thích là cà phê sữa đá."
    adv.reply(user_id=user_id, thread_id=first_thread, message=msg)
    base.reply(user_id=user_id, thread_id=first_thread, message=msg)

    # Thread 2: ask recall question in a fresh thread
    recall_q = "Mình tên gì và đồ uống yêu thích là gì?"
    adv_reply = adv.reply(user_id=user_id, thread_id=second_thread, message=recall_q)
    base_reply = base.reply(user_id=user_id, thread_id=second_thread, message=recall_q)

    # Advanced must remember facts from User.md
    assert "DũngCT" in adv_reply["response"]
    assert "cà phê sữa đá" in adv_reply["response"]

    # Baseline must NOT remember across threads
    assert "DũngCT" not in base_reply["response"]
    assert "cà phê sữa đá" not in base_reply["response"]


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare prompt load of baseline vs advanced on a long thread."""
    cfg = make_config(tmp_path)
    adv = AdvancedAgent(config=cfg, force_offline=True)
    base = BaselineAgent(config=cfg, force_offline=True)

    user_id = "dungct_stress"
    thread_id = "long_context_thread"

    # Run 12 long turns
    long_turns = [
        f"Lượt {i}: Đoạn văn bản dài để kiểm tra độ tăng trưởng ngữ cảnh "
        f"giữa hai kiến trúc memory agent qua nhiều chu kỳ làm việc liên tiếp. " * 3
        for i in range(12)
    ]

    for turn in long_turns:
        adv.reply(user_id=user_id, thread_id=thread_id, message=turn)
        base.reply(user_id=user_id, thread_id=thread_id, message=turn)

    adv_prompt = adv.prompt_token_usage(thread_id)
    base_prompt = base.prompt_token_usage(thread_id)

    # Compact must have triggered on the advanced agent
    assert adv.compaction_count(thread_id) > 0

    # Advanced agent must process significantly fewer prompt tokens than Baseline
    assert adv_prompt < base_prompt, (
        f"Expected Advanced prompt tokens ({adv_prompt}) to be less than Baseline ({base_prompt})"
    )
