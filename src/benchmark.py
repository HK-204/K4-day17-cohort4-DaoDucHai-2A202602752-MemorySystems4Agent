from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversations from disk."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """Return 0, 0.5, or 1.0 depending on how many expected facts appear in answer."""
    if not expected:
        return 1.0
    ans_lower = answer.lower()
    found = sum(1 for exp in expected if exp.lower() in ans_lower)
    if found == len(expected):
        return 1.0
    elif found > 0:
        return 0.5
    return 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Add a lightweight, objective quality score for offline mode."""
    if not answer or not answer.strip():
        return 0.0

    score = 0.0
    # Length and completeness check (up to 0.4)
    stripped_len = len(answer.strip())
    if stripped_len >= 20:
        score += 0.3
    if stripped_len >= 40:
        score += 0.1

    # Factual correctness check (up to 0.4)
    rec = recall_points(answer, expected)
    score += rec * 0.4

    # Structure check (bullet structure, polite informative phrasing) (up to 0.2)
    if any(marker in answer for marker in ["-", "•", "ghi nhớ", "Xin lỗi", "Theo thông tin"]):
        score += 0.2

    return round(min(1.0, score), 2)


def run_agent_benchmark(
    agent_name: str,
    agent: Any,
    conversations: list[dict[str, Any]],
    config: LabConfig,
) -> BenchmarkRow:
    """Evaluate one agent over multiple conversations and recall tests."""
    recalls: list[float] = []
    qualities: list[float] = []
    unique_users: set[str] = set()

    for conv in conversations:
        conv_id = conv["id"]
        user_id = conv["user_id"]
        unique_users.add(user_id)
        chat_thread_id = f"{conv_id}_chat"

        # 1. Feed all dialogue turns to the agent
        for turn in conv.get("turns", []):
            agent.reply(user_id=user_id, thread_id=chat_thread_id, message=turn)

        # 2. Ask recall questions in a FRESH thread
        recall_questions = conv.get("recall_questions", [])
        for q_idx, q in enumerate(recall_questions):
            recall_thread_id = f"{conv_id}_recall_{q_idx}"
            question_text = q["question"]
            expected = q["expected_contains"]

            reply_dict = agent.reply(
                user_id=user_id,
                thread_id=recall_thread_id,
                message=question_text,
            )
            answer = reply_dict.get("response", "")

            recalls.append(recall_points(answer, expected))
            qualities.append(heuristic_quality(answer, expected))

    # Calculate aggregate metrics
    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    avg_quality = sum(qualities) / len(qualities) if qualities else 0.0

    agent_tokens = agent.token_usage()
    prompt_tokens = agent.prompt_token_usage()
    compactions = agent.compaction_count()

    # Memory growth in bytes
    memory_bytes = 0
    if hasattr(agent, "memory_file_size"):
        for u in unique_users:
            memory_bytes += agent.memory_file_size(u)

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent_tokens,
        prompt_tokens_processed=prompt_tokens,
        recall_score=round(avg_recall, 2),
        response_quality=round(avg_quality, 2),
        memory_growth_bytes=memory_bytes,
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Print a clean markdown table comparing benchmark rows."""
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    table_data = [
        [
            r.agent_name,
            r.agent_tokens_only,
            r.prompt_tokens_processed,
            f"{r.recall_score:.2f}",
            f"{r.response_quality:.2f}",
            r.memory_growth_bytes,
            r.compactions,
        ]
        for r in rows
    ]
    try:
        from tabulate import tabulate

        return tabulate(table_data, headers=headers, tablefmt="github")
    except ImportError:
        # Fallback simple markdown table generator
        col_lens = [
            max(len(h), max(len(str(row[i])) for row in table_data))
            for i, h in enumerate(headers)
        ]
        header_line = "| " + " | ".join(h.ljust(col_lens[i]) for i, h in enumerate(headers)) + " |"
        sep_line = "| " + " | ".join("-" * col_lens[i] for i in range(len(headers))) + " |"
        body_lines = [
            "| " + " | ".join(str(row[i]).ljust(col_lens[i]) for i in range(len(headers))) + " |"
            for row in table_data
        ]
        return "\n".join([header_line, sep_line] + body_lines)


def clean_user_profiles(state_dir: Path, user_ids: list[str]) -> None:
    """Helper to remove specific user profile directories before benchmark runs."""
    profiles_dir = state_dir / "profiles"
    if not profiles_dir.exists():
        return
    for u in user_ids:
        slug = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in u.strip())
        target = profiles_dir / slug
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)


def main() -> None:
    """Run both benchmark suites: Standard Benchmark and Long-Context Stress Benchmark."""
    repo_root = Path(__file__).resolve().parent.parent
    config = load_config(repo_root)

    std_data_path = config.data_dir / "conversations.json"
    stress_data_path = config.data_dir / "advanced_long_context.json"

    # ==========================================
    # 1. Standard Benchmark (conversations.json)
    # ==========================================
    print("\n" + "=" * 65)
    print("### 1. Standard Benchmark (data/conversations.json)")
    print("=" * 65)

    clean_user_profiles(config.state_dir, ["dungct"])
    std_convs = load_conversations(std_data_path)

    baseline_std = BaselineAgent(config, force_offline=True)
    row_baseline_std = run_agent_benchmark("Baseline", baseline_std, std_convs, config)

    advanced_std = AdvancedAgent(config, force_offline=True)
    row_advanced_std = run_agent_benchmark("Advanced", advanced_std, std_convs, config)

    print(format_rows([row_baseline_std, row_advanced_std]))

    # =======================================================
    # 2. Long-Context Stress Benchmark (advanced_long_context.json)
    # =======================================================
    print("\n" + "=" * 65)
    print("### 2. Long-Context Stress Benchmark (data/advanced_long_context.json)")
    print("=" * 65)

    clean_user_profiles(config.state_dir, ["dungct_stress"])
    stress_convs = load_conversations(stress_data_path)

    baseline_stress = BaselineAgent(config, force_offline=True)
    row_baseline_stress = run_agent_benchmark("Baseline", baseline_stress, stress_convs, config)

    advanced_stress = AdvancedAgent(config, force_offline=True)
    row_advanced_stress = run_agent_benchmark("Advanced", advanced_stress, stress_convs, config)

    print(format_rows([row_baseline_stress, row_advanced_stress]))
    print("\n" + "=" * 65 + "\n")


if __name__ == "__main__":
    main()
