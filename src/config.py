from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Shared configuration for the lab."""

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables and return a LabConfig."""
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    # Load environment variables from .env in root if present
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()

    # Create state directories
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    profiles_dir = state_dir / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    data_dir = root / "data"

    # Model provider settings
    provider_str = os.getenv("LLM_PROVIDER", "openai")
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))

    # Map provider-specific API keys
    api_key = (
        os.getenv("LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("CUSTOM_API_KEY")
    )
    base_url = os.getenv("LLM_BASE_URL") or os.getenv("CUSTOM_BASE_URL") or os.getenv("OLLAMA_BASE_URL")

    try:
        norm_prov = normalize_provider(provider_str)
    except Exception:
        norm_prov = "openai"

    if norm_prov == "gemini" and os.getenv("GEMINI_API_KEY"):
        api_key = os.getenv("GEMINI_API_KEY")
    elif norm_prov == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
        api_key = os.getenv("ANTHROPIC_API_KEY")
    elif norm_prov == "openrouter" and os.getenv("OPENROUTER_API_KEY"):
        api_key = os.getenv("OPENROUTER_API_KEY")
    elif norm_prov == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    elif norm_prov == "custom":
        base_url = os.getenv("CUSTOM_BASE_URL", "http://localhost:8000/v1")
        api_key = os.getenv("CUSTOM_API_KEY", "EMPTY")

    model_cfg = ProviderConfig(
        provider=provider_str,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )

    # Judge model settings (falls back to main model)
    judge_provider = os.getenv("JUDGE_PROVIDER", provider_str)
    judge_model_name = os.getenv("JUDGE_MODEL", model_name)
    judge_api_key = os.getenv("JUDGE_API_KEY", api_key)
    judge_base_url = os.getenv("JUDGE_BASE_URL", base_url)

    judge_cfg = ProviderConfig(
        provider=judge_provider,
        model_name=judge_model_name,
        temperature=0.0,
        api_key=judge_api_key,
        base_url=judge_base_url,
    )

    # Compact memory thresholds
    compact_threshold = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "600"))
    compact_keep = int(os.getenv("COMPACT_KEEP_MESSAGES", "4"))

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold,
        compact_keep_messages=compact_keep,
        model=model_cfg,
        judge_model=judge_cfg,
    )
