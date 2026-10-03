from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float = 0.0
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Map provider aliases to their canonical name."""
    if not value:
        raise ValueError("Provider value cannot be empty.")
    
    val = value.strip().lower()
    alias_map = {
        "openai": "openai",
        "oai": "openai",
        "custom": "custom",
        "local": "custom",
        "openai-compatible": "custom",
        "gemini": "gemini",
        "google": "gemini",
        "google-genai": "gemini",
        "anthropic": "anthropic",
        "anthorpic": "anthropic",
        "claude": "anthropic",
        "ollama": "ollama",
        "openrouter": "openrouter",
    }
    if val in alias_map:
        return alias_map[val]
    raise ValueError(
        f"Unsupported provider: '{value}'. Supported providers: "
        "openai, custom, gemini, anthropic, ollama, openrouter."
    )


def build_chat_model(config: ProviderConfig) -> Any:
    """Instantiate the real chat model for the selected provider.

    Mapping:
    - `openai` -> `ChatOpenAI`
    - `custom` -> `ChatOpenAI` with `base_url`
    - `gemini` -> `ChatGoogleGenerativeAI`
    - `anthropic` -> `ChatAnthropic`
    - `ollama` -> `ChatOllama`
    - `openrouter` -> `ChatOpenRouter`
    """
    provider = normalize_provider(config.provider)

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        kwargs: dict[str, Any] = {
            "model": config.model_name,
            "temperature": config.temperature,
        }
        if config.api_key:
            kwargs["api_key"] = config.api_key
        return ChatOpenAI(**kwargs)

    elif provider == "custom":
        from langchain_openai import ChatOpenAI
        kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
            "base_url": config.base_url or "http://localhost:8000/v1",
            "api_key": config.api_key or "EMPTY",
        }
        return ChatOpenAI(**kwargs)

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
        }
        if config.api_key:
            kwargs["google_api_key"] = config.api_key
        return ChatGoogleGenerativeAI(**kwargs)

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        kwargs = {
            "model_name": config.model_name,
            "temperature": config.temperature,
        }
        if config.api_key:
            kwargs["api_key"] = config.api_key
        return ChatAnthropic(**kwargs)

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
        }
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOllama(**kwargs)

    elif provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter
        kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
        }
        if config.api_key:
            kwargs["api_key"] = config.api_key
        return ChatOpenRouter(**kwargs)

    raise ValueError(f"Unknown provider: {provider}")
