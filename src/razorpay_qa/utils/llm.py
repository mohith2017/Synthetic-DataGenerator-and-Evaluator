"""LLM model factory — Anthropic Claude generator, OpenAI GPT judge (optional)."""

from __future__ import annotations

import os

PROVIDER = "anthropic"


class LLMUnavailableError(RuntimeError):
    """Raised when the Anthropic model cannot be initialised (e.g. missing key)."""


def get_model(settings_models: dict):
    """Return a DeepEval ``AnthropicModel`` configured from settings + env."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise LLMUnavailableError(
            "ANTHROPIC_API_KEY not set. Export it (or put it in .env) before running."
        )
    from deepeval.models import AnthropicModel

    cfg = settings_models.get("anthropic", {})
    return AnthropicModel(
        model=cfg.get("model", "claude-haiku-4-5-20251001"),
        api_key=api_key,
        temperature=float(cfg.get("temperature", 0.0)),
    )


def get_judge_model(settings_models: dict):
    """Return the judge LLM — GPT-4.1 if OPENAI_API_KEY is set, else falls back to generator model."""
    from dotenv import load_dotenv
    load_dotenv()

    judge_cfg = settings_models.get("judge", {})
    provider = judge_cfg.get("provider", "anthropic")

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            from deepeval.models import GPTModel
            return GPTModel(
                model=judge_cfg.get("model", "gpt-5.5"),
                api_key=api_key,
            )

    return get_model(settings_models)


def model_name(settings_models: dict) -> str:
    return settings_models.get("anthropic", {}).get("model", "claude-haiku-4-5-20251001")


def judge_model_name(settings_models: dict) -> str:
    """Return a human-readable name for the judge model."""
    from dotenv import load_dotenv
    load_dotenv()

    judge_cfg = settings_models.get("judge", {})
    provider = judge_cfg.get("provider", "anthropic")
    if provider == "openai" and os.environ.get("OPENAI_API_KEY"):
        return judge_cfg.get("model", "gpt-5.5")
    return model_name(settings_models)


def model_temperature(settings_models: dict) -> float:
    return float(settings_models.get("anthropic", {}).get("temperature", 0.0))
