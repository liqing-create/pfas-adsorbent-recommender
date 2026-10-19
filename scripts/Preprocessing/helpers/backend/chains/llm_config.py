import os
from pathlib import Path
from typing import Optional, Dict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# ── Load .env ─────────────────────────────────────────────────────────────
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

SUPPORTED_PROVIDERS = {"openai", "together"}

OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
TOGETHER_API_KEY: Optional[str] = os.getenv("TOGETHER_API_KEY")
TOGETHER_API_BASE: str = os.getenv("TOGETHER_API_BASE", "https://api.together.ai/v1")

# ── Shared defaults ────────────────────────────────────────────────────────
COMMON_LLM_PARAMS: Dict = {
    "temperature":       float(os.getenv("LLM_TEMPERATURE", "0.0")),
    "max_tokens":        int  (os.getenv("LLM_MAX_TOKENS",    "16000")),
    "top_p":             float(os.getenv("LLM_TOP_P",         "1.0")),
    "frequency_penalty": float(os.getenv("LLM_FREQUENCY_PENALTY", "0.0")),
    "presence_penalty":  float(os.getenv("LLM_PRESENCE_PENALTY",  "0.0")),
}

# ── Factory ────────────────────────────────────────────────────────────────
def get_llm(provider: str, model_name: str) -> ChatOpenAI:
    """
    Centralized LLM factory.

    provider="openai" uses OpenAI directly.
    provider="together" uses Together AI's OpenAI-compatible endpoint.
    """
    provider = provider.lower().strip()
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported LLM provider: {provider}. "
            f"Supported providers: {sorted(SUPPORTED_PROVIDERS)}"
        )

    # copy so we can pop
    params = COMMON_LLM_PARAMS.copy()
    # OpenAI o* and GPT-5 models only use their built-in defaults.
    if provider == "openai" and model_name.lower().startswith(("o", "gpt-5")):
        # remove any keys unsupported by o-models
        for k in ("temperature", "top_p", "frequency_penalty", "presence_penalty"):
            params.pop(k, None)
    if provider == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set")
        return ChatOpenAI(
            model_name      = model_name,
            openai_api_key  = OPENAI_API_KEY,
            **params,
        )
    if provider == "together":
        if not TOGETHER_API_KEY:
            raise ValueError("TOGETHER_API_KEY is not set")
        return ChatOpenAI(
            model_name      = model_name,
            openai_api_key  = TOGETHER_API_KEY,
            openai_api_base = TOGETHER_API_BASE,
            **params,
        )