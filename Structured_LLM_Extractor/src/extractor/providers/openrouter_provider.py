"""
OpenRouter integration providing unified access to 200+ models.
Includes access to free model tiers and broad fallback coverage.
"""

from typing import Optional
from src.extractor.config import Settings, get_settings
from src.extractor.providers.openai_compatible import OpenAICompatibleProvider


class OpenRouterProvider(OpenAICompatibleProvider):
    """OpenRouter universal gateway provider."""

    name = "openrouter"
    display_name = "OpenRouter"
    badge = "200+ Models & Free Tier"
    description = "Universal multi-provider gateway with access to top open and commercial models."
    default_model = "meta-llama/llama-3.3-70b-instruct:free"
    available_models = [
        "meta-llama/llama-3.3-70b-instruct:free",
        "google/gemini-2.0-flash-exp:free",
        "mistralai/mistral-7b-instruct:free",
        "qwen/qwen-2.5-coder-32b-instruct:free",
        "meta-llama/llama-3.1-8b-instruct:free",
    ]
    base_url = "https://openrouter.ai/api/v1"
    api_key_field = "OPENROUTER_API_KEY"
    default_headers = {
        "HTTP-Referer": "https://github.com/your-org/structured-llm-extractor",
        "X-Title": "Structured LLM Extractor",
    }

    def __init__(self, settings: Optional[Settings] = None, api_key: Optional[str] = None) -> None:
        cfg = settings or get_settings()
        effective_key = api_key if api_key is not None else cfg.openrouter_api_key
        super().__init__(api_key=effective_key, default_model=cfg.openrouter_default_model or self.default_model)
