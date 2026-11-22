"""
GroqCloud integration providing ultra-fast LPU inference (500+ tok/s).
Ideal for high-throughput invoice and resume structured extraction.
"""

from typing import Optional
from src.extractor.config import Settings, get_settings
from src.extractor.providers.openai_compatible import OpenAICompatibleProvider


class GroqProvider(OpenAICompatibleProvider):
    """GroqCloud high-speed inference provider."""

    name = "groq"
    display_name = "GroqCloud"
    badge = "Ultra-Fast (500 T/s)"
    description = "Ultra-fast LPU inference powered by Meta Llama 3.3 70B with JSON mode."
    default_model = "llama-3.3-70b-versatile"
    available_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ]
    base_url = "https://api.groq.com/openai/v1"
    api_key_field = "GROQ_API_KEY"

    def __init__(self, settings: Optional[Settings] = None, api_key: Optional[str] = None) -> None:
        cfg = settings or get_settings()
        effective_key = api_key if api_key is not None else cfg.groq_api_key
        super().__init__(api_key=effective_key, default_model=cfg.groq_default_model or self.default_model)
