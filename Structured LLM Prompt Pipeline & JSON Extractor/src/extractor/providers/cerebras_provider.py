"""
Cerebras Cloud integration providing wafer-scale high-speed inference.
Optimized for low-latency text and document extraction.
"""

from typing import Optional
from src.extractor.config import Settings, get_settings
from src.extractor.providers.openai_compatible import OpenAICompatibleProvider


class CerebrasProvider(OpenAICompatibleProvider):
    """Cerebras Cloud wafer-scale inference provider."""

    name = "cerebras"
    display_name = "Cerebras Cloud"
    badge = "Wafer-Scale Speed"
    description = "Hardware-accelerated wafer-scale inference using Llama 3.3 70B."
    default_model = "llama3.3-70b"
    available_models = [
        "llama3.3-70b",
        "llama3.1-8b",
    ]
    base_url = "https://api.cerebras.ai/v1"
    api_key_field = "CEREBRAS_API_KEY"

    def __init__(self, settings: Optional[Settings] = None, api_key: Optional[str] = None) -> None:
        cfg = settings or get_settings()
        effective_key = api_key if api_key is not None else cfg.cerebras_api_key
        super().__init__(api_key=effective_key, default_model=cfg.cerebras_default_model or self.default_model)
