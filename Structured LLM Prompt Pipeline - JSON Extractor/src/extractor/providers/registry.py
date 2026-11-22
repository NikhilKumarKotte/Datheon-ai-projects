"""
Central registry and intelligent router for LLM providers.
Manages provider lookup, configuration status, and multi-provider failover chains.
"""

import logging
from typing import Any, Dict, List, Optional

from src.extractor.config import Settings, get_settings
from src.extractor.providers.base import LLMProvider
from src.extractor.providers.cerebras_provider import CerebrasProvider
from src.extractor.providers.gemini_provider import GeminiProvider
from src.extractor.providers.groq_provider import GroqProvider
from src.extractor.providers.openrouter_provider import OpenRouterProvider

logger = logging.getLogger("extractor.providers.registry")


class ProviderRegistry:
    """Manages LLM providers, capabilities, and intelligent multi-provider routing."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._providers: Dict[str, LLMProvider] = {}
        self._initialize_providers()

    def _initialize_providers(self) -> None:
        """Register supported providers."""
        self._providers = {
            "gemini": GeminiProvider(settings=self.settings),
            "groq": GroqProvider(settings=self.settings),
            "cerebras": CerebrasProvider(settings=self.settings),
            "openrouter": OpenRouterProvider(settings=self.settings),
        }

    def register(self, provider: LLMProvider) -> None:
        """Register a custom LLM provider."""
        self._providers[provider.name.lower().strip()] = provider

    def get(self, name: str) -> LLMProvider:
        """Retrieve a provider by name."""
        key = name.lower().strip()
        if key not in self._providers:
            valid = list(self._providers.keys())
            raise ValueError(f"Unknown LLM provider '{name}'. Supported providers: {valid}")
        return self._providers[key]

    def list_providers(self) -> List[LLMProvider]:
        """Return all registered provider instances."""
        return list(self._providers.values())

    def get_metadata(self) -> List[Dict[str, Any]]:
        """Return metadata list for API discovery and UI rendering."""
        return [p.get_metadata() for p in self._providers.values()]

    def resolve_provider(
        self,
        requested_provider: Optional[str] = None,
        is_multimodal: bool = False,
    ) -> LLMProvider:
        """
        Intelligently select the best provider based on user preference,
        multimodal requirements, and available API credentials.
        """
        pref = (requested_provider or self.settings.default_provider or "auto").lower().strip()

        # If a specific provider is explicitly requested (not 'auto')
        if pref != "auto" and pref in self._providers:
            provider = self._providers[pref]
            if is_multimodal and not provider.supports_multimodal:
                logger.warning(
                    "Requested provider '%s' does not support multimodal vision. Falling back to Gemini.",
                    pref,
                )
                return self._providers["gemini"]
            return provider

        # Autonomous provider selection
        if is_multimodal:
            return self._providers["gemini"]

        # Check configured providers in priority order: Gemini -> Groq -> Cerebras -> OpenRouter
        priority_order = ["gemini", "groq", "cerebras", "openrouter"]
        for p_name in priority_order:
            p = self._providers.get(p_name)
            if p and p.is_available():
                logger.info("Auto-routed to active provider: %s", p.display_name)
                return p

        # Default fallback to configured default provider or Gemini
        default_name = self.settings.default_provider.lower().strip()
        return self._providers.get(default_name) or self._providers["gemini"]

    def get_fallback_chain(
        self,
        primary_name: str,
        is_multimodal: bool = False,
    ) -> List[LLMProvider]:
        """Return fallback provider candidates if the primary provider fails."""
        if is_multimodal:
            # Only multimodal providers can handle raw images/PDF scans
            return [p for p in self._providers.values() if p.supports_multimodal and p.name != primary_name]

        # Return other configured providers as fallbacks
        primary_key = primary_name.lower().strip()
        return [
            p for name, p in self._providers.items()
            if name != primary_key and p.is_available()
        ]


# Singleton instance
provider_registry = ProviderRegistry()
