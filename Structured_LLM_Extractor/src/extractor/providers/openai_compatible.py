"""
OpenAI-compatible HTTP provider base class using httpx.
Serves as the high-speed backend for Groq, Cerebras, and OpenRouter.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional
import httpx

from src.extractor.core.document_loader import ProcessedDocument
from src.extractor.core.exceptions import MissingAPIKeyError, ModelExecutionError
from src.extractor.providers.base import LLMProvider

logger = logging.getLogger("extractor.providers.openai_compat")


class OpenAICompatibleProvider(LLMProvider):
    """Base class for OpenAI-compatible inference endpoints (Groq, Cerebras, OpenRouter)."""

    base_url: str = ""
    api_key_field: str = ""
    default_headers: Dict[str, str] = {}

    def __init__(self, api_key: str = "", default_model: Optional[str] = None) -> None:
        self.api_key = api_key
        if default_model:
            self.default_model = default_model

    def is_available(self) -> bool:
        k = (self.api_key or "").strip()
        return bool(k and not k.startswith("your_") and k != "replace_with_your_api_key")

    def _clean_json_output(self, content: str) -> str:
        """Strip markdown code blocks or surrounding text to yield pure JSON string."""
        raw = content.strip()
        # Remove ```json ... ``` or ``` ... ```
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        return raw.strip()

    def extract_json(
        self,
        document: ProcessedDocument,
        document_type: str,
        json_schema: Dict[str, Any],
        model_name: Optional[str] = None,
        timeout: float = 45.0,
    ) -> str:
        if not self.is_available():
            raise MissingAPIKeyError(
                f"{self.display_name} API key is not configured. "
                f"Please set {self.api_key_field} in your .env file or environment."
            )

        # Check multimodal capability
        doc_text = document.text_content
        if not doc_text or not doc_text.strip():
            if document.is_multimodal:
                raise ModelExecutionError(
                    f"{self.display_name} does not support direct binary multimodal image extraction. "
                    "Please select Google Gemini for scanned images or OCR documents."
                )
            raise ModelExecutionError(f"Document has no text content to extract with {self.display_name}.")

        system_prompt = self._build_system_prompt(document_type, json_schema)
        user_message = (
            f"Please extract the structured '{document_type}' data from the following document:\n\n"
            f"--- DOCUMENT START ---\n"
            f"{doc_text}\n"
            f"--- DOCUMENT END ---\n"
        )

        effective_model = model_name or self.default_model
        payload = {
            "model": effective_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }

        headers = {
            "Authorization": f"Bearer {self.api_key.strip()}",
            "Content-Type": "application/json",
            **self.default_headers,
        }

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        logger.info("Sending extraction request to %s (%s) at %s", self.display_name, effective_model, url)

        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(url, headers=headers, json=payload)

                if response.status_code == 401:
                    raise MissingAPIKeyError(
                        f"Authentication failed with {self.display_name} (HTTP 401). "
                        f"Verify {self.api_key_field} in .env."
                    )
                if response.status_code == 429:
                    raise ModelExecutionError(
                        f"{self.display_name} rate limit reached (HTTP 429). Retry or switch provider."
                    )

                response.raise_for_status()
                data = response.json()

                choices = data.get("choices") or []
                if not choices:
                    raise ModelExecutionError(f"No completion choices returned by {self.display_name}.")

                content = choices[0].get("message", {}).get("content", "")
                if not content:
                    raise ModelExecutionError(f"Empty response content from {self.display_name}.")

                cleaned = self._clean_json_output(content)
                # Verify valid JSON syntax
                json.loads(cleaned)
                return cleaned

        except httpx.HTTPStatusError as err:
            err_msg = err.response.text if err.response else str(err)
            logger.error("%s API error HTTP %d: %s", self.display_name, err.response.status_code if err.response else 0, err_msg)
            raise ModelExecutionError(f"{self.display_name} HTTP {err.response.status_code}: {err_msg}") from err
        except httpx.RequestError as req_err:
            logger.error("%s connection error: %s", self.display_name, req_err)
            raise ModelExecutionError(f"{self.display_name} network request failed: {req_err}") from req_err
        except json.JSONDecodeError as decode_err:
            logger.error("%s output failed JSON decoding: %s", self.display_name, decode_err)
            raise ModelExecutionError(f"{self.display_name} returned malformed non-JSON output: {decode_err}") from decode_err
