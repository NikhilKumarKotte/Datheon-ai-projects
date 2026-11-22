"""
Comprehensive test suite for multi-provider LLM integrations:
Gemini, GroqCloud, Cerebras, and OpenRouter.
"""

import json
from unittest.mock import MagicMock, patch
import pytest
import httpx

from src.extractor.config import Settings
from src.extractor.core.document_loader import ProcessedDocument
from src.extractor.core.exceptions import MissingAPIKeyError, ModelExecutionError
from src.extractor.core.service import ExtractorService
from src.extractor.providers import (
    CerebrasProvider,
    GeminiProvider,
    GroqProvider,
    OpenRouterProvider,
    ProviderRegistry,
    provider_registry,
)


def test_provider_registry_contains_all_four_providers():
    providers = provider_registry.list_providers()
    names = [p.name for p in providers]
    assert "gemini" in names
    assert "groq" in names
    assert "cerebras" in names
    assert "openrouter" in names


def test_provider_metadata_structure():
    metadata = provider_registry.get_metadata()
    assert len(metadata) == 4
    for item in metadata:
        assert "id" in item
        assert "name" in item
        assert "badge" in item
        assert "default_model" in item
        assert "available_models" in item
        assert "supports_multimodal" in item
        assert "configured" in item


def test_groq_missing_api_key():
    groq = GroqProvider(api_key="")
    assert groq.is_available() is False
    doc = ProcessedDocument(text_content="Sample text", extension=".txt")
    with pytest.raises(MissingAPIKeyError):
        groq.extract_json(doc, "invoice", {"type": "object"})


def test_groq_mock_extraction_success():
    groq = GroqProvider(api_key="gsk_test_mock_key_12345")
    assert groq.is_available() is True

    doc = ProcessedDocument(text_content="Invoice INV-99 Acme Corp Total 500", extension=".txt")
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": '{"invoice_number": "INV-99", "seller": "Acme Corp", "total_amount": 500.0}'
                }
            }
        ]
    }

    with patch("httpx.Client.post", return_value=mock_response):
        raw_json = groq.extract_json(doc, "invoice", {"type": "object"})
        data = json.loads(raw_json)
        assert data["invoice_number"] == "INV-99"
        assert data["total_amount"] == 500.0


def test_cerebras_mock_extraction_success():
    cerebras = CerebrasProvider(api_key="csk_test_cerebras_mock_key")
    assert cerebras.is_available() is True

    doc = ProcessedDocument(text_content="Store receipt Starbucks $5.50", extension=".txt")
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": '```json\n{"merchant_name": "Starbucks", "total_amount": 5.50}\n```'
                }
            }
        ]
    }

    with patch("httpx.Client.post", return_value=mock_response):
        raw_json = cerebras.extract_json(doc, "receipt", {"type": "object"})
        data = json.loads(raw_json)
        assert data["merchant_name"] == "Starbucks"
        assert data["total_amount"] == 5.50


def test_openrouter_mock_extraction_with_headers():
    openrouter = OpenRouterProvider(api_key="sk-or-test-openrouter-key")
    assert openrouter.is_available() is True

    doc = ProcessedDocument(text_content="Candidate Jane Doe Python engineer", extension=".txt")
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": '{"full_name": "Jane Doe", "skills": ["Python"]}'
                }
            }
        ]
    }

    with patch("httpx.Client.post", return_value=mock_response) as mock_post:
        raw_json = openrouter.extract_json(doc, "resume", {"type": "object"})
        data = json.loads(raw_json)
        assert data["full_name"] == "Jane Doe"
        # Verify custom OpenRouter headers were passed
        call_kwargs = mock_post.call_args[1]
        assert "HTTP-Referer" in call_kwargs["headers"]
        assert "X-Title" in call_kwargs["headers"]


def test_provider_multimodal_guard():
    # Groq should reject pure binary multimodal documents without text
    groq = GroqProvider(api_key="mock_key")
    image_doc = ProcessedDocument(
        filename="scan.png",
        is_multimodal=True,
        binary_content=b"\x89PNG\r\n\x1a\n",
        text_content=None,
    )
    with pytest.raises(ModelExecutionError) as exc_info:
        groq.extract_json(image_doc, "invoice", {"type": "object"})
    assert "multimodal" in str(exc_info.value)


def test_service_demo_mode_with_all_providers():
    settings = Settings(gemini_api_key="")
    service = ExtractorService(settings=settings)

    for prov in ["gemini", "groq", "cerebras", "openrouter"]:
        res = service.extract("Sample invoice text", document_type="invoice", provider=prov, demo_mode=True)
        assert res.metadata.provider_used == prov
        assert res.metadata.is_demo is True
        assert res.data["invoice_number"] == "INV-1042"


def test_api_providers_endpoint(test_client):
    response = test_client.get("/api/v1/providers")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 4
    provider_ids = [p["id"] for p in data]
    assert "gemini" in provider_ids
    assert "groq" in provider_ids
    assert "cerebras" in provider_ids
    assert "openrouter" in provider_ids
