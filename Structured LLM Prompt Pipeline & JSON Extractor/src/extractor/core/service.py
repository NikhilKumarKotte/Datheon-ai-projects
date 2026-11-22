"""
Extractor Service orchestrating multi-provider LLM extraction,
autonomous routing, local Pydantic validation, and agentic auditing.
Supports Google Gemini, GroqCloud, Cerebras, and OpenRouter.
"""

import logging
from pathlib import Path
import time
from typing import Any, Dict, Optional, Type, Union
from pydantic import BaseModel, Field, ValidationError

from google import genai

from src.extractor.config import Settings, get_settings
from src.extractor.core.classifier import DocumentClassifier
from src.extractor.core.document_loader import DocumentLoader, ProcessedDocument
from src.extractor.core.exceptions import (
    EmptyDocumentError,
    ExtractorError,
    MissingAPIKeyError,
    ModelExecutionError,
    SchemaValidationError,
    UnsupportedDocumentTypeError,
)
from src.extractor.core.mock_data import get_mock_extraction
from src.extractor.core.verification import VerificationAgent, VerificationReport
from src.extractor.providers import (
    GeminiProvider,
    LLMProvider,
    ProviderRegistry,
    provider_registry as default_provider_registry,
)
from src.extractor.schemas.base import StrictModel
from src.extractor.schemas.registry import registry

logger = logging.getLogger("extractor.service")


class ExtractionMetadata(BaseModel):
    """Metadata about the extraction execution and verification audit."""
    document_type: str
    model_used: str
    duration_ms: float
    provider_used: str = "gemini"
    fallback_used: bool = False
    input_format: str = "text"
    is_multimodal: bool = False
    confidence_score: float = 0.95
    math_verified: Optional[bool] = None
    verification: Optional[VerificationReport] = None
    is_demo: bool = False


class ExtractionResult(BaseModel):
    """Encapsulates the validated data along with operational metadata."""
    data: Dict[str, Any]
    metadata: ExtractionMetadata

    @property
    def verification(self) -> Optional[VerificationReport]:
        """Convenience property accessing the verification audit report."""
        return self.metadata.verification

    def to_model(self, model_cls: Type[StrictModel]) -> StrictModel:
        """Convert underlying dict back into the specific Pydantic model."""
        return model_cls.model_validate(self.data)


class ExtractorService:
    """
    Production service for structured information extraction from any file format
    using multi-provider LLM backends (Gemini, Groq, Cerebras, OpenRouter).
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        client: Optional[genai.Client] = None,
        registry_instance: Optional[ProviderRegistry] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = client
        self.provider_registry = registry_instance or ProviderRegistry(settings=self.settings)

        # If a custom or mock genai.Client was passed (e.g. in test suites), register a GeminiProvider with it
        if self._client is not None:
            self.provider_registry.register(GeminiProvider(settings=self.settings, client=self._client))

    def _resolve_document(
        self,
        document: Union[str, Path, bytes, ProcessedDocument],
        filename: Optional[str] = None,
    ) -> ProcessedDocument:
        """Normalize various input types (text string, file path, bytes, or ProcessedDocument)."""
        if isinstance(document, ProcessedDocument):
            return document

        if isinstance(document, bytes):
            return DocumentLoader.load_from_bytes(file_bytes=document, filename=filename)

        if isinstance(document, Path):
            return DocumentLoader.load_from_path(document)

        if isinstance(document, str):
            # Check if string is an existing file path on disk
            try:
                candidate_path = Path(document)
                if candidate_path.exists() and candidate_path.is_file():
                    return DocumentLoader.load_from_path(candidate_path)
            except (OSError, ValueError):
                pass

            # Otherwise, treat as raw text content
            if not document.strip():
                raise EmptyDocumentError("Input document text cannot be empty or only whitespace.")

            return ProcessedDocument(
                filename=filename or "document.txt",
                extension=".txt",
                mime_type="text/plain",
                text_content=document.strip(),
                is_multimodal=False,
            )

        raise TypeError(f"Unsupported document input type: {type(document).__name__}")

    def extract(
        self,
        document: Optional[Union[str, Path, bytes, ProcessedDocument]] = None,
        document_type: str = "invoice",
        model_name: Optional[str] = None,
        filename: Optional[str] = None,
        document_text: Optional[Union[str, Path, bytes, ProcessedDocument]] = None,
        demo_mode: bool = False,
        provider: Optional[str] = None,
    ) -> ExtractionResult:
        """
        Extract structured data from any file type (PDF, DOCX, Images, Text, CSV, JSON)
        using multi-provider LLM backends (Gemini, Groq, Cerebras, OpenRouter),
        schema constraints, autonomous routing, local validation, and agentic auditing.

        Args:
            document: Raw text string, file path, file bytes, or ProcessedDocument.
            document_type: Registered document type ('invoice', 'resume', 'receipt') or 'auto'.
            model_name: Optional override for model name (or 'demo' for instant mock review).
            filename: Optional filename hint when passing raw bytes or text.
            document_text: Backward-compatible alias for document parameter.
            demo_mode: If True, returns pre-verified mock result with zero API dependency.
            provider: LLM provider to use ('gemini', 'groq', 'cerebras', 'openrouter', or 'auto').

        Returns:
            ExtractionResult containing validated data, metadata, and verification audit report.
        """
        doc_input = document if document is not None else document_text
        if doc_input is None:
            raise EmptyDocumentError("Input document text cannot be empty or only whitespace.")

        # 1. Resolve & normalize document
        proc_doc = self._resolve_document(doc_input, filename=filename)
        if not proc_doc.has_content:
            raise EmptyDocumentError(f"Document '{proc_doc.filename or 'unnamed'}' has no extractable content.")

        # 2. Autonomous Document Classification (Model Chaining & Routing)
        effective_type = document_type.lower().strip()
        if effective_type == "auto":
            effective_type = DocumentClassifier.classify(
                text=proc_doc.text_content or "",
                filename=proc_doc.filename,
            )
            logger.info("Auto-routed document to specialized schema: '%s'", effective_type)

        # 3. Schema lookup
        try:
            schema_cls = registry.get(effective_type)
        except KeyError as err:
            raise UnsupportedDocumentTypeError(str(err)) from err

        json_schema = schema_cls.model_json_schema()
        input_fmt = proc_doc.extension.lstrip(".") or "text"

        # 4. Safe Demo Mode (For live evaluations without API key or network limits)
        is_demo = demo_mode or (model_name == "demo") or (provider == "demo")
        if is_demo:
            mock_data = get_mock_extraction(effective_type)
            validated_model = schema_cls.model_validate(mock_data)
            audit_report = VerificationAgent.audit_extraction(effective_type, validated_model.model_dump())
            return ExtractionResult(
                data=validated_model.model_dump(),
                metadata=ExtractionMetadata(
                    document_type=effective_type,
                    provider_used=provider or "demo",
                    model_used="demo-mode-deterministic",
                    duration_ms=45.0,
                    fallback_used=False,
                    input_format=input_fmt,
                    is_multimodal=proc_doc.is_multimodal,
                    confidence_score=audit_report.confidence_score,
                    math_verified=audit_report.math_verified,
                    verification=audit_report,
                    is_demo=True,
                ),
            )

        # 5. Resolve active provider and construct candidate failover chain
        active_provider = self.provider_registry.resolve_provider(
            requested_provider=provider,
            is_multimodal=proc_doc.is_multimodal,
        )

        candidate_providers: list[LLMProvider] = [active_provider]
        if provider is None or provider.lower().strip() == "auto":
            fallbacks = self.provider_registry.get_fallback_chain(
                active_provider.name,
                is_multimodal=proc_doc.is_multimodal,
            )
            candidate_providers.extend(fallbacks)
        elif not active_provider.is_available():
            # If the requested provider has no API key, chain any available configured providers
            fallbacks = self.provider_registry.get_fallback_chain(
                active_provider.name,
                is_multimodal=proc_doc.is_multimodal,
            )
            candidate_providers.extend(fallbacks)


        raw_json: Optional[str] = None
        used_provider = active_provider
        last_error: Optional[Exception] = None
        fallback_used = False
        start_time = time.perf_counter()

        # 6. Execute model extraction across provider chain
        for idx, current_provider in enumerate(candidate_providers):
            is_failover = idx > 0
            if is_failover:
                fallback_used = True
                logger.warning(
                    "Primary provider failed; attempting failover to '%s'...",
                    current_provider.display_name,
                )

            try:
                raw_json = current_provider.extract_json(
                    document=proc_doc,
                    document_type=effective_type,
                    json_schema=json_schema,
                    model_name=model_name if idx == 0 else None,
                    timeout=self.settings.timeout_seconds,
                )
                used_provider = current_provider
                break

            except (EmptyDocumentError, UnsupportedDocumentTypeError):
                raise
            except MissingAPIKeyError as key_err:
                last_error = key_err
                if len(candidate_providers) == 1:
                    raise
                continue
            except Exception as exc:
                logger.warning("Provider '%s' error: %s", current_provider.display_name, exc)
                last_error = exc
                continue

        if not raw_json:
            raise ModelExecutionError(
                f"All provider extraction attempts failed. Last error: {last_error}"
            ) from last_error

        # 7. Local Pydantic validation (defense in depth)
        try:
            validated_model = schema_cls.model_validate_json(raw_json)
        except ValidationError as val_err:
            raise SchemaValidationError(
                f"Output from '{used_provider.display_name}' failed schema validation: {val_err}"
            ) from val_err

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(
            "Extraction completed successfully in %.2f ms using %s (%s)",
            duration_ms,
            used_provider.display_name,
            model_name or used_provider.default_model,
        )

        # 8. Agentic Verification & Confidence Audit
        audit_report = VerificationAgent.audit_extraction(effective_type, validated_model.model_dump())

        return ExtractionResult(
            data=validated_model.model_dump(),
            metadata=ExtractionMetadata(
                document_type=effective_type,
                provider_used=used_provider.name,
                model_used=model_name or used_provider.default_model,
                duration_ms=duration_ms,
                fallback_used=fallback_used,
                input_format=input_fmt,
                is_multimodal=proc_doc.is_multimodal,
                confidence_score=audit_report.confidence_score,
                math_verified=audit_report.math_verified,
                verification=audit_report,
                is_demo=False,
            ),
        )


# Convenience function for backward compatibility
def extract_document(
    document_text: Union[str, Path, bytes, ProcessedDocument],
    document_type: str = "invoice",
    model_name: Optional[str] = None,
    filename: Optional[str] = None,
    provider: Optional[str] = None,
) -> StrictModel:
    """Backward-compatible extraction function returning the validated Pydantic model."""
    service = ExtractorService()
    result = service.extract(
        document=document_text,
        document_type=document_type,
        model_name=model_name,
        filename=filename,
        provider=provider,
    )
    schema_cls = registry.get(result.metadata.document_type)
    return result.to_model(schema_cls)
