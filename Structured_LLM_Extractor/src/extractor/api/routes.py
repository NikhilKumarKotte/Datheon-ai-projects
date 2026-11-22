import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from src.extractor.config import Settings, get_settings
from src.extractor.core.exceptions import (
    EmptyDocumentError,
    ExtractorError,
    MissingAPIKeyError,
    ModelExecutionError,
    SchemaValidationError,
    UnsupportedDocumentTypeError,
)
from src.extractor.core.service import ExtractionResult, ExtractorService
from src.extractor.schemas.registry import registry

from src.extractor.providers import provider_registry

logger = logging.getLogger("extractor.api")
router = APIRouter()


class ExtractionRequest(BaseModel):
    """Payload for text extraction requests."""
    text: str = Field(
        ...,
        description="Raw document text to parse and extract structured data from",
        examples=["Invoice INV-1042\nSeller: ABC Technologies\nTotal: INR 12,500\nStatus: Paid"],
    )
    document_type: str = Field(
        default="auto",
        description="Type of document schema ('auto', 'invoice', 'resume', 'receipt')",
        examples=["auto"],
    )
    provider: Optional[str] = Field(
        default=None,
        description="LLM provider ('auto', 'gemini', 'groq', 'cerebras', 'openrouter')",
    )
    model: Optional[str] = Field(
        default=None,
        description="Optional model override (e.g. 'llama-3.3-70b-versatile', 'gemini-3.5-flash-lite')",
    )
    demo_mode: bool = Field(
        default=False,
        description="If True, runs in offline pre-verified demo mode with zero API key dependencies",
    )


class HealthResponse(BaseModel):
    """Application health and readiness status."""
    status: str
    version: str
    environment: str
    api_key_configured: bool
    default_model: str
    default_provider: str = "gemini"
    configured_providers: List[str] = []
    supported_schemas: List[str]


def get_extractor_service(settings: Settings = Depends(get_settings)) -> ExtractorService:
    """Dependency injector for ExtractorService."""
    return ExtractorService(settings=settings)


class ApiKeysPayload(BaseModel):
    """Payload for updating API keys directly from the Web UI or API."""
    gemini_api_key: Optional[str] = Field(default=None, description="Google Gemini API key")
    groq_api_key: Optional[str] = Field(default=None, description="GroqCloud API key")
    cerebras_api_key: Optional[str] = Field(default=None, description="Cerebras API key")
    openrouter_api_key: Optional[str] = Field(default=None, description="OpenRouter API key")
    persist_to_env: bool = Field(default=True, description="Persist keys to local .env file")


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Readiness and health check endpoint for container orchestrators and monitoring."""
    configured = [p.name for p in provider_registry.list_providers() if p.is_available()]
    has_any_key = len(configured) > 0

    return HealthResponse(
        status="healthy" if has_any_key else "degraded",
        version="1.0.0",
        environment=settings.environment,
        api_key_configured=has_any_key,
        default_model=settings.default_model,
        default_provider=settings.default_provider,
        configured_providers=configured,
        supported_schemas=registry.list_types(),
    )


@router.get("/api/v1/providers", tags=["Providers"])
def list_providers() -> List[Dict[str, Any]]:
    """List all supported LLM providers (Gemini, Groq, Cerebras, OpenRouter) with metadata and status."""
    return provider_registry.get_metadata()


@router.get("/api/v1/schemas", tags=["Schemas"])
def list_schemas() -> List[Dict[str, Any]]:
    """List all registered document types with their full JSON schemas and descriptions."""
    return registry.get_metadata()


@router.get("/api/v1/settings/keys", tags=["Settings"])
def get_configured_keys(settings: Settings = Depends(get_settings)) -> Dict[str, Any]:
    """Return masked status of configured API keys for all providers."""
    from src.extractor.config import mask_api_key
    return {
        "gemini": {
            "configured": bool(mask_api_key(settings.gemini_api_key)),
            "masked": mask_api_key(settings.gemini_api_key),
        },
        "groq": {
            "configured": bool(mask_api_key(settings.groq_api_key)),
            "masked": mask_api_key(settings.groq_api_key),
        },
        "cerebras": {
            "configured": bool(mask_api_key(settings.cerebras_api_key)),
            "masked": mask_api_key(settings.cerebras_api_key),
        },
        "openrouter": {
            "configured": bool(mask_api_key(settings.openrouter_api_key)),
            "masked": mask_api_key(settings.openrouter_api_key),
        },
    }


@router.post("/api/v1/settings/keys", tags=["Settings"])
def save_api_keys(payload: ApiKeysPayload) -> Dict[str, Any]:
    """Save API keys directly from the interface and apply immediately."""
    from src.extractor.config import mask_api_key, update_api_keys
    updated = update_api_keys(
        gemini_key=payload.gemini_api_key,
        groq_key=payload.groq_api_key,
        cerebras_key=payload.cerebras_api_key,
        openrouter_key=payload.openrouter_api_key,
        persist_to_env=payload.persist_to_env,
    )
    return {
        "status": "success",
        "message": "API keys updated and activated successfully.",
        "keys": {
            "gemini": bool(mask_api_key(updated.gemini_api_key)),
            "groq": bool(mask_api_key(updated.groq_api_key)),
            "cerebras": bool(mask_api_key(updated.cerebras_api_key)),
            "openrouter": bool(mask_api_key(updated.openrouter_api_key)),
        },
    }


@router.post("/api/v1/extract", response_model=ExtractionResult, tags=["Extraction"])
def extract_from_text(
    payload: ExtractionRequest,
    service: ExtractorService = Depends(get_extractor_service),
) -> ExtractionResult:
    """Extract structured data from raw text using schema constraints, auto-classification, and verification."""
    try:
        return service.extract(
            document_text=payload.text,
            document_type=payload.document_type,
            provider=payload.provider,
            model_name=payload.model,
            demo_mode=payload.demo_mode,
        )
    except EmptyDocumentError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except UnsupportedDocumentTypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except MissingAPIKeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{exc} Tip: You can pass demo_mode=True to test the full pipeline in offline review mode.",
        ) from exc
    except SchemaValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Schema validation failed: {exc}",
        ) from exc
    except ModelExecutionError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Extraction model failure: {exc}",
        ) from exc
    except ExtractorError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


@router.post("/api/v1/extract/upload", response_model=ExtractionResult, tags=["Extraction"])
async def extract_from_file_upload(
    file: UploadFile = File(..., description="Document file (PDF, DOCX, PNG, JPG, TXT, CSV, etc.)"),
    document_type: str = Form(default="auto", description="Schema type ('auto', 'invoice', 'resume', 'receipt')"),
    provider: Optional[str] = Form(default=None, description="LLM provider ('auto', 'gemini', 'groq', 'cerebras', 'openrouter')"),
    model: Optional[str] = Form(default=None, description="Optional model override"),
    demo_mode: bool = Form(default=False, description="Run in offline deterministic Demo Mode"),
    service: ExtractorService = Depends(get_extractor_service),
) -> ExtractionResult:
    """Extract structured data from any uploaded file format (PDF, Word DOCX, Image, Text, CSV, JSON)."""
    try:
        content_bytes = await file.read()
        return service.extract(
            document=content_bytes,
            document_type=document_type,
            provider=provider or None,
            model_name=model or None,
            filename=file.filename,
            demo_mode=demo_mode,
        )
    except EmptyDocumentError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except UnsupportedDocumentTypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except MissingAPIKeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{exc} Tip: Click '⚙️ API Keys' in the header to configure your key, or toggle Demo Mode.",
        ) from exc
    except SchemaValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Schema validation failed: {exc}",
        ) from exc
    except ModelExecutionError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Extraction model failure: {exc}",
        ) from exc
    except ExtractorError as exc:
        logger.error("Extractor error during file upload: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.error("Unhandled error during file upload: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error processing document: {exc}",
        ) from exc

