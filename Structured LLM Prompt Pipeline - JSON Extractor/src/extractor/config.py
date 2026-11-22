from functools import lru_cache
from typing import Any, Dict, List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Provider Selection
    default_provider: str = Field(
        default="gemini",
        alias="DEFAULT_PROVIDER",
        description="Default LLM provider: 'gemini', 'groq', 'cerebras', 'openrouter', or 'auto'",
    )

    # Gemini API Credentials & Models
    gemini_api_key: str = Field(
        default="",
        alias="GEMINI_API_KEY",
        description="Google Gemini API key",
    )
    default_model: str = Field(
        default="gemini-3.5-flash-lite",
        alias="DEFAULT_MODEL",
        description="Default Gemini model to use for extraction",
    )
    fallback_models: List[str] = Field(
        default=[
            "gemini-3.5-flash-lite",
            "gemini-flash-lite-latest",
            "gemini-3-flash-preview",
            "gemini-3.8-flash",
        ],
        alias="FALLBACK_MODELS",
        description="List of fallback models if primary model is unavailable or rate-limited",
    )

    # GroqCloud Credentials & Models
    groq_api_key: str = Field(
        default="",
        alias="GROQ_API_KEY",
        description="GroqCloud API key (free access at console.groq.com)",
    )
    groq_default_model: str = Field(
        default="llama-3.3-70b-versatile",
        alias="GROQ_DEFAULT_MODEL",
        description="Default Groq model (e.g., llama-3.3-70b-versatile, llama-3.1-8b-instant)",
    )

    # Cerebras Credentials & Models
    cerebras_api_key: str = Field(
        default="",
        alias="CEREBRAS_API_KEY",
        description="Cerebras Cloud API key (free access at cloud.cerebras.ai)",
    )
    cerebras_default_model: str = Field(
        default="llama3.3-70b",
        alias="CEREBRAS_DEFAULT_MODEL",
        description="Default Cerebras model (e.g., llama3.3-70b, llama3.1-8b)",
    )

    # OpenRouter Credentials & Models
    openrouter_api_key: str = Field(
        default="",
        alias="OPENROUTER_API_KEY",
        description="OpenRouter API key (access 200+ models at openrouter.ai)",
    )
    openrouter_default_model: str = Field(
        default="meta-llama/llama-3.3-70b-instruct:free",
        alias="OPENROUTER_DEFAULT_MODEL",
        description="Default OpenRouter model (e.g., meta-llama/llama-3.3-70b-instruct:free)",
    )

    # Server Configuration
    server_host: str = Field(
        default="0.0.0.0",
        alias="HOST",
        description="API server binding host",
    )
    server_port: int = Field(
        default=8000,
        alias="PORT",
        description="API server binding port",
    )
    cors_origins: List[str] = Field(
        default=["*"],
        alias="CORS_ORIGINS",
        description="Allowed CORS origins",
    )

    # Operational Settings
    max_retries: int = Field(
        default=2,
        alias="MAX_RETRIES",
        description="Maximum retry attempts on transient model failures",
    )
    timeout_seconds: float = Field(
        default=45.0,
        alias="TIMEOUT_SECONDS",
        description="HTTP request timeout for LLM client",
    )
    environment: str = Field(
        default="development",
        alias="ENVIRONMENT",
        description="Deployment environment (development, staging, production)",
    )
    log_level: str = Field(
        default="INFO",
        alias="LOG_LEVEL",
        description="Logging level",
    )


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()


def mask_api_key(key: str) -> Optional[str]:
    """Mask an API key for safe UI and API display (e.g. 'gsk_••••••••1234')."""
    if not key or key.startswith("your_") or key == "replace_with_your_api_key":
        return None
    k = key.strip()
    if len(k) <= 8:
        return "••••••••"
    return f"{k[:4]}••••••••{k[-4:]}"


def update_env_file(key_values: dict) -> None:
    """Safely update or add key-value pairs in the local .env file."""
    from pathlib import Path
    env_path = Path(".env")
    lines = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    elif Path(".env.example").exists():
        lines = Path(".env.example").read_text(encoding="utf-8").splitlines()

    existing_keys = set()
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            k, _ = stripped.split("=", 1)
            k = k.strip()
            if k in key_values:
                existing_keys.add(k)
                new_lines.append(f"{k}={key_values[k]}")
                continue
        new_lines.append(line)

    for k, v in key_values.items():
        if k not in existing_keys:
            new_lines.append(f"{k}={v}")

    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def update_api_keys(
    gemini_key: Optional[str] = None,
    groq_key: Optional[str] = None,
    cerebras_key: Optional[str] = None,
    openrouter_key: Optional[str] = None,
    persist_to_env: bool = True,
) -> Settings:
    """Update runtime API keys in the Settings singleton and optionally persist to .env."""
    from typing import Optional
    settings = get_settings()
    env_updates = {}

    if gemini_key is not None and gemini_key.strip():
        settings.gemini_api_key = gemini_key.strip()
        env_updates["GEMINI_API_KEY"] = gemini_key.strip()

    if groq_key is not None and groq_key.strip():
        settings.groq_api_key = groq_key.strip()
        env_updates["GROQ_API_KEY"] = groq_key.strip()

    if cerebras_key is not None and cerebras_key.strip():
        settings.cerebras_api_key = cerebras_key.strip()
        env_updates["CEREBRAS_API_KEY"] = cerebras_key.strip()

    if openrouter_key is not None and openrouter_key.strip():
        settings.openrouter_api_key = openrouter_key.strip()
        env_updates["OPENROUTER_API_KEY"] = openrouter_key.strip()

    if persist_to_env and env_updates:
        update_env_file(env_updates)

    # Re-initialize providers with updated keys
    try:
        from src.extractor.providers import provider_registry
        provider_registry.settings = settings
        provider_registry._initialize_providers()
    except Exception:
        pass

    return settings
