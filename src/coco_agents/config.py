from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["openai", "anthropic", "google", "mistral", "ollama", "openai-compatible"]


class Settings(BaseSettings):
    model_provider: Provider = "openai"
    model_name: str = "gpt-4o-mini"
    model_base_url: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None
    mistral_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    tmdb_api_key: str = ""
    tavily_api_key: str = ""
    coding_workspace_root: Path = Path.home() / "Documents"
    command_timeout_seconds: int = 30

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    def model_api_key(self) -> str | None:
        return {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "google": self.google_api_key,
            "mistral": self.mistral_api_key,
            "openai-compatible": self.openai_api_key,
        }.get(self.model_provider)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    if settings.command_timeout_seconds <= 0:
        raise ValueError("COMMAND_TIMEOUT_SECONDS must be positive")
    if settings.model_provider != "ollama" and not settings.model_api_key():
        key_name = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "openai-compatible": "OPENAI_API_KEY",
        }[settings.model_provider]
        raise ValueError(f"{key_name} is required for MODEL_PROVIDER={settings.model_provider}")
    if not settings.tmdb_api_key.strip():
        raise ValueError("TMDB_API_KEY is required for the movie agent")
    if not settings.tavily_api_key.strip():
        raise ValueError("TAVILY_API_KEY is required for the web agent")
    return settings
