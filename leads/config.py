"""Configuration, loaded once from the environment / ``.env``.

Design rules:
  * Secrets never have code defaults. A missing key is a ``ConfigError``.
  * Every tunable has a safe default so a fresh clone runs with only keys set.
  * Validation happens at startup (``doctor``), not mid-run.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime settings. Env var names are case-insensitive."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM ---
    openrouter_api_key: str = Field(default="")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1")
    model: str = Field(default="")

    # --- data sources ---
    tavily_api_key: str = Field(default="")
    scraperapi_api_key: str = Field(default="")
    apify_api_token: str = Field(default="")

    # --- runtime ---
    leads_home: Path = Field(default=Path("var"))
    concurrency: int = Field(default=4, ge=1, le=32)
    max_llm_calls: int = Field(default=500, ge=0)
    max_tavily_credits: int = Field(default=900, ge=0)
    http_cache_ttl_hours: int = Field(default=168, ge=0)
    request_timeout: float = Field(default=45.0, gt=0)
    llm_timeout: float = Field(default=180.0, gt=0)
    llm_max_tokens: int = Field(default=1500, ge=256, le=8192)
    llm_max_consecutive_failures: int = Field(default=6, ge=1)
    log_level: str = Field(default="INFO")
    user_agent: str = Field(
        default="webdev-leads/1.0 (+https://example.invalid/contact; lead research bot)"
    )

    # --- pipeline defaults ---
    default_min_rating: float = Field(default=4.5, ge=0, le=5)
    default_min_reviews: int = Field(default=10, ge=0)
    max_results_per_query: int = Field(default=8, ge=1, le=20)

    @field_validator("openrouter_base_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, v: str) -> str:
        v = v.upper()
        if v not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"invalid log level: {v}")
        return v

    # ------------------------------------------------------------------ paths
    @property
    def home(self) -> Path:
        """Directory holding the SQLite database, logs and run artifacts."""
        p = self.leads_home
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p

    @property
    def db_path(self) -> Path:
        return self.home / "leads.sqlite3"

    @property
    def logs_dir(self) -> Path:
        return self.home / "logs"

    @property
    def runs_dir(self) -> Path:
        return self.home / "runs"

    def ensure_dirs(self) -> None:
        for d in (self.home, self.logs_dir, self.runs_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- validation
    @property
    def llm_ready(self) -> bool:
        return bool(self.openrouter_api_key and self.model)

    @property
    def tavily_ready(self) -> bool:
        return bool(self.tavily_api_key)

    def require_llm(self) -> None:
        if not self.openrouter_api_key:
            raise ConfigError(
                "OPENROUTER_API_KEY is not set. Add it to .env (see .env.example)."
            )
        if not self.model:
            raise ConfigError("MODEL is not set. Add it to .env (see .env.example).")

    def require_any_source(self) -> None:
        if not self.tavily_ready and not self.scraperapi_api_key:
            raise ConfigError(
                "No discovery source configured. Set TAVILY_API_KEY (free tier) — "
                "OpenStreetMap discovery works without a key but coverage is thin."
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()


def reset_settings_cache() -> None:
    """Test helper: forget the cached settings."""
    get_settings.cache_clear()
