"""
config.py
---------
Centralized configuration for the Dubai Luxury Real Estate Concierge POC.

All secrets/credentials are pulled exclusively from environment variables
(loaded via a local .env file through python-dotenv). No hardcoded tokens
are permitted anywhere in this codebase.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load variables from a local .env file (if present) into os.environ.
# In production (e.g. Docker, Render, Railway free tiers) these are injected
# directly by the platform, so load_dotenv() is a safe no-op there.
load_dotenv()


def _require_env(key: str, default: str | None = None, required: bool = False) -> str:
    """Fetch an environment variable, optionally enforcing presence."""
    value = os.environ.get(key, default)
    if required and not value:
        raise EnvironmentError(
            f"Missing required environment variable: '{key}'. "
            f"Add it to your .env file or export it before starting the app."
        )
    return value


@dataclass(frozen=True)
class Settings:
    # --- LLM Engine ---------------------------------------------------
    GOOGLE_API_KEY: str = field(default_factory=lambda: _require_env("GOOGLE_API_KEY", required=True))
    GEMINI_MODEL: str = field(default_factory=lambda: _require_env("GEMINI_MODEL", "gemini-2.5-flash"))
    GEMINI_TEMPERATURE: float = field(default_factory=lambda: float(_require_env("GEMINI_TEMPERATURE", "0.4")))

    # --- Webhook / Server ----------------------------------------------
    APP_HOST: str = field(default_factory=lambda: _require_env("APP_HOST", "0.0.0.0"))
    APP_PORT: int = field(default_factory=lambda: int(_require_env("APP_PORT", "8000")))
    WEBHOOK_VERIFY_TOKEN: str = field(default_factory=lambda: _require_env("WEBHOOK_VERIFY_TOKEN", "dev-verify-token"))

    # --- Escalation / Notification --------------------------------------
    # The developer verification device that receives escalation + qualified-lead alerts.
    # NOTE: This is a POC-fixed destination (not a secret), set via env for flexibility.
    ESCALATION_PHONE_NUMBER: str = field(default_factory=lambda: _require_env("ESCALATION_PHONE_NUMBER", "+918484820346"))
    
    NOTIFICATION_TIMEOUT_SECONDS: float = field(default_factory=lambda: float(_require_env("NOTIFICATION_TIMEOUT_SECONDS", "5.0")))

    # --- Storage Paths ----------------------------------------------------
    BASE_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    SQLITE_DB_PATH: str = field(default_factory=lambda: _require_env("SQLITE_DB_PATH", ":memory:"))

    TELEGRAM_BOT_TOKEN: str = field(default_factory=lambda: _require_env("TELEGRAM_BOT_TOKEN", required=True))
    TELEGRAM_ADMIN_CHAT_ID: str = field(default_factory=lambda: _require_env("TELEGRAM_ADMIN_CHAT_ID", required=True))
    TELEGRAM_WEBHOOK_SECRET: str = field(default_factory=lambda: _require_env("TELEGRAM_WEBHOOK_SECRET", "dev-telegram-secret"))
    TELEGRAM_REPLY_TIMEOUT_SECONDS: float = field(default_factory=lambda: float(_require_env("TELEGRAM_REPLY_TIMEOUT_SECONDS", "15.0")))

    QUALIFIED_LEAD_CUSTOMER_REPLY: str = (
        "Wonderful — I have everything I need. One of our senior advisors will "
        "personally reach out within the next 3-4 hours to confirm the final "
        "details and next steps. Thank you for your patience."
    )
    
    # --- Vector Store (Pinecone Serverless) --------------------------------
    # Pinecone serverless index. PINECONE_API_KEY is mandatory — there is no
    # safe default for a credential, so this fails fast at startup if unset.
    PINECONE_API_KEY: str = field(default_factory=lambda: _require_env("PINECONE_API_KEY", required=True))
    PINECONE_INDEX_NAME: str = field(default_factory=lambda: _require_env(
        "PINECONE_INDEX_NAME", "cx-ai-agent-realestate-leadqualifier"
    ))
    PINECONE_NAMESPACE: str = field(default_factory=lambda: _require_env("PINECONE_NAMESPACE", "__default__"))
    PINECONE_INDEX_HOST: str = field(default_factory=lambda: _require_env("PINECONE_INDEX_HOST", required=True))
    GEMINI_EMBEDDING_MODEL: str = field(default_factory=lambda: _require_env("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"))

    # --- Safety / Guardrail Thresholds -------------------------------------
    SENTIMENT_ESCALATION_THRESHOLD: float = field(default_factory=lambda: float(_require_env("SENTIMENT_ESCALATION_THRESHOLD", "-0.4")))
    MAX_MESSAGE_LENGTH: int = field(default_factory=lambda: int(_require_env("MAX_MESSAGE_LENGTH", "2000")))
    MAX_TURNS_BEFORE_SOFT_ESCALATION: int = field(default_factory=lambda: int(_require_env("MAX_TURNS_BEFORE_SOFT_ESCALATION", "12")))

    # Hard keyword triggers for instant human handoff (case-insensitive).
    HUMAN_HANDOFF_KEYWORDS: tuple = (
        "talk to a human agent",
        "talk to a real agent",
        "speak to a human",
        "real person",
        "human agent",
        "manager",
        "this is ridiculous",
        "not helpful",
        "useless bot",
    )

    FALLBACK_NO_DATA_PHRASE: str = field(
        default_factory=lambda: _require_env(
            "FALLBACK_NO_DATA_PHRASE",
            "I currently don't have an active listing matching those exact criteria in our portfolio, but I can check with our off-market sourcing team for you."
        )
    )


settings = Settings()