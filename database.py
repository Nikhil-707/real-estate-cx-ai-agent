"""
database.py
-----------
Owns the POC's persistence and retrieval layers:

1. SQLite (in-memory, thread-safe) — conversation history keyed strictly by
   the user's incoming phone number. PII never touches disk or logs; it
   lives only for the lifetime of the process.

2. Pinecone (managed, serverless) — the property vector index. All property
   facts are retrieved exclusively from the live "cx-ai-agent-realestate-
   leadqualifier" index, embedded on the fly via Google's gemini-embedding-001
   model. ChromaDB has been fully retired from this stack.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from typing import Iterator

from google import genai
from pinecone import Pinecone
from google.genai import types as genai_types

from config import settings
import time
import re

# ---------------------------------------------------------------------------
# SQLite — Thread-safe in-memory conversation store
# ---------------------------------------------------------------------------
# A single shared in-memory connection with `check_same_thread=False` plus a
# module-level lock gives us safe concurrent access from FastAPI's async
# worker threads without standing up a real database server.

_sqlite_lock = threading.Lock()
_sqlite_conn = sqlite3.connect(settings.SQLITE_DB_PATH, check_same_thread=False)


def init_sqlite() -> None:
    """Create the conversation_history table if it does not already exist."""
    with _sqlite_lock:
        _sqlite_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phone_number TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                content TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        _sqlite_conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_conversation_phone
            ON conversation_history(phone_number)
            """
        )
        _sqlite_conn.commit()


@contextmanager
def _cursor() -> Iterator[sqlite3.Cursor]:
    with _sqlite_lock:
        cur = _sqlite_conn.cursor()
        try:
            yield cur
            _sqlite_conn.commit()
        finally:
            cur.close()


def append_message(phone_number: str, role: str, content: str) -> None:
    """Persist a single turn. `content` stays in-process memory only (no file/log writes)."""
    with _cursor() as cur:
        cur.execute(
            "INSERT INTO conversation_history (phone_number, role, content) VALUES (?, ?, ?)",
            (phone_number, role, content),
        )


def get_history(phone_number: str, limit: int = 20) -> list[dict]:
    """Return the most recent `limit` turns for a given phone number, oldest first."""
    with _cursor() as cur:
        cur.execute(
            """
            SELECT role, content FROM conversation_history
            WHERE phone_number = ?
            ORDER BY id DESC LIMIT ?
            """,
            (phone_number, limit),
        )
        rows = cur.fetchall()
    rows.reverse()
    return [{"role": r, "content": c} for r, c in rows]


def clear_history(phone_number: str) -> None:
    """Explicit purge hook — supports data-subject deletion requests."""
    with _cursor() as cur:
        cur.execute("DELETE FROM conversation_history WHERE phone_number = ?", (phone_number,))


# ---------------------------------------------------------------------------
# Pinecone — Managed serverless vector index for property RAG grounding
# ---------------------------------------------------------------------------
# The index itself ("cx-ai-agent-realestate-leadqualifier") is already active
# and seeded out-of-band (768-dim vectors via gemini-embedding-001). This
# module only ever *queries* it — no local vector store, no local embedding
# model, no ChromaDB client of any kind remains in this file.

_embedding_client = genai.Client(api_key=settings.GOOGLE_API_KEY)
pc = Pinecone(api_key=settings.PINECONE_API_KEY)


def query_properties(query_text: str, n_results: int = 3) -> list[dict]:
    """Deterministic RAG lookup — queries the live Pinecone Serverless Index."""
    index = pc.Index(host=settings.PINECONE_INDEX_HOST)

    # 1. Generate the vector via the official Google GenAI Client
    # Retry loop for transient API cold-starts
    for attempt in range(3):
        try:
            response = _embedding_client.models.embed_content(
                model=settings.GEMINI_EMBEDDING_MODEL,
                contents=query_text,
                config=genai_types.EmbedContentConfig(output_dimensionality=768),
            )
            break
        except Exception as e:
            if attempt == 2:
                raise e
            time.sleep(0.5 * (2 ** attempt))
    
    # 2. Resilient list vs object value extraction matching your SDK environment
    embeddings_attr = response.embeddings
    if isinstance(embeddings_attr, list) and embeddings_attr:
        query_vector = list(embeddings_attr[0].values)
    elif hasattr(embeddings_attr, "values"):
        query_vector = list(embeddings_attr.values)
    else:
        raise ValueError("Could not extract dense vector floats from Google API response.")

    # 3. Query the Pinecone cloud cluster index, explicitly targeting the namespace seen in your console
    query_res = index.query(
        vector=query_vector, 
        top_k=n_results, 
        include_metadata=True,
        namespace=settings.PINECONE_NAMESPACE,
    )

    # 4. Map the matches cleanly to avoid breaking downstream LangGraph schemas
    hits = []
    for match in query_res.matches:
        hits.append({
            "document": (match.metadata or {}).get("document", ""),
            "metadata": dict(match.metadata or {}),
            "distance": match.score,
        })
    return hits


# ---------------------------------------------------------------------------
# Mock CRM — local in-memory dictionary updated on lead qualification
# ---------------------------------------------------------------------------
MOCK_CRM: dict[str, dict] = {}


def upsert_lead(phone_number: str, **fields) -> dict:
    """Merge newly-discovered lead attributes into the mock CRM record."""
    record = MOCK_CRM.setdefault(phone_number, {"phone_number": phone_number, "status": "New"})
    record.update({k: v for k, v in fields.items() if v is not None})
    return record


def get_lead(phone_number: str) -> dict | None:
    return MOCK_CRM.get(phone_number)


def mark_contact_requested(session_key: str) -> dict:
    """Idempotency guard so we only ever send the contact-share prompt once per lead."""
    return upsert_lead(phone_number=session_key, contact_requested=True)


def has_requested_contact(session_key: str) -> bool:
    lead = get_lead(session_key)
    return bool(lead and lead.get("contact_requested"))


# ---------------------------------------------------------------------------
# Phone-number parsing / validation (E.164, country code mandatory)
# ---------------------------------------------------------------------------
_E164_RE = re.compile(r"^\+[1-9]\d{7,14}$")  # + then 8-15 digits, first digit never 0

# Arabic-Indic and Eastern Arabic-Indic digits -> ASCII, so "+٩٧١٥٠..." works too.
_DIGIT_TABLE = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

# A number that STARTS with "+" or "00" (country-code marker), optionally
# followed by spaces/dashes/brackets between groups, e.g.
#   +971501234567 | +971 50 123 4567 | + 971 50-123-4567 | 00971 50 123 4567
_PHONE_CANDIDATE_RE = re.compile(r"(?:\+|00)[\s\-().]*\d[\d\s\-().]{6,22}")


def normalize_phone(raw: str, assume_plus: bool = False) -> str | None:
    """
    Returns a clean E.164 string ('+971501234567') or None if `raw` is not a
    valid international number. Spaces, dashes and brackets are ignored, a
    leading '00' is treated as '+'. `assume_plus=True` is only for payloads
    that are guaranteed to already carry a country code (Telegram's native
    contact share sends '971501234567' without the '+').
    """
    if not raw:
        return None
    s = str(raw).translate(_DIGIT_TABLE)
    s = re.sub(r"[\s\-().]", "", s)
    if s.startswith("00"):
        s = "+" + s[2:]
    elif assume_plus and not s.startswith("+"):
        s = "+" + s
    return s if _E164_RE.match(s) else None


def extract_phone_from_text(text: str) -> str | None:
    """Finds a valid international number inside free text, or None."""
    m = _PHONE_CANDIDATE_RE.search((text or "").translate(_DIGIT_TABLE))
    return normalize_phone(m.group(0)) if m else None


def looks_like_number_without_country_code(text: str) -> bool:
    """True for things like '8484820346' or '084 848 20346' — digits, no +/00 prefix."""
    t = (text or "").translate(_DIGIT_TABLE)
    if extract_phone_from_text(t):
        return False
    return bool(re.search(r"\d[\d\s\-().]{5,}\d", t)) and len(re.sub(r"\D", "", t)) >= 7


# ---------------------------------------------------------------------------
# CRM helpers: identity + contact state
# ---------------------------------------------------------------------------
def set_sender_identity(session_key: str, first_name: str, last_name: str, username: str | None) -> dict:
    """
    Captures Telegram's own identity fields (message.from) — display name and
    @handle come for free in every update. This is NOT a phone number.
    """
    display_name = f"{first_name or ''} {last_name or ''}".strip()
    return upsert_lead(
        phone_number=session_key,
        channel="telegram",
        telegram_display_name=display_name or None,
        telegram_username=username,
    )


def set_verified_phone(session_key: str, phone: str) -> dict | None:
    """
    Stores a real phone number the client explicitly gave us. Returns the CRM
    record, or None (and stores NOTHING) if the number is not valid E.164.
    """
    normalized = normalize_phone(phone, assume_plus=True)
    if not normalized:
        return None
    return upsert_lead(phone_number=session_key, verified_phone=normalized, contact_status="shared")


def set_contact_declined(session_key: str) -> dict:
    return upsert_lead(phone_number=session_key, contact_status="refused")


def set_contact_not_provided(session_key: str) -> dict:
    return upsert_lead(phone_number=session_key, contact_status="not_provided")