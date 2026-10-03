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

from config import settings

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
    index = pc.Index(settings.PINECONE_INDEX_NAME)

    # 1. Generate the vector via the official Google GenAI Client
    response = _embedding_client.models.embed_content(
        model=settings.GEMINI_EMBEDDING_MODEL,
        contents=query_text,
    )
    
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
        namespace="_default_"  # Matches the exact metadata partition name from your dashboard screenshot
    )

    # 4. Map the matches cleanly to avoid breaking downstream LangGraph schemas
    hits = []
    for match in query_res.get("matches", []):
        hits.append({
            "document": match.metadata.get("document", ""),
            "metadata": match.metadata,
            "distance": match.score,  # Maps similarity score over cleanly
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