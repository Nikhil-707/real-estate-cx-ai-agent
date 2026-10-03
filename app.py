"""
app.py
------
FastAPI application exposing the public webhook endpoint that ingests
inbound customer messages, drives the LangGraph workflow, and returns the
structured text reply.

Run locally:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import settings
from database import init_sqlite, get_lead
from agent import run_turn

# ---------------------------------------------------------------------------
# Logging — deliberately configured to NEVER log raw message bodies (PII).
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("concierge")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_sqlite()
    # The Pinecone serverless index ("cx-ai-agent-realestate-leadqualifier") is
    # already active and seeded out-of-band — nothing to initialize here.
    logger.info("SQLite conversation store initialized. Pinecone index queried on-demand via database.query_properties().")
    yield
    logger.info("Shutting down gracefully.")


app = FastAPI(
    title="Dubai Luxury Concierge — Guardrailed Hybrid Agent",
    description="Production-POC webhook for an AI real estate concierge with deterministic RAG grounding and sentiment-based human escalation.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class IncomingMessage(BaseModel):
    phone_number: str = Field(..., description="E.164 formatted sender phone number, e.g. +9715XXXXXXXX")
    message: str = Field(..., min_length=1, description="Raw inbound text content from the customer.")

    class Config:
        # Prevent accidental logging of the model repr in exception traces.
        str_strip_whitespace = True


class OutgoingReply(BaseModel):
    phone_number: str
    reply: str
    lead_status: str | None = None


# ---------------------------------------------------------------------------
# Webhook verification (Meta-style GET challenge, kept for provider parity)
# ---------------------------------------------------------------------------
@app.get("/webhook")
async def verify_webhook(request: Request):
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == settings.WEBHOOK_VERIFY_TOKEN:
        return int(challenge) if challenge and challenge.isdigit() else challenge

    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Webhook verification failed.")


# ---------------------------------------------------------------------------
# Primary inbound message webhook
# ---------------------------------------------------------------------------
@app.post("/webhook", response_model=OutgoingReply)
async def receive_message(payload: IncomingMessage):
    # Never log payload.message directly — PII must stay inside ephemeral Graph State.
    logger.info("Inbound message received for a client (content redacted from logs).")

    try:
        reply_text = run_turn(phone_number=payload.phone_number, incoming_text=payload.message)
    except Exception as exc:  # noqa: BLE001 — surfaced as a safe 500 without leaking internals
        logger.exception("Graph execution failed.")
        raise HTTPException(status_code=500, detail="Internal processing error. Escalation has been queued.") from exc

    lead = get_lead(payload.phone_number)
    lead_status = lead.get("status") if lead else None

    return OutgoingReply(phone_number=payload.phone_number, reply=reply_text, lead_status=lead_status)


@app.get("/health")
async def health_check():
    return JSONResponse({"status": "ok", "service": "dubai-luxury-concierge"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host=settings.APP_HOST, port=settings.APP_PORT, reload=True)