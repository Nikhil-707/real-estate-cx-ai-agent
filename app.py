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
from database import init_sqlite, get_lead, set_verified_phone, set_sender_identity
from agent import run_turn, send_telegram_reply, handle_contact_reply, pop_pending_keyboard

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

class TelegramUpdate(BaseModel):
    # Telegram's actual update schema is much larger; we only need these paths.
    update_id: int
    message: dict | None = None

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

@app.post("/telegram-webhook/{secret}")
async def telegram_webhook(secret: str, request: Request):
    if secret != settings.TELEGRAM_WEBHOOK_SECRET:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid webhook secret.")

    body = await request.json()
    message = body.get("message")
    if not message:
        # No message payload at all (e.g. some other update type) — ignore.
        return JSONResponse({"ok": True})

    chat_id = str(message["chat"]["id"])

    # Telegram's own identity fields (message.from) arrive with every update.
    sender = message.get("from") or {}
    set_sender_identity(
        chat_id,
        first_name=sender.get("first_name", ""),
        last_name=sender.get("last_name", ""),
        username=sender.get("username"),
    )

    # --- Native contact share (the "Share my contact" button) --------------
    # Must be checked before the "text" guard: contact updates carry no text.
    if message.get("contact"):
        shared_phone = message["contact"].get("phone_number")
        logger.info("Client shared a contact (value redacted from logs).")
        outcome = handle_contact_reply(chat_id, shared_phone=shared_phone or "")
        reply_text, remove_kb = outcome
        send_telegram_reply(chat_id=chat_id, text=reply_text, remove_keyboard=remove_kb)
        return JSONResponse({"ok": True})

    if "text" not in message:
        return JSONResponse({"ok": True})

    text = message["text"]

    # --- Contact stage: a typed number / "skip" / country-code nudge --------
    # Only active while the client has been asked for a number. Everything
    # else (including all normal conversation) falls through to run_turn().
    outcome = handle_contact_reply(chat_id, text=text)
    if outcome is not None:
        reply_text, remove_kb = outcome
        send_telegram_reply(chat_id=chat_id, text=reply_text, remove_keyboard=remove_kb)
        return JSONResponse({"ok": True})

    logger.info("Inbound Telegram message received for a client (content redacted from logs).")

    try:
        reply_text = run_turn(phone_number=chat_id, incoming_text=text)
    except Exception:
        import traceback
        print("FATAL_ERROR_START")
        traceback.print_exc()
        print("FATAL_ERROR_END")
        logger.exception("run_turn failed")
        reply_text = "Apologies, something went wrong on our end. Our team has been notified."

    # If this turn qualified the lead, the reply is the "please share your
    # number" message and it must carry the native share button.
    send_telegram_reply(chat_id=chat_id, text=reply_text, reply_markup=pop_pending_keyboard(chat_id))

    return JSONResponse({"ok": True})

# ---------------------------------------------------------------------------
# Primary inbound message webhook
# ---------------------------------------------------------------------------
@app.post("/webhook", response_model=OutgoingReply)
async def receive_message(payload: IncomingMessage):
    # Never log payload.message directly — PII must stay inside ephemeral Graph State.
    logger.info("Inbound message received for a client (content redacted from logs).")

    # On this channel the session key IS a real phone number; keep it as the contact.
    set_verified_phone(payload.phone_number, payload.phone_number)

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
    return JSONResponse({"status": "ok", "service": "Real Estate - CX AI Agent"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host=settings.APP_HOST, port=settings.APP_PORT, reload=True)