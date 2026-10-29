"""WhatsApp confirmations via the Twilio WhatsApp Sandbox (free for testing).

The recipient must first join the sandbox by sending the join code to the sandbox
number from their WhatsApp. Without Twilio credentials, messages are only logged.
"""

import os

import httpx
from loguru import logger

SANDBOX_FROM = "whatsapp:+14155238886"


async def send_whatsapp(to_e164: str, body: str) -> tuple[str, str | None]:
    """Send a WhatsApp message. Returns (status, error) where status is
    "sent", "failed" or "not_configured"."""
    sid = os.getenv("TWILIO_ACCOUNT_SID")
    token = os.getenv("TWILIO_AUTH_TOKEN")
    if not (sid and token):
        logger.info(f"[whatsapp not configured] to {to_e164}: {body}")
        return "not_configured", None

    url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
    data = {
        "From": os.getenv("TWILIO_WHATSAPP_FROM", SANDBOX_FROM),
        "To": f"whatsapp:{to_e164}",
        "Body": body,
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, data=data, auth=(sid, token))
        if resp.status_code >= 400:
            error = resp.json().get("message", resp.text)
            logger.warning(f"WhatsApp send failed: {error}")
            return "failed", error
        return "sent", None
    except httpx.HTTPError as e:
        logger.warning(f"WhatsApp send failed: {e}")
        return "failed", str(e)
