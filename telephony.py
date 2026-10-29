"""Twilio phone calls: the inbound webhook plus REST helpers to dial out and transfer.

Twilio streams call audio over a WebSocket to the Pipecat runner's /ws endpoint. Stream
<Parameter>s carry call details (caller ID, direction, who we're calling and why) to the
bot, where they arrive as call_data.body.

The same code would work with Exotel or Plivo for Indian numbers; Pipecat supports
both as transports. Only this module's TwiML and REST calls are Twilio-specific.
"""

import os
from xml.sax.saxutils import quoteattr

import httpx
from fastapi import Request
from fastapi.responses import Response
from loguru import logger

from pipecat.runner.run import app

API = "https://api.twilio.com/2010-04-01"
FINAL_STATUSES = {"completed", "busy", "no-answer", "failed", "canceled"}


def stream_twiml(host: str, parameters: dict[str, str]) -> str:
    """TwiML that connects the call's audio to the bot's WebSocket."""
    params = "".join(
        f"<Parameter name={quoteattr(k)} value={quoteattr(str(v))}/>" for k, v in parameters.items()
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?><Response><Connect>'
        f'<Stream url="wss://{host}/ws">{params}</Stream>'
        "</Connect></Response>"
    )


@app.post("/twilio/voice")
async def inbound_call(request: Request):
    """Webhook for the Twilio number's "A call comes in" setting."""
    form = await request.form()
    # Twilio reaches us through the public tunnel, so its Host header is the public host.
    host = request.headers.get("x-forwarded-host") or request.headers["host"]
    logger.info(f"Inbound call from {form.get('From')} to {form.get('To')}")
    twiml = stream_twiml(
        host,
        {"direction": "inbound", "from_number": form.get("From", ""), "to_number": form.get("To", "")},
    )
    return Response(content=twiml, media_type="application/xml")


def _auth() -> tuple[str, str]:
    return os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"]


def _calls_url() -> str:
    return f"{API}/Accounts/{os.environ['TWILIO_ACCOUNT_SID']}/Calls"


async def place_call(to_e164: str, twiml: str, ring_secs: int = 25) -> str:
    """Start an outbound call and return its Call SID."""
    data = {"To": to_e164, "From": os.environ["TWILIO_PHONE_NUMBER"], "Twiml": twiml, "Timeout": ring_secs}
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(f"{_calls_url()}.json", data=data, auth=_auth())
    if resp.status_code >= 400:
        raise RuntimeError(resp.json().get("message", resp.text))
    return resp.json()["sid"]


async def call_status(call_sid: str) -> dict:
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(f"{_calls_url()}/{call_sid}.json", auth=_auth())
    resp.raise_for_status()
    return resp.json()


async def transfer_call(call_sid: str, to_number: str) -> None:
    """Move a live call off the bot and ring a staff phone."""
    twiml = f'<?xml version="1.0" encoding="UTF-8"?><Response><Dial>{to_number}</Dial></Response>'
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(f"{_calls_url()}/{call_sid}.json", data={"Twiml": twiml}, auth=_auth())
    resp.raise_for_status()
