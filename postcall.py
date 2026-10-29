"""Post-call pipeline: after every call, turn the transcript into CRM fields.

For each finished call:
1. Booking status comes from the tools' event log (what actually happened), not the LLM.
2. One LLM request reads the transcript and returns the caller's name, intent, outcome,
   sentiment, a short summary and any follow-up for staff, as JSON.
3. Results are saved to the database, data/calls.csv is rewritten, and a row is appended
   to the Google Sheet if one is configured.

The bot runs this automatically when a call ends. To (re)process calls by hand:

    uv run postcall.py          # calls whose analysis is pending or failed
    uv run postcall.py --all    # every call, e.g. after changing the prompt
"""

import argparse
import asyncio
import csv
import json
from pathlib import Path

from loguru import logger

from clinic import db, sheets
from llm import json_completion

CSV_PATH = Path("data/calls.csv")

INTENTS = ["book_appointment", "reschedule", "cancel", "clinic_info", "pricing", "emergency", "complaint", "other"]
OUTCOMES = [
    "booked", "rescheduled", "cancelled", "info_given", "handed_off", "lead_saved",
    "not_interested", "call_back_later", "wrong_number", "do_not_call", "no_conversation", "other",
]
SENTIMENTS = ["positive", "neutral", "negative"]

# Outcomes the event log proves, in priority order after any booking change.
# They override the LLM's label.
EVENT_OUTCOMES = [("handoff", "handed_off"), ("do_not_call", "do_not_call"), ("lead_saved", "lead_saved")]

SYSTEM = f"""You review phone calls handled by Priya, the AI receptionist of a dental clinic.
Read the transcript and the list of actions her tools actually performed, then return a JSON
object with exactly these keys:
- caller_name: the caller's name if they said it, else null
- intent: the caller's main reason for calling, one of {INTENTS}
- outcome: how the call ended, one of {OUTCOMES}
- sentiment: the caller's overall mood, one of {SENTIMENTS}
- summary: one or two plain English sentences for clinic staff
- follow_up: a short instruction for staff if anything needs doing, else null
Trust the actions list over the transcript for what was booked or changed. The transcript
comes from speech recognition and may contain errors or Hindi in Devanagari."""

CSV_FIELDS = [
    "id", "started_at", "channel", "direction", "phone", "duration_secs", "caller_name", "intent",
    "booking_status", "outcome", "sentiment", "summary", "follow_up", "recording_path",
]


def booking_status(events: list[dict]) -> str:
    changes = [e["type"] for e in events if e["type"] in ("booked", "rescheduled", "cancelled")]
    return changes[-1] if changes else "none"


def transcript_text(transcript: list[dict]) -> str:
    speaker = {"user": "Caller", "assistant": "Priya"}
    return "\n".join(
        f"{speaker[t['role']]}: {t['text']}" + (" [interrupted]" if t.get("interrupted") else "")
        for t in transcript
    )


def _pick(value, allowed: list[str], fallback: str) -> str:
    return value if value in allowed else fallback


async def analyse(call) -> dict:
    transcript = json.loads(call["transcript"] or "[]")
    events = json.loads(call["events"] or "[]")
    fields = {"booking_status": booking_status(events)}

    if not any(t["role"] == "user" and t["text"].strip() for t in transcript):
        # Nobody spoke: no LLM call needed.
        return fields | {"intent": "other", "outcome": "no_conversation", "sentiment": "neutral",
                         "summary": "The caller did not say anything.", "caller_name": None, "follow_up": None}

    facts = json.dumps([{k: v for k, v in e.items() if k != "at"} for e in events])
    result = await json_completion(SYSTEM, f"Actions performed:\n{facts}\n\nTranscript:\n{transcript_text(transcript)}")

    outcome = _pick(result.get("outcome"), OUTCOMES, "other")
    proven = {e["type"] for e in events}
    if fields["booking_status"] != "none":
        outcome = fields["booking_status"]
    else:
        outcome = next((label for event, label in EVENT_OUTCOMES if event in proven), outcome)
        if outcome in ("booked", "rescheduled", "cancelled"):
            # The transcript may sound like a booking ("I'll lock that in"), but the tools
            # never made one. Staff must not think an appointment exists when it doesn't.
            outcome = "other"
    return fields | {
        "caller_name": result.get("caller_name") or None,
        "intent": _pick(result.get("intent"), INTENTS, "other"),
        "outcome": outcome,
        "sentiment": _pick(result.get("sentiment"), SENTIMENTS, "neutral"),
        "summary": (result.get("summary") or "").strip() or None,
        "follow_up": result.get("follow_up") or None,
    }


def _csv_row(call) -> dict:
    phone = call["to_number"] if call["direction"] == "outbound" else call["from_number"]
    return {k: call[k] for k in CSV_FIELDS if k != "phone"} | {"phone": phone}


def export_csv() -> None:
    """Rewrite data/calls.csv from the database (opens directly in Excel or Google Sheets)."""
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for call in reversed(db.all_calls()):
            writer.writerow(_csv_row(call))


async def process_call(call_id: int) -> None:
    call = db.get_call(call_id)
    try:
        fields = await analyse(call)
    except Exception as e:
        logger.exception(f"Post-call analysis failed for call {call_id}: {e}")
        db.save_analysis(call_id, "failed", error=str(e))
        return
    db.save_analysis(call_id, "done", **fields)
    logger.info(f"Call {call_id} analysed: {fields['outcome']} / {fields['sentiment']} - {fields['summary']}")

    export_csv()
    if sheets.enabled():
        try:
            row = _csv_row(db.get_call(call_id))
            await sheets.append_row(CSV_FIELDS, [row[k] for k in CSV_FIELDS])
            logger.info(f"Call {call_id} added to Google Sheet")
        except Exception as e:
            logger.exception(f"Google Sheet update failed for call {call_id}: {e}")


async def main(include_done: bool) -> None:
    calls = db.calls_needing_analysis(include_done)
    logger.info(f"Processing {len(calls)} call(s)")
    for call in calls:
        await process_call(call["id"])
    export_csv()


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(override=True)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="Reprocess every call, not just pending/failed ones")
    asyncio.run(main(parser.parse_args().all))
