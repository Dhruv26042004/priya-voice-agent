"""Outbound dialer: calls patients from a CSV list, one at a time, with retries.

    uv run outbound.py data/call_list.csv --host <public-host>

The CSV needs columns: name, phone, and optionally purpose. Each call is connected
to the running bot (uv run bot.py), which talks to the patient.

Rules applied before every call:
- Numbers on the do-not-call list are never dialled (patients can ask to be added
  during a call, or you can add them with --add-dnc).
- Calls are only placed within calling hours (10 AM-7 PM IST, Mon-Sat). This sits
  inside TRAI's 9 AM-9 PM window for commercial calls in India.
- No answer, busy or failed calls are retried later, up to MAX_ATTEMPTS times.
"""

import argparse
import asyncio
import csv
import sys
from datetime import timedelta

from dotenv import load_dotenv
from loguru import logger

load_dotenv(override=True)

from clinic import db  # noqa: E402
from clinic.schedule import is_open, normalize_phone, now  # noqa: E402
from telephony import FINAL_STATUSES, call_status, place_call, stream_twiml  # noqa: E402

MAX_ATTEMPTS = 3
CALL_HOURS = (10, 19)
DEFAULT_PURPOSE = "their six-month dental checkup is due"
RETRY_RESULTS = {"busy", "no-answer", "failed"}


def load_list(path: str) -> None:
    added = skipped = 0
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            phone = normalize_phone(row.get("phone", ""))
            if not phone:
                logger.warning(f"Skipping {row.get('name')!r}: invalid number {row.get('phone')!r}")
                skipped += 1
                continue
            if db.add_outbound(row["name"].strip(), phone, (row.get("purpose") or DEFAULT_PURPOSE).strip()):
                added += 1
    logger.info(f"Queued {added} new numbers ({skipped} invalid)")


def within_calling_hours() -> bool:
    current = now()
    return is_open(current.date()) and CALL_HOURS[0] <= current.hour < CALL_HOURS[1]


async def dial(row, host: str, retry_minutes: int) -> None:
    attempt = row["attempts"] + 1
    twiml = stream_twiml(
        host,
        {"direction": "outbound", "outbound_id": row["id"], "to_number": row["phone"]},
    )
    logger.info(f"Calling {row['name']} ({row['phone']}), attempt {attempt}/{MAX_ATTEMPTS}")
    db.update_outbound(row["id"], status="in_progress", attempts=attempt)
    try:
        sid = await place_call(row["phone"], twiml)
    except Exception as e:
        logger.error(f"Could not place call: {e}")
        result = "failed"
        sid = None
    else:
        db.update_outbound(row["id"], last_call_sid=sid)
        status = {}
        while status.get("status") not in FINAL_STATUSES:
            await asyncio.sleep(3)
            status = await call_status(sid)
        result = status["status"]
        logger.info(f"Call to {row['name']} ended: {result} ({status.get('duration', 0)}s)")

    if result in RETRY_RESULTS and attempt < MAX_ATTEMPTS:
        # Back off: each retry waits longer than the last.
        next_at = now() + timedelta(minutes=retry_minutes * attempt)
        db.update_outbound(row["id"], status="retry", last_result=result, next_attempt_at=next_at.isoformat())
        logger.info(f"Will retry {row['name']} after {next_at:%H:%M}")
    else:
        final = "completed" if result == "completed" else "failed"
        db.update_outbound(row["id"], status=final, last_result=result, next_attempt_at=None)


async def run(host: str, retry_minutes: int, ignore_hours: bool) -> None:
    while db.waiting_outbound():
        if not ignore_hours and not within_calling_hours():
            logger.info("Outside calling hours (10 AM-7 PM IST, Mon-Sat). Stopping; run again later.")
            return
        due = db.due_outbound()
        if not due:
            await asyncio.sleep(30)
            continue
        for row in due:
            if db.is_dnc(row["phone"]):
                logger.info(f"Skipping {row['name']}: on the do-not-call list")
                db.update_outbound(row["id"], status="dnc", last_result="do-not-call")
                continue
            await dial(row, host, retry_minutes)
    logger.info("Call list finished")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="?", help="CSV of name, phone[, purpose] to add to the queue")
    parser.add_argument("--host", help="Public host of the running bot, e.g. abc.trycloudflare.com")
    parser.add_argument("--retry-minutes", type=int, default=30, help="Base wait before a retry (default 30)")
    parser.add_argument("--ignore-hours", action="store_true", help="Testing only: call outside calling hours")
    parser.add_argument("--add-dnc", metavar="PHONE", help="Add a number to the do-not-call list and exit")
    args = parser.parse_args()

    if args.add_dnc:
        phone = normalize_phone(args.add_dnc)
        if not phone:
            sys.exit(f"Invalid number: {args.add_dnc}")
        db.add_dnc(phone, "added manually")
        logger.info(f"Added {phone} to the do-not-call list")
        return
    if args.csv:
        load_list(args.csv)
    if not args.host:
        sys.exit("--host is required to place calls (the public host the bot is reachable on)")
    asyncio.run(run(args.host.removeprefix("https://").rstrip("/"), args.retry_minutes, args.ignore_hours))


if __name__ == "__main__":
    main()
