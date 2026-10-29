"""Tools the receptionist LLM can call.

Each is a Pipecat "direct function": the docstring and type hints become the tool
schema the LLM sees. Every tool reports back with a dict containing "ok", so the
LLM can recover from mistakes (wrong date, unconfirmed details) instead of crashing.
"""

import asyncio
import os
import re
import unicodedata
from datetime import datetime, timedelta
from datetime import time as dtime

from loguru import logger

from pipecat.frames.frames import EndWorkerFrame, FunctionCallResultProperties, TTSSpeakFrame

from clinic import db
from clinic.calendars import get_calendar
from clinic.language import speaks_hindi
from clinic.schedule import (
    PERIODS,
    SLOT,
    combine,
    day_slots,
    is_open,
    normalize_phone,
    now,
    overlaps,
    parse_date,
    parse_time,
    slot_error,
    spoken,
)
from clinic.whatsapp import send_whatsapp
from prompts import CLINIC_NAME

# Short on purpose: a long list gets read out in full, and callers can't hold six options.
MAX_SLOTS_RETURNED = 3
CONFIRM_FIRST = (
    "Not done yet. Read these details back to the caller in one sentence and ask if they "
    "are correct. After they clearly say yes, call again with the same details and "
    "caller_confirmed true."
)


NOT_HEARD = (
    "The caller has not said this mobile number. Never guess details. Ask the caller for "
    "their 10-digit mobile number and wait for their answer."
)

# Spoken digits as STT may write them, in English and Hindi.
DIGIT_WORDS = {
    "zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "shunya": "0", "ek": "1", "do": "2", "teen": "3", "char": "4", "chaar": "4", "paanch": "5",
    "panch": "5", "chhe": "6", "chhah": "6", "saat": "7", "aath": "8", "nau": "9",
    # Devanagari: speech-to-text writes Hindi-spoken digits this way ("नौ आठ एक एक ...").
    "शून्य": "0", "ज़ीरो": "0", "जीरो": "0", "एक": "1", "दो": "2", "तीन": "3", "चार": "4", "पांच": "5", "पाँच": "5",
    "छह": "6", "छः": "6", "छे": "6", "सात": "7", "आठ": "8", "नौ": "9",
}
REPEATS = {"double": 2, "triple": 3}


def _err(message: str) -> dict:
    return {"ok": False, "error": message}


def _caller_speaks_hindi(params) -> bool:
    said = _caller_messages(params)
    return bool(said) and speaks_hindi(said[-1])


def _hinglish_when(start: datetime) -> str:
    """e.g. "Thursday 29 October ko shaam 5:30 baje"."""
    period = "subah" if start.hour < 12 else "dopahar" if start.hour < 16 else "shaam"
    hour = start.hour % 12 or 12
    clock = f"{hour}:{start.minute:02d}" if start.minute else str(hour)
    return f"{start:%A} {start.day} {start:%B} ko {period} {clock} baje"


def _read_back(params, name: str, reason: str, start: datetime, phone_e164: str) -> str:
    number = _number_in_words(phone_e164)
    if _caller_speaks_hindi(params):
        # Digits stay as English words: that's how numbers are usually read out in India.
        return f"Toh {name} ji, {reason} ke liye, {_hinglish_when(start)}, aur aapka number hai {number}. Sahi hai?"
    return f"So that's {name}, for {reason}, on {spoken(start)}, and your number is {number}. Is that correct?"


def _booked_line(params, name: str, start: datetime, whatsapp_sent: bool) -> str:
    if _caller_speaks_hindi(params):
        line = f"Ho gaya {name} ji, aapka appointment {_hinglish_when(start)} confirm hai."
        return line + (" WhatsApp par confirmation bhej diya hai." if whatsapp_sent else "")
    line = f"You're all set, {name}. Your appointment is confirmed for {spoken(start)}."
    return line + (" A confirmation is on its way on WhatsApp." if whatsapp_sent else "")


def _number_in_words(phone_e164: str) -> str:
    """e.g. "nine eight one one one, two two three three four" for +919811122334."""
    words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    digits = phone_e164[-10:]
    return f"{' '.join(words[int(d)] for d in digits[:5])}, {' '.join(words[int(d)] for d in digits[5:])}"


def _spoken_digits(text: str) -> str:
    """All digits in the text, including spoken ones like "double five" or "paanch"."""
    digits, repeat = [], 1
    # Devanagari vowel signs aren't \w, so include the whole block (minus the danda punctuation).
    for token in re.findall(r"[\wऀ-ॣ०-ॿ]+", text.lower()):
        if token in REPEATS:
            repeat = REPEATS[token]
            continue
        if token.isdigit():
            # Handles Devanagari digits too.
            value = "".join(str(unicodedata.digit(ch)) for ch in token)
        else:
            value = DIGIT_WORDS.get(token, "")
        if value:
            digits.append(value[0] * repeat + value[1:])
        repeat = 1
    return "".join(digits)


def _text(message) -> str:
    content = message.get("content")
    if isinstance(content, list):
        content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def _caller_messages(params) -> list[str] | None:
    """Everything the caller has said so far, or None when there's no conversation context."""
    context = getattr(params, "context", None)
    if context is None:
        return None
    return [_text(m) for m in context.messages if isinstance(m, dict) and m.get("role") == "user"]


def _last_readback(params) -> str:
    """What the bot said just before the caller's latest message."""
    messages = [m for m in getattr(params, "context").messages if isinstance(m, dict)]
    last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)
    for m in reversed(messages[:max(last_user, 0)]):
        if m.get("role") == "assistant" and _text(m).strip():
            return _text(m)
    return ""


def _number_caller_said_last(params) -> str | None:
    """The last 10 digits the caller spoke, if they form a mobile number. On a call, people say
    a number in groups with pauses, so it arrives across several messages, and the LLM often
    re-assembles it wrongly. The digits themselves are right."""
    # Only messages that are mostly a number: "ek" (one) and "do" (two) are also everyday
    # Hindi words, as in "ek aur appointment chahiye".
    groups = [d for d in (_spoken_digits(m) for m in _caller_messages(params) or []) if len(d) >= 3]
    digits = "".join(groups)
    return normalize_phone(digits[-10:]) if len(digits) >= 10 else None


def _caller_said(params, phone_e164: str) -> bool:
    """Guard against the LLM inventing a number: it must appear in what the caller said,
    or be the number they're calling from."""
    said = _caller_messages(params)
    if said is None or phone_e164 == _session(params).get("caller_phone"):
        return True
    return phone_e164[-10:] in _spoken_digits(" ".join(said))


def _confirmed_after_readback(params, action: tuple, caller_confirmed: bool, read_back) -> bool:
    """Guard against acting before the caller confirms.

    Goes through only if the LLM says the caller confirmed AND the caller has actually
    answered a read-back of these details: either the bot's message just before the
    caller's latest reply passes `read_back(text)`, or an earlier attempt with identical
    details was held and the caller has spoken since. Otherwise the attempt is held as
    pending and the LLM is told to read the details back first.
    """
    said = _caller_messages(params)
    if said is None:
        return caller_confirmed
    session = _session(params)
    pending = session.get("pending")
    answered_pending = pending and pending["action"] == action and len(said) > pending["turn"]
    if caller_confirmed and (answered_pending or read_back(_last_readback(params))):
        session.pop("pending", None)
        return True
    session["pending"] = {"action": action, "turn": len(said)}
    return False


def _asks_to_confirm(text: str) -> bool:
    return "?" in text


def _session(params) -> dict:
    """Per-call scratch state, shared across tool calls via PipelineWorker app_resources."""
    resources = getattr(params, "app_resources", None)
    return resources if isinstance(resources, dict) else {}


def _log_event(params, kind: str, **data) -> None:
    """Record what a tool actually did, so the post-call summary rests on facts."""
    _session(params).setdefault("events", []).append({"type": kind, "at": now().isoformat(timespec="seconds"), **data})


def _confirmation_note(status: str) -> str:
    if status == "sent":
        return "Tell the caller a WhatsApp confirmation has been sent."
    return "Do not mention WhatsApp or SMS to the caller; no message was sent."


async def _busy_on(day) -> list:
    start, end = combine(day, dtime.min), combine(day + timedelta(days=1), dtime.min)
    return db.booked_between(start, end) + await get_calendar().busy(start, end)


async def _free_slots(day, time_preference: str = "any") -> list:
    busy = await _busy_on(day)
    window = PERIODS.get(time_preference.lower())
    free = []
    for slot in day_slots(day):
        if slot <= now() or overlaps(slot, busy):
            continue
        if window and not (window[0] <= slot.time() < window[1]):
            continue
        free.append(slot)
    return free


def _parse_slot(date: str, time: str):
    try:
        return combine(parse_date(date), parse_time(time)), None
    except ValueError:
        return None, "Use date as YYYY-MM-DD and time as HH:MM in 24-hour format."


async def _confirm_on_whatsapp(appointment_id: int, phone: str, body: str) -> str:
    status, error = await send_whatsapp(phone, body)
    db.add_message(appointment_id, "whatsapp", phone, body, status, error)
    return status


async def check_availability(params, date: str, time_preference: str = "any", preferred_time: str | None = None):
    """Find free appointment slots on a date. Always call this before offering times.

    Args:
        date: The date to check as YYYY-MM-DD. Use the date table in your instructions
            to convert words like "kal" or "next Thursday".
        time_preference: One of "morning", "afternoon", "evening" or "any".
        preferred_time: The exact time the caller asked for, as HH:MM 24-hour, if they named one.
    """
    try:
        day = parse_date(date)
    except ValueError:
        return await params.result_callback(_err("Use date as YYYY-MM-DD."))
    if day < now().date():
        return await params.result_callback(_err("That date is in the past."))

    result = {"ok": True, "date": date}
    if not is_open(day):
        result["note"] = "The clinic is closed on Sundays."
        free = result["free_times"] = []
    elif preferred_time:
        # Answer the caller's actual question: is that time free, and what's closest?
        try:
            wanted = combine(day, parse_time(preferred_time))
        except ValueError:
            return await params.result_callback(_err("Use preferred_time as HH:MM in 24-hour format."))
        free = await _free_slots(day)
        result["requested_time"] = wanted.strftime("%H:%M")
        result["requested_time_free"] = wanted in free
        if error := slot_error(wanted):
            result["requested_time_note"] = error
        free = sorted(free, key=lambda s: abs(s - wanted))[:MAX_SLOTS_RETURNED]
        result["nearest_free_times"] = sorted(s.strftime("%H:%M") for s in free)
    else:
        free = await _free_slots(day, time_preference)
        result["free_times"] = [s.strftime("%H:%M") for s in free[:MAX_SLOTS_RETURNED]]
        if len(free) > MAX_SLOTS_RETURNED:
            # Without these the LLM reads a truncated list as "nothing later is free".
            result["more_free_times_later_that_day"] = len(free) - MAX_SLOTS_RETURNED
            result["latest_free_time"] = free[-1].strftime("%H:%M")

    if not free:
        # Suggest the next day that has room, so the caller isn't left at a dead end.
        for offset in range(1, 8):
            next_day = day + timedelta(days=offset)
            next_free = await _free_slots(next_day, time_preference)
            if next_free:
                result["next_available"] = {
                    "date": next_day.isoformat(),
                    "day": f"{next_day:%A} {next_day.day} {next_day:%B}",
                    "free_times": [s.strftime("%H:%M") for s in next_free[:3]],
                }
                break
    await params.result_callback(result)


async def book_appointment(
    params, name: str, date: str, time: str, reason: str, caller_confirmed: bool, phone: str | None = None
):
    """Book an appointment once the caller has confirmed every detail.

    Args:
        name: The patient's name.
        date: Appointment date as YYYY-MM-DD.
        time: Appointment start time as HH:MM, 24-hour.
        reason: Short reason for the visit, e.g. "cleaning" or "tooth pain".
        caller_confirmed: True only after you read the details back and the caller said yes.
        phone: The patient's 10-digit Indian mobile number. Leave empty to use the number this
            call is on (when it's known), unless the caller gives a different one.
    """
    phone_e164 = normalize_phone(phone) if phone else _session(params).get("caller_phone")
    if phone_e164 and not _caller_said(params, phone_e164):
        # Use what the caller actually said. The read-back below makes them confirm it.
        phone_e164 = _number_caller_said_last(params)
    if not phone_e164:
        return await params.result_callback(_err(NOT_HEARD))
    start, error = _parse_slot(date, time)
    if error or (error := slot_error(start)):
        return await params.result_callback(_err(error))
    if overlaps(start, await _busy_on(start.date())):
        return await params.result_callback(
            _err("That slot was just taken. Call check_availability and offer other times.")
        )
    # A caller who already has an appointment more likely wants to move it than add one.
    session = _session(params)
    existing = db.upcoming_for_phone(phone_e164)
    if existing and phone_e164 not in session.setdefault("existing_checked", []):
        session["existing_checked"].append(phone_e164)
        return await params.result_callback(
            {
                "ok": False,
                "status": "not booked yet",
                "existing_appointments": [
                    {"appointment_id": a["id"], "when": spoken(parse_iso(a["start"])), "reason": a["reason"]} for a in existing
                ],
                "next_step": "This number already has the appointment(s) above. Ask the caller whether they want "
                "to move that appointment (use reschedule_appointment) or book an additional one (call "
                "book_appointment again).",
            }
        )
    action = ("book", name.strip().lower(), phone_e164, start.isoformat())
    def read_back_number(text: str) -> bool:
        # A booking read-back must include the phone number, so the caller could correct it.
        return phone_e164[-10:] in _spoken_digits(text)

    if not _confirmed_after_readback(params, action, caller_confirmed, read_back_number):
        # The read-back is written here, not by the LLM: LLMs drop digits when spelling
        # out numbers (e.g. "one one one two" for 1112 2), which the caller may not catch.
        return await params.result_callback(
            {
                "ok": False,
                "status": "not booked yet",
                "say_to_caller": _read_back(params, name, reason, start, phone_e164),
                "next_step": "Say say_to_caller to the caller word for word and end your turn. If they clearly "
                "say yes, call book_appointment again with the same details and caller_confirmed true.",
            }
        )

    end = start + SLOT
    try:
        event_id = await get_calendar().create(
            f"{reason.capitalize()} - {name}",
            f"Phone: {phone_e164}\nBooked by the AI receptionist.",
            start,
            end,
        )
    except Exception as e:
        logger.exception(f"Calendar create failed: {e}")
        return await params.result_callback(
            _err("The calendar is not responding. Offer to have the staff call back.")
        )
    appointment_id = db.add_appointment(name, phone_e164, reason, start, end, event_id)
    logger.info(f"Booked appointment #{appointment_id}: {name} {spoken(start)}")
    _log_event(params, "booked", appointment_id=appointment_id, name=name, when=spoken(start), reason=reason)

    whatsapp = await _confirm_on_whatsapp(
        appointment_id,
        phone_e164,
        f"Hi {name}, your appointment at {CLINIC_NAME} is confirmed for {spoken(start)}."
        f" Booking ID: {appointment_id}. Reply here if you need to change it.",
    )
    await params.result_callback(
        {
            "ok": True,
            "appointment_id": appointment_id,
            "say_to_caller": _booked_line(params, name, start, whatsapp == "sent"),
            "next_step": "Say say_to_caller, then ask if there's anything else.",
        }
    )


async def find_appointments(params, phone: str):
    """Look up a caller's upcoming appointments, before rescheduling or cancelling.

    Args:
        phone: The caller's 10-digit Indian mobile number.
    """
    phone_e164 = normalize_phone(phone)
    if not phone_e164:
        return await params.result_callback(_err("Invalid mobile number. Ask for it again."))
    rows = db.upcoming_for_phone(phone_e164)
    await params.result_callback(
        {
            "ok": True,
            "appointments": [
                {
                    "appointment_id": r["id"],
                    "name": r["name"],
                    "reason": r["reason"],
                    "when": spoken(parse_iso(r["start"])),
                }
                for r in rows
            ],
        }
    )


async def reschedule_appointment(
    params, appointment_id: int, new_date: str, new_time: str, caller_confirmed: bool
):
    """Move an existing appointment to a new slot.

    Args:
        appointment_id: The ID from find_appointments.
        new_date: New date as YYYY-MM-DD.
        new_time: New start time as HH:MM, 24-hour.
        caller_confirmed: True only after you read the new time back and the caller said yes.
    """
    appt = db.get_appointment(int(appointment_id))
    if not appt or appt["status"] != "booked":
        return await params.result_callback(_err("No active appointment with that ID."))
    start, error = _parse_slot(new_date, new_time)
    if error or (error := slot_error(start)):
        return await params.result_callback(_err(error))
    # The appointment's own current slot doesn't block the move.
    old = (parse_iso(appt["start"]), parse_iso(appt["end"]))
    busy = [b for b in await _busy_on(start.date()) if b != old]
    if overlaps(start, busy):
        return await params.result_callback(
            _err("That slot is taken. Call check_availability and offer other times.")
        )
    action = ("move", appt["id"], start.isoformat())
    if not _confirmed_after_readback(params, action, caller_confirmed, _asks_to_confirm):
        return await params.result_callback(_err(CONFIRM_FIRST))

    end = start + SLOT
    try:
        await get_calendar().move(appt["calendar_event_id"], start, end)
    except Exception as e:
        logger.exception(f"Calendar move failed: {e}")
        return await params.result_callback(
            _err("The calendar is not responding. Offer to have the staff call back.")
        )
    db.reschedule(appt["id"], start, end)
    logger.info(f"Rescheduled appointment #{appt['id']} to {spoken(start)}")
    _log_event(params, "rescheduled", appointment_id=appt["id"], name=appt["name"], when=spoken(start))

    whatsapp = await _confirm_on_whatsapp(
        appt["id"],
        appt["phone"],
        f"Hi {appt['name']}, your appointment at {CLINIC_NAME} has been moved to"
        f" {spoken(start)}. Booking ID: {appt['id']}.",
    )
    await params.result_callback(
        {"ok": True, "when": spoken(start), "tell_caller": _confirmation_note(whatsapp)}
    )


async def cancel_appointment(params, appointment_id: int, caller_confirmed: bool):
    """Cancel an existing appointment.

    Args:
        appointment_id: The ID from find_appointments.
        caller_confirmed: True only after the caller clearly confirmed the cancellation.
    """
    appt = db.get_appointment(int(appointment_id))
    if not appt or appt["status"] != "booked":
        return await params.result_callback(_err("No active appointment with that ID."))
    action = ("cancel", appt["id"])
    if not _confirmed_after_readback(params, action, caller_confirmed, _asks_to_confirm):
        return await params.result_callback(_err(CONFIRM_FIRST))
    try:
        await get_calendar().delete(appt["calendar_event_id"])
    except Exception as e:
        logger.exception(f"Calendar delete failed: {e}")
        return await params.result_callback(
            _err("The calendar is not responding. Offer to have the staff call back.")
        )
    db.cancel(appt["id"])
    logger.info(f"Cancelled appointment #{appt['id']}")
    _log_event(params, "cancelled", appointment_id=appt["id"], name=appt["name"])

    whatsapp = await _confirm_on_whatsapp(
        appt["id"],
        appt["phone"],
        f"Hi {appt['name']}, your appointment at {CLINIC_NAME} on"
        f" {spoken(parse_iso(appt['start']))} has been cancelled.",
    )
    await params.result_callback({"ok": True, "tell_caller": _confirmation_note(whatsapp)})


async def save_caller_details(
    params, name: str, phone: str, interest: str, notes: str | None = None
):
    """Save a caller's details when they are interested but not booking right now.

    Args:
        name: The caller's name.
        phone: The caller's 10-digit Indian mobile number.
        interest: What they are interested in, e.g. "braces" or "whitening price".
        notes: Anything else useful for the staff, such as the best time to call.
    """
    phone_e164 = normalize_phone(phone)
    if not phone_e164:
        return await params.result_callback(_err("Invalid mobile number. Ask for it again."))
    if not _caller_said(params, phone_e164):
        return await params.result_callback(_err(NOT_HEARD))
    lead_id = db.add_lead(name, phone_e164, interest, notes)
    logger.info(f"Saved lead #{lead_id}: {name} ({interest})")
    _log_event(params, "lead_saved", lead_id=lead_id, name=name, interest=interest)
    await params.result_callback({"ok": True})


async def request_human_handoff(
    params, reason: str, urgency: str = "normal", name: str | None = None, phone: str | None = None
):
    """Hand the caller over to clinic staff. Use for dental emergencies, complaints,
    upset callers, or anything you cannot handle.

    Args:
        reason: Why the caller needs a person, in one sentence.
        urgency: "urgent" for pain, swelling, bleeding or an angry caller, otherwise "normal".
        name: The caller's name, if known.
        phone: The caller's mobile number, if known.
    """
    phone_e164 = (normalize_phone(phone) if phone else None) or _session(params).get("caller_phone")
    session = _session(params)
    # Calling again in the same call (e.g. once the number is known) updates the task
    # instead of creating a duplicate.
    if callback_id := session.get("callback_id"):
        db.update_callback(callback_id, name or None, phone_e164 or phone or None, reason, urgency)
    else:
        callback_id = db.add_callback(name or None, phone_e164 or phone or None, reason, urgency)
        session["callback_id"] = callback_id
    logger.warning(f"Handoff #{callback_id} ({urgency}): {reason}")

    # On a phone call during clinic hours, transfer live to the staff phone if one is set.
    call_sid, staff = session.get("call_sid"), os.getenv("STAFF_PHONE_NUMBER")
    transfer = bool(call_sid and staff and session.get("can_transfer") and _clinic_open_now())
    _log_event(params, "handoff", callback_id=callback_id, urgency=urgency, reason=reason, live_transfer=transfer)
    if transfer:
        # Keep a reference so the task isn't garbage-collected mid-wait.
        session["transfer_task"] = asyncio.create_task(_transfer_after_goodbye(session, call_sid, staff))
        return await params.result_callback(
            {
                "ok": True,
                "transferring": True,
                "tell_caller": "Say one short line that you are connecting them to the clinic staff now.",
            }
        )

    # Otherwise (browser, after hours, no staff phone) staff get a callback task.
    when = "within 15 minutes" if urgency == "urgent" else "within 2 hours, during clinic hours"
    await params.result_callback(
        {
            "ok": True,
            "callback_id": callback_id,
            "say_to_caller": f"A staff member will call you back {when}.",
            "next_step": "Say say_to_caller (in the caller's language)."
            + (" Then ask for their mobile number so staff can call back." if not phone_e164 else ""),
            "need_phone": not phone_e164,
        }
    )


def _clinic_open_now() -> bool:
    current = now()
    return is_open(current.date()) and combine(current.date(), dtime(10)) <= current < combine(
        current.date(), dtime(19)
    )


async def _transfer_after_goodbye(session: dict, call_sid: str, staff: str) -> None:
    from telephony import transfer_call

    # Give the bot time to say "connecting you now" before the line moves.
    await asyncio.sleep(4)
    # When the bot's media stream closes, Pipecat normally hangs up the call. After a
    # transfer that would drop the caller, so turn it off first.
    serializer = session.get("serializer")
    if serializer is not None:
        serializer._params.auto_hang_up = False
    try:
        await transfer_call(call_sid, staff)
        logger.info(f"Transferred call {call_sid} to staff")
    except Exception as e:
        logger.exception(f"Transfer failed: {e}")


async def end_call(params, goodbye: str):
    """Hang up after saying goodbye. Use when the conversation is finished, or after
    a wrong number or an outbound call that isn't convenient.

    Args:
        goodbye: One short closing sentence to say before hanging up, in the caller's language.
    """
    await params.result_callback({"ok": True}, properties=FunctionCallResultProperties(run_llm=False))
    _log_event(params, "assistant_hung_up")
    # Speak the goodbye, then end the pipeline once it has played (frames flush in order).
    await params.llm.push_frame(TTSSpeakFrame(goodbye))
    await params.llm.push_frame(EndWorkerFrame(reason="call ended by assistant"))


async def add_to_do_not_call(params, phone: str | None = None):
    """Put the caller on the do-not-call list when they ask not to be called again.

    Args:
        phone: Their 10-digit mobile number. Leave empty to use the number they are calling from.
    """
    phone_e164 = (normalize_phone(phone) if phone else None) or _session(params).get("caller_phone")
    if not phone_e164:
        return await params.result_callback(_err("No number known. Ask for their mobile number."))
    db.add_dnc(phone_e164, "requested by caller")
    logger.info(f"Added {phone_e164} to do-not-call list")
    _log_event(params, "do_not_call", phone=phone_e164)
    await params.result_callback(
        {"ok": True, "tell_caller": "Confirm they won't receive calls from the clinic again."}
    )


async def stop_recording(params):
    """Stop recording this call and delete the recording, when the caller asks not to be recorded."""
    session = _session(params)
    recorder = session.get("recorder")
    if recorder:
        session["recording_opted_out"] = True
        await recorder.stop_recording()
    logger.info("Caller opted out of recording")
    _log_event(params, "recording_opt_out")
    await params.result_callback(
        {"ok": True, "tell_caller": "Confirm the call is no longer being recorded, then carry on."}
    )


def parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


TOOLS = [
    check_availability,
    book_appointment,
    find_appointments,
    reschedule_appointment,
    cancel_appointment,
    save_caller_details,
    request_human_handoff,
    add_to_do_not_call,
    stop_recording,
    end_call,
]
