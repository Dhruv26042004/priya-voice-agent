"""Clinic hours, appointment slots and date helpers. All times are IST."""

import re
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Asia/Kolkata")
OPEN = time(10, 0)
CLOSE = time(19, 0)
SLOT = timedelta(minutes=30)
CLOSED_WEEKDAYS = {6}  # Sunday

# Rough windows for "subah" / "dopahar" / "shaam" preferences.
PERIODS = {
    "morning": (time(10, 0), time(12, 0)),
    "afternoon": (time(12, 0), time(16, 0)),
    "evening": (time(16, 0), time(19, 0)),
}


def now() -> datetime:
    return datetime.now(TZ)


def parse_date(value: str) -> date:
    return date.fromisoformat(value.strip())


def parse_time(value: str) -> time:
    """Accept "16:30", "4:30 PM" or "4 pm"."""
    value = value.strip().upper().replace(".", "")
    for fmt in ("%H:%M", "%I:%M %p", "%I:%M%p", "%I %p", "%I%p"):
        try:
            return datetime.strptime(value, fmt).time()
        except ValueError:
            continue
    raise ValueError(f"Unrecognised time: {value!r}")


def combine(day: date, at: time) -> datetime:
    return datetime.combine(day, at, tzinfo=TZ)


def is_open(day: date) -> bool:
    return day.weekday() not in CLOSED_WEEKDAYS


def day_slots(day: date) -> list[datetime]:
    """All slot start times on a day, ignoring bookings."""
    if not is_open(day):
        return []
    slots, start, end = [], combine(day, OPEN), combine(day, CLOSE)
    while start + SLOT <= end:
        slots.append(start)
        start += SLOT
    return slots


def slot_error(start: datetime) -> str | None:
    """Why a slot can't be booked, or None if it's a valid slot."""
    if not is_open(start.date()):
        return "The clinic is closed on Sundays."
    if start not in day_slots(start.date()):
        return "Appointments start on the hour or half hour, between 10 AM and 6:30 PM."
    if start <= now():
        return "That time has already passed."
    return None


def overlaps(start: datetime, busy: list[tuple[datetime, datetime]]) -> bool:
    end = start + SLOT
    return any(start < b_end and b_start < end for b_start, b_end in busy)


def spoken(dt: datetime) -> str:
    """e.g. "Thursday 29 October at 4:30 PM" — the LLM turns this into speech."""
    clock = dt.strftime("%I:%M %p").lstrip("0").replace(":00", "")
    return f"{dt:%A} {dt.day} {dt:%B} at {clock}"


def upcoming_days(n: int = 10) -> str:
    """A date lookup table for the prompt, so the LLM never does calendar maths."""
    today = now().date()
    lines = []
    for i in range(n):
        day = today + timedelta(days=i)
        label = {0: " (today / aaj)", 1: " (tomorrow / kal)", 2: " (day after / parson)"}.get(i, "")
        closed = " - CLOSED" if not is_open(day) else ""
        lines.append(f"{day:%A} {day.day} {day:%B} = {day.isoformat()}{label}{closed}")
    return "\n".join(lines)


def normalize_phone(raw: str) -> str | None:
    """Turn a spoken/typed Indian mobile number into +91XXXXXXXXXX, or None if invalid."""
    digits = re.sub(r"\D", "", raw)
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    elif len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]
    if len(digits) != 10 or digits[0] not in "6789":
        return None
    return f"+91{digits}"
