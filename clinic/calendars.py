"""Calendar backends.

- LocalCalendar: no external calendar; bookings live only in SQLite. Works with zero setup.
- GoogleCalendar: mirrors bookings into a Google Calendar and treats any other events
  on it (staff leave, blocked time) as busy. Enabled when GOOGLE_SERVICE_ACCOUNT_FILE
  and GOOGLE_CALENDAR_ID are set.
"""

import asyncio
import os
from datetime import datetime

from loguru import logger

from clinic.schedule import TZ


class LocalCalendar:
    name = "local"

    async def busy(self, start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
        return []

    async def create(self, summary: str, description: str, start: datetime, end: datetime):
        return None

    async def move(self, event_id: str | None, start: datetime, end: datetime) -> None:
        pass

    async def delete(self, event_id: str | None) -> None:
        pass


class GoogleCalendar:
    name = "google"

    def __init__(self, service_account_file: str, calendar_id: str):
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        creds = service_account.Credentials.from_service_account_file(
            service_account_file, scopes=["https://www.googleapis.com/auth/calendar"]
        )
        self._events = build("calendar", "v3", credentials=creds, cache_discovery=False)
        self._calendar_id = calendar_id

    # The Google client is synchronous, so each call runs in a thread to keep audio flowing.

    async def busy(self, start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
        body = {
            "timeMin": start.isoformat(),
            "timeMax": end.isoformat(),
            "timeZone": str(TZ),
            "items": [{"id": self._calendar_id}],
        }
        resp = await asyncio.to_thread(lambda: self._events.freebusy().query(body=body).execute())
        periods = resp["calendars"][self._calendar_id].get("busy", [])
        return [
            (datetime.fromisoformat(p["start"]), datetime.fromisoformat(p["end"])) for p in periods
        ]

    async def create(self, summary: str, description: str, start: datetime, end: datetime):
        event = {
            "summary": summary,
            "description": description,
            "start": {"dateTime": start.isoformat(), "timeZone": str(TZ)},
            "end": {"dateTime": end.isoformat(), "timeZone": str(TZ)},
        }
        created = await asyncio.to_thread(
            lambda: self._events.events().insert(calendarId=self._calendar_id, body=event).execute()
        )
        return created["id"]

    async def move(self, event_id: str | None, start: datetime, end: datetime) -> None:
        if not event_id:
            return
        patch = {
            "start": {"dateTime": start.isoformat(), "timeZone": str(TZ)},
            "end": {"dateTime": end.isoformat(), "timeZone": str(TZ)},
        }
        await asyncio.to_thread(
            lambda: self._events.events()
            .patch(calendarId=self._calendar_id, eventId=event_id, body=patch)
            .execute()
        )

    async def delete(self, event_id: str | None) -> None:
        if not event_id:
            return
        await asyncio.to_thread(
            lambda: self._events.events()
            .delete(calendarId=self._calendar_id, eventId=event_id)
            .execute()
        )


_calendar = None


def get_calendar():
    global _calendar
    if _calendar is None:
        sa_file = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
        calendar_id = os.getenv("GOOGLE_CALENDAR_ID")
        if sa_file and calendar_id:
            _calendar = GoogleCalendar(sa_file, calendar_id)
        else:
            _calendar = LocalCalendar()
        logger.info(f"Calendar backend: {_calendar.name}")
    return _calendar
