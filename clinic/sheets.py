"""Append call results to a Google Sheet (optional).

Uses the same service account as Google Calendar. Enabled when GOOGLE_SERVICE_ACCOUNT_FILE
and GOOGLE_SHEET_ID are set; share the sheet with the service account's email as Editor.
"""

import asyncio
import os

_sheets = None


def enabled() -> bool:
    return bool(os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE") and os.getenv("GOOGLE_SHEET_ID"))


def _client():
    global _sheets
    if _sheets is None:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        creds = service_account.Credentials.from_service_account_file(
            os.environ["GOOGLE_SERVICE_ACCOUNT_FILE"], scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        _sheets = build("sheets", "v4", credentials=creds, cache_discovery=False).spreadsheets()
    return _sheets


def _append(header: list[str], row: list) -> None:
    sheet_id, sheets = os.environ["GOOGLE_SHEET_ID"], _client()
    first = sheets.values().get(spreadsheetId=sheet_id, range="A1:A1").execute()
    rows = [row] if first.get("values") else [header, row]
    sheets.values().append(
        spreadsheetId=sheet_id, range="A1", valueInputOption="RAW", insertDataOption="INSERT_ROWS",
        body={"values": rows},
    ).execute()


async def append_row(header: list[str], row: list) -> None:
    # The Google client is synchronous; keep it off the event loop that carries call audio.
    await asyncio.to_thread(_append, header, row)
