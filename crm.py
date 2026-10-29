"""A simple CRM view of calls, served by the bot at http://localhost:7860/crm.

Lists every call with its post-call analysis, and shows each call's summary, transcript,
tool actions and recording. It holds patients' personal data, so it only answers
requests made on this computer, unless ADMIN_PASSWORD is set (e.g. when deployed),
in which case remote visitors must log in with that password.
"""

import base64
import json
import os
import secrets
from html import escape
from pathlib import Path

from fastapi import Depends, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

from clinic import db
from pipecat.runner.run import app
from postcall import CSV_PATH, export_csv

TUNNEL_HEADERS = ("cf-connecting-ip", "x-forwarded-for", "x-forwarded-host")


def _password_ok(request: Request, password: str) -> bool:
    scheme, _, encoded = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "basic":
        return False
    try:
        _user, _, given = base64.b64decode(encoded).decode().partition(":")
    except (ValueError, UnicodeDecodeError):
        return False
    return secrets.compare_digest(given.encode(), password.encode())


def local_only(request: Request) -> None:
    """Patient data: open on this computer, or anywhere with ADMIN_PASSWORD (when deployed)."""
    host = request.client.host if request.client else ""
    if host in ("127.0.0.1", "::1", "localhost") and not any(h in request.headers for h in TUNNEL_HEADERS):
        return
    password = os.getenv("ADMIN_PASSWORD")
    if not password:
        raise HTTPException(status_code=403, detail="Only available on this computer. Set ADMIN_PASSWORD to allow remote access.")
    if not _password_ok(request, password):
        # Makes the browser show its login box; any username, the password is ADMIN_PASSWORD.
        raise HTTPException(status_code=401, detail="Login required", headers={"WWW-Authenticate": 'Basic realm="Priya admin"'})


STYLE = """
:root { --bg:#f7f7f5; --card:#fff; --ink:#1d1d1b; --muted:#6b6b66; --line:#e4e3de; --accent:#2f6f5e;
        --good:#2f7d4f; --warn:#a86b00; --bad:#b3372e; --chip:#efeee9; color-scheme: light; }
@media (prefers-color-scheme: dark) { :root { --bg:#161615; --card:#1f1f1d; --ink:#ecebe6; --muted:#9a9993;
        --line:#33322f; --accent:#6fbfa6; --good:#6fcf97; --warn:#e0b04f; --bad:#ef7d72; --chip:#2a2a27; color-scheme: dark; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width:1100px; margin:0 auto; padding:24px 16px 48px; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:16px; margin:24px 0 8px; }
a { color:var(--accent); text-decoration:none; } a:hover { text-decoration:underline; }
.muted { color:var(--muted); }
.stats { display:flex; flex-wrap:wrap; gap:12px; margin:16px 0 20px; }
.stat { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px 14px; min-width:120px; }
.stat b { display:block; font-size:22px; }
.table-wrap { overflow-x:auto; background:var(--card); border:1px solid var(--line); border-radius:10px; }
table { border-collapse:collapse; width:100%; min-width:820px; }
th, td { text-align:left; padding:9px 12px; border-bottom:1px solid var(--line); vertical-align:top; }
th { font-size:12px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); font-weight:600; }
tr:last-child td { border-bottom:0; }
.chip { display:inline-block; padding:1px 8px; border-radius:99px; background:var(--chip); font-size:12px; white-space:nowrap; }
.good { color:var(--good); } .warn { color:var(--warn); } .bad { color:var(--bad); }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }
.grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(170px, 1fr)); gap:10px 18px; }
.grid div span { display:block; font-size:12px; color:var(--muted); }
.turn { margin:8px 0; max-width:80%; padding:8px 12px; border-radius:12px; background:var(--chip); }
.turn.assistant { margin-left:auto; background:color-mix(in srgb, var(--accent) 16%, var(--card)); }
.turn small { display:block; color:var(--muted); font-size:11px; }
audio { width:100%; margin-top:6px; }
"""

OUTCOME_TONE = {"booked": "good", "rescheduled": "good", "info_given": "good", "lead_saved": "good",
                "handed_off": "warn", "call_back_later": "warn", "cancelled": "warn",
                "do_not_call": "bad", "not_interested": "bad", "wrong_number": "bad"}
SENTIMENT_TONE = {"positive": "good", "neutral": "", "negative": "bad"}


def page(title: str, body: str) -> HTMLResponse:
    return HTMLResponse(
        f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{escape(title)}</title><style>{STYLE}</style></head><body><main>{body}</main></body></html>"
    )


def chip(value, tones: dict) -> str:
    if not value:
        return "<span class='muted'>-</span>"
    return f"<span class='chip {tones.get(value, '')}'>{escape(str(value).replace('_', ' '))}</span>"


def phone_of(call) -> str:
    return (call["to_number"] if call["direction"] == "outbound" else call["from_number"]) or "-"


def when(call) -> str:
    return escape((call["started_at"] or "")[:16].replace("T", " "))


@app.get("/crm", dependencies=[Depends(local_only)])
async def crm_index():
    calls = db.all_calls()
    done = [c for c in calls if c["analysis_status"] == "done"]

    def count(outcome: str) -> int:
        return sum(c["outcome"] == outcome for c in done)

    stats = [("Calls", len(calls)), ("Booked", count("booked") + count("rescheduled")),
             ("Handed off", count("handed_off")), ("Negative", sum(c["sentiment"] == "negative" for c in done))]
    rows = "".join(
        f"<tr><td><a href='/crm/calls/{c['id']}'>{when(c)}</a></td>"
        f"<td>{escape(c['direction'])}<br><span class='muted'>{escape(c['channel'].replace('_', ' '))}</span></td>"
        f"<td>{escape(c['caller_name'] or '-')}<br><span class='muted'>{escape(phone_of(c))}</span></td>"
        f"<td>{chip(c['intent'], {})}</td><td>{chip(c['outcome'], OUTCOME_TONE)}</td>"
        f"<td>{chip(c['sentiment'], SENTIMENT_TONE)}</td>"
        f"<td>{escape(c['summary'] or ('Analysing...' if c['analysis_status'] == 'pending' else '-'))}</td></tr>"
        for c in calls
    ) or "<tr><td colspan='7' class='muted'>No calls yet. Talk to Priya in the browser or with phone_sim.py.</td></tr>"
    body = (
        "<h1>Smile Care - Calls</h1><div class='muted'>Every call, summarised after it ends. "
        "<a href='/crm/calls.csv'>Download CSV</a></div>"
        "<div class='stats'>" + "".join(f"<div class='stat'><b>{v}</b>{escape(k)}</div>" for k, v in stats) + "</div>"
        "<div class='table-wrap'><table><tr><th>When</th><th>Call</th><th>Caller</th><th>Intent</th>"
        f"<th>Outcome</th><th>Mood</th><th>Summary</th></tr>{rows}</table></div>"
    )
    return page("Calls - Smile Care", body)


@app.get("/crm/calls/{call_id}", dependencies=[Depends(local_only)])
async def crm_call(call_id: int):
    call = db.get_call(call_id)
    if not call:
        raise HTTPException(status_code=404)
    transcript = json.loads(call["transcript"] or "[]")
    events = json.loads(call["events"] or "[]")
    fields = [("Caller", call["caller_name"] or "-"), ("Phone", phone_of(call)),
              ("Direction", f"{call['direction']} ({call['channel'].replace('_', ' ')})"),
              ("Duration", f"{call['duration_secs'] or 0} s"), ("Intent", (call["intent"] or "-").replace("_", " ")),
              ("Booking", call["booking_status"] or "-")]
    grid = "".join(f"<div><span>{escape(k)}</span>{escape(str(v))}</div>" for k, v in fields)
    grid += f"<div><span>Outcome</span>{chip(call['outcome'], OUTCOME_TONE)}</div>"
    grid += f"<div><span>Mood</span>{chip(call['sentiment'], SENTIMENT_TONE)}</div>"

    if call["recording_path"] and Path(call["recording_path"]).exists():
        audio = f"<audio controls preload='none' src='/crm/calls/{call_id}/recording'></audio>"
    elif call["recording_status"] == "opted_out":
        audio = "<span class='muted'>Not recorded: the caller opted out.</span>"
    else:
        audio = "<span class='muted'>No recording.</span>"

    turns = "".join(
        f"<div class='turn {t['role']}'><small>{'Priya' if t['role'] == 'assistant' else 'Caller'} - "
        f"{escape(t.get('at', '')[11:19])}{' - interrupted' if t.get('interrupted') else ''}</small>{escape(t['text'])}</div>"
        for t in transcript if t["text"].strip()
    ) or "<span class='muted'>No transcript.</span>"
    actions = "".join(
        f"<li>{escape(e['type'].replace('_', ' '))}: "
        f"{escape(', '.join(f'{k} {v}' for k, v in e.items() if k not in ('type', 'at')))}</li>"
        for e in events
    ) or "<li class='muted'>None</li>"
    follow_up = f"<p><b>Follow-up:</b> {escape(call['follow_up'])}</p>" if call["follow_up"] else ""
    error = f"<p class='bad'>Analysis failed: {escape(call['analysis_error'] or '')}</p>" if call["analysis_status"] == "failed" else ""

    body = (
        f"<a href='/crm'>&larr; All calls</a><h1>Call {call_id}</h1><div class='muted'>{when(call)}</div>"
        f"<h2>Summary</h2><div class='card'><p>{escape(call['summary'] or 'Not analysed yet.')}</p>{follow_up}{error}"
        f"<div class='grid'>{grid}</div></div>"
        f"<h2>Recording</h2><div class='card'>{audio}</div>"
        f"<h2>Actions taken</h2><div class='card'><ul>{actions}</ul></div>"
        f"<h2>Transcript</h2><div class='card'>{turns}</div>"
    )
    return page(f"Call {call_id} - Smile Care", body)


@app.get("/crm/calls/{call_id}/recording", dependencies=[Depends(local_only)])
async def crm_recording(call_id: int):
    call = db.get_call(call_id)
    if not call or not call["recording_path"] or not Path(call["recording_path"]).exists():
        raise HTTPException(status_code=404)
    return FileResponse(call["recording_path"], media_type="audio/wav")


@app.get("/crm/calls.csv", dependencies=[Depends(local_only)])
async def crm_csv():
    export_csv()
    return FileResponse(CSV_PATH, media_type="text/csv", filename="calls.csv")
