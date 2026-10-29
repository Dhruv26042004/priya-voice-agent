"""SQLite storage: appointments, leads, callback requests and sent messages.

This is the source of truth for bookings; the calendar holds a mirror for staff.
"""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

from clinic.schedule import now

DB_PATH = Path(os.getenv("CLINIC_DB", "data/clinic.db"))

SCHEMA = """
create table if not exists appointments (
    id integer primary key,
    name text not null,
    phone text not null,
    reason text,
    start text not null,
    end text not null,
    status text not null default 'booked',   -- booked | cancelled
    calendar_event_id text,
    created_at text not null,
    updated_at text not null
);
create table if not exists leads (
    id integer primary key,
    name text,
    phone text,
    interest text,
    notes text,
    created_at text not null
);
create table if not exists callbacks (
    id integer primary key,
    name text,
    phone text,
    reason text not null,
    urgency text not null,
    status text not null default 'open',     -- open | done
    created_at text not null
);
create table if not exists calls (
    id integer primary key,
    channel text not null,                   -- browser | phone
    direction text not null,                 -- inbound | outbound
    call_sid text,
    from_number text,
    to_number text,
    outbound_id integer,
    recording_path text,
    recording_status text not null default 'recording',  -- recording | saved | opted_out
    started_at text not null,
    ended_at text
);
create table if not exists do_not_call (
    phone text primary key,
    reason text,
    added_at text not null
);
create table if not exists outbound (
    id integer primary key,
    name text not null,
    phone text not null,
    purpose text not null,
    status text not null default 'pending',  -- pending | retry | in_progress | completed | failed | dnc
    attempts integer not null default 0,
    last_result text,
    last_call_sid text,
    next_attempt_at text,
    updated_at text not null
);
create table if not exists turn_latency (
    id integer primary key,
    call_id integer not null,
    at text not null,
    total_secs real not null,                -- end of caller speech -> first bot audio
    stages text not null                     -- JSON {stage label: seconds}
);
create table if not exists eval_runs (
    id integer primary key,
    mode text not null,                      -- text | voice
    started_at text not null,
    finished_at text,
    scenarios integer not null default 0,
    passed integer not null default 0
);
create table if not exists eval_results (
    id integer primary key,
    run_id integer not null,
    scenario text not null,
    category text not null,
    title text not null,
    status text not null,                    -- pass | fail | error
    checks text not null,                    -- JSON list of {name, ok, detail}
    transcript text,
    outcome text,
    booking_status text,
    handed_off integer,
    task_completed integer,                  -- null when the scenario has no task
    turns integer,
    duration_secs real,
    avg_latency_secs real,
    max_latency_secs real,
    llm_tokens integer,
    call_id integer,                         -- voice mode: the call row
    created_at text not null
);
create table if not exists messages (
    id integer primary key,
    appointment_id integer,
    channel text not null,
    to_number text not null,
    body text not null,
    status text not null,                    -- sent | failed | not_configured
    error text,
    created_at text not null
);
"""

# Post-call results (Phase 5), added to existing databases on startup.
CALL_ANALYSIS_COLUMNS = {
    "transcript": "text",        # JSON list of {role, text, at, interrupted}
    "events": "text",            # JSON list of what the tools actually did
    "duration_secs": "integer",
    "caller_name": "text",
    "intent": "text",
    "booking_status": "text",    # booked | rescheduled | cancelled | none
    "outcome": "text",
    "sentiment": "text",         # positive | neutral | negative
    "summary": "text",
    "follow_up": "text",
    "analysis_status": "text",   # pending | done | failed
    "analysis_error": "text",
    # Usage and cost (Phase 6)
    "llm_prompt_tokens": "integer",
    "llm_completion_tokens": "integer",
    "tts_chars": "integer",
    "stt_secs": "real",
    "cost_usd": "real",
}

_conn: sqlite3.Connection | None = None


def use_database(path: str | Path) -> None:
    """Switch to another database file, e.g. a throwaway one per evaluation scenario."""
    global _conn, DB_PATH
    if _conn is not None:
        _conn.close()
    _conn, DB_PATH = None, Path(path)


def conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.executescript(SCHEMA)
        existing = {r["name"] for r in _conn.execute("pragma table_info(calls)")}
        for column, kind in CALL_ANALYSIS_COLUMNS.items():
            if column not in existing:
                _conn.execute(f"alter table calls add column {column} {kind}")
        _conn.commit()
    return _conn


def _stamp() -> str:
    return now().isoformat(timespec="seconds")


def add_appointment(name, phone, reason, start: datetime, end: datetime, event_id) -> int:
    with conn() as c:
        cur = c.execute(
            "insert into appointments (name, phone, reason, start, end, calendar_event_id,"
            " created_at, updated_at) values (?, ?, ?, ?, ?, ?, ?, ?)",
            (name, phone, reason, start.isoformat(), end.isoformat(), event_id, _stamp(), _stamp()),
        )
        return cur.lastrowid


def get_appointment(appointment_id: int) -> sqlite3.Row | None:
    return conn().execute("select * from appointments where id = ?", (appointment_id,)).fetchone()


def upcoming_for_phone(phone: str) -> list[sqlite3.Row]:
    return conn().execute(
        "select * from appointments where phone = ? and status = 'booked' and start > ?"
        " order by start",
        (phone, now().isoformat()),
    ).fetchall()


def booked_between(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    rows = conn().execute(
        "select start, end from appointments where status = 'booked' and start < ? and end > ?",
        (end.isoformat(), start.isoformat()),
    ).fetchall()
    return [(datetime.fromisoformat(r["start"]), datetime.fromisoformat(r["end"])) for r in rows]


def reschedule(appointment_id: int, start: datetime, end: datetime) -> None:
    with conn() as c:
        c.execute(
            "update appointments set start = ?, end = ?, updated_at = ? where id = ?",
            (start.isoformat(), end.isoformat(), _stamp(), appointment_id),
        )


def cancel(appointment_id: int) -> None:
    with conn() as c:
        c.execute(
            "update appointments set status = 'cancelled', updated_at = ? where id = ?",
            (_stamp(), appointment_id),
        )


def add_lead(name, phone, interest, notes) -> int:
    with conn() as c:
        cur = c.execute(
            "insert into leads (name, phone, interest, notes, created_at) values (?, ?, ?, ?, ?)",
            (name, phone, interest, notes, _stamp()),
        )
        return cur.lastrowid


def add_callback(name, phone, reason, urgency) -> int:
    with conn() as c:
        cur = c.execute(
            "insert into callbacks (name, phone, reason, urgency, created_at)"
            " values (?, ?, ?, ?, ?)",
            (name, phone, reason, urgency, _stamp()),
        )
        return cur.lastrowid


def update_callback(callback_id, name, phone, reason, urgency) -> None:
    with conn() as c:
        c.execute(
            "update callbacks set name = coalesce(?, name), phone = coalesce(?, phone),"
            " reason = ?, urgency = ? where id = ?",
            (name, phone, reason, urgency, callback_id),
        )


def add_message(appointment_id, channel, to_number, body, status, error) -> None:
    with conn() as c:
        c.execute(
            "insert into messages (appointment_id, channel, to_number, body, status, error,"
            " created_at) values (?, ?, ?, ?, ?, ?, ?)",
            (appointment_id, channel, to_number, body, status, error, _stamp()),
        )


# --- Calls ---


def start_call(channel, direction, call_sid, from_number, to_number, outbound_id) -> int:
    with conn() as c:
        cur = c.execute(
            "insert into calls (channel, direction, call_sid, from_number, to_number,"
            " outbound_id, started_at) values (?, ?, ?, ?, ?, ?, ?)",
            (channel, direction, call_sid, from_number, to_number, outbound_id, _stamp()),
        )
        return cur.lastrowid


def end_call(call_id: int, recording_path: str | None, recording_status: str, transcript: list, events: list) -> None:
    ended = now()
    row = get_call(call_id)
    duration = int((ended - datetime.fromisoformat(row["started_at"])).total_seconds())
    with conn() as c:
        c.execute(
            "update calls set ended_at = ?, recording_path = ?, recording_status = ?, transcript = ?,"
            " events = ?, duration_secs = ?, analysis_status = 'pending' where id = ?",
            (ended.isoformat(timespec="seconds"), recording_path, recording_status,
             json.dumps(transcript, ensure_ascii=False), json.dumps(events), duration, call_id),
        )


def get_call(call_id: int) -> sqlite3.Row | None:
    return conn().execute("select * from calls where id = ?", (call_id,)).fetchone()


def all_calls() -> list[sqlite3.Row]:
    return conn().execute("select * from calls order by id desc").fetchall()


def calls_needing_analysis(include_done: bool = False) -> list[sqlite3.Row]:
    statuses = "('pending', 'failed', 'done')" if include_done else "('pending', 'failed')"
    return conn().execute(
        f"select * from calls where ended_at is not null and analysis_status in {statuses} order by id"
    ).fetchall()


def save_analysis(call_id: int, status: str, error: str | None = None, **fields) -> None:
    fields |= {"analysis_status": status, "analysis_error": error}
    update_call(call_id, **fields)


def update_call(call_id: int, **fields) -> None:
    assignments = ", ".join(f"{k} = ?" for k in fields)
    with conn() as c:
        c.execute(f"update calls set {assignments} where id = ?", (*fields.values(), call_id))


def call_by_sid(call_sid: str) -> sqlite3.Row | None:
    return conn().execute("select * from calls where call_sid = ? order by id desc", (call_sid,)).fetchone()


def add_turn_latency(call_id: int, total_secs: float, stages: dict) -> None:
    with conn() as c:
        c.execute(
            "insert into turn_latency (call_id, at, total_secs, stages) values (?, ?, ?, ?)",
            (call_id, _stamp(), total_secs, json.dumps(stages)),
        )


def turn_latencies(call_ids: list[int] | None = None) -> list[sqlite3.Row]:
    if call_ids is None:
        return conn().execute("select * from turn_latency order by id").fetchall()
    marks = ",".join("?" * len(call_ids)) or "null"
    return conn().execute(f"select * from turn_latency where call_id in ({marks}) order by id", call_ids).fetchall()


# --- Evaluation runs ---


def start_eval_run(mode: str) -> int:
    with conn() as c:
        return c.execute("insert into eval_runs (mode, started_at) values (?, ?)", (mode, _stamp())).lastrowid


def finish_eval_run(run_id: int) -> None:
    with conn() as c:
        c.execute(
            "update eval_runs set finished_at = ?,"
            " scenarios = (select count(*) from eval_results where run_id = ?),"
            " passed = (select count(*) from eval_results where run_id = ? and status = 'pass') where id = ?",
            (_stamp(), run_id, run_id, run_id),
        )


def add_eval_result(run_id: int, **fields) -> None:
    fields |= {"run_id": run_id, "created_at": _stamp()}
    columns = ", ".join(fields)
    with conn() as c:
        c.execute(f"insert into eval_results ({columns}) values ({', '.join('?' * len(fields))})", tuple(fields.values()))


def eval_runs() -> list[sqlite3.Row]:
    return conn().execute("select * from eval_runs order by id desc").fetchall()


def eval_results(run_id: int) -> list[sqlite3.Row]:
    return conn().execute("select * from eval_results where run_id = ? order by id", (run_id,)).fetchall()


# --- Do-not-call list ---


def is_dnc(phone: str) -> bool:
    return conn().execute("select 1 from do_not_call where phone = ?", (phone,)).fetchone() is not None


def add_dnc(phone: str, reason: str) -> None:
    with conn() as c:
        c.execute(
            "insert or ignore into do_not_call (phone, reason, added_at) values (?, ?, ?)",
            (phone, reason, _stamp()),
        )


# --- Outbound call list ---


def add_outbound(name, phone, purpose, status="pending") -> int | None:
    """Queue a number unless it's already queued and not finished. Returns the new id, or None."""
    with conn() as c:
        exists = c.execute(
            "select 1 from outbound where phone = ? and status in ('pending', 'retry', 'in_progress')",
            (phone,),
        ).fetchone()
        if exists:
            return None
        cur = c.execute(
            "insert into outbound (name, phone, purpose, status, updated_at) values (?, ?, ?, ?, ?)",
            (name, phone, purpose, status, _stamp()),
        )
        return cur.lastrowid


def due_outbound() -> list[sqlite3.Row]:
    return conn().execute(
        "select * from outbound where status in ('pending', 'retry')"
        " and (next_attempt_at is null or next_attempt_at <= ?) order by id",
        (now().isoformat(),),
    ).fetchall()


def waiting_outbound() -> int:
    return conn().execute(
        "select count(*) from outbound where status in ('pending', 'retry')"
    ).fetchone()[0]


def get_outbound(outbound_id: int) -> sqlite3.Row | None:
    return conn().execute("select * from outbound where id = ?", (outbound_id,)).fetchone()


def update_outbound(outbound_id: int, **fields) -> None:
    fields["updated_at"] = _stamp()
    assignments = ", ".join(f"{k} = ?" for k in fields)
    with conn() as c:
        c.execute(f"update outbound set {assignments} where id = ?", (*fields.values(), outbound_id))
