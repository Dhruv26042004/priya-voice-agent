"""Run the scripted test calls and record the results for the dashboard.

    uv run python -m evals.run                    # text mode: every scenario, no audio (fast, cheap)
    uv run python -m evals.run --mode voice       # voice mode: real phone audio through the running bot
    uv run python -m evals.run --only rude_caller,pricing

Text mode drives Priya's real prompt, tools and safety checks directly with text, using a
throwaway database per scenario. Voice mode calls the running bot (uv run bot.py) over its
phone WebSocket with synthesised caller speech, background noise and interruptions, so it also
measures end-to-end latency. Results appear on the dashboard at http://localhost:7860/dashboard.
"""

import argparse
import asyncio
import json
import os
import tempfile
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import yaml
from dotenv import load_dotenv
from loguru import logger

load_dotenv(override=True)
# Evaluations never touch real calendars or send real WhatsApp messages.
for var in ("GOOGLE_SERVICE_ACCOUNT_FILE", "GOOGLE_CALENDAR_ID", "GOOGLE_SHEET_ID", "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN"):
    os.environ.pop(var, None)

from clinic import db  # noqa: E402
from clinic.schedule import SLOT, combine, is_open, normalize_phone, now, parse_time  # noqa: E402
from clinic.speech_filter import MARKERS, TOOL_TALK, DropJsonFilter  # noqa: E402
from evals.agents import caller_line, judge  # noqa: E402

SCENARIOS = Path(__file__).with_name("scenarios.yaml")
MAIN_DB = db.DB_PATH
MAX_TURNS = 12


# --- Scenarios ---


def open_days() -> tuple[date, date]:
    """The next two days the clinic is open, starting tomorrow."""
    days, day = [], now().date()
    while len(days) < 2:
        day += timedelta(days=1)
        if is_open(day):
            days.append(day)
    return days[0], days[1]


def load_scenarios(mode: str, only: set[str] | None) -> list[dict]:
    day1, day2 = open_days()
    names = {"open_day": f"{day1:%A}", "open_day_2": f"{day2:%A} {day2.day} {day2:%B}"}
    scenarios = []
    for s in yaml.safe_load(SCENARIOS.read_text(encoding="utf-8")):
        if only and s["id"] not in only:
            continue
        if mode not in s.get("modes", ["text"]):
            continue
        s["persona"] = s["persona"].format(**names)
        s["_days"] = {"open_day": day1, "open_day_2": day2}
        scenarios.append(s)
    return scenarios


def call_context(s: dict) -> dict:
    caller = normalize_phone(s["caller_id"]) if s.get("caller_id") else None
    if s.get("direction") == "outbound":
        return {"direction": "outbound", "caller_phone": caller, "patient_name": s["patient"]["name"],
                "purpose": s["patient"]["purpose"]}
    return {"direction": "inbound", "caller_phone": caller}


def apply_setup(s: dict) -> list[int]:
    ids = []
    for appt in s.get("setup", []):
        start = combine(s["_days"][appt["day"]], parse_time(appt["time"]))
        ids.append(db.add_appointment(appt["name"], normalize_phone(appt["phone"]), appt["reason"], start, start + SLOT, None))
    return ids


# --- Hard checks ---


def booking_status(events: list[dict]) -> str:
    changes = [e["type"] for e in events if e["type"] in ("booked", "rescheduled", "cancelled")]
    return changes[-1] if changes else "none"


def check_expectations(s: dict, events: list[dict], outcome: str, caller_phone: str | None) -> list[dict]:
    expect, checks = s.get("expect", {}), []
    types = [e["type"] for e in events]

    def add(name, ok, detail=""):
        checks.append({"name": name, "ok": bool(ok), "detail": detail, "task_check": True})

    if "booking_status" in expect:
        got = booking_status(events)
        add(f"booking status is {expect['booking_status']}", got == expect["booking_status"], f"got {got}")
    if "outcome_in" in expect:
        add(f"outcome in {expect['outcome_in']}", outcome in expect["outcome_in"], f"got {outcome}")
    for t in expect.get("events_include", []):
        add(f"Priya used {t.replace('_', ' ')}", t in types, f"actions: {types}")
    for t in expect.get("events_exclude", []):
        add(f"Priya did not {t.replace('_', ' ')}", t not in types, f"actions: {types}")
    if want := expect.get("appointment"):
        ids = [e["appointment_id"] for e in events if e["type"] in ("booked", "rescheduled")]
        appt = db.get_appointment(ids[-1]) if ids else None
        if appt is None:
            add("appointment exists", False, "no appointment booked or moved")
        else:
            start = datetime.fromisoformat(appt["start"])
            if "time" in want:
                add(f"appointment at {want['time']}", start.strftime("%H:%M") == want["time"], f"got {start:%H:%M}")
            if "day" in want:
                day = s["_days"][want["day"]]
                add(f"appointment on {day:%A %d %b}", start.date() == day, f"got {start:%A %d %b}")
            if want.get("not_sunday"):
                add("appointment not on a Sunday", start.weekday() != 6, f"got {start:%A}")
    if expect.get("dnc"):
        add("number is on the do-not-call list", caller_phone and db.is_dnc(caller_phone))
    return checks


def summarise(s: dict, checks: list[dict], events: list[dict], outcome: str, transcript: list[dict],
              latencies: list[float], tokens: int, started: float, call_id=None) -> dict:
    required = [c for c in checks if not c.get("warning")]
    # Task completion: did the thing get done (the scenario's expected outcome checks), not
    # whether every turn was smooth; that's what the voice and judge checks are for.
    hard_ok = all(c["ok"] for c in required if c.get("task_check"))
    return {
        "scenario": s["id"], "category": s["category"], "title": s["title"],
        "status": "pass" if all(c["ok"] for c in required) else "fail",
        "checks": json.dumps(checks, ensure_ascii=False),
        "transcript": json.dumps(transcript, ensure_ascii=False),
        "outcome": outcome, "booking_status": booking_status(events),
        "handed_off": int(any(e["type"] == "handoff" for e in events)),
        "task_completed": int(hard_ok) if s.get("task") else None,
        "turns": sum(t["role"] == "user" for t in transcript),
        "duration_secs": round(time.monotonic() - started, 1),
        "avg_latency_secs": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "max_latency_secs": round(max(latencies), 3) if latencies else None,
        "llm_tokens": tokens, "call_id": call_id,
    }


# --- Text mode ---


class _Context:
    def __init__(self, messages):
        self.messages = messages


class _LLM:
    """Stands in for the pipeline when a tool pushes frames (end_call speaks a goodbye)."""

    def __init__(self):
        self.spoken = []

    async def push_frame(self, frame, *_):
        if hasattr(frame, "text"):
            self.spoken.append(frame.text)


class _ToolParams:
    def __init__(self, messages, session):
        self.context, self.app_resources, self.llm, self.result = _Context(messages), session, _LLM(), None

    async def result_callback(self, result, **_):
        self.result = result


async def _tool_schemas():
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.services.groq.llm import GroqLLMService

    from clinic.tools import TOOLS

    params = await GroqLLMService(api_key="unused").get_llm_adapter().get_llm_invocation_params(
        LLMContext(tools=TOOLS), system_instruction=None, convert_developer_to_user=True
    )
    return params["tools"], {f.__name__: f for f in TOOLS}


async def _bot_reply(messages, session, schemas, tools) -> tuple[str, bool, int]:
    """One Priya turn: run the LLM, executing tool calls, until she speaks. Returns
    (what she said, whether she hung up, tokens used)."""
    from llm import create_with_failover, default_targets
    from prompts import language_hint

    used = 0
    for _ in range(6):
        # Same per-turn language reminder as the live bot (FailoverGroqLLMService).
        system = {"role": "system", "content": messages[0]["content"] + language_hint(messages)}
        resp = await create_with_failover(
            default_targets(), {"messages": [system] + messages[1:], "tools": schemas, "reasoning_effort": "low"},
            max_wait=120,
        )
        used += resp.usage.total_tokens if resp.usage else 0
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True, exclude={"reasoning"}))
        if not msg.tool_calls:
            text = msg.content or ""
            if "{" in text or MARKERS.search(text) or TOOL_TALK.search(text):
                # The live bot's speech filter drops this, but it's still a model error worth tracking.
                session.setdefault("json_leaks", []).append(text[:160])
            return (await DropJsonFilter().filter(text)).strip(), False, used
        goodbye = None
        for tc in msg.tool_calls:
            p = _ToolParams(messages, session)
            try:
                await tools[tc.function.name](p, **json.loads(tc.function.arguments or "{}"))
            except Exception as e:  # unknown tool or bad arguments: report back like Pipecat does
                p.result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            if tc.function.name == "end_call" and p.llm.spoken:
                goodbye = p.llm.spoken[0]
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(p.result)})
        if goodbye:
            messages.append({"role": "assistant", "content": goodbye})
            return goodbye, True, used
    return "", False, used


async def run_text(s: dict, schemas, tools) -> dict:
    from postcall import analyse
    from prompts import INBOUND_OPENING, build_system_prompt, outbound_opening

    started = time.monotonic()
    db.use_database(Path(tempfile.mkdtemp()) / "eval.db")
    apply_setup(s)
    call = call_context(s)
    session = {"caller_phone": call.get("caller_phone"), "events": []}
    opening = outbound_opening(call["patient_name"]) if call["direction"] == "outbound" else INBOUND_OPENING
    messages = [{"role": "system", "content": build_system_prompt(call)}, {"role": "assistant", "content": opening}]
    transcript = [{"role": "assistant", "text": opening}]
    latencies, tokens = [], 0

    for _ in range(MAX_TURNS):
        say, hang_up = await caller_line(s["persona"], transcript)
        if not say:
            break
        messages.append({"role": "user", "content": say})
        transcript.append({"role": "user", "text": say})
        # Priya still processes the caller's last words, even if they hang up after them.
        t = time.monotonic()
        reply, ended, used = await _bot_reply(messages, session, schemas, tools)
        latencies.append(time.monotonic() - t)
        tokens += used
        if reply:
            transcript.append({"role": "assistant", "text": reply})
        if ended or hang_up:
            break

    events = session["events"]
    analysis = await analyse({"transcript": json.dumps(transcript), "events": json.dumps(events)})
    checks = check_expectations(s, events, analysis["outcome"], call.get("caller_phone"))
    leaks = session.get("json_leaks", [])
    # A warning, not a failure: the speech filter strips it before TTS, so callers never hear it.
    checks.append({"name": "model wrote no tool JSON/markup in replies (filtered before speech)", "ok": not leaks,
                   "warning": True, "detail": f"{len(leaks)} reply(s), e.g. {leaks[0]!r}" if leaks else ""})
    db.use_database(MAIN_DB)
    checks += await judge(s.get("judge", []), transcript, events)
    return summarise(s, checks, events, analysis["outcome"], transcript, latencies, tokens, started)


# --- Runner ---


async def main(mode: str, only: set[str] | None, append_to: int | None = None, cooldown: float = 60) -> None:
    scenarios = load_scenarios(mode, only)
    if not scenarios:
        raise SystemExit("No scenarios match.")
    run_id = append_to or db.start_eval_run(mode)
    logger.info(f"Eval run {run_id}: {len(scenarios)} scenario(s) in {mode} mode")
    if mode == "text":
        schemas, tools = await _tool_schemas()
    else:
        from evals.voice import run_voice

    for i, s in enumerate(scenarios, 1):
        if mode == "voice" and i > 1 and cooldown:
            # Voice calls share the free LLM tiers with the caller, judge and post-call analysis;
            # let the per-minute limits recover so one call's usage doesn't starve the next.
            await asyncio.sleep(cooldown)
        logger.info(f"[{i}/{len(scenarios)}] {s['id']}: {s['title']}")
        try:
            result = await (run_text(s, schemas, tools) if mode == "text" else run_voice(s))
        except Exception as e:
            logger.exception(f"{s['id']} errored: {e}")
            db.use_database(MAIN_DB)
            result = {"scenario": s["id"], "category": s["category"], "title": s["title"], "status": "error",
                      "checks": json.dumps([{"name": "scenario ran", "ok": False, "detail": f"{type(e).__name__}: {e}"}])}
        db.add_eval_result(run_id, **result)
        failed = [c["name"] for c in json.loads(result["checks"]) if not c["ok"]]
        logger.info(f"    {result['status'].upper()}" + (f": {'; '.join(failed)}" if failed else ""))
    db.finish_eval_run(run_id)
    run = db.eval_runs()[0]
    logger.info(f"Run {run_id} done: {run['passed']}/{run['scenarios']} passed. See http://localhost:7860/dashboard")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["text", "voice"], default="text")
    parser.add_argument("--only", help="Comma-separated scenario ids")
    parser.add_argument("--append", type=int, metavar="RUN_ID", help="Add results to an existing run (run in batches)")
    parser.add_argument("--cooldown", type=float, default=60, help="Voice mode: seconds between calls (free-tier limits)")
    args = parser.parse_args()
    asyncio.run(main(args.mode, set(args.only.split(",")) if args.only else None, args.append, args.cooldown))
