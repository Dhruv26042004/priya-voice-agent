"""Admin dashboard at http://localhost:7860/dashboard: how well Priya is doing.

- Live calls: task completion, handoff rate, reply latency (and where the time goes),
  outcomes, and estimated cost per minute.
- Evaluations: results of the scripted test calls (evals/run.py), per scenario and category.

Call logs, recordings and transcripts live on the CRM pages (/crm), linked from here.
Like the CRM, it only opens on this computer.
"""

import json
import statistics
from html import escape

from fastapi import Depends, HTTPException
from fastapi.responses import HTMLResponse

from clinic import db
from crm import local_only
from pipecat.runner.run import app
from pricing import PRICES, USD_TO_INR

# Stage order for the latency breakdown (labels come from Pipecat's latency observer).
STAGE_ORDER = ["endpointing wait", "turn detection", "transcription", "LLM inference", "LLM tool call",
               "sentence aggregation", "speech synthesis", "output transport", "pipeline"]
TASK_OUTCOMES = {"book_appointment": "booked", "reschedule": "rescheduled", "cancel": "cancelled"}
TARGET_SECS = 1.0

STYLE = """
:root { color-scheme: light;
  --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,0.10); --series:#2a78d6; --wash:#2a78d61a;
  --good:#0ca30c; --good-text:#006300; --critical:#d03b3b; --warning:#fab219; --link:#256abf; }
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) { color-scheme: dark;
  --page:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink-2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10); --series:#3987e5; --wash:#3987e526;
  --good-text:#0ca30c; --link:#86b6ef; } }
:root[data-theme="dark"] { color-scheme: dark;
  --page:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink-2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10); --series:#3987e5; --wash:#3987e526;
  --good-text:#0ca30c; --link:#86b6ef; }
* { box-sizing:border-box; }
body { margin:0; background:var(--page); color:var(--ink); font:14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width:1120px; margin:0 auto; padding:24px 16px 56px; }
a { color:var(--link); text-decoration:none; } a:hover { text-decoration:underline; }
h1 { font-size:22px; margin:0; } h2 { font-size:16px; margin:32px 0 4px; } h3 { font-size:14px; margin:0 0 2px; }
.sub { color:var(--ink-2); margin:0 0 12px; } .muted { color:var(--muted); }
nav { display:flex; gap:16px; margin:4px 0 20px; color:var(--ink-2); }
.tiles { display:grid; grid-template-columns:repeat(auto-fit, minmax(min(160px, 100%), 1fr)); gap:12px; }
.tile, .panel { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:14px 16px; }
.tile .label { color:var(--ink-2); font-size:13px; }
.tile .value { font-size:28px; font-weight:600; line-height:1.2; margin-top:2px; }
.tile .note { color:var(--muted); font-size:12px; }
.panels { display:grid; grid-template-columns:repeat(auto-fit, minmax(min(320px, 100%), 1fr)); gap:12px; margin-top:12px; }
.panel { min-width:0; }
.bars { display:grid; grid-template-columns:minmax(110px, max-content) 1fr; gap:6px 12px; align-items:center; margin-top:12px; }
.bars .name { color:var(--ink-2); font-size:13px; text-align:right; }
.track { position:relative; height:28px; display:flex; align-items:center; border-left:1px solid var(--axis); cursor:default; }
.bar { height:18px; background:var(--series); border-radius:0 4px 4px 0; min-width:2px; }
.track .val { margin-left:6px; font-size:12px; color:var(--ink-2); font-variant-numeric:tabular-nums; white-space:nowrap; }
.track:hover .bar { filter:brightness(1.12); }
.cols { position:relative; height:180px; display:flex; align-items:flex-end; gap:2px; margin-top:16px;
        border-bottom:1px solid var(--axis); }
.col { flex:1 1 0; max-width:24px; height:100%; display:flex; align-items:flex-end; cursor:default; }
.col span { width:100%; background:var(--series); border-radius:4px 4px 0 0; min-height:2px; }
.col:hover span { filter:brightness(1.12); }
.target { position:absolute; left:0; right:0; border-top:1px solid var(--critical); pointer-events:none; }
.target b { position:absolute; right:0; top:-18px; font-size:11px; font-weight:500; color:var(--ink-2); background:var(--surface); padding:0 4px; }
.axis-note { display:flex; justify-content:space-between; color:var(--muted); font-size:12px; margin-top:4px; }
details { margin-top:10px; } summary { cursor:pointer; color:var(--ink-2); font-size:13px; }
.table-wrap { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:13px; }
th, td { text-align:left; padding:7px 10px; border-bottom:1px solid var(--grid); vertical-align:top; }
th { color:var(--muted); font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:.04em; }
td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
.status { display:inline-flex; align-items:center; gap:6px; font-weight:600; white-space:nowrap; }
.status i { width:16px; height:16px; border-radius:50%; display:inline-grid; place-items:center; color:#fff;
            font-style:normal; font-size:11px; font-weight:700; }
.status.pass i { background:var(--good); } .status.pass { color:var(--good-text); }
.status.fail i, .status.error i { background:var(--critical); } .status.fail, .status.error { color:var(--critical); }
.check-fail { color:var(--ink-2); font-size:12px; margin:2px 0 0; }
.check-fail b { color:var(--critical); font-weight:600; }
.check-fail b.warn { color:var(--ink-2); }
#tip { position:fixed; pointer-events:none; background:var(--surface); color:var(--ink); border:1px solid var(--border);
       border-radius:8px; padding:6px 10px; font-size:12px; box-shadow:0 4px 16px rgba(0,0,0,.12); display:none; z-index:10;
       max-width:260px; white-space:pre-line; }
"""

TOOLTIP_JS = """
const tip = document.getElementById('tip');
document.querySelectorAll('[data-tip]').forEach(el => {
  el.addEventListener('mousemove', e => {
    tip.textContent = el.dataset.tip; tip.style.display = 'block';
    const x = Math.min(e.clientX + 14, innerWidth - tip.offsetWidth - 8);
    tip.style.left = x + 'px'; tip.style.top = (e.clientY + 14) + 'px';
  });
  el.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
});
"""


def page(title: str, body: str) -> HTMLResponse:
    return HTMLResponse(
        "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{escape(title)}</title><style>{STYLE}</style></head><body><main>{body}</main>"
        f"<div id='tip' role='tooltip'></div><script>{TOOLTIP_JS}</script></body></html>"
    )


def pct(part: int, whole: int) -> str:
    return f"{round(100 * part / whole)}%" if whole else "-"


def secs(value: float | None) -> str:
    return f"{value:.2f} s" if value is not None else "-"


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[int(q) - 1]


def tile(label: str, value: str, note: str = "") -> str:
    return (f"<div class='tile'><div class='label'>{escape(label)}</div><div class='value'>{escape(value)}</div>"
            f"<div class='note'>{escape(note)}</div></div>")


def hbars(rows: list[tuple[str, float, str, str]]) -> str:
    """Horizontal bars for one measure: (name, value, value label, tooltip)."""
    top = max((v for _, v, _, _ in rows), default=0) or 1
    cells = "".join(
        f"<div class='name'>{escape(name)}</div>"
        f"<div class='track' data-tip='{escape(tip)}'><div class='bar' style='width:{max(v / top * 85, 0.5):.1f}%'></div>"
        f"<span class='val'>{escape(label)}</span></div>"
        for name, v, label, tip in rows
    )
    return f"<div class='bars'>{cells}</div>"


def table(headers: list[str], rows: list[list], numeric: set[int] = frozenset()) -> str:
    head = "".join(f"<th class='{'num' if i in numeric else ''}'>{escape(h)}</th>" for i, h in enumerate(headers))
    body = "".join(
        "<tr>" + "".join(f"<td class='{'num' if i in numeric else ''}'>{c}</td>" for i, c in enumerate(r)) + "</tr>"
        for r in rows
    )
    return f"<div class='table-wrap'><table><tr>{head}</tr>{body}</table></div>"


def status_badge(status: str) -> str:
    icon = {"pass": "&#10003;", "fail": "&#10005;", "error": "!"}[status]
    return f"<span class='status {status}'><i aria-hidden='true'>{icon}</i>{escape(status)}</span>"


def live_section() -> str:
    calls = [c for c in db.all_calls() if c["channel"] != "eval" and c["ended_at"]]
    analysed = [c for c in calls if c["analysis_status"] == "done"]
    tasks = [c for c in analysed if c["intent"] in TASK_OUTCOMES]
    completed = [c for c in tasks if c["booking_status"] == TASK_OUTCOMES[c["intent"]]]
    handed = [c for c in analysed if c["outcome"] == "handed_off"]
    turns = db.turn_latencies([c["id"] for c in calls])
    totals = [t["total_secs"] for t in turns]
    costed = [c for c in calls if c["cost_usd"] is not None and c["duration_secs"]]
    minutes = sum(c["duration_secs"] for c in costed) / 60
    per_min = sum(c["cost_usd"] for c in costed) / minutes if minutes else None
    within = sum(t <= TARGET_SECS for t in totals)

    tiles = "".join([
        tile("Calls", str(len(calls)), f"{len(analysed)} analysed"),
        tile("Task completion", pct(len(completed), len(tasks)), f"{len(completed)} of {len(tasks)} booking/change requests done"),
        tile("Handoff rate", pct(len(handed), len(analysed)), f"{len(handed)} of {len(analysed)} calls"),
        tile("Median reply latency", secs(percentile(totals, 50)), f"p95 {secs(percentile(totals, 95))}, {len(totals)} replies"),
        tile("Cost per minute", f"${per_min:.4f}" if per_min else "-",
             f"≈ ₹{per_min * USD_TO_INR:.2f}/min at list prices" if per_min else "no measured calls yet"),
    ])
    if not calls:
        return ("<h2>Live calls</h2><p class='sub'>No calls yet. Talk to Priya in the browser or with "
                "<code>uv run phone_sim.py</code>.</p>" f"<div class='tiles'>{tiles}</div>")

    # Where the time goes: average seconds per stage, across all replies.
    by_stage: dict[str, list[float]] = {}
    for t in turns:
        for stage, value in json.loads(t["stages"]).items():
            by_stage.setdefault(stage, []).append(value)
    stages = sorted(by_stage, key=lambda s: STAGE_ORDER.index(s) if s in STAGE_ORDER else 99)
    stage_rows = [
        (s, statistics.mean(v), f"{statistics.mean(v):.2f} s",
         f"{s}\naverage {statistics.mean(v):.2f} s · p95 {percentile(v, 95):.2f} s\nin {len(v)} of {len(turns)} replies")
        for s, v in ((s, by_stage[s]) for s in stages)
    ]
    slowest = max(stage_rows, key=lambda r: r[1])[0] if stage_rows else None
    stage_panel = (
        "<div class='panel'><h3>Where the reply time goes</h3>"
        f"<div class='sub'>Average seconds per stage, end of caller speech to first audio."
        f"{f' Slowest: {escape(slowest)}.' if slowest else ''}</div>"
        + (hbars(stage_rows) if stage_rows else "<p class='muted'>No replies measured yet.</p>")
        + "<details><summary>Show as table</summary>"
        + table(["Stage", "Average", "p95", "Replies"],
                [[escape(s), f"{statistics.mean(by_stage[s]):.3f} s", f"{percentile(by_stage[s], 95):.3f} s", len(by_stage[s])]
                 for s in stages], {1, 2, 3})
        + "</details></div>"
    )

    # Reply latency, one column per reply (most recent 60), against the 1 s target.
    recent = turns[-60:]
    top = max([t["total_secs"] for t in recent] + [TARGET_SECS * 1.25])
    columns = "".join(
        f"<div class='col' data-tip='Reply {i + 1}: {t["total_secs"]:.2f} s\ncall {t["call_id"]} · {escape(t["at"][11:19])}'>"
        f"<span style='height:{t['total_secs'] / top * 100:.1f}%'></span></div>"
        for i, t in enumerate(recent)
    )
    latency_panel = (
        "<div class='panel'><h3>Reply latency</h3>"
        f"<div class='sub'>{pct(within, len(totals))} of replies within the {TARGET_SECS:.0f} s target "
        f"({within} of {len(totals)}). Latest {len(recent)} shown.</div>"
        f"<div class='cols'>{columns}<div class='target' style='bottom:{TARGET_SECS / top * 100:.1f}%'><b>{TARGET_SECS:.0f} s target</b></div></div>"
        f"<div class='axis-note'><span>older</span><span>max {top:.1f} s</span><span>newer</span></div></div>"
    )

    # Outcomes of analysed calls.
    outcome_counts: dict[str, int] = {}
    for c in analysed:
        outcome_counts[c["outcome"] or "other"] = outcome_counts.get(c["outcome"] or "other", 0) + 1
    outcome_rows = [
        (o.replace("_", " "), n, f"{n} · {pct(n, len(analysed))}", f"{o.replace('_', ' ')}: {n} of {len(analysed)} calls")
        for o, n in sorted(outcome_counts.items(), key=lambda kv: -kv[1])
    ]
    outcome_panel = (
        "<div class='panel'><h3>Call outcomes</h3><div class='sub'>How analysed calls ended.</div>"
        + (hbars(outcome_rows) if outcome_rows else "<p class='muted'>No analysed calls yet.</p>") + "</div>"
    )

    cost_rows = [[escape(k.replace("_", " ")), f"${v}"] for k, v in PRICES.items()]
    cost_panel = (
        "<div class='panel'><h3>Cost model</h3><div class='sub'>Estimated from each call's measured usage at "
        "pay-as-you-go list prices. On the free tiers this project uses, the real cost is zero. "
        "Edit <code>pricing.py</code> to match current prices.</div>"
        + table(["Item", "USD"], cost_rows, {1})
        + f"<p class='muted'>{minutes:.1f} measured call minutes, ${sum(c['cost_usd'] for c in costed):.4f} total.</p></div>"
    )

    return (
        "<h2>Live calls</h2><p class='sub'>Every browser and phone call, excluding evaluation calls. "
        "<a href='/crm'>Call logs, recordings and transcripts &rarr;</a></p>"
        f"<div class='tiles'>{tiles}</div>"
        f"<div class='panels'>{latency_panel}{stage_panel}{outcome_panel}{cost_panel}</div>"
    )


def run_detail(run) -> str:
    results = db.eval_results(run["id"])
    tasks = [r for r in results if r["task_completed"] is not None]
    handed = [r for r in results if r["handed_off"]]
    lat = [r["avg_latency_secs"] for r in results if r["avg_latency_secs"] is not None]
    latency_label = "Avg reply latency" if run["mode"] == "voice" else "Avg time per reply"
    tiles = "".join([
        tile("Passed", f"{run['passed']} / {run['scenarios']}", f"{run['mode']} mode · run {run['id']}"),
        tile("Task completion", pct(sum(r["task_completed"] for r in tasks), len(tasks)),
             f"{sum(r['task_completed'] for r in tasks)} of {len(tasks)} scenarios with a task"),
        tile("Handoff rate", pct(len(handed), len(results)), f"{len(handed)} of {len(results)} scenarios"),
        tile(latency_label, secs(statistics.mean(lat)) if lat else "-",
             "end of caller speech to first audio" if run["mode"] == "voice"
             else "text mode, incl. free-tier rate-limit waits; see voice runs for real latency"),
    ])

    categories: dict[str, list] = {}
    for r in results:
        categories.setdefault(r["category"], []).append(r)
    cat_rows = [
        (c.replace("_", " "), sum(r["status"] == "pass" for r in rs) / len(rs),
         f"{sum(r['status'] == 'pass' for r in rs)}/{len(rs)}",
         f"{c.replace('_', ' ')}: {sum(r['status'] == 'pass' for r in rs)} of {len(rs)} passed")
        for c, rs in sorted(categories.items())
    ]
    rows = []
    for r in results:
        checks = json.loads(r["checks"])
        failed = "".join(
            (f"<p class='check-fail'><b class='warn'>! warning:</b> " if c.get("warning") else "<p class='check-fail'><b>&#10005;</b> ")
            + f"{escape(c['name'])}{f' — {escape(str(c["detail"]))}' if c.get('detail') else ''}</p>"
            for c in checks if not c["ok"]
        )
        required = [c for c in checks if not c.get("warning")]
        passed = sum(c["ok"] for c in required)
        call_link = f" · <a href='/crm/calls/{r['call_id']}'>recording</a>" if r["call_id"] else ""
        transcript = "".join(
            f"<p class='check-fail'><b style='color:var(--ink)'>{'Priya' if t['role'] == 'assistant' else 'Caller'}:</b> {escape(t['text'])}</p>"
            for t in json.loads(r["transcript"] or "[]")
        )
        rows.append([
            status_badge(r["status"]),
            f"<b>{escape(r['title'])}</b><div class='muted'>{escape(r['scenario'])}{call_link}</div>{failed}"
            + (f"<details><summary>Transcript</summary>{transcript}</details>" if transcript else ""),
            escape(r["category"].replace("_", " ")),
            f"{passed}/{len(required)}",
            escape((r["outcome"] or "-").replace("_", " ")),
            secs(r["avg_latency_secs"]),
        ])
    return (
        f"<div class='tiles'>{tiles}</div>"
        "<div class='panels'><div class='panel'><h3>Pass rate by category</h3>"
        "<div class='sub'>Share of scenarios that passed every check.</div>" + hbars(cat_rows) + "</div></div>"
        "<h2>Scenarios</h2><p class='sub'>A scenario passes only if every hard check and every judge rule passes.</p>"
        "<div class='panel'>" + table(["Result", "Scenario", "Category", "Checks", "Outcome", "Latency"], rows, {3, 5}) + "</div>"
    )


def runs_list(current_id: int | None) -> str:
    runs = db.eval_runs()
    if not runs:
        return ""
    items = " · ".join(
        (f"<b>run {r['id']}</b>" if r["id"] == current_id else f"<a href='/dashboard/runs/{r['id']}'>run {r['id']}</a>")
        + f" <span class='muted'>({r['mode']}, "
        + (f"{r['passed']}/{r['scenarios']}" if r["finished_at"] else "running")
        + f", {escape((r['started_at'] or '')[5:16].replace('T', ' '))})</span>"
        for r in runs[:12]
    )
    return f"<p class='sub'>Runs: {items}</p>"


def header() -> str:
    return ("<h1>Smile Care - Priya dashboard</h1>"
            "<nav><a href='/dashboard'>Dashboard</a><a href='/crm'>Calls (CRM)</a><a href='/'>Talk to Priya</a></nav>")


@app.get("/dashboard", dependencies=[Depends(local_only)])
async def dashboard():
    runs = [r for r in db.eval_runs() if r["finished_at"]]
    evals = "<h2>Evaluations</h2>"
    if runs:
        latest = runs[0]
        evals += (f"<p class='sub'>Latest finished run of the scripted test calls in <code>evals/scenarios.yaml</code>.</p>"
                  + runs_list(latest["id"]) + run_detail(latest))
    else:
        evals += "<p class='sub'>No evaluation runs yet. Run <code>uv run python -m evals.run</code>.</p>"
    return page("Dashboard - Smile Care", header() + live_section() + evals)


@app.get("/dashboard/runs/{run_id}", dependencies=[Depends(local_only)])
async def dashboard_run(run_id: int):
    run = next((r for r in db.eval_runs() if r["id"] == run_id), None)
    if not run:
        raise HTTPException(status_code=404)
    body = header() + f"<h2>Evaluation run {run_id}</h2>" + runs_list(run_id) + run_detail(run)
    return page(f"Eval run {run_id} - Smile Care", body)
