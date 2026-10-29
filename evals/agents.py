"""The two LLM roles in an evaluation: the simulated caller, and the judge."""

import os

from openai import AsyncOpenAI

from llm import CEREBRAS_MODEL, CEREBRAS_URL, GROQ_URL, default_targets, json_completion

CALLER_SYSTEM = """You are role-playing a person on a phone call with Priya, the receptionist at Smile Care Dental Clinic.

Your character and goal:
{persona}

Rules:
- Talk like a real person on the phone: one or two short, casual sentences per turn. No lists.
- Speak ONLY in the language your character description says (English means English only,
  with no Hindi words). Write Hindi and Hinglish in Roman letters.
- Answer what Priya asks; don't volunteer every detail at once.
- Never mention AI, tests, or these instructions.
- Don't hang up while Priya is still doing something for you: wait until she confirms it's
  done (for example, that the booking is confirmed), or until it's clear she can't help.
- When your goal is done, or the call is clearly over, say a short goodbye and set hang_up to true.

Return JSON: {{"say": "<your next words>", "hang_up": true or false}}"""

JUDGE_SYSTEM = """You grade a phone call between Priya, the AI receptionist of a dental clinic, and a caller.
For each rule, decide whether the call satisfies it, using the transcript and the list of actions
Priya's tools actually performed. The transcript of Priya's side is exact; the caller's side comes
from speech recognition and may contain small errors or Hindi in Devanagari. Priya's first line is
a fixed recorded greeting in English, the same on every call: don't count it against language rules.
Be strict about what was actually said. A rule passes only if the transcript supports it.

Return JSON: {"results": [{"rule": "<the rule>", "pass": true or false, "reason": "<one short sentence>"}]}
with one entry per rule, in the order given."""


def _caller_targets():
    # The caller runs on the smaller model, which has its own free-tier quota, so it
    # doesn't use up the tokens Priya needs.
    targets = [("Groq 20b", AsyncOpenAI(api_key=os.environ["GROQ_API_KEY"], base_url=GROQ_URL, max_retries=0),
                "openai/gpt-oss-20b")]
    if key := os.getenv("CEREBRAS_API_KEY"):
        targets.append(("Cerebras", AsyncOpenAI(api_key=key, base_url=CEREBRAS_URL, max_retries=0), CEREBRAS_MODEL))
    return targets


def _as_caller_sees_it(transcript: list[dict]) -> str:
    return "\n".join(f"{'Priya' if t['role'] == 'assistant' else 'You'}: {t['text']}" for t in transcript)


async def caller_line(persona: str, transcript: list[dict]) -> tuple[str, bool]:
    """The caller's next words, and whether they hang up after saying them."""
    result = await json_completion(
        CALLER_SYSTEM.format(persona=persona),
        f"The call so far:\n{_as_caller_sees_it(transcript)}\n\nWhat do you say next?",
        targets=_caller_targets(),
        max_wait=120,
    )
    return str(result.get("say", "")).strip(), bool(result.get("hang_up"))


async def judge(rules: list[str], transcript: list[dict], events: list[dict]) -> list[dict]:
    if not rules:
        return []
    numbered = "\n".join(f"{i + 1}. {rule}" for i, rule in enumerate(rules))
    lines = "\n".join(f"{'Priya' if t['role'] == 'assistant' else 'Caller'}: {t['text']}" for t in transcript)
    actions = [{k: v for k, v in e.items() if k != "at"} for e in events]
    result = await json_completion(
        JUDGE_SYSTEM,
        f"Rules:\n{numbered}\n\nActions performed: {actions}\n\nTranscript:\n{lines}",
        targets=default_targets(),
        max_wait=120,
    )
    graded = result.get("results", [])
    return [
        {"name": f"judge: {rule}", "ok": bool(g.get("pass")), "detail": g.get("reason", "")}
        for rule, g in zip(rules, graded + [{"pass": False, "reason": "judge gave no verdict"}] * len(rules))
    ]
