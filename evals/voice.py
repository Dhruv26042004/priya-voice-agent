"""Voice-mode evaluation: a real phone call to the running bot.

The caller LLM's lines are synthesised (Cartesia, 8 kHz), optionally mixed with background
noise, and streamed to the bot's /ws endpoint in Twilio's protocol, 20 ms at a time, like a
real phone line. Priya's replies are transcribed (Deepgram) so the caller only knows what it
actually heard. This measures what text mode can't: speech recognition under noise, turn
detection, interruptions, and true end-of-speech to first-audio latency.

Calls are tagged as evals, so they stay out of the live stats; anything they book is
cancelled afterwards.
"""

import asyncio
import base64
import json
import os
import time
import uuid
from collections import deque

import httpx
import numpy as np
import websockets
from loguru import logger

from clinic import db
from clinic.language import speaks_hindi
from evals.agents import caller_line, judge
from evals.run import apply_setup, call_context, check_expectations, summarise
from phone_sim import pcm_to_ulaw, ulaw_to_pcm

BOT_WS = os.getenv("EVAL_BOT_WS", "ws://localhost:7860/ws")
RATE, FRAME = 8000, 160          # 8 kHz phone audio, 20 ms frames
QUIET_SECS = 1.5                 # this much bot silence ends its turn
REPLY_TIMEOUT_SECS = 20
SPEECH_RMS = 2500                # typical loudness of the synthesised caller voice
MAX_TURNS = 10
INTERJECTION = "Sorry, sorry, one second please."


def synth(text: str) -> np.ndarray:
    # Hinglish lines in a Hindi voice; English lines in English, so digits stay English digits.
    language = "hi" if speaks_hindi(text) else "en"
    resp = httpx.post(
        "https://api.cartesia.ai/tts/bytes",
        headers={"X-API-Key": os.environ["CARTESIA_API_KEY"], "Cartesia-Version": "2026-03-01"},
        json={
            "model_id": "sonic-3.6", "transcript": text, "language": language,
            "voice": {"mode": "id", "id": os.getenv("EVAL_CALLER_VOICE_ID") or os.environ["CARTESIA_VOICE_ID"]},
            "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": RATE},
        },
        timeout=30,
    )
    resp.raise_for_status()
    return np.frombuffer(resp.content, dtype=np.int16)


async def transcribe(pcm: bytes) -> str:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.deepgram.com/v1/listen?model=nova-3-general&language=multi&smart_format=true"
            f"&encoding=linear16&sample_rate={RATE}",
            headers={"Authorization": f"Token {os.environ['DEEPGRAM_API_KEY']}", "Content-Type": "application/octet-stream"},
            content=pcm,
        )
    resp.raise_for_status()
    return resp.json()["results"]["channels"][0]["alternatives"][0]["transcript"]


def pink_noise(seconds: float, rms: float, seed: int = 7) -> np.ndarray:
    """Background noise with a 1/f spectrum, closer to a street or cafe than white noise."""
    rng = np.random.default_rng(seed)
    n = int(seconds * RATE)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    spectrum /= np.sqrt(np.maximum(np.arange(len(spectrum)), 1))
    noise = np.fft.irfft(spectrum, n)
    return noise / np.sqrt(np.mean(noise**2)) * rms


class PhoneLine:
    """Streams the caller's side of the call: speech when there is some, otherwise
    silence, with background noise mixed in throughout, paced in real time."""

    def __init__(self, ws, stream_sid: str, snr_db: float | None):
        self._ws, self._sid = ws, stream_sid
        self._frames: deque[np.ndarray] = deque()
        self._noise = pink_noise(20, SPEECH_RMS / 10 ** (snr_db / 20)) if snr_db is not None else None
        self._noise_pos = 0
        self.drained = asyncio.Event()
        self.drained.set()

    def say(self, pcm: np.ndarray) -> None:
        pcm = np.pad(pcm, (0, -len(pcm) % FRAME))
        self._frames.extend(pcm.reshape(-1, FRAME))
        self.drained.clear()

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        next_at = loop.time()
        while True:
            frame = self._frames.popleft().astype(np.float64) if self._frames else np.zeros(FRAME)
            if not self._frames:
                self.drained.set()
            if self._noise is not None:
                frame = frame + self._noise[self._noise_pos:self._noise_pos + FRAME]
                self._noise_pos = (self._noise_pos + FRAME) % (len(self._noise) - FRAME)
            payload = pcm_to_ulaw(np.clip(frame, -32768, 32767).astype(np.int16))
            await self._ws.send(json.dumps(
                {"event": "media", "streamSid": self._sid, "media": {"payload": base64.b64encode(payload).decode()}}
            ))
            next_at += FRAME / RATE
            await asyncio.sleep(max(0, next_at - loop.time()))


class BotEar:
    """Collects the bot's audio and notices when it stops talking or is interrupted."""

    def __init__(self, ws):
        self._ws = ws
        self.chunks: list[tuple[float, bytes]] = []
        self.clears: list[float] = []
        self.closed = asyncio.Event()

    async def run(self) -> None:
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                if msg.get("event") == "media":
                    pcm = ulaw_to_pcm(base64.b64decode(msg["media"]["payload"])).tobytes()
                    self.chunks.append((time.monotonic(), pcm))
                elif msg.get("event") == "clear":
                    self.clears.append(time.monotonic())
        except websockets.ConnectionClosed:
            pass
        self.closed.set()

    async def next_turn(self, after: float, timeout: float = REPLY_TIMEOUT_SECS) -> tuple[float, bytes] | None:
        """Wait for the bot to start speaking after `after`, then to go quiet. Returns
        (time its first audio arrived, its audio), or None if it never spoke."""
        while True:
            mine = [c for c in self.chunks if c[0] > after]
            if mine and (time.monotonic() - mine[-1][0] > QUIET_SECS or self.closed.is_set()):
                return mine[0][0], b"".join(pcm for _, pcm in mine)
            if not mine and (time.monotonic() - after > timeout or self.closed.is_set()):
                return None
            await asyncio.sleep(0.05)


async def run_voice(s: dict) -> dict:
    started = time.monotonic()
    call = call_context(s)
    setup_ids = apply_setup(s)
    stream_sid, call_sid = f"MZeval{uuid.uuid4().hex[:24]}", f"CAeval{uuid.uuid4().hex[:24]}"
    params = {"simulated": "true", "eval_scenario": s["id"], "direction": call["direction"]}
    outbound_id = None
    if call["direction"] == "outbound":
        outbound_id = db.add_outbound(call["patient_name"], call["caller_phone"], call["purpose"], status="in_progress")
        params |= {"outbound_id": str(outbound_id), "to_number": call["caller_phone"]}
    elif call.get("caller_phone"):
        params["from_number"] = call["caller_phone"]

    heard: list[dict] = []           # the call as the caller heard it
    latencies: list[float] = []      # caller stops talking -> first bot audio
    missed_replies = 0
    barge_in_secs = None
    interjection = await asyncio.to_thread(synth, INTERJECTION) if s.get("barge_in") else None

    async with websockets.connect(BOT_WS) as ws:
        await ws.send(json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"}))
        await ws.send(json.dumps({
            "event": "start", "sequenceNumber": "1", "streamSid": stream_sid,
            "start": {"streamSid": stream_sid, "callSid": call_sid, "accountSid": "ACeval", "customParameters": params,
                      "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": RATE, "channels": 1}},
        }))
        line, ear = PhoneLine(ws, stream_sid, s.get("noise_snr_db")), BotEar(ws)
        tasks = [asyncio.create_task(line.run()), asyncio.create_task(ear.run())]
        try:
            opening = await ear.next_turn(after=started)
            if opening:
                heard.append({"role": "assistant", "text": await transcribe(opening[1])})
            for turn in range(MAX_TURNS):
                say, hang_up = await caller_line(s["persona"], heard)
                if not say or ear.closed.is_set():
                    break
                line.say(await asyncio.to_thread(synth, say))
                await line.drained.wait()
                spoke_until = time.monotonic()
                heard.append({"role": "user", "text": say})

                if interjection is not None and barge_in_secs is None and turn == 0:
                    # Talk over Priya one second into her reply, and time how fast she stops.
                    while not [c for c in ear.chunks if c[0] > spoke_until] and not ear.closed.is_set():
                        await asyncio.sleep(0.02)
                    await asyncio.sleep(1.0)
                    barge_at = time.monotonic()
                    line.say(interjection)
                    await line.drained.wait()
                    heard.append({"role": "user", "text": INTERJECTION})
                    stops = [t for t in ear.clears if t > barge_at]
                    barge_in_secs = round(stops[0] - barge_at, 3) if stops else None
                    spoke_until = time.monotonic()

                reply = await ear.next_turn(after=spoke_until, timeout=10 if hang_up else REPLY_TIMEOUT_SECS)
                if reply is None:
                    if not hang_up:
                        missed_replies += 1
                else:
                    latencies.append(reply[0] - spoke_until)
                    heard.append({"role": "assistant", "text": await transcribe(reply[1])})
                if hang_up or ear.closed.is_set():
                    break
        finally:
            if not ear.closed.is_set():
                await ws.send(json.dumps({"event": "stop", "streamSid": stream_sid}))
            for task in tasks:
                task.cancel()

    # The bot writes the call record and its post-call analysis; wait for it.
    row = None
    for _ in range(180):
        row = db.call_by_sid(call_sid)
        if row and row["analysis_status"] in ("done", "failed"):
            break
        await asyncio.sleep(0.5)
    if not row or not row["transcript"]:
        raise RuntimeError("The bot didn't record this call. Is `uv run bot.py` running?")
    transcript, events = json.loads(row["transcript"]), json.loads(row["events"] or "[]")

    checks = check_expectations(s, events, row["outcome"], call.get("caller_phone"))
    checks.append({"name": "Priya answered every turn", "ok": missed_replies == 0,
                   "detail": f"{missed_replies} turn(s) got no reply"})
    if s.get("barge_in"):
        checks.append({"name": "Priya stopped talking within 1.5 s when interrupted",
                       "ok": barge_in_secs is not None and barge_in_secs <= 1.5, "detail": f"stopped after {barge_in_secs} s"})
    checks += await judge(s.get("judge", []), transcript, events)

    # Leave the real schedule as it was.
    for appt_id in {e["appointment_id"] for e in events if e["type"] in ("booked", "rescheduled")} | set(setup_ids):
        db.cancel(appt_id)
    if outbound_id:
        db.update_outbound(outbound_id, status="completed", last_result="eval")

    tokens = (row["llm_prompt_tokens"] or 0) + (row["llm_completion_tokens"] or 0)
    result = summarise(s, checks, events, row["outcome"], transcript, latencies, tokens, started, call_id=row["id"])
    logger.info(f"    latency per turn: {[round(x, 2) for x in latencies]}"
                + (f", barge-in stop: {barge_in_secs}s" if s.get("barge_in") else ""))
    return result
