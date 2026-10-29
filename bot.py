"""Dental clinic voice receptionist.

Pipeline: caller audio -> Deepgram STT -> Groq LLM (+ tools) -> Cartesia TTS -> caller,
streaming at every stage. One server handles both the browser and phone calls:

    uv run bot.py

- Browser: open http://localhost:7860 and click Connect.
- Phone: point a Twilio number's "A call comes in" webhook at https://<public-host>/twilio/voice
  (see README), or dial out with outbound.py.
"""

import asyncio
import csv
import os
import time
import wave
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.observers.user_bot_latency_observer import MeasuredFrom, UserBotLatencyObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.runner.types import RunnerArguments, WebSocketRunnerArguments
from pipecat.runner.utils import _create_telephony_transport, create_transport, parse_telephony_websocket
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.cerebras.llm import CerebrasLLMService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

import crm  # noqa: F401  (registers the /crm pages on the runner's server)
import dashboard  # noqa: F401  (registers the /dashboard pages)
import postcall
import telephony  # noqa: F401  (registers the /twilio/voice webhook on the runner's server)
from clinic import db
from clinic.schedule import normalize_phone
from clinic.speech_filter import DropJsonFilter
from clinic.tools import TOOLS
from llm import FailoverGroqLLMService
from pricing import UsageObserver, call_cost
from prompts import INBOUND_OPENING, LLM_FAILED_LINE, build_system_prompt, outbound_opening

load_dotenv(override=True)

LATENCY_LOG = Path("logs/latency.csv")
RECORDINGS_DIR = Path("data/recordings")
# Post-call analyses still running; held so they aren't garbage-collected mid-way.
_postcall_tasks: set[asyncio.Task] = set()


def log_latency(breakdown) -> None:
    """Print a per-stage latency breakdown and append it to logs/latency.csv.

    One row per stage, since the set of stages varies from turn to turn.
    """
    for line in breakdown.turn_contribution_lines():
        logger.info(f"[latency] {line}")

    LATENCY_LOG.parent.mkdir(exist_ok=True)
    new_file = not LATENCY_LOG.exists()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with LATENCY_LOG.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["timestamp", "stage", "secs"])
        for c in breakdown.contributions:
            writer.writerow([timestamp, c.label, f"{c.duration_secs:.3f}"])
        writer.writerow([timestamp, "TOTAL", f"{breakdown.total_secs:.3f}"])


async def make_transport(runner_args):
    """Browser sessions use WebRTC; phone calls (and the phone simulator) arrive on /ws."""
    if not isinstance(runner_args, WebSocketRunnerArguments):
        return await create_transport(
            runner_args, {"webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True)}
        )

    transport_type, call_data = await parse_telephony_websocket(runner_args.websocket)
    runner_args.call_data = call_data
    params = FastAPIWebsocketParams(audio_in_enabled=True, audio_out_enabled=True, add_wav_header=False)
    if transport_type != "twilio":
        # Exotel, Plivo, Telnyx: let Pipecat build the provider's serializer.
        return await _create_telephony_transport(runner_args.websocket, params, transport_type, call_data)

    # Built here rather than by Pipecat so a simulated call (phone_sim.py) doesn't try to
    # hang up through Twilio's API; it has no Twilio call to hang up.
    simulated = (call_data.body or {}).get("simulated") == "true"
    params.serializer = TwilioFrameSerializer(
        stream_sid=call_data.stream_id,
        call_sid=call_data.call_id,
        account_sid=os.getenv("TWILIO_ACCOUNT_SID"),
        auth_token=os.getenv("TWILIO_AUTH_TOKEN"),
        params=TwilioFrameSerializer.InputParams(auto_hang_up=not simulated),
    )
    return FastAPIWebsocketTransport(websocket=runner_args.websocket, params=params)


def call_info(runner_args) -> dict:
    """Who's on the line and why, from the browser session or the phone stream's parameters."""
    data = getattr(runner_args, "call_data", None)
    if data is None:
        return {"channel": "browser", "direction": "inbound"}
    body = data.body or {}
    direction = body.get("direction", "inbound")
    outbound = db.get_outbound(int(body["outbound_id"])) if body.get("outbound_id") else None
    remote = body.get("to_number") if direction == "outbound" else body.get("from_number")
    return {
        # Eval calls (evals/run.py --mode voice) are kept out of the live call stats.
        "channel": "eval" if body.get("eval_scenario") else "phone_sim" if body.get("simulated") == "true" else "phone",
        "direction": direction,
        "call_sid": data.call_id,
        "from_number": body.get("from_number"),
        "to_number": body.get("to_number"),
        "caller_phone": normalize_phone(remote or ""),
        "outbound_id": outbound["id"] if outbound else None,
        "patient_name": outbound["name"] if outbound else None,
        "purpose": outbound["purpose"] if outbound else None,
    }


def save_recording(path: Path, audio: bytes, sample_rate: int, num_channels: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(num_channels)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(audio)


def make_llm(call: dict):
    """The LLM. Both providers serve gpt-oss-120b on free tiers. By default Groq is
    primary and Cerebras (if CEREBRAS_API_KEY is set) takes over on rate limits.
    LLM_PROVIDER=cerebras uses Cerebras alone."""
    # Built per call so "today", the date table and the caller's details are current.
    prompt = build_system_prompt(call)
    if os.getenv("LLM_PROVIDER", "groq").lower() == "cerebras":
        return CerebrasLLMService(
            api_key=os.environ["CEREBRAS_API_KEY"],
            settings=CerebrasLLMService.Settings(
                model=os.getenv("LLM_MODEL", "gpt-oss-120b"),
                system_instruction=prompt,
                extra={"reasoning_effort": "low"},
            ),
        )
    return FailoverGroqLLMService(
        backup_api_key=os.getenv("CEREBRAS_API_KEY"),
        backup_base_url="https://api.cerebras.ai/v1",
        backup_model="gpt-oss-120b",
        api_key=os.environ["GROQ_API_KEY"],
        settings=FailoverGroqLLMService.Settings(
            model=os.getenv("LLM_MODEL", "openai/gpt-oss-120b"),
            system_instruction=prompt,
            # Low reasoning keeps time-to-first-token short for voice.
            reasoning_effort="low",
        ),
    )


async def bot(runner_args: RunnerArguments):
    transport = await make_transport(runner_args)
    call = call_info(runner_args)
    logger.info(f"New {call['channel']} call ({call['direction']}) from {call.get('caller_phone')}")

    # "multi" lets Nova-3 transcribe English, Hindi and code-switched Hinglish.
    stt = DeepgramSTTService(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        settings=DeepgramSTTService.Settings(
            model="nova-3-general",
            language=os.getenv("STT_LANGUAGE", "multi"),
            # Writes spoken numbers as digits, so phone numbers reach the LLM cleanly.
            smart_format=True,
        ),
    )

    llm = make_llm(call)

    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            voice=os.environ["CARTESIA_VOICE_ID"],
            language=os.getenv("TTS_LANGUAGE", "en"),
        ),
        # Never speak JSON, if the LLM ever writes a tool call as text.
        text_filters=[DropJsonFilter()],
    )

    context = LLMContext(tools=TOOLS)
    # Silero VAD detects speech; the smart-turn model decides when the caller has
    # actually finished, so a mid-sentence pause doesn't end the turn. If it judges a
    # turn unfinished, it still ends it after 2 s of silence (default 3 s felt sluggish).
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            user_turn_strategies=UserTurnStrategies(
                stop=[
                    TurnAnalyzerUserTurnStopStrategy(
                        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams(stop_secs=2.0))
                    )
                ]
            ),
        ),
    )

    # Stereo recording: caller on the left channel, bot on the right.
    recorder = AudioBufferProcessor(num_channels=2)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            aggregators.user(),
            llm,
            tts,
            transport.output(),
            recorder,
            aggregators.assistant(),
        ]
    )

    latency = UserBotLatencyObserver()
    usage = UsageObserver()

    @latency.event_handler("on_latency_breakdown")
    async def on_latency_breakdown(_observer, breakdown):
        log_latency(breakdown)
        # Reply latency only: the opening line is timed from connection, not from speech.
        if breakdown.measured_from == MeasuredFrom.USER_SILENCE:
            stages = {c.label: round(c.duration_secs, 3) for c in breakdown.contributions}
            db.add_turn_latency(call_id, round(breakdown.total_secs, 3), stages)

    call_id = db.start_call(
        call["channel"], call["direction"], call.get("call_sid"),
        call.get("from_number"), call.get("to_number"), call.get("outbound_id"),
    )
    recording_path = RECORDINGS_DIR / f"call_{call_id}.wav"
    # Per-call state shared by the tools: who's calling, the recorder, the phone line.
    session = {
        "caller_phone": call.get("caller_phone"),
        "call_sid": call.get("call_sid"),
        "can_transfer": call["channel"] == "phone",
        "recorder": recorder,
        "serializer": getattr(getattr(transport, "_params", None), "serializer", None),
        "events": [],
    }
    transcript: list[dict] = []

    @aggregators.user().event_handler("on_user_turn_message_added")
    async def on_user_turn(_aggregator, message):
        transcript.append({"role": "user", "text": message.content, "at": message.timestamp})

    @aggregators.assistant().event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn(_aggregator, message):
        if message.content:
            transcript.append(
                {"role": "assistant", "text": message.content, "at": message.timestamp, "interrupted": message.interrupted}
            )

    @recorder.event_handler("on_audio_data")
    async def on_audio_data(_recorder, audio, sample_rate, num_channels):
        # Fires once when recording stops: at hang-up, or when the caller opts out.
        if session.get("recording_opted_out") or not audio:
            return
        save_recording(recording_path, audio, sample_rate, num_channels)
        logger.info(f"Saved recording {recording_path}")

    # Phone audio is 8 kHz; the browser runs at the pipeline defaults.
    on_phone = call["channel"] in ("phone", "phone_sim", "eval")
    rates = {"audio_in_sample_rate": 8000, "audio_out_sample_rate": 8000} if on_phone else {}
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True, **rates),
        observers=[latency, usage],
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        app_resources=session,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(_transport, _client):
        logger.info("Caller connected")
        # Recording starts with the opening line, so the notice itself is on the recording.
        await recorder.start_recording()
        opening = (
            outbound_opening(call["patient_name"])
            if call["direction"] == "outbound" and call.get("patient_name")
            else INBOUND_OPENING
        )
        # A fixed line rather than LLM output: instant, and the notice can't be skipped.
        await worker.queue_frames([TTSSpeakFrame(opening)])

    @worker.event_handler("on_pipeline_error")
    async def on_pipeline_error(_worker, frame):
        # If the LLM fails for a turn (e.g. both free tiers rate limited), say something
        # rather than leave the caller in silence. Their words are lost, so ask again.
        if getattr(frame, "processor", None) is llm:
            logger.warning(f"LLM failed this turn, asking the caller to repeat: {frame.error}")
            await worker.queue_frames([TTSSpeakFrame(LLM_FAILED_LINE)])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_transport, _client):
        logger.info("Caller disconnected")
        await worker.cancel()

    @worker.event_handler("on_pipeline_finished")
    async def on_pipeline_finished(_worker, _frame):
        opted_out = session.get("recording_opted_out")
        saved = recording_path.exists() and not opted_out
        db.end_call(
            call_id, str(recording_path) if saved else None, "opted_out" if opted_out else "saved",
            transcript, session["events"],
        )
        phone_secs = db.get_call(call_id)["duration_secs"] if on_phone else 0
        db.update_call(
            call_id,
            llm_prompt_tokens=usage.prompt_tokens,
            llm_completion_tokens=usage.completion_tokens,
            tts_chars=usage.tts_chars,
            stt_secs=round(usage.stt_secs, 1),
            cost_usd=round(call_cost(usage.stt_secs, usage.prompt_tokens, usage.completion_tokens,
                                     usage.tts_chars, phone_secs), 5),
        )
        # Summarise in the background; the server keeps serving other calls meanwhile.
        task = asyncio.create_task(postcall.process_call(call_id))
        _postcall_tasks.add(task)
        task.add_done_callback(_postcall_tasks.discard)

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    await runner.run()


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
