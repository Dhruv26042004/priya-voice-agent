"""Phone-call simulator: talk to Priya through the bot's real phone path, using your
mic and speakers instead of a phone line. No Twilio account needed.

    uv run bot.py                               # in one terminal
    uv run phone_sim.py                         # you call the clinic
    uv run phone_sim.py --outbound --name Dhruv # the clinic calls you (reminder call)

It connects to the bot's /ws endpoint and speaks Twilio's Media Streams protocol, with
audio converted to 8 kHz mu-law exactly as on a real call. So the bot runs its phone
code unchanged: caller ID, recording notice, recording, tools, hang-up. Press Ctrl+C
to hang up. Use headphones, or --half-duplex if you only have speakers.
"""

import argparse
import asyncio
import base64
import json
import threading
import uuid

import numpy as np
import sounddevice as sd
import soxr
import websockets

PHONE_RATE = 8000
FRAME = PHONE_RATE // 50  # 20 ms, the frame size Twilio uses
BIAS = 0x84


# --- G.711 mu-law, the phone network's audio codec ---


def _decode_table() -> np.ndarray:
    u = ~np.arange(256, dtype=np.int32) & 0xFF
    exponent, mantissa = (u >> 4) & 0x07, u & 0x0F
    magnitude = (((mantissa << 3) + BIAS) << exponent) - BIAS
    return np.where(u & 0x80, -magnitude, magnitude).astype(np.int16)


ULAW_TO_PCM = _decode_table()


def pcm_to_ulaw(pcm: np.ndarray) -> bytes:
    # The reference G.711 encoder works on 14-bit samples.
    x = pcm.astype(np.int32) >> 2
    mask = np.where(x < 0, 0x7F, 0xFF)
    x = np.minimum(np.abs(x), 8159) + (BIAS >> 2)
    segment = np.clip(np.floor(np.log2(x)).astype(np.int32) - 5, 0, 7)
    mantissa = (x >> (segment + 1)) & 0x0F
    code = np.where(x > 0x1FFF, 0x7F, (segment << 4) | mantissa)  # loudest samples saturate
    return (code ^ mask).astype(np.uint8).tobytes()


def ulaw_to_pcm(data: bytes) -> np.ndarray:
    return ULAW_TO_PCM[np.frombuffer(data, dtype=np.uint8)]


# --- Speaker playback with a small jitter buffer ---


class Speaker:
    def __init__(self, rate: int):
        self._buf = np.zeros(0, dtype=np.int16)
        self._lock = threading.Lock()
        self._upsample = soxr.ResampleStream(PHONE_RATE, rate, 1, dtype="int16")
        self._stream = sd.OutputStream(samplerate=rate, channels=1, dtype="int16", callback=self._play)

    def _play(self, outdata, frames, _time, _status):
        with self._lock:
            chunk, self._buf = self._buf[:frames], self._buf[frames:]
        outdata[: len(chunk), 0] = chunk
        outdata[len(chunk):, 0] = 0

    def add(self, pcm8k: np.ndarray) -> None:
        pcm = self._upsample.resample_chunk(pcm8k)
        with self._lock:
            self._buf = np.concatenate([self._buf, pcm])

    def clear(self) -> None:
        # The bot sends "clear" when you interrupt it: drop what it hasn't said yet.
        with self._lock:
            self._buf = np.zeros(0, dtype=np.int16)

    @property
    def playing(self) -> bool:
        with self._lock:
            return len(self._buf) > 0

    def __enter__(self):
        self._stream.start()
        return self

    def __exit__(self, *_):
        self._stream.stop()
        self._stream.close()


async def call(args) -> None:
    in_rate = int(sd.query_devices(kind="input")["default_samplerate"])
    out_rate = int(sd.query_devices(kind="output")["default_samplerate"])
    loop = asyncio.get_running_loop()
    mic_chunks: asyncio.Queue[np.ndarray] = asyncio.Queue()
    downsample = soxr.ResampleStream(in_rate, PHONE_RATE, 1, dtype="int16")

    def on_mic(indata, _frames, _time, _status):
        loop.call_soon_threadsafe(mic_chunks.put_nowait, indata[:, 0].copy())

    stream_sid, call_sid = f"MZsim{uuid.uuid4().hex[:24]}", f"CAsim{uuid.uuid4().hex[:24]}"
    parameters = {"simulated": "true", "direction": "outbound" if args.outbound else "inbound"}
    if args.outbound:
        parameters |= {"outbound_id": str(args.outbound_id), "to_number": args.number}
    else:
        parameters |= {"from_number": args.number, "to_number": "+910000000000"}

    async with websockets.connect(args.url) as ws:
        await ws.send(json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"}))
        await ws.send(json.dumps({
            "event": "start", "sequenceNumber": "1", "streamSid": stream_sid,
            "start": {"streamSid": stream_sid, "callSid": call_sid, "accountSid": "ACsimulated",
                      "customParameters": parameters,
                      "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": PHONE_RATE, "channels": 1}},
        }))
        print("Connected. Talk to Priya. Press Ctrl+C to hang up.\n")

        with Speaker(out_rate) as speaker, sd.InputStream(
            samplerate=in_rate, channels=1, dtype="int16", blocksize=in_rate // 50, callback=on_mic
        ):

            speaking = False

            async def receive():
                nonlocal speaking
                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("event") == "media":
                        speaker.add(ulaw_to_pcm(base64.b64decode(msg["media"]["payload"])))
                        if not speaking:
                            print("  Priya is speaking...")
                            speaking = True
                    elif msg.get("event") == "clear":
                        speaker.clear()

            async def send():
                nonlocal speaking
                pending = np.zeros(0, dtype=np.int16)
                while True:
                    pending = np.concatenate([pending, downsample.resample_chunk(await mic_chunks.get())])
                    if speaking and not speaker.playing:
                        speaking = False
                    while len(pending) >= FRAME:
                        frame, pending = pending[:FRAME], pending[FRAME:]
                        if args.half_duplex and speaker.playing:
                            # Without headphones the mic hears the bot; send silence meanwhile.
                            frame = np.zeros(FRAME, dtype=np.int16)
                        await ws.send(json.dumps({
                            "event": "media", "streamSid": stream_sid,
                            "media": {"payload": base64.b64encode(pcm_to_ulaw(frame)).decode()},
                        }))

            receiver, sender = asyncio.create_task(receive()), asyncio.create_task(send())
            try:
                await receiver
                print("\nPriya hung up.")
            except websockets.ConnectionClosed:
                print("\nCall ended.")
            finally:
                sender.cancel()
                if not receiver.done():
                    receiver.cancel()
                    # Tell the bot the caller hung up, like Twilio does.
                    await ws.send(json.dumps({"event": "stop", "streamSid": stream_sid}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--number", default="+919876543210", help="Your simulated phone number (caller ID)")
    parser.add_argument("--outbound", action="store_true", help="Simulate the clinic calling you")
    parser.add_argument("--name", default="Dhruv", help="Patient name for --outbound")
    parser.add_argument("--purpose", default="their six-month dental checkup is due", help="Reason for --outbound")
    parser.add_argument("--half-duplex", action="store_true", help="Mute your mic while Priya speaks (no headphones)")
    parser.add_argument("--url", default="ws://localhost:7860/ws", help="Bot's phone WebSocket")
    args = parser.parse_args()

    outbound_id = None
    if args.outbound:
        from dotenv import load_dotenv

        from clinic import db
        from clinic.schedule import normalize_phone

        load_dotenv(override=True)
        args.number = normalize_phone(args.number) or args.number
        # Logged like a real reminder call, but marked in progress so the dialer never picks it up.
        outbound_id = db.add_outbound(args.name, args.number, args.purpose, status="in_progress")
        if outbound_id is None:
            raise SystemExit(f"{args.number} is already in the outbound queue; use a different --number.")
        args.outbound_id = outbound_id

    try:
        asyncio.run(call(args))
    except KeyboardInterrupt:
        print("\nYou hung up.")
    except OSError as e:
        print(f"Could not reach the bot at {args.url}. Is `uv run bot.py` running? ({e})")
    finally:
        if outbound_id is not None:
            db.update_outbound(outbound_id, status="completed", last_result="simulated")


if __name__ == "__main__":
    main()
