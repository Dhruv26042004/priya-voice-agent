---
title: Priya Voice Receptionist
emoji: 🦷
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Smile Care — AI Voice Receptionist

A voice agent for a dental clinic that talks naturally in English, Hindi and Hinglish.

**Stack:** Pipecat · Deepgram Nova-3 (speech-to-text) · gpt-oss-120b on Groq with Cerebras failover ·
Cartesia Sonic (text-to-speech) · Twilio-compatible phone streaming · SQLite · FastAPI

**Status:** Phase 6 — 31 scripted test calls and an admin dashboard (latency, task completion, handoffs, cost).

## What Priya can do

- Check free slots (Mon–Sat, 10 AM–7 PM, 30-minute appointments, IST)
- Book, reschedule and cancel appointments, after reading the details back and getting a "yes"
- Mirror every booking into **Google Calendar** (optional)
- Send a **WhatsApp confirmation** through the Twilio sandbox (optional)
- Save details of callers who aren't ready to book (leads)
- Hand off to staff for emergencies or complaints by creating an urgent callback task

On the phone:
- Answers inbound calls on a Twilio number and knows the caller's number
- Calls patients from a CSV list (e.g. checkup reminders), with retries for no-answer/busy
- Never calls numbers on the do-not-call list, and only within calling hours (10 AM–7 PM IST)
- Every call opens with a fixed recording notice; callers can opt out, and recordings
  are saved to `data/recordings/` only if they don't
- Transfers to a staff phone during clinic hours (if `STAFF_PHONE_NUMBER` is set), and hangs up politely

After every call:
- Full transcript, plus a log of what the tools actually did (booked, cancelled, handed off…)
- An LLM extracts the caller's name, intent, outcome, sentiment, a short summary and a follow-up for staff.
  Booking status and key outcomes come from the tool log, not the LLM's reading of the transcript.
- Results appear on the **CRM page** at http://localhost:7860/crm (with transcript and recording playback),
  in `data/calls.csv`, and optionally in a Google Sheet

Safety checks in code, not just in the prompt:
- No booking without `caller_confirmed`, a valid slot and a free calendar
- A phone number is rejected unless the caller actually said it, so the model can't invent one
- The bot only mentions WhatsApp when a message was really sent

## How it works

```
Mic → Deepgram (speech-to-text) → Groq LLM → Cartesia (text-to-speech) → Speaker
```

Built with [Pipecat](https://github.com/pipecat-ai/pipecat). Every stage streams, so the
bot starts speaking before the full reply is written.

- **Turn detection:** Silero VAD + Pipecat's smart-turn model, so the bot waits while
  you pause mid-sentence.
- **Barge-in:** start talking while the bot is speaking and it stops.
- **Latency logging:** each reply's time is split into stages (endpointing, STT, LLM, TTS),
  printed in the terminal and saved to `logs/latency.csv`.

## Run it

1. Install [uv](https://docs.astral.sh/uv/).
2. Get free API keys from Deepgram, Groq and Cartesia, and pick a Cartesia voice ID.
3. Copy `.env.example` to `.env` and fill in the keys.
4. Start the bot:
   ```
   uv run bot.py
   ```
5. Open http://localhost:7860, click **Connect**, allow the microphone, and talk.

Use headphones so the bot doesn't hear itself.

## Deploy for free (Hugging Face Spaces)

A Docker Space on the free CPU tier (2 vCPU, 16 GB RAM) runs everything: the voice agent, the phone
WebSocket, the CRM and the dashboard, at a public `https://<you>-<space>.hf.space` address.

**One command:** create a free Hugging Face account, make an access token with **Write** permission
(Settings → Access Tokens), add `HF_TOKEN=<token>` to `.env`, then run:
```
uv run --with huggingface_hub deploy_hf.py
```
It creates the Space, copies your keys from `.env` into its secrets, sets an `ADMIN_PASSWORD`
(saved to `.env`), uploads the git-tracked files and waits until it's live.

**Or by hand:**

1. Create a free account at [huggingface.co](https://huggingface.co) and a **New Space**: SDK **Docker**,
   template **Blank**, hardware **CPU basic (free)**.
2. In the Space's **Settings → Variables and secrets**, add as **secrets**: `DEEPGRAM_API_KEY`,
   `GROQ_API_KEY`, `CEREBRAS_API_KEY`, `CARTESIA_API_KEY`, `CARTESIA_VOICE_ID`, and `ADMIN_PASSWORD`
   (protects the CRM and dashboard; log in with any username and this password).
3. Push this repository to the Space (create a write token under Settings → Access Tokens):
   ```
   git remote add space https://huggingface.co/spaces/<you>/<space>
   git push space main
   ```
   The Space builds the `Dockerfile` and starts in a few minutes.
4. Talk to Priya over the phone path from your laptop:
   ```
   uv run phone_sim.py --url wss://<you>-<space>.hf.space/ws
   ```
   The dashboard is at `https://<you>-<space>.hf.space/dashboard`.

Free-tier notes: the Space sleeps after 48 hours without visitors and wakes on the next visit; its
disk is temporary, so calls and recordings reset when it restarts (fine for a demo, use a database
for real use). The browser demo page needs a TURN server to carry audio from a hosted Space; the
phone path above works without one. To enable it, get free TURN credentials (e.g. a free
[Metered](https://www.metered.ca/tools/openrelay/) account) and add them as a secret:
`PIPECAT_ICE_SERVERS=[{"urls":"turn:<host>:443?transport=tcp","username":"<user>","credential":"<pass>"}]`.
The phone WebSocket is open to anyone who knows the address, and each call uses your API quota,
so share the address carefully.

To run the same container anywhere else: `docker build -t priya . && docker run -p 7860:7860 --env-file .env priya`.

## Optional: Google Calendar

1. In [Google Cloud Console](https://console.cloud.google.com), create a project and enable the **Google Calendar API**.
2. Go to **IAM & Admin → Service Accounts**, create one, open it, then **Keys → Add key → JSON**.
   Save the file in this folder as `google-service-account.json`.
3. In [Google Calendar](https://calendar.google.com), open your calendar's **Settings and sharing**, then under
   **Share with specific people** add the service account's email with **Make changes to events**.
4. Copy the **Calendar ID** from the same page (for your main calendar, it's your Gmail address).
5. Add both settings to `.env`:
   ```
   GOOGLE_SERVICE_ACCOUNT_FILE=google-service-account.json
   GOOGLE_CALENDAR_ID=you@gmail.com
   ```

## Phone-call simulator (no Twilio needed)

Talk to Priya through the bot's real phone path, using your mic and speakers. Audio is
converted to 8 kHz phone quality and sent using Twilio's own protocol, so caller ID,
the recording notice, recording, tools and hang-up all behave as on a real call.

```
uv run bot.py                                 # terminal 1
uv run phone_sim.py                           # terminal 2: you call the clinic
uv run phone_sim.py --outbound --name Dhruv   # or: the clinic calls you with a checkup reminder
```

Options: `--number +9198XXXXXXXX` sets your caller ID, `--half-duplex` mutes your mic while
Priya talks (if you're not using headphones). Press Ctrl+C to hang up.

## Phone calls (Twilio trial)

1. Sign up at [twilio.com/try-twilio](https://www.twilio.com/try-twilio) and verify your own mobile number.
   Trial accounts can only call numbers you've verified, and play a short trial message first.
2. Get a trial phone number (Console → Phone Numbers → Buy a number; it's paid from trial credit).
3. Allow calls to India: Console → Voice → Settings → **Geo permissions** → tick India.
4. Add `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN` and `TWILIO_PHONE_NUMBER` to `.env`.
5. Start the bot, then give it a public address with a free Cloudflare tunnel:
   ```
   uv run bot.py
   cloudflared tunnel --url http://localhost:7860
   ```
   Copy the `https://….trycloudflare.com` address it prints (it changes each time).

**Inbound:** on your Twilio number's page, set **A call comes in** → Webhook →
`https://<tunnel-host>/twilio/voice` (HTTP POST), then call the number.

**Outbound:** copy `call_list.example.csv` to `data/call_list.csv`, put your verified number in it, and run:
```
uv run outbound.py data/call_list.csv --host <tunnel-host>
```
Add `--ignore-hours` to test outside 10 AM–7 PM. Add a number to the do-not-call list with
`uv run outbound.py --add-dnc 98XXXXXXXX`.

**For a real Indian clinic:** use an Indian provider such as Exotel or Plivo (Pipecat supports both)
with a KYC-verified number. Service calls to existing patients should come from TRAI's 160-series
numbers and promotional calls from the 140 series; scrub lists against the national DND registry
and keep calls within 9 AM–9 PM. Check current TRAI rules before going live.

## Call results: CRM page, CSV and Google Sheet

Open http://localhost:7860/crm while the bot is running. It lists every call with its outcome and
summary; click one for the transcript, the actions taken and the recording. It holds patient data, so it
only opens on this computer, even when a tunnel is running.

To also send each call to a Google Sheet: create a sheet, share it with your service account's email
(see Google Calendar above) as **Editor**, and add its ID (the long part of the URL between `/d/` and `/edit`):
```
GOOGLE_SERVICE_ACCOUNT_FILE=google-service-account.json
GOOGLE_SHEET_ID=1AbC...xyz
```

If an analysis fails (for example, both free LLM tiers are busy), the call is marked failed and you can rerun it:
```
uv run postcall.py          # pending or failed calls
uv run postcall.py --all    # every call
```

## Evaluation and dashboard

`evals/scenarios.yaml` holds 31 scripted test calls: bookings in English, Hinglish and Hindi, wrong dates
(Sunday, 31 February, yesterday, 9 PM), corrections at the read-back, a bad phone number, a taken slot,
reschedule and cancel, pricing and services questions, emergencies, medical-advice requests, a rude caller,
a complaint, prompt injection, recording opt-out, outbound reminders, do-not-call, background noise and
interruptions. An LLM plays each caller; each call is graded by **hard checks** (what really ended up in the
database) and an **LLM judge** (rules like "never named a medicine").

```
uv run python -m evals.run                     # text mode: 29 scenarios, no audio
uv run python -m evals.run --mode voice        # voice mode: 9 scenarios as real phone audio (bot must be running)
uv run python -m evals.run --only pricing,rude_caller
```

Text mode drives the real prompt, tools and safety checks with text, on a throwaway database per scenario.
Voice mode calls the running bot over its phone line with synthesised caller speech, background noise and
an interruption, and measures true end-of-speech to first-audio latency. Anything an eval books is cancelled.

**Latest results** (free tiers):
- Text mode: **28/29 scenarios passed, 18/18 tasks completed** (the one miss: no apology on a
  "bad time" reminder call).
- Voice mode over real phone audio: 3/9 passed, 3/5 tasks completed; interruptions stop Priya in
  **0.5 s**; replies ~1–2 s when the LLM free tiers have quota. Most voice failures were turns where
  both free LLM tiers were rate limited at once (the evaluation's own caller and judge use the same
  quota), so a paid LLM tier is the first upgrade for real deployments.

Bugs the evaluations found and fixed: wrong availability from a truncated slot list; dropped digits
in phone read-backs (read-backs are now written by code); tool JSON and model markup that would have
been spoken (filtered); English replies to Hindi callers (per-turn language hint and Hinglish
read-backs); a second booking instead of a reschedule; Hindi-spoken phone numbers not recognised;
numbers mis-assembled from pauses; post-call labels saying "booked" when nothing was booked.

The **dashboard** at http://localhost:7860/dashboard shows live-call task completion, handoff rate, reply
latency against the 1 s target and where that time goes (per stage), call outcomes, estimated cost per
minute (list prices in `pricing.py`), and every evaluation run with each scenario's checks and transcript.
Like the CRM, it only opens on this computer.

## Optional: WhatsApp confirmations

1. Create a free [Twilio](https://www.twilio.com/try-twilio) account.
2. Open **Messaging → Try it out → Send a WhatsApp message**, and from your phone send the join code to the sandbox number.
3. Add `TWILIO_ACCOUNT_SID` and `TWILIO_AUTH_TOKEN` to `.env`.

## Files

- `bot.py` — the voice pipeline
- `prompts.py` — the receptionist's personality, clinic info and booking rules
- `llm.py` — Groq LLM that fails over to Cerebras on free-tier rate limits
- `telephony.py` — Twilio webhook, outbound dialling and call transfer
- `outbound.py` — outbound dialer with retries, calling hours and do-not-call list
- `phone_sim.py` — phone-call simulator using your mic and speakers
- `postcall.py` — post-call analysis: summary, extracted fields, CSV and Google Sheet export
- `crm.py` — the CRM pages at `/crm`
- `clinic/sheets.py` — Google Sheets uploader
- `clinic/speech_filter.py` — stops tool JSON or model markup from ever being spoken
- `dashboard.py` — the admin dashboard at `/dashboard`
- `pricing.py` — usage tracking and cost-per-minute estimates
- `evals/` — scenarios, LLM caller and judge, text and voice runners
- `clinic/tools.py` — the tools Priya can call
- `clinic/schedule.py` — clinic hours, slots, dates, phone numbers
- `clinic/db.py` — SQLite storage (`data/clinic.db`): appointments, leads, callbacks, messages
- `clinic/calendars.py` — local or Google Calendar backend
- `clinic/whatsapp.py` — Twilio WhatsApp sender
- `logs/latency.csv` — latency per reply (created on first run)
