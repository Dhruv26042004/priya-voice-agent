"""Deploy to a Hugging Face Space in one command (Docker Spaces need a PRO account).

    uv run --with huggingface_hub deploy_hf.py
    uv run --with huggingface_hub deploy_hf.py --space my-clinic-bot

Needs HF_TOKEN in .env: a Hugging Face access token with write permission
(huggingface.co -> Settings -> Access Tokens -> New token -> Write).

It creates the Space if needed, copies your API keys from .env into the Space's secrets
(they never go into the code), sets an ADMIN_PASSWORD for the CRM and dashboard, uploads
every file tracked by git (so .env, data and recordings are never uploaded), and waits
until the Space is running. Run it again after changes to redeploy.
"""

import argparse
import secrets
import subprocess
import time
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import HfApi

# Copied from .env into the Space's secrets, when set.
SECRET_KEYS = [
    "DEEPGRAM_API_KEY", "GROQ_API_KEY", "CEREBRAS_API_KEY", "CARTESIA_API_KEY", "CARTESIA_VOICE_ID",
    "LLM_PROVIDER", "STT_LANGUAGE", "TTS_LANGUAGE", "PIPECAT_ICE_SERVERS",
]
FINAL_STAGES = {"RUNNING", "BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--space", default="priya-voice-agent", help="Space name (default: priya-voice-agent)")
    args = parser.parse_args()

    env = {k: (v or "").strip() for k, v in dotenv_values(".env").items()}
    if not env.get("HF_TOKEN"):
        raise SystemExit("Add HF_TOKEN=<your write token> to .env first (see the top of this file).")
    api = HfApi(token=env["HF_TOKEN"])
    user = api.whoami()["name"]
    repo_id = f"{user}/{args.space}"

    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    print(f"Space: https://huggingface.co/spaces/{repo_id}")

    for key in SECRET_KEYS:
        if env.get(key):
            api.add_space_secret(repo_id, key, env[key])
    admin_password = env.get("ADMIN_PASSWORD")
    if not admin_password:
        # Generated once and saved to .env, so you can always look it up.
        admin_password = secrets.token_urlsafe(12)
        with Path(".env").open("a", encoding="utf-8") as f:
            f.write(f"\nADMIN_PASSWORD={admin_password}\n")
    api.add_space_secret(repo_id, "ADMIN_PASSWORD", admin_password)
    print("Secrets set: " + ", ".join(k for k in SECRET_KEYS if env.get(k)) + ", ADMIN_PASSWORD")

    files = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout.split()
    api.upload_folder(folder_path=".", repo_id=repo_id, repo_type="space", allow_patterns=files,
                      commit_message="Deploy from deploy_hf.py")
    print(f"Uploaded {len(files)} files. Building (first build takes a few minutes)...")

    stage, started = None, time.monotonic()
    while time.monotonic() - started < 25 * 60:
        runtime = api.get_space_runtime(repo_id)
        if runtime.stage != stage:
            stage = runtime.stage
            print(f"  {time.monotonic() - started:5.0f}s  {stage}")
        if stage in FINAL_STAGES:
            break
        time.sleep(10)

    host = f"{user}-{args.space}".lower().replace("_", "-").replace(".", "-")
    if stage == "RUNNING":
        print(f"\nLive at https://{host}.hf.space")
        print(f"Dashboard: https://{host}.hf.space/dashboard  (any username, password in .env as ADMIN_PASSWORD)")
        print(f"Call it:   uv run phone_sim.py --url wss://{host}.hf.space/ws")
    else:
        print(f"\nSpace ended in {stage}. See the logs at https://huggingface.co/spaces/{repo_id}?logs=container")


if __name__ == "__main__":
    main()
