# Runs the voice agent: browser demo, phone WebSocket, CRM and dashboard on port 7860.
# Built for Hugging Face Spaces (Docker SDK, free CPU tier), but works on any Docker host.
FROM python:3.12-slim

# onnxruntime (voice activity and turn detection models) needs OpenMP.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.11.19 /uv /usr/local/bin/uv

# Hugging Face Spaces run containers as user 1000.
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never PYTHONUNBUFFERED=1

# Dependencies first, so code changes don't reinstall them.
COPY --chown=user pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project --no-dev
COPY --chown=user . .

EXPOSE 7860
CMD ["uv", "run", "--frozen", "--no-sync", "bot.py", "--host", "0.0.0.0", "--port", "7860"]
