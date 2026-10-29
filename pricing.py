"""Cost per call at pay-as-you-go list prices, from measured usage.

These are ESTIMATES to show a client what the agent would cost at scale; on the free
tiers this project uses, the real cost is zero. Prices change, so check each provider's
pricing page and edit the numbers below before quoting them.
"""

from pipecat.metrics.metrics import LLMUsageMetricsData, STTUsageMetricsData, TTSUsageMetricsData
from pipecat.frames.frames import MetricsFrame
from pipecat.observers.base_observer import BaseObserver, FramePushed

# USD. Edit to match current pricing.
PRICES = {
    "stt_per_min": 0.0092,         # Deepgram Nova-3 multilingual, streaming
    "llm_input_per_m": 0.15,       # gpt-oss-120b input tokens (per million)
    "llm_output_per_m": 0.60,      # gpt-oss-120b output tokens (per million)
    "tts_per_1k_chars": 0.04,      # Cartesia Sonic, pay-as-you-go
    "phone_per_min": 0.01,         # telephony; Indian providers bill per minute in INR
}
USD_TO_INR = 84.0


def call_cost(stt_secs: float, prompt_tokens: int, completion_tokens: int, tts_chars: int, phone_secs: float) -> float:
    return (
        stt_secs / 60 * PRICES["stt_per_min"]
        + prompt_tokens / 1e6 * PRICES["llm_input_per_m"]
        + completion_tokens / 1e6 * PRICES["llm_output_per_m"]
        + tts_chars / 1000 * PRICES["tts_per_1k_chars"]
        + phone_secs / 60 * PRICES["phone_per_min"]
    )


class UsageObserver(BaseObserver):
    """Adds up the usage metrics Pipecat's services report during a call."""

    def __init__(self):
        super().__init__()
        self.prompt_tokens = self.completion_tokens = self.tts_chars = 0
        self.stt_secs = 0.0

    async def on_push_frame(self, data: FramePushed):
        # Each frame passes many processors; count it only the first time it's pushed.
        if not isinstance(data.frame, MetricsFrame) or not data.first_push:
            return
        for metric in data.frame.data:
            if isinstance(metric, LLMUsageMetricsData):
                self.prompt_tokens += metric.value.prompt_tokens
                self.completion_tokens += metric.value.completion_tokens
            elif isinstance(metric, TTSUsageMetricsData):
                self.tts_chars += metric.value
            elif isinstance(metric, STTUsageMetricsData):
                self.stt_secs += metric.value.audio_seconds
