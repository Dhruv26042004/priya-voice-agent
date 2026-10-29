"""Last line of defence before text is spoken.

Occasionally an LLM writes a tool call or tool result as plain text instead of calling the
tool, e.g. `{"name": "Farah", ...}{"ok": true, ...}`, or leaks its tool-calling syntax, e.g.
`to=functions.book_appointment` (gpt-oss's internal format). Spoken aloud, that's nonsense to
a caller. This filter drops anything inside {curly braces} and those markers before text
reaches TTS. Text arrives in sentence-sized chunks, so a blob can span chunks; the brace depth
is carried across them.
"""

import re

from pipecat.utils.text.base_text_filter import BaseTextFilter

# gpt-oss tool-call routing ("to=functions.x") and special tokens ("<|call|>").
MARKERS = re.compile(r"\s*(?:to=functions\.[\w.]+|<\|[^|>]*\|>)")
# A sentence that names a tool is the model thinking aloud ("We need to call end_call."):
# real speech never contains snake_case identifiers.
TOOL_TALK = re.compile(r"[^.!?]*\b[a-z]+_[a-z_]+\b[^.!?]*[.!?]?")
# Asides and stage directions, "(Please share the number.)" or "[awaiting]": on a phone
# call everything is spoken, so anything bracketed is never meant for the caller.
ASIDES = re.compile(r"\s*(?:\([^()]*\)|\[[^\[\]]*\])")


class DropJsonFilter(BaseTextFilter):
    def __init__(self):
        super().__init__()
        self._depth = 0

    async def update_settings(self, settings):
        pass

    async def filter(self, text: str) -> str:
        kept = []
        for ch in text:
            if ch == "{":
                self._depth += 1
            elif ch == "}" and self._depth:
                self._depth -= 1
            elif not self._depth:
                kept.append(ch)
        return ASIDES.sub("", TOOL_TALK.sub("", MARKERS.sub("", "".join(kept))))

    async def handle_interruption(self):
        self._depth = 0

    async def reset_interruption(self):
        self._depth = 0
