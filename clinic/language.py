"""Telling Hinglish from English in what the caller said.

The LLM is told to answer in the caller's language, but it drifts back to English after a
few turns. So each turn we check the caller's last message here and, if it has Hindi, add a
one-line reminder to that request (see prompts.language_hint).
"""

import re

# Common Hindi words in Roman script, to tell Hinglish from English.
HINDI_WORDS = {
    "haan", "nahi", "nahin", "ji", "hai", "hain", "kal", "parson", "aaj", "mera", "meri", "mujhe", "aap", "aapka",
    "kya", "karna", "karwana", "chahiye", "theek", "thik", "baje", "subah", "shaam", "dopahar", "bilkul", "kab",
    "kitne", "kaunsa", "naam", "bol", "raha", "rahi", "hoon", "acha", "achha", "dhanyavaad", "shukriya", "ke", "ko",
}


def speaks_hindi(text: str) -> bool:
    if any("ऀ" <= ch <= "ॿ" for ch in text):  # Devanagari, as speech-to-text writes Hindi
        return True
    return len(set(re.findall(r"[a-z]+", text.lower())) & HINDI_WORDS) >= 2


def message_text(message) -> str:
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, list):
        content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def last_caller_text(messages) -> str:
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "user":
            return message_text(m)
    return ""
