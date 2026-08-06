"""
Layer 2 guardrails: cheap, always-on checks that run before the LLM is
even called, plus lightweight cleanup on what it generates. No model
calls here — this entire module should run in low single-digit
milliseconds, which is what makes a blocked reply faster than a normal
one (Guide 4's design rule).

Layer 1 (per-character refusal lines, in characters.py) and Layer 3
(a small model check on the first sentence) are deliberately separate —
this module only does the free stuff.
"""

import re

MAX_INPUT_CHARS = 500
MAX_REPLY_SENTENCES = 6

# Patterns aimed at getting the model to drop its system prompt or
# character. Deliberately broad/lowercase-substring matching rather than
# exact phrasing — attackers vary wording, but these cores tend to recur.
INJECTION_PATTERNS = [
    "ignore previous",
    "ignore the above",
    "ignore your instructions",
    "disregard your instructions",
    "disregard the above",
    "you are now",
    "pretend you are",
    "pretend to be",
    "act as if",
    "system prompt",
    "new instructions",
    "your real instructions",
    "reveal your prompt",
    "print your prompt",
    "you are no longer",
    "forget you are",
    "forget your character",
    "developer mode",
    "i am a developer",
    "i am the developer",
    "this is a test",
    "for testing purposes",
]

# Deliberately small and blunt. This is NOT meant to catch every adult
# or harmful topic — nuanced cases are what Layer 3 (a real model) is
# for. This list exists to catch the cheap, obvious cases for ~free,
# so the expensive check only has to handle what slips past it.
BLOCKLIST_TERMS = [
    "kill yourself",
    "how to make a bomb",
    "how to make a weapon",
]


def check_input(user_text: str) -> tuple[bool, str | None]:
    """
    Returns (allowed, reason). reason is None when allowed=True, and a
    short machine-readable code when False — never the matched text
    itself, so logs don't become a copy of whatever was blocked.
    """
    if not user_text or not user_text.strip():
        return False, "empty_input"

    if len(user_text) > MAX_INPUT_CHARS:
        return False, "input_too_long"

    lowered = user_text.lower()

    for pattern in INJECTION_PATTERNS:
        if pattern in lowered:
            return False, "injection_pattern"

    for term in BLOCKLIST_TERMS:
        if term in lowered:
            return False, "blocklist_term"

    return True, None


_AI_LEAKAGE_PATTERNS = [
    re.compile(r"\bas an ai\b", re.IGNORECASE),
    re.compile(r"\bas a language model\b", re.IGNORECASE),
    re.compile(r"\bi('m| am) an ai\b", re.IGNORECASE),
    re.compile(r"\bi don't have (personal )?(feelings|opinions|beliefs)\b", re.IGNORECASE),
]


def clean_sentence(sentence: str) -> str:
    """Strips common AI-persona leakage from a single generated sentence.
    Applied per-sentence during streaming, not just at the end, so a
    leak in sentence 2 doesn't ship before sentence 4 gets cleaned."""
    cleaned = sentence
    for pattern in _AI_LEAKAGE_PATTERNS:
        cleaned = pattern.sub("", cleaned).strip()
    return cleaned