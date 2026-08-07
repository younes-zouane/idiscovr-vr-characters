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

import logging
import os
import re
import time

from openai import OpenAI

from . import config

MAX_INPUT_CHARS = 500
MAX_REPLY_SENTENCES = 6

log = logging.getLogger(__name__)  # reuse if guardrails.py already has this from Layer 2; don't redeclare

GUARD_MODEL_NAME = os.environ.get("GUARD_MODEL_NAME", "llama-guard3:1b")
# Default OFF: measured ~2.2s per call for llama-guard3:1b on this hardware
# (5 real single-attempt runs, 2.190-2.207s on the 4 warm ones) — ~7x over
# the 300ms budget this project's guide sets. See KNOWN_ISSUES.md for the
# full write-up. Set GUARD_ENABLED=true to re-enable if faster hardware or
# a smaller/quantized guard model changes this later.
GUARD_ENABLED = os.environ.get("GUARD_ENABLED", "false").lower() == "true"
# GUARD_BUDGET_SECONDS: the design threshold from Guide 4 ("hard budget:
# 300ms, if it can't hit that, drop it"). This is NOT the API call's
# timeout — it's what logged guard_check timings get compared against to
# decide whether GUARD_ENABLED should stay true. Conflating this with the
# actual request timeout previously guaranteed every real call would time
# out before finishing, regardless of the model's true speed.
GUARD_BUDGET_SECONDS = float(os.environ.get("GUARD_BUDGET_SECONDS", "0.3"))

# The actual per-call network timeout — generous, so a real inference call
# has a genuine chance to complete and be measured. This is a safety net
# against a truly hung request, not an enforcement of the budget above.
GUARD_REQUEST_TIMEOUT_SECONDS = float(os.environ.get("GUARD_REQUEST_TIMEOUT_SECONDS", "5.0"))

# max_retries=0: a slow/failed guard call should fail once and fail open
# immediately, not retry 2x with exponential backoff — that turns one
# timeout into a guaranteed multi-second stall regardless of the model's
# real speed, which is exactly what was happening before this was set.
_guard_client = OpenAI(api_key="ollama", base_url=config.OLLAMA_BASE_URL, max_retries=0)

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


def check_output_with_guard_model(user_text: str, generated_sentence: str) -> tuple[bool, str | None]:
    """
    Layer 3: asks a small guard model whether the first generated sentence
    is safe, given the user's message that prompted it. Only meant to be
    called once per reply, on the first sentence — see pipeline.py.

    Fails OPEN (returns allowed=True) on timeout, error, or unparseable
    output — a slow/broken guard check should never add latency to a
    normal reply. If this fires often, GUARD_ENABLED should be set to
    false and the reason documented in KNOWN_ISSUES.md, per the guide's
    own instruction to drop Layer 3 rather than miss the latency budget.
    """
    if not GUARD_ENABLED:
        return True, None

    t0 = time.time()
    try:
        response = _guard_client.chat.completions.create(
            model=GUARD_MODEL_NAME,
            messages=[
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": generated_sentence},
            ],
            timeout=GUARD_REQUEST_TIMEOUT_SECONDS,
        )
        elapsed = time.time() - t0
        over_budget = elapsed > GUARD_BUDGET_SECONDS
        log.info(
            "TIMING guard_check=%.3fs (budget=%.2fs, %s)",
            elapsed,
            GUARD_BUDGET_SECONDS,
            "OVER BUDGET" if over_budget else "within budget",
        )

        verdict = (response.choices[0].message.content or "").strip().lower()
        if verdict.startswith("unsafe"):
            return False, "guard_model_unsafe"
        return True, None

    except Exception as e:
        elapsed = time.time() - t0
        log.warning("Guard model check failed/timed out after %.3fs, failing open: %s", elapsed, e)
        return True, None
