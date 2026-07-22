import re

_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")
MIN_SENTENCE_LENGTH = 15


def split_into_sentences(buffer: str):
    """
    Extract complete sentences from the front of an accumulating text buffer.

    A sentence boundary is punctuation (. ! ?) followed by whitespace. Short
    fragments under MIN_SENTENCE_LENGTH chars (e.g. "Dr.", "1.") are folded
    into the next fragment instead of being treated as their own sentence —
    this is what stops "Dr. Smith" or "1. First," from being split early.

    Returns (sentences, remainder) — sentences ready to speak, and whatever
    text is left over (either incomplete, or a short fragment waiting to be
    merged with what arrives next).
    """
    if not buffer:
        return [], buffer

    parts = _SENTENCE_END_RE.split(buffer)
    if len(parts) < 2:
        return [], buffer  # no sentence boundary yet

    remainder = parts[-1]
    candidates = parts[:-1]

    sentences = []
    pending = ""
    for part in candidates:
        pending = f"{pending} {part}".strip() if pending else part
        if len(pending) >= MIN_SENTENCE_LENGTH:
            sentences.append(pending)
            pending = ""

    if pending:
        remainder = f"{pending} {remainder}".strip()

    return sentences, remainder
