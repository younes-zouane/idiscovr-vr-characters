import logging

from openai import OpenAI

from . import config
from .characters import CHARACTERS

log = logging.getLogger(__name__)

client = OpenAI(api_key="ollama", base_url=config.OLLAMA_BASE_URL)

MAX_HISTORY_MESSAGES = 20  # system prompt + last 10 user/assistant exchanges


def init_conversation_histories():
    """Build a fresh, per-session history dict — one system-prompted list per character."""
    return {name: [{"role": "system", "content": data["prompt"]}] for name, data in CHARACTERS.items()}


def stream_character_reply(character_name, message, history):
    """
    Stream a reply from Ollama, yielding text deltas as they arrive.

    History bookkeeping happens exactly once, here, in the `finally` block —
    whether the stream completes normally or dies partway through. This is
    the single source of truth for "does the reply land in history," so
    callers never need to append to history themselves.
    """
    history.append({"role": "user", "content": message})
    full_reply = ""

    try:
        stream = client.chat.completions.create(
            model=config.MODEL,
            max_tokens=300,
            messages=history,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                full_reply += delta
                yield delta
    except Exception as e:
        log.error(f"LLM stream failed for {character_name} mid-reply: {e}", exc_info=True)
        # Deliberately no re-raise — whatever was generated before the
        # failure still gets saved below, so the user doesn't lose partial
        # progress they may have already heard.
    finally:
        if full_reply:
            history.append({"role": "assistant", "content": full_reply})
            if len(history) > MAX_HISTORY_MESSAGES:
                history[:] = [history[0]] + history[-(MAX_HISTORY_MESSAGES - 1) :]


def ask_character(character_name, message, history):
    history.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model=config.MODEL,
        max_tokens=300,
        messages=history,
    )
    reply = response.choices[0].message.content
    history.append({"role": "assistant", "content": reply})

    # keep the system prompt + only the most recent exchanges
    if len(history) > MAX_HISTORY_MESSAGES:
        history[:] = [history[0]] + history[-(MAX_HISTORY_MESSAGES - 1) :]

    return reply


def prewarm_model():
    log.info("Pre-warming LLM model...")
    client.chat.completions.create(
        model=config.MODEL,
        max_tokens=1,
        messages=[{"role": "user", "content": "hi"}],
    )
    log.info("Model warm and ready.")
