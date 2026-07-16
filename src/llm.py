from openai import OpenAI

from . import config
from .characters import CHARACTERS

client = OpenAI(api_key="ollama", base_url=config.OLLAMA_BASE_URL)

MAX_HISTORY_MESSAGES = 20  # system prompt + last 10 user/assistant exchanges


def init_conversation_histories():
    """Build a fresh, per-session history dict — one system-prompted list per character."""
    return {
        name: [{"role": "system", "content": data["prompt"]}]
        for name, data in CHARACTERS.items()
    }


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
        history[:] = [history[0]] + history[-(MAX_HISTORY_MESSAGES - 1):]

    return reply


def prewarm_model():
    print("Pre-warming LLM model...")
    client.chat.completions.create(
        model=config.MODEL,
        max_tokens=1,
        messages=[{"role": "user", "content": "hi"}],
    )
    print("Model warm and ready.")