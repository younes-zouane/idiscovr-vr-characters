from openai import OpenAI

from . import config
from .characters import CHARACTERS

client = OpenAI(api_key="ollama", base_url=config.OLLAMA_BASE_URL)

conversation_histories = {}
for name, data in CHARACTERS.items():
    conversation_histories[name] = [{"role": "system", "content": data["prompt"]}]

def ask_character(character_name, message, history=None):
    if history is None:
        history = conversation_histories[character_name]
    history.append({"role": "user", "content": message})
    response = client.chat.completions.create(
        model=config.MODEL,
        max_tokens=300,
        messages=history,
    )
    reply = response.choices[0].message.content
    history.append({"role": "assistant", "content": reply})
    return reply

def prewarm_model():
    print("Pre-warming LLM model...")
    client.chat.completions.create(
        model=config.MODEL,
        max_tokens=1,
        messages=[{"role": "user", "content": "hi"}],
    )
    print("Model warm and ready.")