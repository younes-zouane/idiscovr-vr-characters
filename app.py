import os
import base64
import requests
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

load_dotenv()

client = OpenAI(
    api_key=os.environ["DEEPINFRA_API_KEY"],
    base_url="https://api.deepinfra.com/v1/openai",
)

MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Each character has ONE fixed voice name. Never changes. No "instruction" field.
CHARACTERS = {
    "Genie": {
        "prompt": """You are the Genie of the lamp, straight out of the 
        One Thousand and One Nights. Loud, theatrical, a show-off. You 
        grant "wishes" by answering questions with flair and humor. 
        Keep replies short (2-4 sentences). Never break character.""",
        "voice": "Ryan",
    },
    "Aladdin": {
        "prompt": """You are Aladdin, a quick, cheeky, street-smart young 
        man. Friendly and a bit of a charmer. Keep replies short 
        (2-4 sentences). Never break character.""",
        "voice": "Dylan",
    },
    "The Princess": {
        "prompt": """You are a sharp, independent princess who knows a 
        great deal and refuses to be talked down to. Witty and 
        confident. Keep replies short (2-4 sentences). Never break character.""",
        "voice": "Serena",
    },
    "Iago": {
        "prompt": """You are Iago, a sarcastic parrot who complains about 
        everything. Comic relief, dry wit, never impressed. Keep replies 
        short (1-3 sentences). Never break character.""",
        "voice": "Uncle_Fu",
    },
    "The Sorcerer": {
        "prompt": """You are a smooth, slightly menacing sorcerer who 
        answers in riddles. Mysterious and calculating. Keep replies 
        short (2-4 sentences). Never break character.""",
        "voice": "Eric",
    },
    "The Cave of Wonders": {
        "prompt": """You are the Cave of Wonders, an ancient, booming, 
        magical voice — not a person, but the voice of the cave itself. 
        You speak in dramatic warnings and riddles about who is worthy 
        to enter. Example tone: "WHO DISTURBS MY SLUMBER?" Keep replies 
        short (1-3 sentences), deep and theatrical. Never break character.""",
        "voice": "Eric",
    },
}

conversation_histories = {
    name: [{"role": "system", "content": data["prompt"]}]
    for name, data in CHARACTERS.items()
}

def transcribe(filepath):
    with open(filepath, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="openai/whisper-large-v3",
            file=audio_file
        )
    return transcript.text

def ask_character(character_name, message):
    history = conversation_histories[character_name]
    history.append({"role": "user", "content": message})
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=300,
        messages=history
    )
    reply = response.choices[0].message.content
    history.append({"role": "assistant", "content": reply})
    return reply

from pydub import AudioSegment
from pydub.effects import normalize

def add_cave_echo(filename):
    sound = AudioSegment.from_wav(filename)

    # Create a few delayed, quieter copies layered on top = echo effect
    echo1 = sound - 8   # 8 dB quieter
    echo2 = sound - 14
    delay1 = AudioSegment.silent(duration=180) + echo1
    delay2 = AudioSegment.silent(duration=350) + echo2

    # Overlay original + echoes
    combined = sound.overlay(delay1).overlay(delay2)
    combined = normalize(combined)

    # Slightly lower pitch for a deeper, more menacing tone
    octaves = -0.15
    new_sample_rate = int(combined.frame_rate * (2 ** octaves))
    deeper = combined._spawn(combined.raw_data, overrides={"frame_rate": new_sample_rate})
    deeper = deeper.set_frame_rate(sound.frame_rate)

    deeper.export(filename, format="wav")

def speak(text, character_name, filename="reply.wav"):
    voice = CHARACTERS[character_name]["voice"]
    response = requests.post(
        "https://api.deepinfra.com/v1/inference/Qwen/Qwen3-TTS",
        headers={
            "Authorization": f"Bearer {os.environ['DEEPINFRA_API_KEY']}",
            "Content-Type": "application/json"
        },
        json={"input": text, "voice": voice}
    )
    result = response.json()
    audio_b64 = result["audio"].split(",", 1)[1]
    audio_bytes = base64.b64decode(audio_b64)
    with open(filename, "wb") as f:
        f.write(audio_bytes)

    if character_name == "The Cave of Wonders":
        add_cave_echo(filename)

    return filename

def chat_with_character(character_name, mic_audio):
    if mic_audio is None:
        return "No audio recorded.", None

    user_text = transcribe(mic_audio)
    reply = ask_character(character_name, user_text)
    audio_path = speak(reply, character_name)
    return f"You said: {user_text}\n\n{character_name}: {reply}", audio_path

with gr.Blocks(title="Talking AI Characters - VR Demo") as demo:
    gr.Markdown("# 🏺 Talking AI Characters")
    gr.Markdown("Pick a character, record your message, and hear them reply.")

    character_dropdown = gr.Dropdown(
        choices=list(CHARACTERS.keys()),
        value="Genie",
        label="Choose your character"
    )
    mic_input = gr.Audio(sources=["microphone"], type="filepath", label="Speak here")
    talk_button = gr.Button("Talk")
    text_output = gr.Textbox(label="Conversation")
    audio_output = gr.Audio(label="Character's reply", autoplay=True)

    talk_button.click(
        fn=chat_with_character,
        inputs=[character_dropdown, mic_input],
        outputs=[text_output, audio_output]
    )

if __name__ == "__main__":
    demo.launch()