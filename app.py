import os
import time
import base64
import requests
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from faster_whisper import WhisperModel
from pydub import AudioSegment
from pydub.effects import normalize
import gradio as gr

load_dotenv()

# ── NVIDIA DLL registration (needed for faster-whisper on Windows) ──
root = Path(__file__).resolve().parent / "venv" / "Lib" / "site-packages" / "nvidia"
for sub in list(root.glob("*/bin")) + list(root.glob("*/lib")):
    os.add_dll_directory(str(sub.resolve()))
    os.environ["PATH"] = str(sub.resolve()) + os.pathsep + os.environ["PATH"]

# ── LLM: local Ollama ──
client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1",
)
MODEL = "llama3.1:8b"

# ── STT: local faster-whisper ──
stt_model = WhisperModel(
    "medium",
    device="cuda",
    compute_type="float16"
)

# ── CHARACTERS: prompt + fixed Qwen3-TTS voice per character ──
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
        "voice": "Aiden",
    },
}

# ── Conversation memory: one separate history list per character ──
conversation_histories = {}
for name, data in CHARACTERS.items():
    conversation_histories[name] = [
        {"role": "system", "content": data["prompt"]}
    ]

# ── STT ──
def transcribe(filepath):
    segments, info = stt_model.transcribe(filepath, beam_size=5)
    return " ".join(segment.text for segment in segments).strip()

# ── LLM ──
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

# ── Cave of Wonders audio post-processing ──
def add_cave_echo(filename):
    sound = AudioSegment.from_wav(filename)
    echo1 = sound - 8
    echo2 = sound - 14
    delay1 = AudioSegment.silent(duration=180) + echo1
    delay2 = AudioSegment.silent(duration=350) + echo2
    combined = sound.overlay(delay1).overlay(delay2)
    combined = normalize(combined)
    octaves = -0.15
    new_sample_rate = int(combined.frame_rate * (2 ** octaves))
    deeper = combined._spawn(combined.raw_data, overrides={"frame_rate": new_sample_rate})
    deeper = deeper.set_frame_rate(sound.frame_rate)
    deeper.export(filename, format="wav")

# ── TTS: Qwen3-TTS via DeepInfra ──
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

# ── Single character chat ──
def chat_with_character(character_name, mic_audio):
    if mic_audio is None:
        return "No audio recorded.", None

    t0 = time.time()
    user_text = transcribe(mic_audio)
    t1 = time.time()

    reply = ask_character(character_name, user_text)
    t2 = time.time()

    audio_path = speak(reply, character_name)
    t3 = time.time()

    print(f"STT (Whisper):  {t1-t0:.2f}s")
    print(f"LLM (Ollama):   {t2-t1:.2f}s")
    print(f"TTS (Qwen3):    {t3-t2:.2f}s")
    print(f"Total latency:  {t3-t0:.2f}s")

    return f"You said: {user_text}\n\n{character_name}: {reply}", audio_path

# ── Two characters talking to each other ──
def characters_talk(char_a, char_b, opening_line, num_turns=6):
    transcript_lines = []
    audio_segments = []

    current_speaker = char_a
    other_speaker = char_b
    last_line = opening_line

    for i in range(num_turns):
        reply = ask_character(current_speaker, last_line)
        transcript_lines.append(f"{current_speaker}: {reply}")

        turn_audio_path = f"turn_{i}.wav"
        speak(reply, current_speaker, filename=turn_audio_path)
        audio_segments.append(AudioSegment.from_wav(turn_audio_path))

        last_line = reply
        current_speaker, other_speaker = other_speaker, current_speaker

    combined = AudioSegment.silent(duration=300)
    for seg in audio_segments:
        combined += seg + AudioSegment.silent(duration=400)
    combined.export("conversation.wav", format="wav")

    for i in range(num_turns):
        os.remove(f"turn_{i}.wav")

    return "\n\n".join(transcript_lines), "conversation.wav"

# ── Pre-warm LLM so first real user gets fast response ──
def prewarm_model():
    print("Pre-warming LLM model...")
    client.chat.completions.create(
        model=MODEL,
        max_tokens=1,
        messages=[{"role": "user", "content": "hi"}]
    )
    print("Model warm and ready.")

# ── Gradio UI ──
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

    gr.Markdown("## 🎭 Let two characters talk to each other")
    with gr.Row():
        char_a_dropdown = gr.Dropdown(
            choices=list(CHARACTERS.keys()),
            value="Genie",
            label="Character A"
        )
        char_b_dropdown = gr.Dropdown(
            choices=list(CHARACTERS.keys()),
            value="Iago",
            label="Character B"
        )
    opening_line_input = gr.Textbox(
        label="Opening line / topic",
        value="What do you think of this whole 'wish granting' business?"
    )
    turns_slider = gr.Slider(minimum=2, maximum=10, step=2, value=6, label="Number of turns")
    debate_button = gr.Button("Let them talk")
    debate_transcript = gr.Textbox(label="Conversation transcript", lines=10)
    debate_audio = gr.Audio(label="Full conversation", autoplay=True)

    debate_button.click(
        fn=characters_talk,
        inputs=[char_a_dropdown, char_b_dropdown, opening_line_input, turns_slider],
        outputs=[debate_transcript, debate_audio]
    )

if __name__ == "__main__":
    prewarm_model()
    demo.launch()