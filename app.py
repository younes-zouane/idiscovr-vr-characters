import os
import time
import sysconfig
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from faster_whisper import WhisperModel
from pydub import AudioSegment
from pydub.effects import normalize
import gradio as gr
from kokoro import KPipeline
import soundfile as sf
import numpy as np
import torch

load_dotenv()

# ── NVIDIA DLL registration (needed for faster-whisper on Windows) ──
root = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
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

# ── TTS: local Kokoro ──
kokoro_pipeline = KPipeline(lang_code='a', device="cuda" if torch.cuda.is_available() else "cpu")

CHARACTERS = {
    "Genie": {
        "prompt": """You are the Genie of the lamp, straight out of the
        One Thousand and One Nights. Loud, theatrical, a show-off. You
        grant "wishes" by answering questions with flair and humor.
        Keep replies short (2-4 sentences). Never break character.""",
    },
    "Aladdin": {
        "prompt": """You are Aladdin, a quick, cheeky, street-smart young
        man. Friendly and a bit of a charmer. Keep replies short
        (2-4 sentences). Never break character.""",
    },
    "The Princess": {
        "prompt": """You are a sharp, independent princess who knows a
        great deal and refuses to be talked down to. Witty and
        confident. Keep replies short (2-4 sentences). Never break character.""",
    },
    "Iago": {
        "prompt": """You are Iago, a sarcastic parrot who complains about
        everything. Comic relief, dry wit, never impressed. Keep replies
        short (1-3 sentences). Never break character.""",
    },
    "The Sorcerer": {
        "prompt": """You are a smooth, slightly menacing sorcerer who
        answers in riddles. Mysterious and calculating. Keep replies
        short (2-4 sentences). Never break character.""",
    },
    "The Cave of Wonders": {
        "prompt": """You are the Cave of Wonders, an ancient, booming,
        magical voice — not a person, but the voice of the cave itself.
        You speak in dramatic warnings and riddles about who is worthy
        to enter. Example tone: "WHO DISTURBS MY SLUMBER?" Keep replies
        short (1-3 sentences), deep and theatrical. Never break character.""",
    },
}

AUDIO_ONLY_CHARACTERS = {"Iago", "The Cave of Wonders"}

CHARACTER_IMAGES = {
    "Genie":               "character_images/genie.jpg",
    "Aladdin":             "character_images/aladdin.jpg",
    "The Princess":        "character_images/princess.jpg",
    "Iago":                "character_images/iago.jpg",
    "The Sorcerer":        "character_images/sorcerer.jpg",
    "The Cave of Wonders": "character_images/cave.jpg",
}

IDLE_LOOPS = {
    "Genie": "idle_loops/genie_idle_loop.mp4",
    "Aladdin": "idle_loops/aladdin_idle_loop.mp4",
    "The Princess": "idle_loops/princess_idle_loop.mp4",
    "The Sorcerer": "idle_loops/sorcerer_idle_loop.mp4",
    # Iago and The Cave of Wonders don't have idle loops yet —
    # they'll just show no video / stay on whatever character_video already shows
}

# ── Conversation memory: one separate history list per character ──
conversation_histories = {}
for name, data in CHARACTERS.items():
    conversation_histories[name] = [
        {"role": "system", "content": data["prompt"]}
    ]

import sys
sys.path.insert(0, "wav2lip-onnx-256")
from lipsync_local import LocalLipSync

lip_sync = LocalLipSync(checkpoint_path="wav2lip-onnx-256/checkpoints/wav2lip_256.onnx", device="cuda")

def generate_talking_video(character_name, audio_path, output_path="reply_video.mp4"):
    image_path = CHARACTER_IMAGES[character_name]
    return lip_sync.generate(image_path, audio_path, output_path)

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

# Voice mapping for Kokoro-82M
# Available voices: af_heart, af_bella, af_sarah, af_nicole (female)
#                   am_adam, am_michael, am_echo, am_liam (male)
VOICE_MAP = {
    "Genie":              "am_adam",    # deep, authoritative
    "Aladdin":            "am_michael", # young, casual
    "The Princess":       "af_sarah",   # clear, confident
    "Iago":               "am_liam",    # lighter, slightly nasal
    "The Sorcerer":       "am_echo",    # deeper, resonant
    "The Cave of Wonders": "am_adam",   # deepest available
}

def speak(text, character_name, filename="reply.wav"):
    voice = VOICE_MAP[character_name]
    audio_chunks = []
    for _, _, audio in kokoro_pipeline(text, voice=voice, speed=1.0):
        audio_chunks.append(audio)
    full_audio = np.concatenate(audio_chunks)
    sf.write(filename, full_audio, 24000)

    if character_name == "The Cave of Wonders":
        add_cave_echo(filename)

    return filename

# ── Single character chat ──
def chat_with_character(character_name, mic_audio):
    if mic_audio is None:
        return "No audio recorded.", None, None

    t0 = time.time()
    user_text = transcribe(mic_audio)
    t1 = time.time()

    reply = ask_character(character_name, user_text)
    t2 = time.time()

    audio_path = speak(reply, character_name)
    t3 = time.time()

    video_path = None
    if character_name not in AUDIO_ONLY_CHARACTERS:
        try:
            video_path = generate_talking_video(character_name, audio_path)
        except Exception as e:
            print(f"Video generation failed for {character_name}: {e}")
            video_path = None
    t4 = time.time()

    print(f"STT (Whisper):  {t1-t0:.2f}s")
    print(f"LLM (Ollama):   {t2-t1:.2f}s")
    print(f"TTS (Kokoro):   {t3-t2:.2f}s")
    print(f"Video gen:      {t4-t3:.2f}s")
    print(f"Total latency:  {t4-t0:.2f}s")

    return f"You said: {user_text}\n\n{character_name}: {reply}", audio_path, video_path
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

    with gr.Row():
        with gr.Column(scale=1):
            character_video = gr.Video(
                label="Character",
                height=350,
                autoplay=True,
                loop=True,
                value=IDLE_LOOPS.get("Genie")
            )
        with gr.Column(scale=2):
            character_dropdown = gr.Dropdown(
                choices=list(CHARACTERS.keys()),
                value="Genie",
                label="Choose your character"
            )
            mic_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="Speak here"
            )
            talk_button = gr.Button("🗣️ Talk", variant="primary")
            text_output = gr.Textbox(label="Conversation", lines=4)
            audio_output = gr.Audio(
                label="Character's reply",
                autoplay=True
            )

    # NEW — swap in the idle loop whenever the character changes
    def on_character_change(character_name):
        return IDLE_LOOPS.get(character_name)

    character_dropdown.change(
        fn=on_character_change,
        inputs=character_dropdown,
        outputs=character_video
    )

    talk_button.click(
        fn=chat_with_character,
        inputs=[character_dropdown, mic_input],
        outputs=[text_output, audio_output, character_video]
    )

    gr.Markdown("---")
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
    turns_slider = gr.Slider(
        minimum=2, maximum=10, step=2, value=6,
        label="Number of turns"
    )
    debate_button = gr.Button("🎬 Let them talk", variant="primary")
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
