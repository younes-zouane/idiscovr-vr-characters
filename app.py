import os
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
from faster_whisper import WhisperModel

load_dotenv()

#client = OpenAI(
#    api_key=os.environ["DEEPINFRA_API_KEY"],
#    base_url="https://api.deepinfra.com/v1/openai",
#)
#
#MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# (local Ollama):
client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1",
)
MODEL = "llama3.1:8b"

# Load once at startup (this takes a few seconds — that's normal, it's loading the model into VRAM)
stt_model = WhisperModel("medium", device="cuda", compute_type="float16")

# Each character has ONE fixed voice name. Never changes.
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

conversation_histories = {}  # start with empty dict

for name, data in CHARACTERS.items():
    conversation_histories[name] = [
        {"role": "system", "content": data["prompt"]}
    ]

def transcribe(filepath):
    segments, info = stt_model.transcribe(filepath, beam_size=5)
    text = " ".join(segment.text for segment in segments).strip()
    return text

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

import subprocess

VOICE_MODELS = {
    "Genie": "voices/en_US-ryan-high.onnx",
    "Aladdin": "voices/en_US-joe-medium.onnx",
    "The Princess": "voices/en_US-amy-medium.onnx",
    "Iago": "voices/en_US-danny-low.onnx",
    "The Sorcerer": "voices/en_US-norman-medium.onnx",
    "The Cave of Wonders": "voices/en_US-norman-medium.onnx",
}

def speak(text, character_name, filename="reply.wav"):
    voice = VOICE_MODELS[character_name]
    subprocess.run([
        "piper", "--model", voice, "--output_file", filename
    ], input=text.encode("utf-8"))

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

########################################################################
from pydub import AudioSegment

def characters_talk(char_a, char_b, opening_line, num_turns=6):
    """
    char_a starts by responding to opening_line.
    Then char_b responds to char_a's line, and so on, alternating.
    Returns a full transcript string and a path to a combined audio file.
    """
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

        # swap who's speaking, and feed this reply as the next prompt
        last_line = reply
        current_speaker, other_speaker = other_speaker, current_speaker

    # stitch all turns into one file with a short pause between lines
    combined = AudioSegment.silent(duration=300)
    for seg in audio_segments:
        combined += seg + AudioSegment.silent(duration=400)

    combined.export("conversation.wav", format="wav")

    # cleanup the individual turn files
    for i in range(num_turns):
        os.remove(f"turn_{i}.wav")

    return "\n\n".join(transcript_lines), "conversation.wav"
##############################################################################

with gr.Blocks(title="Talking AI Characters - VR Demo") as demo:
    gr.Markdown("# 🏺 Talking AI Characters")
    gr.Markdown("Pick a character, record your message, and hear them reply.")

    character_dropdown = gr.Dropdown(
        choices=list(CHARACTERS.keys()), value="Genie", label="Choose your character"
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
        char_a_dropdown = gr.Dropdown(choices=list(CHARACTERS.keys()), value="Genie", label="Character A")
        char_b_dropdown = gr.Dropdown(choices=list(CHARACTERS.keys()), value="Iago", label="Character B")
    opening_line_input = gr.Textbox(label="Opening line / topic", value="What do you think of this whole 'wish granting' business?")
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
    demo.launch()