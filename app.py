import os
import time

import gradio as gr
from pydub import AudioSegment

from src.characters import AUDIO_ONLY_CHARACTERS, CHARACTERS, IDLE_LOOPS
from src.lipsync import generate_talking_video
from src.llm import ask_character, init_conversation_histories, prewarm_model
from src.stt import transcribe
from src.tts import speak


def chat_with_character(character_name, mic_audio, session_histories):
    if mic_audio is None:
        return "No audio recorded.", None, None, session_histories

    t0 = time.time()
    try:
        user_text = transcribe(mic_audio)
    except Exception as e:
        print(f"Transcription failed: {e}")
        return "Sorry, I couldn't understand that audio. Please try recording again.", None, None, session_histories
    t1 = time.time()

    if not user_text:
        return "I didn't catch that — could you try speaking again?", None, None, session_histories

    try:
        reply = ask_character(character_name, user_text, session_histories[character_name])
    except Exception as e:
        print(f"LLM call failed for {character_name}: {e}")
        return f"You said: {user_text}\n\n{character_name} is having trouble responding right now. Please try again in a moment.", None, None, session_histories
    t2 = time.time()

    try:
        audio_path = speak(reply, character_name)
    except Exception as e:
        print(f"TTS failed for {character_name}: {e}")
        return f"You said: {user_text}\n\n{character_name}: {reply}\n\n(Voice generation failed, showing text only.)", None, None, session_histories
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

    return f"You said: {user_text}\n\n{character_name}: {reply}", audio_path, video_path, session_histories

def characters_talk(char_a, char_b, opening_line, num_turns=6):
    num_turns = int(num_turns)
    transcript_lines = []
    audio_segments = []

    debate_histories = {
        char_a: [{"role": "system", "content": CHARACTERS[char_a]["prompt"]}],
        char_b: [{"role": "system", "content": CHARACTERS[char_b]["prompt"]}],
    }

    current_speaker = char_a
    other_speaker = char_b
    last_line = opening_line

    for i in range(num_turns):
        reply = ask_character(current_speaker, last_line, history=debate_histories[current_speaker])
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

# ── Gradio UI ──
with gr.Blocks(title="Talking AI Characters - VR Demo") as demo:
    session_histories = gr.State(value=init_conversation_histories)
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
        inputs=[character_dropdown, mic_input, session_histories],
        outputs=[text_output, audio_output, character_video, session_histories]
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
