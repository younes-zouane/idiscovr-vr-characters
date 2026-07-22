import logging
import os
import queue
import tempfile
import threading
import time

import gradio as gr
from pydub import AudioSegment

from src.characters import AUDIO_ONLY_CHARACTERS, CHARACTERS, IDLE_LOOPS
from src.lipsync import generate_talking_video
from src.llm import ask_character, init_conversation_histories, prewarm_model, stream_character_reply
from src.sentence_splitter import split_into_sentences
from src.stt import transcribe
from src.tts import speak

log = logging.getLogger(__name__)


def _llm_producer(character_name, message, history, sentence_queue):
    """
    Runs in a background thread. Streams the LLM reply, cuts it into
    sentences as text arrives, and pushes each finished sentence onto
    sentence_queue immediately — so the main thread can start TTS on
    sentence 1 while this thread is still generating sentence 2. Pushes
    None as a sentinel when there's nothing more coming (success or error).
    """
    buffer = ""
    try:
        for delta in stream_character_reply(character_name, message, history):
            buffer += delta
            sentences, buffer = split_into_sentences(buffer)
            for sentence in sentences:
                sentence_queue.put(sentence)
        if buffer.strip():
            sentence_queue.put(buffer.strip())  # flush a trailing fragment with no closing punctuation
    except Exception as e:
        log.error(f"LLM stream failed for {character_name}: {e}", exc_info=True)
    finally:
        sentence_queue.put(None)


def chat_with_character(character_name, mic_audio, session_histories):
    if mic_audio is None:
        yield "No audio recorded.", None, None, session_histories
        return

    t0 = time.time()
    try:
        user_text = transcribe(mic_audio)
    except Exception as e:
        log.error(f"Transcription failed: {e}", exc_info=True)
        yield "Sorry, I couldn't understand that audio. Please try recording again.", None, None, session_histories
        return
    t1 = time.time()

    if not user_text:
        yield "I didn't catch that — could you try speaking again?", None, None, session_histories
        return

    history = session_histories[character_name]
    sentence_queue = queue.Queue()
    producer = threading.Thread(
        target=_llm_producer, args=(character_name, user_text, history, sentence_queue), daemon=True
    )
    producer.start()

    displayed_text = f"You said: {user_text}\n\n{character_name}: "
    sentence_audio_paths = []
    first_audio_time = None

    while True:
        sentence = sentence_queue.get()
        if sentence is None:
            break

        displayed_text += sentence + " "

        try:
            sentence_audio_path = speak(sentence, character_name)
        except Exception as e:
            log.error(f"TTS failed for a sentence ({character_name}): {e}", exc_info=True)
            yield displayed_text, None, None, session_histories
            continue

        if first_audio_time is None:
            first_audio_time = time.time()
            log.info(f"Time to FIRST audio:  {first_audio_time - t0:.2f}s")

        sentence_audio_paths.append(sentence_audio_path)
        clip_duration = len(AudioSegment.from_wav(sentence_audio_path)) / 1000.0  # seconds

        yield displayed_text, sentence_audio_path, None, session_histories
        # Gradio's real audio-streaming mode proved unreliable in this
        # browser/version (JS errors in its own player component). Instead
        # we yield discrete files to a normal autoplay component — but that
        # means we must not advance to the next clip before this one has
        # actually finished playing, or we'd cut it off mid-sentence. We
        # know the exact clip length we just generated, so pace ourselves.
        time.sleep(clip_duration)

    producer.join()
    t2 = time.time()

    if not sentence_audio_paths:
        log.error(f"No audio was generated for {character_name} — LLM or TTS failed entirely.")
        yield displayed_text + "\n\n(Voice generation failed, showing text only.)", None, None, session_histories
        return

    # Step 4: concatenate the per-sentence audio SAMPLES (not just filenames)
    # into one file, and generate the video once from that — exactly like
    # before streaming existed. This file is never re-sent to the audio
    # player (it was already heard sentence-by-sentence) — it only feeds
    # the lip-sync video generator.
    combined = AudioSegment.empty()
    for path in sentence_audio_paths:
        combined += AudioSegment.from_wav(path)  # samples, not filename concatenation
    combined_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    combined.export(combined_path, format="wav")

    for path in sentence_audio_paths:
        os.remove(path)

    video_path = None
    if character_name not in AUDIO_ONLY_CHARACTERS:
        try:
            video_path = generate_talking_video(character_name, combined_path)
        except Exception as e:
            log.error(f"Video generation failed for {character_name}: {e}", exc_info=True)
    t3 = time.time()

    log.info(f"STT (Whisper):        {t1 - t0:.2f}s")
    log.info(f"Time to first audio:  {(first_audio_time or t2) - t0:.2f}s")
    log.info(f"Full reply + audio:   {t2 - t0:.2f}s")
    log.info(f"Video gen:            {t3 - t2:.2f}s")
    log.info(f"Total latency:        {t3 - t0:.2f}s")

    yield displayed_text, gr.skip(), video_path, session_histories


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

    turn_audio_paths = []
    for i in range(num_turns):
        reply = ask_character(current_speaker, last_line, history=debate_histories[current_speaker])
        transcript_lines.append(f"{current_speaker}: {reply}")

        turn_audio_path = speak(reply, current_speaker)
        turn_audio_paths.append(turn_audio_path)
        audio_segments.append(AudioSegment.from_wav(turn_audio_path))

        last_line = reply
        current_speaker, other_speaker = other_speaker, current_speaker

    combined = AudioSegment.silent(duration=300)
    for seg in audio_segments:
        combined += seg + AudioSegment.silent(duration=400)
    combined_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    combined.export(combined_path, format="wav")

    for path in turn_audio_paths:
        os.remove(path)

    return "\n\n".join(transcript_lines), combined_path


# ── Gradio UI ──
with gr.Blocks(title="Talking AI Characters - VR Demo") as demo:
    session_histories = gr.State(value=init_conversation_histories)
    gr.Markdown("# 🏺 Talking AI Characters")
    gr.Markdown("Pick a character, record your message, and hear them reply.")

    with gr.Row():
        with gr.Column(scale=1):
            character_video = gr.Video(
                label="Character", height=350, autoplay=True, loop=True, value=IDLE_LOOPS.get("Genie")
            )
        with gr.Column(scale=2):
            character_dropdown = gr.Dropdown(
                choices=list(CHARACTERS.keys()), value="Genie", label="Choose your character"
            )
            mic_input = gr.Audio(sources=["microphone"], type="filepath", label="Speak here")
            talk_button = gr.Button("🗣️ Talk", variant="primary")
            text_output = gr.Textbox(label="Conversation", lines=4)
            audio_output = gr.Audio(label="Character's reply", autoplay=True)

    # NEW — swap in the idle loop whenever the character changes
    def on_character_change(character_name):
        return IDLE_LOOPS.get(character_name)

    character_dropdown.change(fn=on_character_change, inputs=character_dropdown, outputs=character_video)

    talk_button.click(
        fn=chat_with_character,
        inputs=[character_dropdown, mic_input, session_histories],
        outputs=[text_output, audio_output, character_video, session_histories],
    )

    gr.Markdown("---")
    gr.Markdown("## 🎭 Let two characters talk to each other")
    with gr.Row():
        char_a_dropdown = gr.Dropdown(choices=list(CHARACTERS.keys()), value="Genie", label="Character A")
        char_b_dropdown = gr.Dropdown(choices=list(CHARACTERS.keys()), value="Iago", label="Character B")
    opening_line_input = gr.Textbox(
        label="Opening line / topic", value="What do you think of this whole 'wish granting' business?"
    )
    turns_slider = gr.Slider(minimum=2, maximum=10, step=2, value=6, label="Number of turns")
    debate_button = gr.Button("🎬 Let them talk", variant="primary")
    debate_transcript = gr.Textbox(label="Conversation transcript", lines=10)
    debate_audio = gr.Audio(label="Full conversation", autoplay=True)

    debate_button.click(
        fn=characters_talk,
        inputs=[char_a_dropdown, char_b_dropdown, opening_line_input, turns_slider],
        outputs=[debate_transcript, debate_audio],
    )

if __name__ == "__main__":
    prewarm_model()
    demo.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"))
