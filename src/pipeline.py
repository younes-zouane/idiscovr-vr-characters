"""
Shared orchestration for the "stream LLM reply -> speak each sentence ->
combine audio -> optional lip-sync video" flow.

Before this existed, app.py's chat_with_character/_llm_producer and both
vr_service.py endpoints each reimplemented this independently. Three
copies of the same logic drift — this is the one place it lives now.

Design note: TTS-per-sentence is deliberately NOT folded in here. app.py
runs LLM-streaming in a background thread so it can synthesize sentence 1
while the LLM is still generating sentence 2; vr_service.py just does it
inline. Baking speak() into this generator would force both callers into
the same (non-)concurrency model, so stream_reply_sentences() only yields
sentence text — callers call speak() themselves, on whatever thread makes
sense for them.
"""

import logging
import tempfile

from pydub import AudioSegment

from src.characters import AUDIO_ONLY_CHARACTERS, CHARACTERS
from src.lipsync import generate_talking_video
from src.llm import stream_character_reply
from src.sentence_splitter import split_into_sentences
from src.guardrails import (
    MAX_REPLY_SENTENCES,
    check_input,
    check_output_with_guard_model,
    clean_sentence,
)

log = logging.getLogger(__name__)


def stream_reply_sentences(character_name, user_text, history, on_first_token=None):
    """
    Streams the character's LLM reply and yields each completed sentence
    as plain text.

    Guardrails, in order:
    - Layer 2 (check_input): a blocked input never reaches the LLM at all —
      yields the character's own refusal line and returns immediately,
      which is why a blocked reply is faster than a normal one.
    - Layer 3 (check_output_with_guard_model): runs exactly once, on the
      FIRST completed sentence only, before it's yielded. If the guard
      model flags it, the refusal line is yielded instead and generation
      stops there — nothing unsafe ever reaches speak(). Every sentence
      after the first is not re-checked (cost/latency tradeoff — see
      guardrails.py docstring).
    """
    allowed, reason = check_input(user_text)
    if not allowed:
        yield CHARACTERS[character_name]["refusal"]
        return

    buffer = ""
    first_token_seen = False
    first_sentence_checked = False
    sentence_count = 0
    for delta in stream_character_reply(character_name, user_text, history):
        if not first_token_seen:
            first_token_seen = True
            if on_first_token:
                on_first_token()
        buffer += delta
        sentences, buffer = split_into_sentences(buffer)
        for sentence in sentences:
            if sentence_count >= MAX_REPLY_SENTENCES:
                return  # cap hit — stop pulling more from the LLM stream

            if not first_sentence_checked:
                first_sentence_checked = True
                safe, guard_reason = check_output_with_guard_model(user_text, sentence)
                if not safe:
                    yield CHARACTERS[character_name]["refusal"]
                    return

            cleaned = clean_sentence(sentence)
            if cleaned:
                sentence_count += 1
                yield cleaned

    if buffer.strip() and sentence_count < MAX_REPLY_SENTENCES:
        # Note: if the ENTIRE reply was one unpunctuated fragment (no
        # sentence boundary ever hit), first_sentence_checked is still
        # False here — this fragment never goes through Layer 3. Same
        # tradeoff as above; flagging it rather than hiding it.
        cleaned = clean_sentence(buffer.strip())
        if cleaned:
            yield cleaned


def combine_audio(sentence_audio_paths, output_path=None):
    """
    Concatenates per-sentence wav SAMPLES (not filenames) into one file.
    Writes to output_path if given, otherwise a fresh temp file. Returns
    the resulting path either way.

    Does not delete the input files — callers differ on when that's safe
    (app.py removes them right after combining; the SSE endpoint has
    already copied each one out to a public URL by this point).
    """
    combined = AudioSegment.empty()
    for path in sentence_audio_paths:
        combined += AudioSegment.from_wav(path)
    if output_path is None:
        output_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    combined.export(output_path, format="wav")
    return output_path


def generate_video_if_applicable(character_name, combined_audio_path, output_path=None):
    """
    Returns (video_path_or_None, error_code_or_None).

    Audio-only characters (Iago, Cave of Wonders) -> (None, None): expected,
    not an error. A real generation failure -> (None, "video_failed") so
    callers can still return the audio that already succeeded — the same
    fallback vr_service.py's sync endpoint already relied on.
    """
    if character_name in AUDIO_ONLY_CHARACTERS:
        return None, None
    try:
        if output_path:
            video_path = generate_talking_video(character_name, combined_audio_path, output_path=output_path)
        else:
            video_path = generate_talking_video(character_name, combined_audio_path)
        return video_path, None
    except Exception as e:
        log.error("Video generation failed for %s: %s", character_name, e, exc_info=True)
        return None, "video_failed"