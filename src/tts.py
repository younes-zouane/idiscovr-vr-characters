import tempfile

import numpy as np
import soundfile as sf
import torch
from kokoro import KPipeline

from .audio_effects import add_cave_echo
from .characters import VOICE_MAP

kokoro_pipeline = KPipeline(lang_code="a", device="cuda" if torch.cuda.is_available() else "cpu")


def speak(text, character_name, filename=None):
    if filename is None:
        filename = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    voice = VOICE_MAP[character_name]
    audio_chunks = []
    for _, _, audio in kokoro_pipeline(text, voice=voice, speed=1.0):
        audio_chunks.append(audio)
    full_audio = np.concatenate(audio_chunks)
    sf.write(filename, full_audio, 24000)

    if character_name == "The Cave of Wonders":
        add_cave_echo(filename)

    return filename
