import numpy as np
import soundfile as sf

from src.audio_effects import add_cave_echo


def test_add_cave_echo_produces_valid_wav(tmp_path):
    # generate a short, real sine-wave wav file
    filepath = tmp_path / "test_tone.wav"
    sample_rate = 24000
    duration_sec = 0.5
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(str(filepath), tone, sample_rate)

    add_cave_echo(str(filepath))

    # file must still exist and be readable as valid audio afterward
    data, sr = sf.read(str(filepath))
    assert len(data) > 0
    assert sr == sample_rate