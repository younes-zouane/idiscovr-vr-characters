from pydub import AudioSegment
from pydub.effects import normalize


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