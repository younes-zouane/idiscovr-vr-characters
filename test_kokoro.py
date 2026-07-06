# test_kokoro.py
from kokoro import KPipeline
import soundfile as sf

pipeline = KPipeline(lang_code='a')  # 'a' = American English

# Test Genie voice
generator = pipeline(
    "AHHHH MASTER! You have summoned me at last! Your wish is my command!",
    voice='am_adam',  # deep male voice
    speed=0.9
)

for i, (gs, ps, audio) in enumerate(generator):
    sf.write(f'test_kokoro_{i}.wav', audio, 24000)
    print(f"Generated segment {i}")

print("Done - check test_kokoro_0.wav")