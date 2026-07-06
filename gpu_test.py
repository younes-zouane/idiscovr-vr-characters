import os
import glob

if os.name == "nt":
    base = r"C:\Users\PC\idiscovr-vr-characters\venv\Lib\site-packages\nvidia"

    for folder in (
        glob.glob(os.path.join(base, "*", "bin"))
        + glob.glob(os.path.join(base, "*", "lib"))
    ):
        print("Registering:", folder)
        os.add_dll_directory(folder)

from faster_whisper import WhisperModel

print("Loading model...")

model = WhisperModel(
    "medium",
    device="cuda",
    compute_type="float16"
)

print("Loaded.")

segments, info = model.transcribe("input.wav")

print(info.language)

for s in segments:
    print(s.text)