import os
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["DEEPINFRA_API_KEY"],
    base_url="https://api.deepinfra.com/v1/openai",
)

SAMPLE_RATE = 16000

def record_audio(filename="input.wav", duration=5):
    print(f"Recording for {duration} seconds... speak now!")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    sf.write(filename, audio, SAMPLE_RATE)
    print("Done recording.")
    return filename

def transcribe(filename="input.wav"):
    with open(filename, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="openai/whisper-large-v3",
            file=audio_file
        )
    return transcript.text

if __name__ == "__main__":
    record_audio(duration=5)
    text = transcribe()
    print("You said:", text)