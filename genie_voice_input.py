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

MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
SAMPLE_RATE = 16000

GENIE_SYSTEM_PROMPT = """You are the Genie of the lamp, straight out of the 
One Thousand and One Nights. You are loud, theatrical, and a bit of a 
show-off. You grant "wishes" by answering questions with flair and humor. 
Keep replies short (2-4 sentences) and full of personality. Never break 
character."""

conversation_history = [
    {"role": "system", "content": GENIE_SYSTEM_PROMPT}
]

def record_audio(filename="input.wav", duration=5):
    print(f"Recording for {duration} seconds... speak now!")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    sf.write(filename, audio, SAMPLE_RATE)
    return filename

def transcribe(filename="input.wav"):
    with open(filename, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="openai/whisper-large-v3",
            file=audio_file
        )
    return transcript.text

def ask_genie(message):
    conversation_history.append({"role": "user", "content": message})
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=300,
        messages=conversation_history
    )
    reply = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": reply})
    return reply

if __name__ == "__main__":
    print("The Genie awaits your wishes... (Ctrl+C to exit)\n")
    while True:
        record_audio(duration=5)
        text = transcribe()
        print("You said:", text)

        reply = ask_genie(text)
        print("Genie:", reply, "\n")