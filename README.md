# Talking AI Characters — iDISCOVR VR Internship Project

A small app where you pick a character from the world of *One Thousand and One Nights*, 
talk to them out loud, and hear them reply in character with their own distinct voice.

Built as part of an 8-week internship assignment for iDISCOVR.

## Characters
- **The Genie** — loud, theatrical showman
- **Aladdin** — quick, cheeky, street-smart
- **The Princess** — sharp, independent, knowledgeable
- **Iago** — sarcastic parrot, comic relief
- **The Sorcerer** — smooth, menacing, speaks in riddles
- **The Cave of Wonders** — booming voice of the room itself

## How it works
1. You speak into your microphone
2. Speech is transcribed to text (Whisper, via DeepInfra)
3. The text is sent to an LLM with a character-specific system prompt (Llama 3.1 8B, via DeepInfra)
4. The character's reply is converted to speech with a fixed, distinct voice per character (Qwen3-TTS, via DeepInfra)
5. Everything runs inside a simple Gradio web page

## How to run it

### 1. Clone the repo
```bash
git clone https://github.com/younes-zouane/idiscovr-vr-characters.git
cd idiscovr-vr-characters
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install openai python-dotenv requests gradio sounddevice soundfile
```

### 4. Add your API key
Create a `.env` file in the project root:
DEEPINFRA_API_KEY=your_key_here
Get a key at [deepinfra.com](https://deepinfra.com).

### 5. Run the app
```bash
python app.py
```
Open the local URL shown in the terminal (usually `http://127.0.0.1:7860`).

### 6. Use it
- Pick a character from the dropdown
- Click the microphone, speak, then stop recording
- Click "Talk"
- Read the transcript and hear the character's reply

## What's next (Phase 2)
Move the AI off the cloud and onto local hardware:
- **Brain**: Ollama running Llama 3.1 8B locally
- **Ears**: faster-whisper for local speech-to-text
- **Voice**: Coqui XTTS-v2 (or Piper/Kokoro) for local text-to-speech

This removes all per-call cost and lets the whole thing run fully offline.

## Notes
- Each character keeps memory within a session (remembers what you told them earlier)
- Speech auto-detects language — speak French, get a French reply