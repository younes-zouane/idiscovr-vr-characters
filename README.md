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

## Running Phase 2 (local, offline version)

**Requirements:** an NVIDIA GPU with CUDA support (tested on 16GB VRAM). CPU-only fallback is possible for each component but noticeably slower.

### 1. Install Ollama and pull the model

Install from [ollama.com](https://ollama.com), then:
```bash
ollama pull llama3.1:8b
```
Keep Ollama running (`ollama serve`, or the desktop app) before starting `app.py`.

### 2. GPU libraries for faster-whisper

On Windows, faster-whisper's GPU mode needs cuBLAS/cuDNN DLLs on PATH:
```powershell
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
$env:PATH = "<path-to-venv>\Lib\site-packages\nvidia\cublas\bin;<path-to-venv>\Lib\site-packages\nvidia\cudnn\bin;" + $env:PATH
```
(Run `python -c "import nvidia.cublas.lib as m; print(m.__path__[0])"` to find your exact path.)

If this doesn't work on your machine, faster-whisper falls back to CPU by changing `device="cuda"` to `device="cpu"` in `app.py` — slower, but functional.

### 3. Download Piper voice models

```bash
python -m piper.download_voices en_US-ryan-high
python -m piper.download_voices en_US-joe-medium
python -m piper.download_voices en_US-amy-medium
python -m piper.download_voices en_US-danny-low
python -m piper.download_voices en_US-norman-medium
```

### 4. Run it

```bash
python app.py
```

Open `http://127.0.0.1:7860`, pick a character, hit Talk. Try turning off your wifi — it keeps working.

## Use it

- Pick a character from the dropdown
- Click the microphone, speak, then stop recording
- Click "Talk"
- Read the transcript and hear the character's reply

## Stretch feature: two characters talking to each other

A second tab lets you pick two characters, give them an opening line or topic, and watch them argue/banter back and forth automatically — no mic needed. Good for demos since it shows off personality and voice differences without requiring live input.

- Pick **Character A** and **Character B** from the dropdowns
- Enter an opening line/topic (e.g. *"What's the best wish anyone's ever asked you for?"*)
- Set the number of turns (each turn = one character's reply; defaults to 6, so 3 lines each)
- Click **"Let them talk"**

Each character keeps its own separate conversation memory (from the main tab) and the turn-taking logic passes the previous reply as the next character's prompt, alternating back and forth. All turns are stitched into a single playable audio file (`conversation.wav`) with short pauses between lines, alongside a full text transcript.

Genie vs. Iago is a good pairing to start with — theatrical showman vs. sarcastic parrot gives strong contrast in both writing style and voice.

## Notes

- Each character keeps memory within a session (remembers what you told them earlier).
- Speech auto-detects language — speak French, get a French reply (Phase 2/faster-whisper).
- The Cave of Wonders gets a post-processing echo/pitch effect for extra atmosphere.
- Voice model files (`.onnx`) and Ollama models are not committed to this repo — they're downloaded locally per the setup steps above.

## What I learned / what was hard

- Windows + CUDA library paths are finicky — faster-whisper needs cuBLAS/cuDNN DLLs explicitly on PATH, which pip alone doesn't set up.
- Not every ML package supports the newest Python version immediately — Coqui XTTS-v2 doesn't yet support Python 3.14, which is why this project uses Piper instead for local TTS.
- Piper needs voice models downloaded ahead of time (via `piper.download_voices`) rather than fetching them automatically on first use.
- Budgeting GPU memory across three models running simultaneously (LLM + STT + TTS) is a real constraint worth planning for, even on a comfortable 16GB card.

This removes all per-call cost and lets the whole thing run fully offline.

## Notes
- Each character keeps memory within a session (remembers what you told them earlier)
- Speech auto-detects language — speak French, get a French reply
