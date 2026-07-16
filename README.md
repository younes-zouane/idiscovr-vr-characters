# Talking AI Characters — iDISCOVR VR Internship Project

A local, fully offline app where you pick a character from the world of *One Thousand and One
Nights*, talk to them out loud, and watch/hear them reply in character — with their own voice
and a lip-synced talking-head video generated on your own GPU.

Built as part of an internship assignment for iDISCOVR.

## Characters

- **The Genie** — loud, theatrical showman
- **Aladdin** — quick, cheeky, street-smart
- **The Princess** — sharp, independent, knowledgeable
- **Iago** — sarcastic parrot, comic relief
- **The Sorcerer** — smooth, menacing, speaks in riddles
- **The Cave of Wonders** — booming voice of the room itself (not a person)

## Project structure
app.py                          # Entry point: Gradio UI, orchestrates the pipeline
src/
config.py                     # Env loading, CUDA/DLL setup, constants
characters.py                 # Character prompts, images, voices, idle loops
stt.py                        # faster-whisper wrapper
llm.py                        # Ollama client, conversation history, capping
tts.py                        # Kokoro wrapper
lipsync.py                    # Wav2Lip wrapper
audio_effects.py               # Cave of Wonders echo effect
tests/                          # pytest suite (15 tests), mocked ML deps
wav2lip-onnx-256/                # Vendored Wav2Lip ONNX inference code (see NOTICE.md)

Split into a `src/` package so each concern — speech-to-text, the LLM,
text-to-speech, lip sync — is independently testable and readable, rather
than one large `app.py`.

## How it works (current: fully local pipeline)

1. You speak into your microphone.
2. Speech is transcribed to text locally with **faster-whisper** (`medium`, float16, CUDA).
3. The text is sent to a local LLM with a character-specific system prompt — **Llama 3.1 8B**,
   served by **Ollama** (`http://localhost:11434`).
4. The character's reply is converted to speech with a fixed, distinct voice per character using
   **Kokoro-82M** (local TTS).
5. The reply audio drives a lip-synced talking-head video of the character's portrait using
   **Wav2Lip (ONNX, 256px)**, composited back onto the full source image with a feathered blend.
6. Everything runs inside a Gradio web page — no internet connection needed once models are
   downloaded and Ollama is running.

No API keys, no per-call cost, works with wifi off.

> This project went through an earlier cloud-based phase (DeepInfra APIs + Qwen3-TTS) before
> moving everything local. If you find references to that anywhere else, this README describes
> the current, local-only version.

## Requirements

- An NVIDIA GPU with CUDA support. Tested on an **RTX 5060 Ti (16GB VRAM, Blackwell)**.
- Windows (setup steps below are PowerShell-specific; adjust for Mac/Linux where noted).
- [Ollama](https://ollama.com) installed.
- Python 3.11.

CPU-only fallback is possible for each component but noticeably slower — expect video
generation in particular to go from single-digit seconds to tens of seconds on CPU.

## Setup

### 1. Clone the repo

```powershell
git clone https://github.com/younes-zouane/idiscovr-vr-characters.git
cd idiscovr-vr-characters
```

### 2. Create a virtual environment

```powershell
python -m venv venv311
venv311\Scripts\Activate.ps1        # Windows
source venv311/bin/activate         # Mac/Linux
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Install PyTorch matched to your GPU

**If you're on an RTX 50-series (Blackwell) card**, you need cu128 or newer — older CUDA
builds don't include Blackwell kernels and will silently fail at runtime rather than at
install time:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify:
```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_capability(0))"
```
Expect `True` and `(12, 0)` for Blackwell.

### 5. Install Ollama and pull the model

```powershell
ollama pull llama3.1:8b
```

Keep Ollama running (`ollama serve`, or the desktop app) before starting `app.py`.

### 6. GPU libraries for faster-whisper and Wav2Lip (onnxruntime-gpu)

On Windows, GPU execution needs CUDA runtime DLLs available to Python. This project uses
pip-installed CUDA packages rather than a system-wide CUDA Toolkit install:

```powershell
pip install onnxruntime-gpu==1.26.0
pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cufft-cu12
```
> **Important:** `onnxruntime-gpu` must stay pinned to `1.26.0`. Version 1.27+
> dropped CUDA 12 support and requires CUDA 13, which conflicts with the CUDA
> 12.8 torch build above. Don't install a separate `nvidia-cudnn-cu12` package
> either — cuDNN comes from torch's own bundled copy, shared with onnxruntime
> via `preload_dlls()` in `wav2lip-onnx-256/lipsync_local.py`. Full story in
> `KNOWN_ISSUES.md` if you hit DLL loading errors.

`app.py` and `wav2lip-onnx-256/lipsync_local.py` locate these automatically via
`sysconfig.get_paths()["purelib"]` and register them with the DLL loader at startup — no
manual `$env:PATH` edits needed if the packages are installed correctly. See the
**Troubleshooting** section below if GPU acceleration doesn't kick in.

### 7. Run it

```powershell
python app.py
```

Open the local URL shown in the terminal (usually `http://127.0.0.1:7860`).

## Using it

- Pick a character from the dropdown.
- Click the microphone, speak, then stop recording.
- Click **"Talk"**.
- Read the transcript, hear the character's reply, and watch the lip-synced video.

### Stretch feature: two characters talking to each other

A second tab lets you pick two characters, give them an opening line or topic, and watch them
banter back and forth automatically — no mic needed. Good for demos since it shows off
personality and voice differences without requiring live input.

- Pick Character A and Character B from the dropdowns.
- Enter an opening line/topic (e.g. *"What's the best wish anyone's ever asked you for?"*).
- Set the number of turns (each turn = one character's reply; defaults to 6, so 3 lines each).
- Click **"Let them talk"**.

Each character keeps its own separate conversation memory and the turn-taking logic passes
the previous reply as the next character's prompt, alternating back and forth. All turns are
stitched into a single playable audio file (`conversation.wav`) with short pauses between
lines, alongside a full text transcript.

Genie vs. Iago is a good pairing to start with — theatrical showman vs. sarcastic parrot gives
strong contrast in both writing style and voice.

## Notes

- Each character keeps memory **per browser session** (via Gradio's `gr.State`) —
  two people using the app at the same time, or the same person in two tabs,
  get fully separate conversation histories. Conversation history is also
  automatically capped (oldest messages trimmed, system prompt always kept)
  so long conversations don't grow the LLM context unboundedly.
- Speech auto-detects language — speak French, get a French reply.
- The Cave of Wonders gets a post-processing echo/pitch effect for extra atmosphere.
- Model weights (`.onnx`, `.pth`) and the Ollama model are **not committed to this repo** —
  they're downloaded locally per the setup steps above (see `.gitignore`).
- **Iago and The Cave of Wonders have no human face** — any face-detection-based feature
  (see Idle Loops below) doesn't apply to them; they only work through the standard
  audio-only conversation flow.

## Idle-loop head motion (experimental, 4 of 6 characters)

Separately from the per-turn lip-sync, there's an experimental pipeline using
[FasterLivePortrait](https://github.com/warmshao/FasterLivePortrait) to generate looping
"idle" videos (subtle head motion, blinking) for Genie, Aladdin, Princess, and Sorcerer —
Iago and Cave of Wonders are excluded since they have no detectable human face for
LivePortrait to drive.

This lives in a completely separate folder/venv (`FasterLivePortrait-windows/`, not tracked
in this repo — see `.gitignore`) and is not yet wired into live speaking segments; it only
produces standalone idle-loop clips intended for `character_video` while waiting for a reply.
The `warping_spade` model in that pipeline currently runs on CPU (not GPU) due to an
unresolved ONNX GridSample op limitation on the CUDA execution provider — video generation
there is slow (~1.5s/frame) but fine for one-time, offline idle-loop generation.

## Known quality limits

- Wav2Lip-256 generates the mouth region at a native 256×256px, then upscales into the face
  crop — this is a real resolution ceiling, not a bug. Quality varies by character: more
  photorealistic-human characters (Aladdin) sync noticeably better than heavily stylized ones
  (Genie), since Wav2Lip is trained on real human video, not painted illustration.
- We tried GFPGAN face restoration to sharpen the mouth region and reverted it — GFPGAN is
  trained on real photographic faces (FFHQ) and measurably *reduced* sharpness on our
  stylized art (Laplacian-variance sharpness dropped from ~93 to ~83), because it imposes a
  photorealistic-face prior that fights illustrated styles. Not worth it unless character art
  becomes more photorealistic.

## Testing

```powershell
pip install pytest pytest-mock
python -m pytest tests/ -v
```

15 tests covering character data integrity, LLM conversation history and
capping, STT/TTS wrapper behavior, and real (unmocked) audio processing for
the Cave of Wonders echo effect. Heavy ML dependencies (faster-whisper,
Kokoro, the Wav2Lip ONNX session) are mocked via `tests/conftest.py` so the
suite runs in under 2 seconds with no GPU required — safe to run in CI or on
a machine without the model weights downloaded.

## Docker

A `docker-compose.yml` sets up two containerized services: `ollama` (the
LLM) and `app` (everything else). GPU passthrough is confirmed working.

```powershell
docker compose build
docker compose up -d
docker exec idiscovr-ollama ollama pull llama3.1:8b
```

**Current status:** the `app` container currently hangs during startup
(somewhere in Kokoro/spaCy model loading) — see `KNOWN_ISSUES.md` for the
full investigation and next steps. The `ollama` service works standalone.
Native (non-Docker) setup per the steps above is the reliable path for now.

## Typical latency (RTX 5060 Ti, 16GB)

```
STT (faster-whisper, medium, fp16):   ~0.8s
LLM (Ollama, llama3.1:8b):            ~3.0s
TTS (Kokoro-82M):                     ~1.1s
Video gen (Wav2Lip-256, GPU):         ~7-9s
Total:                                ~12-14s per turn
```

Video generation is the largest single cost. The underlying ONNX export has a hardcoded
batch size of 1, so batching multiple frames per inference call isn't possible without
re-exporting the model — per-frame inference is the current ceiling. See `OPTIMIZATION.md`
for the optimization work (single-pass video encoding, streaming UI responses, and the
onnxruntime-gpu/torch GPU fix) that got the pipeline to these numbers.

## Troubleshooting / Environment Gotchas (local pipeline)

These bit us hard enough during setup that they're worth documenting for future reference.

**torch / torchvision / torchaudio must be reinstalled together, always.**
Never `pip install --upgrade torch` (or `--force-reinstall`) on its own. torchvision's native
extensions are compiled against a specific torch ABI version — installing torch alone leaves
torchvision mismatched, and anything touching torchvision (including `transformers`, which
Kokoro imports) breaks with cryptic errors like `RuntimeError: operator torchvision::nms does
not exist`. Always reinstall all three from the same index in one command:

```powershell
pip install torch torchvision torchaudio --force-reinstall --no-cache-dir --index-url https://download.pytorch.org/whl/cu128
```

**RTX 50-series (Blackwell) needs cu128 or newer, not cu121.**
The 5060 Ti's compute capability is sm_120. Older PyTorch CUDA builds (cu121 and earlier)
don't include Blackwell kernels — torch will "install successfully" then silently fail with
`RuntimeError: No CUDA GPUs are available` or similar the moment it actually tries to use the
GPU. Verify after any torch reinstall:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_capability(0))"
```

Expect `True` and `(12, 0)`.

**cuDNN must come from exactly one place: torch's own bundled copy.**
Two different cuDNN 9.x builds installed side-by-side (torch's bundled copy vs. a separately
pip-installed `nvidia-cudnn-cu12` package) are *not* binary-compatible with each other, even
though both report "cuDNN 9" — different file sizes, different internal symbols. Having both
on the DLL search path crashes torch's own import with `OSError: [WinError 127] ... Error
loading "torch\lib\cudnn_cnn64_9.dll"`. Fix: don't install `nvidia-cudnn-cu12` at all. Point
onnxruntime's cuDNN loading directly at torch's own `torch/lib` folder instead:

```python
onnxruntime.preload_dlls(cuda=True, cudnn=False, directory=_cuda_dir)
onnxruntime.preload_dlls(cuda=False, cudnn=True, directory=_torch_lib_dir)  # torch's own copy
```

See `wav2lip-onnx-256/lipsync_local.py` for the full implementation, and `KNOWN_ISSUES.md`
for the complete debugging story.

**`onnxruntime-gpu` must stay pinned to `1.26.0`, not the latest version.**
Version 1.27+ dropped CUDA 12 support entirely and requires CUDA 13, which doesn't match this
project's CUDA 12.8 torch build. `onnxruntime` (CPU) and `onnxruntime-gpu` also share the same
import name, so whichever installs *last* wins, regardless of what `pip show` says about the
other. If GPU execution providers go missing after any pip churn, check:

```powershell
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
```

If it only shows `CPUExecutionProvider`, or complains about CUDA 13, do a clean reinstall:

```powershell
pip uninstall onnxruntime onnxruntime-gpu -y
pip install onnxruntime-gpu==1.26.0
```

**If Ollama connection errors show up (`Connection refused` / `APIConnectionError`):**
Ollama's background server isn't running. Start it with `ollama serve` in a separate
terminal (or via the desktop app) before launching `app.py`.

**GFPGAN was tried for lip-sync sharpening and reverted** — see "Known quality limits" above.
If you want to try it again in the future, expect dependency friction: `basicsr`/`facexlib`
conflict with modern torchvision (`functional_tensor` module removed in torchvision 0.17+,
needs a manual shim) and both packages have loosely pinned `torch` dependencies that can
silently swap out a working CUDA-enabled torch install for a CPU-only one.

## What I learned / what was hard

- Windows + CUDA library paths are finicky — GPU execution needs CUDA/cuDNN DLLs explicitly
  discoverable, which pip alone doesn't fully set up without the preload logic described above.
- Not every ML package supports the newest hardware/software immediately — RTX 50-series
  (Blackwell) support in PyTorch's CUDA builds required specifically targeting cu128+, and
  TensorRT/onnxruntime-gpu compatibility across CUDA 12 vs 13 needed careful version matching.
- Budgeting GPU memory across four-plus models running simultaneously (LLM + STT + TTS +
  Wav2Lip, sometimes plus FasterLivePortrait) is a real constraint worth planning for, even
  on a comfortable 16GB card.
- Not every post-processing model generalizes to every art style — GFPGAN's photorealistic
  face prior actively fought our stylized character portraits rather than helping them.
- Moving everything local removes all per-call API cost and lets the whole thing run fully
  offline — worth the setup pain for a demo environment that can't rely on a stable internet
  connection.

## License

This project depends on two components with non-commercial, research-only
license restrictions: **Wav2Lip** and **insightface's pre-trained models**.
Because of this, the pipeline as a whole cannot be used commercially without
separately licensing both from their original authors. Full details,
including direct quotes from both projects' licensing terms, are in
[`NOTICE.md`](./NOTICE.md).
