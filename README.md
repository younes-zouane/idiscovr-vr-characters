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
pip install nvidia-cuda-runtime nvidia-cublas nvidia-cudnn-cu13 nvidia-cufft
```

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

- Each character keeps memory within a session (remembers what you told them earlier).
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

## Typical latency (RTX 5060 Ti, 16GB)

```
STT (faster-whisper, medium, fp16):   ~0.9s
LLM (Ollama, llama3.1:8b):            ~2.9s
TTS (Kokoro-82M):                     ~1.3s
Video gen (Wav2Lip-256, GPU):         ~5.5s
Total:                                ~10.6s per turn
```

Video generation is the largest single cost. The underlying ONNX export has a hardcoded
batch size of 1, so batching multiple frames per inference call isn't possible without
re-exporting the model — per-frame inference is the current ceiling.

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

**Don't preload cuDNN via `onnxruntime.preload_dlls()` in the same process as torch.**
`lipsync_local.py` needs `onnxruntime-gpu` to find its CUDA runtime DLLs (via pip-installed
`nvidia-cuda-runtime` / `nvidia-cublas` / `nvidia-cufft` packages, not a system CUDA Toolkit
install). But do **not** also preload cuDNN this way — torch already loads its own bundled
cuDNN 9 DLLs at import time, and forcing a second cuDNN load from the pip package's directory
causes both libraries to silently fail (`WinError 127`), breaking Kokoro's TTS with
`RuntimeError: GET was unable to find an engine to execute this computation`. Only preload
the plain CUDA runtime, not cuDNN:

```python
onnxruntime.preload_dlls(cuda=True, cudnn=False, directory=_cuda_dir)
# do NOT also call preload_dlls(cudnn=True, ...) — let torch own cuDNN
```

**onnxruntime and onnxruntime-gpu can't coexist.**
They share the same import name (`onnxruntime`), so whichever gets installed *last* wins,
regardless of what `pip show` says about the other. If GPU execution providers
(`CUDAExecutionProvider`) go missing after any pip install/uninstall churn, check:

```powershell
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
```

If it only shows `CPUExecutionProvider`, do a clean reinstall:

```powershell
pip uninstall onnxruntime onnxruntime-gpu -y
pip install onnxruntime-gpu
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
