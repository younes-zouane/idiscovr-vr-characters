import os
import sysconfig
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── NVIDIA DLL registration (needed for faster-whisper on Windows) ──
_root = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
for sub in list(_root.glob("*/bin")) + list(_root.glob("*/lib")):
    if "cudnn" in sub.parts:
        continue  # let torch's own bundled cuDNN be the only one in the process
    os.add_dll_directory(str(sub.resolve()))
    os.environ["PATH"] = str(sub.resolve()) + os.pathsep + os.environ["PATH"]

# ── LLM: local Ollama ──
OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL = "llama3.1:8b"

# ── STT: local faster-whisper ──
WHISPER_MODEL_SIZE = "medium"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"

# ── Lip sync ──
WAV2LIP_DIR = "wav2lip-onnx-256"
WAV2LIP_CHECKPOINT = f"{WAV2LIP_DIR}/checkpoints/wav2lip_256.onnx"