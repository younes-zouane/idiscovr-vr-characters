import sys

# Ubuntu 22.04's python3.11 package is a pre-release build (3.11.0rc1) missing
# sys.get_int_max_str_digits, added right at the 3.11.0 final release. torch's
# internals expect it to exist. This restores it with CPython's real default.
if not hasattr(sys, "get_int_max_str_digits"):
    sys.set_int_max_str_digits = lambda maxdigits: None
    sys.get_int_max_str_digits = lambda: 4300

import os
import sysconfig
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── NVIDIA DLL registration (needed for faster-whisper on Windows only —
#    os.add_dll_directory doesn't exist on Linux, and isn't needed there
#    since the CUDA base image already configures library paths correctly) ──
if os.name == "nt":
    _root = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    for sub in list(_root.glob("*/bin")) + list(_root.glob("*/lib")):
        if "cudnn" in sub.parts:
            continue  # avoid conflict between torch's bundled cuDNN (cu12) and onnxruntime-gpu's expected cuDNN (cu13)
        os.add_dll_directory(str(sub.resolve()))
        os.environ["PATH"] = str(sub.resolve()) + os.pathsep + os.environ["PATH"]
# ── LLM: local Ollama ──
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MODEL = "llama3.1:8b"

# ── STT: local faster-whisper ──
WHISPER_MODEL_SIZE = "medium"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"

# ── Lip sync ──
WAV2LIP_DIR = "wav2lip-onnx-256"
WAV2LIP_CHECKPOINT = f"{WAV2LIP_DIR}/checkpoints/wav2lip_256.onnx"