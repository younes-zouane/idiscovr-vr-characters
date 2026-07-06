import os
import ctypes
from pathlib import Path

root = Path.cwd() / "venv" / "Lib" / "site-packages" / "nvidia"

for folder in [
    "cublas/bin",
    "cuda_runtime/bin",
    "cuda_runtime/lib",
    "cuda_nvrtc/bin",
    "cudnn/bin",
]:
    p = (root / folder).resolve()
    if p.exists():
        os.add_dll_directory(str(p))
        print("Registered", p)

dll = root / "cublas" / "bin" / "cublas64_12.dll"

print("Loading:", dll)

ctypes.CDLL(str(dll))

print("SUCCESS")