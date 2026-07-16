# Known Issues

## onnxruntime-gpu vs torch cuDNN conflict — RESOLVED

**Status (native Windows venv):** ✅ Resolved. Video generation confirmed
running on GPU again: 6.9-8.8s (down from 24-48s on CPU fallback), matching
the originally intended ~5.5s figure in the README.

**Root cause (fully understood now):** two separate but related conflicts,
solved in sequence:

1. **onnxruntime-gpu version mismatch.** `onnxruntime-gpu==1.27.0` dropped
   CUDA 12 support entirely and defaults to CUDA 13 (confirmed via the
   official changelog: *"Support for CUDA 12 will be removed in 1.27.0"*).
   Your torch build is `2.11.0+cu128` (CUDA 12.8) — a hard version mismatch,
   not a DLL-path problem. **Fix:** pinned to `onnxruntime-gpu==1.26.0`, the
   last version defaulting to CUDA 12, matching torch.

2. **cuDNN build mismatch, even after (1).** Once both packages wanted
   "CUDA 12 + cuDNN 9," two *different* cuDNN 9.x builds were still present
   in the environment — torch's own bundled copy (`torch/lib/cudnn_cnn64_9.dll`,
   2,984,560 bytes) and a separately pip-installed `nvidia-cudnn-cu12`
   package's copy (`nvidia/cudnn/bin/cudnn_cnn64_9.dll`, 2,994,288 bytes).
   Same filename, same major version, different builds — not binary
   compatible with each other. Loading both crashed torch's own import with
   `OSError: [WinError 127] ... Error loading "torch\lib\cudnn_cnn64_9.dll"`.
   **Fix:** don't install a second cuDNN at all. Point onnxruntime's
   `preload_dlls(cudnn=True, directory=...)` directly at torch's own
   `torch/lib` folder instead, so only one cuDNN 9 build (torch's, already
   proven working) is ever resident in the process. See
   `wav2lip-onnx-256/lipsync_local.py`.

**Final working setup:**
- `torch==2.11.0+cu128`, unchanged
- `onnxruntime-gpu==1.26.0` (pinned, not the default `1.27.0`)
- `nvidia-cuda-runtime-cu12`, `nvidia-cublas-cu12`, `nvidia-cufft-cu12`
  installed for onnxruntime's CUDA (not cuDNN) DLLs
- No separate `nvidia-cudnn-cu12` package — cuDNN comes from torch's own
  bundled copy, shared via `onnxruntime.preload_dlls(cudnn=True, directory=<torch's lib dir>)`
- `src/config.py`'s DLL-directory registration loop skips both `cu13` (stale,
  uninstalled) and `cudnn` (handled explicitly, not via the generic loop)

**Docker status:** the Docker environment (see below) was already
GPU-accelerated via a different, equally valid approach — a CUDA 13.2 base
image with everything (torch, onnxruntime-gpu) aligned to CUDA 13 instead of
12. Both fixes are legitimate; they just align on different CUDA major
versions depending on the environment's constraints.

---

## Docker: onnxruntime-gpu vs torch CUDA alignment — RESOLVED

**Status:** ✅ Resolved. Building the container on a
`nvidia/cuda:13.2.1-cudnn-runtime-ubuntu22.04` base image (matching both the
driver's max supported CUDA version and onnxruntime-gpu's real CUDA 13
requirement) gives a clean environment with no conflicting cuDNN/onnxruntime
installs. Verified: `onnxruntime.get_available_providers()` returns
`['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`
inside the container. See `Dockerfile` for the exact fix (forcing
`onnxruntime-gpu` to win over the CPU `onnxruntime` pulled in transitively by
`faster-whisper`, via an explicit uninstall+reinstall step).

---

## Docker app container hangs during startup (Kokoro/spaCy model loading)

**Status:** Open, not yet debugged further since first discovered.

**Symptom:** `docker compose up -d app` starts the container, but it hangs
indefinitely somewhere in the `from kokoro import KPipeline` import chain —
specifically around spaCy's `en_core_web_sm` model download/load (used by
Kokoro's `misaki` text-processing dependency). The exact hang point is
inconsistent between runs: sometimes it completes the `en-core-web-sm`
download and hangs after, sometimes it hangs before starting the download
at all, even on a restart where the model should already be cached.

**What's been ruled out:**
- Not a build problem — the image builds successfully every time.
- Not a Python version issue — fixed separately (Ubuntu 22.04's `python3.11`
  apt package is an unpatched `3.11.0rc1` missing `sys.get_int_max_str_digits`;
  worked around with a compatibility shim in `src/config.py` rather than
  fighting the deadsnakes PPA, which silently failed to register in this
  build environment).
- Not a general network outage — `curl` to huggingface.co from inside the
  running (hung) container returns a normal `200`.
- `docker stats` showed real (if modest) CPU and a `CLOSE_WAIT` TCP socket
  with a stuck send queue during one hang, suggesting a stalled/half-dead
  HTTP connection somewhere in the download chain, but this wasn't confirmed
  as the root cause since `py-spy` (attempted for a real stack trace) was
  blocked by Docker's default ptrace security restrictions.

**What works fine, confirmed:**
- Docker Desktop itself (was fully broken — `hypervisorlaunchtype` was set to
  `Off` at the Windows boot level; fixed via `bcdedit /set hypervisorlaunchtype auto` + reboot).
- GPU passthrough (`--gpus all`) — confirmed working via `nvidia-smi` in a
  test container.
- `ollama` service runs standalone with no issues; `llama3.1:8b` pulled and
  working.

**Next steps to try:**
1. Pre-download the spaCy model and Kokoro/Whisper weights **at Docker build
   time** (as a `RUN` step in the Dockerfile) instead of at first container
   startup — removes the runtime network dependency entirely, and would
   isolate whether this is a download issue or a load/compile issue.
2. Add explicit timeouts to whatever's making the network call, so it fails
   fast and retries instead of hanging forever.
3. Try running the container with more allocated resources (Docker Desktop
   → Settings → Resources) in case something is being OOM-throttled silently.
4. Revisit with `docker exec ... py-spy dump --pid 1 --nonblocking` or by
   adding `--cap-add=SYS_PTRACE` to the compose service definition, to get
   an actual Python stack trace of the hang instead of inferring from
   network state.
