# Known Issues

## Docker app container hangs during startup (Kokoro/spaCy model loading)

**Status:** Open, actively being debugged (Phase 5).

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

**What works fine, confirmed today:**
- Docker Desktop itself (was fully broken — `hypervisorlaunchtype` was set to
  `Off` at the Windows boot level; fixed via `bcdedit /set hypervisorlaunchtype auto` + reboot).
- GPU passthrough (`--gpus all`) — confirmed working via `nvidia-smi` in a
  test container.
- The `onnxruntime-gpu` vs `torch` CUDA conflict that's still an open,
  documented workaround **natively** is fully resolved inside this container
  (see the other entry in this file) — `nvidia/cuda:13.2.1-cudnn-runtime-ubuntu22.04`
  as the base image, matching both the driver's max supported CUDA version and
  onnxruntime-gpu's real CUDA 13 requirement, was the fix.
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



**Status (native Windows venv):** Open — workaround in place (see below), real fix
deferred to Phase 6 (GPU optimization).

**Status (Docker, Phase 5):** ✅ Resolved. Building the container on a
`nvidia/cuda:13.2.1-cudnn-runtime-ubuntu22.04` base image (matching both the
driver's max supported CUDA version and onnxruntime-gpu's real CUDA 13
requirement) gives a clean environment with no conflicting cuDNN/onnxruntime
installs. Verified: `onnxruntime.get_available_providers()` returns
`['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`
inside the container, vs. CPU-only fallback natively. See `Dockerfile` for the
exact fix (forcing `onnxruntime-gpu` to win over the CPU `onnxruntime` pulled
in transitively by `faster-whisper`, via an explicit uninstall+reinstall step).
This is a strong argument for running the app via Docker going forward rather
than the native venv, once Phase 5 is fully wired up.

**Symptom:** `onnxruntime.InferenceSession` for the Wav2Lip model falls back to
`CPUExecutionProvider` instead of `CUDAExecutionProvider`. Lip sync video generation
still works, but is significantly slower than it should be on the RTX 5060 Ti.

**Root cause:**
- `torch==2.11.0+cu128` bundles its own cuDNN 9 DLLs, built against CUDA 12.x.
- `onnxruntime-gpu==1.27.0` requires cuDNN 9.x built against **CUDA 13.x**
  (installed via the separate `nvidia-cudnn-cu13` pip package).
- Both packages ship DLLs with identical filenames (e.g. `cudnn_cnn64_9.dll`,
  `cudnn64_9.dll`). When both are discoverable on the Windows DLL search path,
  whichever loads first "wins" for the whole process — and the two are not
  binary-compatible with each other's callers.
- If `nvidia-cudnn-cu13` is installed and its `bin` folder is added to the DLL
  search path (see `src/config.py`), **torch crashes on import** with:
  ```
  OSError: [WinError 127] The specified procedure could not be found. Error loading
  "...\torch\lib\cudnn_cnn64_9.dll" or one of its dependencies.
  ```
- If `nvidia-cudnn-cu13` is uninstalled (current workaround), torch imports fine,
  but onnxruntime-gpu can no longer find a compatible cuDNN 13 and silently falls
  back to CPU for the CUDA execution provider.

**Current workaround (in `src/config.py`):**
```python
for sub in list(_root.glob("*/bin")) + list(_root.glob("*/lib")):
    if "cudnn" in sub.parts:
        continue  # avoid conflict between torch's bundled cuDNN (cu12) and
                  # onnxruntime-gpu's expected cuDNN (cu13)
    os.add_dll_directory(str(sub.resolve()))
    os.environ["PATH"] = str(sub.resolve()) + os.pathsep + os.environ["PATH"]
```
Combined with **not** having `nvidia-cudnn-cu13` installed in the venv. This keeps
the app crash-free but loses GPU acceleration for lip sync specifically (STT, LLM,
and TTS are unaffected — they don't go through onnxruntime's CUDA provider).

**Real fix options (pick one in Phase 6):**
1. Downgrade `onnxruntime-gpu` to a version built against CUDA 12.x, matching
   the current torch build. Need to check onnxruntime's release notes for the
   last CUDA-12-compatible 1.x version.
2. Upgrade torch to a CUDA 13.x build, if/when one exists that still supports
   the RTX 5060 Ti (Blackwell, sm_120). Would also require re-verifying every
   other GPU-dependent piece of the pipeline (FasterLivePortrait, Kokoro TTS).
3. Investigate whether onnxruntime-gpu's `preload_dlls()` (already used
   elsewhere in `wav2lip-onnx-256/lipsync_local.py` for the CUDA libs) can be
   pointed at an isolated, non-PATH-polluting cuDNN 13 install so it never
   collides with torch's bundled one.

**How this was discovered:** During Phase 1/2 fresh-clone testing, a `pip install
onnxruntime-gpu --force-reinstall` pulled in `nvidia-cudnn-cu13` as a transitive
dependency, which hadn't been present before and triggered the crash described
above. Uninstalling it and adding the `config.py` exclusion restored a working
(CPU-fallback) state.
