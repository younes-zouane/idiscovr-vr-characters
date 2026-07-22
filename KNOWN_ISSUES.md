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

## Docker app container startup — RESOLVED (was misdiagnosed as a hang)

**Status:** ✅ Resolved. What was previously documented as an indefinite
hang during Kokoro/spaCy model loading was actually a slow first-run cold
start (downloading ~350MB+ of models — Kokoro weights, spaCy's
`en_core_web_sm` — over the network), not a real hang. Confirmed by
patiently timing a full `docker compose up -d app` run: it reached
`Model warm and ready` and a responsive Gradio server in ~3 minutes, which
had previously been mistaken for "stuck" in earlier sessions that didn't
wait long enough or track elapsed time. The build now also pre-loads
Kokoro at image build time (`RUN python -c "from kokoro import KPipeline;
KPipeline(lang_code='a', device='cpu')"` in the `Dockerfile`), per the
guide's suggested fix, further reducing first-run cost.

Two genuinely new bugs were found and fixed while doing full end-to-end
verification of the container (these were never part of the original
"hang," they only became visible once the app was reachable and actually
used):

**1. Gradio only listening on 127.0.0.1 inside the container.**
`demo.launch()` defaulted to binding `127.0.0.1`, which is fine natively
(browser and app share the same machine) but unreachable from the host
through Docker's port mapping — the container's `127.0.0.1` is not the
same as the host's. Symptom: `ERR_EMPTY_RESPONSE` in the browser despite
the container running and logs showing no errors. **Fix:**
`demo.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"))`
in `app.py`, with `GRADIO_SERVER_NAME=0.0.0.0` set in `docker-compose.yml`'s
`app` service. Native behavior unchanged (env var unset → still defaults
to `127.0.0.1`).

**2. faster-whisper (CTranslate2) needs CUDA 12's cuBLAS specifically.**
Once the app was actually reachable and a real transcription was
attempted, it failed with `RuntimeError: Library libcublas.so.12 is not
found or cannot be loaded`. This container's CUDA stack (torch,
onnxruntime-gpu) is CUDA 13, matching the base image — but CTranslate2
4.8.1 (faster-whisper's inference backend) is compiled against CUDA 12
and has no CUDA-13-compatible release. **Fix:** install `nvidia-cublas-cu12`
specifically and point `LD_LIBRARY_PATH` at it, alongside (not replacing)
the CUDA 13 stack everything else uses:
```dockerfile
RUN python -m pip install --no-cache-dir nvidia-cublas-cu12
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.11/dist-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH}"
```

**Verification performed (per the Next Steps Guide's Part 2 checklist):**
- Added an `ollama` healthcheck (`ollama list`, since the base image has
  no `curl`/`wget`) and `depends_on: condition: service_healthy` on `app`,
  so `app` genuinely waits for Ollama to be ready instead of racing it.
- Did a real clean-state test: `docker compose down -v` (removes the
  `ollama_data` volume too), deleted both images
  (`idiscovr-vr-characters-app` and `ollama/ollama`), then
  `docker compose up -d` from nothing. Both images rebuilt/re-pulled
  correctly, `llama3.1:8b` re-pulled successfully, both containers came
  up healthy, and a full conversation with video worked end-to-end on the
  very first try.
- Confirmed a restart (not a rebuild) reuses cached model weights rather
  than re-downloading — cold start ~3 min, warm restart a few seconds.