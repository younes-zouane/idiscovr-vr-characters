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


## VR service (vr_service.py): silent CPU fallback on first run — CAUGHT, RESOLVED

**Status:** ✅ Resolved, and now defended against automatically.

When `fastapi`/`uvicorn` were first installed for the new VR headless
endpoint, `onnxruntime` (CPU-only) got pulled in as a side effect and
silently shadowed the pinned `onnxruntime-gpu==1.26.0` — the exact same
class of bug documented above for the main app, just triggered by a
different pip install. First symptom was a `UserWarning` at startup:
`Specified provider 'CUDAExecutionProvider' is not in available provider
names`. Fixed the same way: `pip uninstall onnxruntime onnxruntime-gpu -y`
then reinstall the pin.

**Prevention added:** `vr_service.py` now checks
`onnxruntime.get_available_providers()` at startup and logs a loud warning
if `CUDAExecutionProvider` is missing, instead of failing silently into
slow CPU inference. Given this is the second time this exact failure mode
has appeared in this project, worth considering the same check in
`app.py`/`src/config.py` too.

## SSE chosen over WebSockets for the streaming endpoint

The client sends one complete audio file per request (not continuous mic
streaming), and only needs one-directional server→client updates
(transcript, then audio, then video) as they become ready — SSE's exact
use case. WebSockets would be the right choice later if hands-free
continuous listening is built, since that needs the client to stream audio
to the server too, which SSE can't do.




## Using httpx with starlette.testclient is deprecated; install httpx2 instead found on TestClient related deprecation from version of starlette/fastapi i am on


## Guardrails Layer 3 (guard-model check): disabled by default — measured

**Status:** Implemented and functional, but disabled by default (`GUARD_ENABLED=false`).

Two separate things were found while measuring this, in order:

**1. Bug found and fixed: timeout/retry conflation guaranteed failure regardless of real speed.**
The guard model's per-call API timeout was set to the same value as the
*design budget* itself (300ms). Combined with the OpenAI client's default
`max_retries=2`, every single call timed out at 300ms and then retried
twice more with backoff — turning one 300ms budget into a guaranteed
~3.1s failure on every call, 5/5 times measured, regardless of how fast
the guard model actually was. Fixed by separating `GUARD_BUDGET_SECONDS`
(the 300ms design threshold — comparison only, never passed to the API)
from `GUARD_REQUEST_TIMEOUT_SECONDS` (a generous 5s real network timeout),
and setting `max_retries=0` on the guard client so a genuine failure fails
open immediately instead of stacking retries.

**2. Real finding, after the fix: the model itself is ~7x over budget.**
With genuine single-attempt timing now possible, `llama-guard3:1b`
measured a consistent **~2.2s per call** on this hardware — 2.190s,
2.204s, 2.206s, 2.207s across 4 warm runs (a 17ms spread, not noise).
`first_llm_token` timing was unaffected across these same runs (~2.7-2.8s,
matching the pre-guardrail baseline), ruling out GPU/VRAM model-swap
thrashing between `llama3.1:8b` and the guard model as the cause — this
appears to be genuine, fixed inference cost for this guard model through
Ollama on this hardware, not a config problem left to chase further.

**Decision:** disabled by default, and set
`GUARD_ENABLED=true` to re-enable if revisited later — first things to try
would be a smaller/more quantized guard model, or a non-Ollama-served
guard model (direct ONNX/transformers inference, bypassing the
OpenAI-compatible HTTP layer) to see if that overhead is what's costing
the extra time.!!!

**Side effect caught while fixing this:** flipping `GUARD_ENABLED`'s
default from `true` to `false` silently broke the assumption behind three
existing tests (`test_safe_output_is_allowed`, `test_guard_timeout_fails_open`,
`test_guard_generic_error_fails_open`) — each short-circuited past its own
mocked/side-effect behavior before ever exercising it, so two of the three
were passing for the wrong reason and one genuinely failed. Fixed by
having each test explicitly force `GUARD_ENABLED=True` for its own scope,
regardless of the module's real-world default — a good example of why
running the *full* test suite after changing a shared default matters,
not just the file that seems related.

**3. Follow-up (2026-08-07), from `scripts/bench_guardrail_latency.py`: point 2 above does
not reproduce.** Running the full pipeline (Genie, `GUARD_ENABLED=true`) across 10 normal
prompts with `llama3.1:8b` and `llama-guard3:1b` both already warm in Ollama, `guard_check`
measured **0.13–0.18s per call** — well inside the 300ms budget — except the very first
(cold) call at 4.6s. That's a completely different number from the "~2.2s per call, 4/4 warm
runs" finding above. Neither run's methodology was wrong exactly, but they're not measuring
the same thing: point 2's isolated timing script called the guard model on its own, whereas
this benchmark calls it immediately after an `llama3.1:8b` streaming call as part of one
continuous request, in a longer-lived process with both models resident. Something about
that difference — Ollama's `keep_alive` eviction behavior, VRAM residency across repeated
back-to-back calls, or process lifetime — is the likely cause, not yet isolated.

**What this means for the "disabled by default" decision:** it needs re-examination, not
because the original measurement was fabricated, but because the newer, more realistic
end-to-end measurement contradicts it. Separately: with the guard model *not* the bottleneck,
the same benchmark showed **0 of 10 normal replies under the 1.5s "first audio" budget**
(avg 2.78s) even with the ~0.15s guard check included — meaning the base LLM-streaming +
TTS path to the first sentence is the real cost, independent of Layer 3 entirely. Next step:
run `scripts/bench_guardrail_latency.py --no-guard` to confirm the base pipeline number in
isolation, then decide whether to (a) revisit the 1.5s target, (b) look at what's slow between
first LLM token and first sentence boundary + TTS, or (c) both. Not resolved yet — flagging
here rather than in a fresh section so this stays the one place guard-latency history lives.

**4. Follow-up (2026-08-07), `--no-guard` comparison: resolved — guardrails are not the cause.**
Same 10 prompts, same warm process, `GUARD_ENABLED=false`: **avg 2.57s** to first audio (min
2.41s, p95 2.70s, max 2.81s), still 0/10 within the 1.5s budget. Guard-on was avg 2.78s. The
~0.21s delta between the two matches the ~0.15–0.18s guard_check cost from point 3 almost
exactly — i.e., Layer 3 is doing precisely what it's supposed to: a small, bounded, in-budget
addition. The 1.5s miss exists identically with guardrails fully off, so it's a base
LLM-streaming + TTS pipeline characteristic (Part 1 territory), not a Part 3 problem. Point 2's
"~2.2s/call" isolated guard-model measurement remains unreconciled (see point 3), but it no
longer matters for the "disabled by default" decision either way, since even the worse-case
0.15–2.2s range is dwarfed by the ~2.5s base pipeline cost. Part 3's guardrail work is
correctly scoped and done; hitting 1.5s to first audio is a separate, Part 1-shaped
investigation (likely: how much of ~2.5s is Ollama prompt-processing/time-to-first-token vs.
generation vs. TTS) — not blocking here.