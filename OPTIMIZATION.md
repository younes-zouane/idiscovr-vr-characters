# Optimization Log (Phase 6)

## Baseline — before any Phase 6 changes

Measured natively on Windows (venv311), 4 runs, same character (Genie), similar
short spoken message, fresh conversation each time. Numbers are the pipeline's
own printed per-stage timings.

| Run | STT (Whisper) | LLM (Ollama) | TTS (Kokoro) | Video gen | Total |
|---|---|---|---|---|---|
| 1 | 1.18s | 2.97s | 2.21s¹ | 48.18s | 54.53s |
| 2 | 0.77s | 3.16s | 0.19s | 32.87s | 36.99s |
| 3 | 0.87s | 2.87s | 0.19s | 24.34s | 28.27s |
| 4 | 0.77s | 3.13s | 0.21s | 34.82s | 38.93s |
| **Average** | **0.90s** | **3.03s** | **0.70s¹** | **35.05s** | **39.68s** |

¹ Run 1's TTS time is a clear outlier, most likely Kokoro's one-time pipeline
warmup cost on first call. Excluding it, TTS is consistently ~0.2s.

**Key finding:** video generation (lip sync) accounts for ~88% of total
latency. This is by far the highest-value target for optimization.

**Important caveat:** these baseline numbers were captured while running on
**CPU fallback** for the lip sync ONNX session, due to the documented
onnxruntime-gpu/torch cuDNN conflict (see `KNOWN_ISSUES.md`). Earlier the same
day, with GPU acceleration briefly working, video generation measured
5-9 seconds instead of 24-48 seconds. That gap is a GPU-availability issue,
not something single-pass encoding alone can fix — it's called out here so
gains from the changes below aren't misattributed to the wrong cause.

## Optimization 1: Single-pass video encoding

**Before:** `LocalLipSync.generate()` writes every generated frame to a
temporary `.avi` file via `cv2.VideoWriter`, then runs `ffmpeg` as a *second*
pass to mux the audio track in and produce the final `.mp4`. That's two full
video-writing passes plus a temp file.

**After:** Start a single `ffmpeg` process up front (`subprocess.Popen`),
streaming each generated frame directly into its stdin as raw bytes as soon
as it's produced, muxing in the audio track in the same pass. No temp `.avi`
file, no second encoding pass. Implemented in `wav2lip-onnx-256/lipsync_local.py`.

**Measured results** (4 runs each, same methodology as baseline):

| | Before (avg) | After (avg) | Change |
|---|---|---|---|
| Video gen | 35.05s | 33.54s | ~4% faster |
| Total latency | 39.68s | 38.47s | ~3% faster |

**Honest assessment:** the improvement is real but modest. This change
removes one full disk write+read cycle, but the actual bottleneck is the
CPU-bound ONNX inference loop generating each frame (running on CPU fallback,
see `KNOWN_ISSUES.md`) — single-pass encoding doesn't touch that cost at all.
For comparison, video gen measured 5-9s earlier when GPU acceleration was
briefly working. Fixing the onnxruntime-gpu/torch CUDA conflict natively
would deliver a far larger win than this optimization alone. This change is
still worth keeping — it's strictly better with no downside, just not the
main lever for speed.

## Optimization 2: Streaming responses in the UI

**Before:** The Gradio UI waits for the entire pipeline (STT → LLM → TTS →
video) to complete before showing anything at all — the user stares at a
blank "Conversation" box for the full latency (avg ~40s) with no feedback.

**After:** `chat_with_character` converted from a function that `return`s
once into a generator that `yield`s progressively: text appears as soon as
the LLM responds (~4s in), audio starts playing as soon as TTS finishes
(~5s in), and video arrives last as a progressive enhancement rather than
blocking everything else. Gradio natively supports generator functions for
`.click()` callbacks — no changes needed to the UI wiring itself.

**Measured results:** total compute time is unchanged (~38s at the time this
was measured, before the GPU fix below — as expected, since this optimization
doesn't make any stage faster). What changes is *perceived* latency: the user
sees a response in ~4 seconds instead of ~38. Verified visually in the Gradio
UI — text, then audio, then video, arriving in separate stages rather than
all at once at the end.

**Honest assessment:** this is a pure UX win with no downside and no real
engineering complexity — Gradio's generator support made it nearly free to
implement. It matters even more now that video gen is fast again (see the
GPU fix below) — a ~4s wait for text feels responsive either way, but it
mattered a lot more back when total latency was ~38s.

## Update: the GPU fix (found while writing this document)

While finishing Phase 7 docs, the onnxruntime-gpu/torch cuDNN conflict
described throughout this document and in `KNOWN_ISSUES.md` was actually
fixed — not just documented. Full details in `KNOWN_ISSUES.md`; summary:
pinning `onnxruntime-gpu==1.26.0` (matching torch's CUDA 12 build) and
pointing its cuDNN loading at torch's own bundled copy instead of installing
a conflicting second one restored GPU-accelerated video generation.

**Real before/after, with the GPU fix included:**

| | Video gen | Total latency |
|---|---|---|
| Baseline (CPU fallback) | 35.05s | 39.68s |
| + Single-pass encoding (still CPU) | 33.54s | 38.47s |
| + Streaming UI (still CPU) | 33.54s | 38.47s (perceived: ~4s) |
| **+ GPU fix** | **~7-9s** | **~12-14s** |

## Summary

Three changes, three different kinds of win:
- **Single-pass encoding** — small, real reduction in compute time (~4%)
- **Streaming UI** — large improvement in perceived responsiveness, zero
  change in actual compute time
- **GPU fix** — by far the biggest win, ~75-80% reduction in video gen time,
  found and fixed during Phase 7 after being correctly identified back in
  Phase 6 as "the real lever" but left as a known issue at the time

This turned out well: Phase 6's honest assessment — that single-pass encoding
and streaming were real but secondary next to the GPU problem — held up, and
that GPU problem got properly fixed rather than staying a documented
limitation. All three changes are complementary and are all kept.