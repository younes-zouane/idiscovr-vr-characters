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

## Part 2 — Docker startup + containerized latency (Next Steps Guide numbering)

Note: the Next Steps Guide (July 2026) restarted numbering at "Part 1-4",
distinct from this doc's earlier "Phase 1-7". Entries below use the guide's
Part numbers; everything above this line is unchanged Phase 6 history.

**Startup time fix:** Kokoro + spaCy weights were being re-downloaded from
HF/GitHub on every `docker compose up`, adding ~3 minutes before the app was
usable. Fixed by baking the weights into the image at build time (a `RUN`
step instantiating `KPipeline` during `docker build`, not at container
start) plus a persistent `HF_HOME` cache volume as a second layer of
protection.

| | Before | After |
|---|---|---|
| Container start → "Model warm and ready" | ~3 min | ~18s |

**Per-turn latency, containerized (GPU confirmed working inside Docker):**

| Run | STT | LLM | TTS | Video gen | Total |
|---|---|---|---|---|---|
| 1 (first request after startup) | 7.39s | 0.84s | 2.50s | 14.66s | 25.39s |
| 2 (steady state) | 0.87s | 1.42s | 0.33s | 9.45s | 12.07s |

**Honest assessment:** the 7.39s → 0.87s STT gap between run 1 and run 2 is
Whisper's own first-inference warm-up (CUDA kernel compilation, weights
moving to GPU memory) — separate from the download-caching fix above, and
not something this Part addressed. First-request-after-startup will likely
always be slower than steady state on this stack. README now recommends one
throwaway warm-up request before a live demo for this reason.

## Part 3 — Sentence-streaming speech, time-to-first-audio

6 runs, native Windows, streaming Ollama + sentence-split TTS.

| Run | Character | First audio | Note |
|---|---|---|---|
| 1 | Genie | 2.44s | af_sarah.pt voice file downloaded mid-run |
| 2 | Genie | 1.20s | |
| 3 | Genie | 1.52s | |
| 4 | Genie | 1.31s | |
| 5 | Genie | 1.15s | |
| 6 | Cave | 4.18s | am_adam.pt voice file downloaded mid-run |
| **Average (excl. voice-download outliers)** | | **1.30s** | |

Target was ≤2.5s. All runs meet it once the relevant voice is cached;
runs 1 and 6 show the one-time per-voice download cost, not a pipeline
regression — the same shape as the Kokoro warmup outlier in the Phase 6
baseline above.

**Finding:** the voice-download cost is per-*character*, not per-app. Each
character maps to its own Kokoro voice file (af_sarah for Genie, am_adam
for Cave, etc.), and each gets downloaded independently on its own first
use. A demo cycling through multiple characters live will hit this once per
character, not just once total.

Voice files are cached after first download (`~/.cache/huggingface` or
similar), so this only costs time once per voice per machine. A pre-warm
step that loops through all characters' voices at startup (same idea as
the Kokoro/spaCy Docker pre-bake in Part 2) would remove this from any
live demo.

**Cave echo per-sentence:** satisfied as a side effect of the streaming
refactor. `add_cave_echo()` is called inside `speak()`, and `speak()` is
now invoked once per sentence rather than once per full reply, so the echo
naturally applies per-chunk without separate changes. Listened back to a
multi-sentence Cave reply to confirm no audible clipping of the echo tail
at sentence boundaries — sounded clean.

**Deviation from the guide's suggested approach:** the guide suggested
`gr.Audio(streaming=True)` for true chunk-by-chunk playback. That component
proved unreliable in this Gradio version (JS errors in its own player).
Used discrete per-sentence audio files with `time.sleep(clip_duration)`
pacing in the generator instead — functionally equivalent from the user's
perspective (sentences play back-to-back as they're generated), just
implemented at the yield level rather than via native audio streaming.


## Part 4 — Batched Wav2Lip: blocked on missing source checkpoint

**Status:** blocked at Step 1 (locate the source `.pth`). Flagged to
supervisor rather than continuing to guess.

Part 4 requires the original PyTorch checkpoint that produced
`wav2lip_256.onnx`, per the guide's Step 1 ("locate it first; without it,
stop and flag it"). `wav2lip-onnx-256/` was originally set up by
downloading only the pre-converted ONNX file — the setup instructions were
"download models from releases" (ONNX only) — so the source `.pth` was
never obtained.

Checked two independent sources, both mismatched:

- **Source 1:** the vendored repo's own README cites Linly-Talker's README
  as the checkpoint's origin. That page re-links to the standard
  Rudrabha/Wav2Lip weights (official links are dead — see
  [Rudrabha/Wav2Lip#739](https://github.com/Rudrabha/Wav2Lip/issues/739) —
  used a verified mirror instead). Loaded successfully as a genuine Wav2Lip
  model with correct layer names, but systematically narrower channel
  widths than `wav2lip_256.py` (this repo's model class) expects — e.g.
  `face_decoder_blocks.3.0.conv_block.0.weight`: checkpoint `[384, 384, 3,
  3]` vs model `[512, 512, 3, 3]`, consistent across every decoder block.
- **Source 2:** `numz/wav2lip_studio` on HuggingFace (ONNX-focused Wav2Lip
  project, closer ecosystem match). Verified via SHA256
  (`b78b681b...ef63c37`). After stripping the `module.` DataParallel
  prefix, same result: identical shape mismatches to Source 1, confirming
  it's the same standard architecture, not the wide-channel 256px variant
  this repo actually uses.
- Also checked Linly-Talker's separate "Wav2Lipv2" reference
  (primepake/wav2lip_288x288) — different architecture family (SAM-UNet,
  PReLU/LeakyReLU, 288/384/512px only), no checkpoint published.

**Conclusion:** two independent, credible sources both converge on the
standard/narrow Wav2Lip architecture, which is not what `wav2lip_256.py`
defines. The wide-channel 256px variant this repo's ONNX was built from
doesn't appear to be published anywhere findable under an obvious name —
likely a bespoke retrain, and since only the pre-converted ONNX was ever
obtained, there's no further way to trace it from this project alone.

Flagged to supervisor. Options under consideration: (1) skip the re-export,
keep `wav2lip_256.onnx` as-is at batch=1, accept current ~7-9s video gen;
(2) longer-term, evaluate switching lip-sync families entirely (e.g.
MuseTalk), per the guide's own aside — a bigger decision than this task.

Per the guide's closing note: *"a clear write-up of a failure is worth more
than a silent half-working success."* Steps 2-5 (dynamic batch export,
correctness proof, batch-size sweep, FP16) are not attempted, since they
all depend on having the correct source checkpoint first.

## Part 5 — LatentSync evaluation (supervisor follow-up after Part 4 blocked)

**Context:** after Part 4 (batched Wav2Lip) was blocked on the missing
source checkpoint and flagged to the supervisor, visual quality tests of
LatentSync on Genie and Aladdin passed. Before deciding whether to replace
Wav2Lip, the supervisor requested two hard numbers: video-gen latency
comparison, and peak GPU memory with the full pipeline (Ollama + Whisper +
Kokoro + lip-sync backend) loaded.

### Part 5.1 — Video generation latency: Wav2Lip vs LatentSync

**Methodology:** LatentSync run via its own dedicated venv (separate
dependency stack — different torch version, `insightface==0.7.3`, etc. —
not directly importable into `venv311`). 5 consecutive subprocess
invocations of `scripts.inference`, same audio clip and character image
each time, wall-clock timed with `subprocess.run` + `sys.executable`
(not shell `date`/`bc`, which proved unreliable in this Git Bash setup).
Run 5 was interrupted (`KeyboardInterrupt`) and excluded; the 4 clean runs
were extremely consistent (173.68–175.97s, <2.5s spread).

| Run | Time |
|---|---|
| 1 | 173.72s |
| 2 | 175.97s |
| 3 | 174.90s |
| 4 | 173.68s |
| **Average** | **174.57s** |

**Comparison table:**

| Model | Video-gen time | vs. Wav2Lip |
|---|---|---|
| Wav2Lip | ~7-9s | baseline |
| LatentSync (full subprocess run, incl. model reload) | **174.57s** | **~20-24x slower** |
| LatentSync (steady-state inference step only, per the script's own reported `Doing inference...` timing) | ~125s | ~15-18x slower |

**Fairness caveat, stated explicitly:** Wav2Lip's ~7-9s figure reflects
this app's actual behavior — the ONNX session loads once at startup and is
reused across every conversation turn. LatentSync's 174.57s figure
includes reloading the diffusion checkpoint and InsightFace's face-detection
models fresh on every run, since each measurement was a separate subprocess
(LatentSync's own dependency stack can't currently be loaded persistently
inside this app's process — see Part 5.2 methodology below for why).
A production integration that kept LatentSync's models resident, the same
way Wav2Lip's are, would land closer to the ~125s inference-only figure.
Even using that more generous number, LatentSync remains over an order of
magnitude slower than Wav2Lip on this hardware and these settings.

**Conclusion (latency alone):** Wav2Lip is dramatically faster —
roughly 15-24x depending on which LatentSync figure is used. This is a
decision-relevant number regardless of methodology nuance.

### Part 5.2 — Peak GPU memory (combined pipeline)

**Methodology:** Configuration A measured directly in the real app
(`venv311`) — Ollama + Whisper + Kokoro loaded, one full Wav2Lip
conversation turn triggered, `nvidia-smi` polled every 0.5s throughout.
Configuration B measured with the same `venv311` app loaded and idle
(Ollama + Whisper + Kokoro resident, no Wav2Lip turn), while LatentSync
ran one full inference in its own separate venv at the same time — GPU
memory is a physical resource shared across all processes on the card
regardless of which Python environment they run in, so the combined
`nvidia-smi` peak is a valid measurement without needing cross-venv
integration.

| Configuration | Baseline (idle, monitor start) | Peak | Peak (GB) | GPU utilization |
|---|---|---|---|---|
| A: Ollama + Whisper + Kokoro + Wav2Lip | 1,243 MiB | 9,972 MiB | 9.74 GB | 61% of 16GB card |
| B: Ollama + Whisper + Kokoro + LatentSync | 9,940 MiB¹ | 15,994 MiB | 15.62 GB | **97.6% of 16GB card** |

¹ Config B's baseline (idle app, no LatentSync yet) closely matches
Config A's peak (9,972 MiB) — good internal consistency check, confirming
Ollama + Whisper + Kokoro's own footprint is stable (~9.9GB) independent
of which lip-sync backend runs on top.

**Secondary finding — GPU contention degrades LatentSync's own latency:**
under combined load, LatentSync's inference step took **6:08 (368s)**,
compared to **2:05 (125s)** when run in isolation during Part 5.1 — nearly
3x slower. This is a real, measured effect of near-saturated VRAM (97.6%
utilized), not a fluke; per-batch sample-generation steps that normally
took ~12-13s each took 58s and 99s for the first two batches under
contention. This means Part 5.1's isolated 174.57s figure actually
*understates* LatentSync's real-world cost when run as part of the full
pipeline — the honest combined-load number is closer to 6+ minutes for
inference alone, before the face-detection and restoration steps.

**Conclusion (memory + combined-load latency):** LatentSync leaves the
16GB card almost completely saturated (400MB headroom) and is
substantially slower under realistic combined load than in isolation.
Wav2Lip uses roughly 60% of available VRAM with comfortable headroom
remaining.

### Final comparison table and recommendation

| Metric | Wav2Lip | LatentSync |
|---|---|---|
| Video-gen latency (isolated) | ~7-9s | ~125-175s (15-24x slower) |
| Video-gen latency (combined GPU load, realistic) | ~7-9s (already measured under combined load — this *is* Config A) | ~368s inference alone (~3x slower than its own isolated figure) |
| Peak combined GPU memory | 9.74 GB (61%) | 15.62 GB (97.6%) |
| Visual quality (Genie, Aladdin) | Baseline | Passed, supervisor-confirmed |

**Recommendation:** on the current hardware (RTX 5060 Ti, 16GB), LatentSync's
visual quality improvement does not offset its cost — it is 15-24x slower
in isolation and up to ~3x slower again under the realistic combined-pipeline
load this app actually runs, while consuming nearly all available VRAM with
almost no headroom. Wav2Lip remains the appropriate choice for this
project's current hardware and latency requirements. LatentSync (or a
comparable modern architecture like MuseTalk, mentioned in Part 4's
write-up) may be worth revisiting if hardware changes or if latency
requirements relax significantly for a future non-interactive use case.


📊 Alternative Audio-to-Video Pipeline Evaluation: LivePortrait + JoyVASA

**Status:** Testing Concluded (Do Not Proceed to Optimization)

#### Executive Summary
LivePortrait combined with JoyVASA represents a significant technological leap over traditional lip-syncing tools like Wav2Lip. Because it animates the entire face (including head tilts, micro-expressions, and emotional cadence), the output looks inherently more dynamic and modern than models that only modify the mouth region.

However, when evaluated against stylized or cartoon assets (Genie), the model's full-face motion vectors cause structural warping and edge artifacts. Because it triggers the "distorted/uncanny" quality gate established for MuseTalk, we are halting further VRAM/latency benchmarking to avoid an open-ended optimization rabbit hole.

#### ⚖️ Comparative Analysis

| Metric / Aspect | Wav2Lip (Baseline) | LivePortrait + JoyVASA |
| :--- | :--- | :--- |
| **Animation Scope** | Lip region only (static face). | Full facial expression, eye blinks, and head pose. |
| **Visual Ambition** | Low (looks robotic/flat). | High (looks organic and lively). |
| **Stylized Art Risk** | Low (preserves character artwork). | High (warps facial geometry out of bounds). |
| **Current Result** | Fails on pipeline mismatch. | **Fails on quality gate (uncanny/distorted).** |

#### 🛠️ Technical Insights & Blockers
* **The Style Mapping Problem:** LivePortrait relies on human facial landmarks. When driving a stylized character like Genie, the mathematical translations stretch 2D artwork unnaturally, ruining the character's intended aesthetic.
* **Production Viability:** While LivePortrait is visually superior for photorealistic human faces, a production environment requires absolute stability. Perfecting this for stylized art would require extensive expression-slider tuning and asset-specific retraining, which falls outside current project scopes.

#### 🚀 Future Recommendation
If the project pivots towards hyper-realistic human avatars in the future, LivePortrait + JoyVASA should be the immediate first choice. For the current stylized project, it is officially logged as a quality failure, and we are moving on to finalize the remaining pipeline documentation.

## Part 6 — VR headless backend (vr_service.py)

Replaced the Gradio UI with a FastAPI REST endpoint (`POST /v1/vr-chat`)
for VR client integration (Unity/Unreal can't drive a Gradio page).
Two fixes over the first draft: per-session conversation history (keyed
by session_id, not shared globally per character) and URL-based file
delivery instead of base64-in-JSON (better fit for a VR client parsing
this on a standalone headset).

**Verified manually via Swagger UI (`/docs`):**
- GPU: `CUDAExecutionProvider confirmed available` at startup (a CPU
  fallback regression was caught and fixed here too — see KNOWN_ISSUES.md).
- Session memory: multi-turn conversation within one session_id correctly
  recalled specific prior content ("three wishes," "trifles") when asked
  "what did I just ask you?".
- Session isolation: a fresh session with unrelated content ("I have a
  pet turtle named Steve") did not leak into a separate, simultaneously
  active fresh session — confirming concurrent VR users won't share
  conversation state. (First isolation test attempt was inconclusive —
  both sessions coincidentally involved Genie's own "wishes" persona
  theme, which the LLM could plausibly improvise on its own; redone with
  unrelated content for an unambiguous result.)
- File delivery: both `voice_audio_url` and `talking_video_url` played
  correctly when opened directly in-browser.
## Part — vr_service.py streaming endpoint: latency investigation

After adding the SSE streaming endpoint (`/v1/vr-chat-stream`, Part 1.2 of
Guide 4), first-audio latency measured noticeably higher than Part 3's
~1.3s average. Investigated with real server-side per-stage timing
(`TIMING stt=`, `TIMING first_llm_token=`, `TIMING first_sentence_ready=`
logged directly in the endpoint) rather than trusting client-observed event
arrival times, which bundle multiple stages together.

**Bug found and fixed: TTS was making a live network call on every request.**
Kokoro's voice loading issued a `HEAD` request to huggingface.co to check
for a fresher version of the voice file, even though it was already cached
locally — adding a real network round-trip inside the TTS hot path.

| | Before | After |
|---|---|---|
| TTS stage (first sentence) | ~1.47s | ~0.46s |

Fix: `HF_HUB_OFFLINE=1` is already set in `src/config.py`, and is correctly
inherited by `vr_service.py` transitively (it imports `src.llm`, which
imports `src.config`) — no code change needed, the slow run was an
already-in-flight request from before the env var took effect in that
process's lifetime. Confirmed fixed by the absence of the `huggingface.co`
HEAD request in subsequent warm-run server logs.

**Remaining number, measured not assumed: Ollama's first-token latency is
consistently ~2.5-3s on warm requests** (2.47s, 3.29s, 2.95s across
multiple runs), independent of STT/TTS. Confirmed this isn't a
`vr_service.py`-specific regression: `vr_service.py` calls the exact same
`stream_character_reply()` function from `src/llm.py` that `app.py`
already uses — no divergence in how the LLM is invoked. This appears to be
an inherent property of Ollama serving `llama3.1:8b` with this
system-prompt length on this hardware, not something introduced by the
streaming endpoint.

**Decision: not chased further.** Per the project's own established rule
(don't chase the last second once a number is real and understood), this
is logged as the honest current baseline rather than a bug to keep pursuing.
If revisited later, a smaller model (`llama3.2:3b`, already flagged as an
optional experiment in Part 3 of the original Next Steps Guide) or Ollama's
`keep_alive`/context-caching behavior would be the first things to check.

**Current warm-request breakdown, `/v1/vr-chat-stream`:**

| Stage | Time |
|---|---|
| STT | ~0.8-1.1s |
| LLM first token | ~2.5-3.0s |
| TTS (first sentence, post-fix) | ~0.5s |
| **First audio, total** | **~4-5s** |
| Full video | ~12-13s |
