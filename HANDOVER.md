# HANDOVER.md

## Running it

**Native (Windows, venv311):**
1. `python -m venv venv311 && venv311\Scripts\activate`
2. `pip install -r requirements.txt`
3. In a separate terminal: `ollama serve`, then `ollama pull llama3.1:8b` (and `ollama pull llama-guard3:1b` if you plan to re-enable Layer 3 guardrails)
4. `python app.py` for the Gradio UI, or `python vr_service.py` for the headless REST API
5. Open `http://127.0.0.1:7860` (Gradio) or `http://127.0.0.1:8000/docs` (VR service Swagger UI)

**Docker:**
1. `docker compose up -d`
2. First run takes ~3 min (models download/bake into the image); a warm restart is ~18s
3. `docker compose logs -f app` — wait for "Model warm and ready"
4. Same URLs as native, through the mapped container ports
5. `docker compose down -v` for a full clean-state teardown (also drops the cached model volume)

*(Verify these against your actual `requirements.txt`/`docker-compose.yml` — reconstructed from OPTIMIZATION.md and KNOWN_ISSUES.md, not run directly.)*

## What's finished

- Full pipeline (STT → LLM → TTS → lip-sync) working end-to-end, GPU-accelerated natively and in Docker
- Sentence-level streaming: TTS starts on sentence 1 while the LLM is still generating sentence 2; SSE endpoint (`/v1/vr-chat-stream`) for VR clients
- Guardrails: Layer 1 (character prompt), Layer 2 (input pattern check), Layer 3 (guard-model check) implemented; red-team suite scores 41/50, published in README
- Per-session conversation memory and isolation in `vr_service.py`, with a cleanup sweep for stale sessions/output files
- Docker: GPU-accelerated inside the container, models pre-baked at build time, healthchecks between `ollama` and `app`

## What's not finished

- **1.5s time-to-first-audio target: missed.** Currently ~2.5–2.8s regardless of guardrails — this is the base LLM-streaming + TTS cost, not a guardrail problem (confirmed via `--no-guard` A/B). Root cause not yet isolated — the next investigation is splitting Ollama prompt-processing vs. generation vs. TTS.
- **Batched Wav2Lip export: blocked, not abandoned.** The source PyTorch checkpoint behind the deployed wide-channel 256px ONNX model was never obtained — the repo was originally set up from a pre-converted ONNX only. Two independently-sourced Wav2Lip checkpoints were checked; both have mismatched channel widths against this repo's model definition. Flagged to supervisor, unresolved.
- **Guardrail Layer 3: disabled by default, and the reason why is itself unresolved.** An early isolated measurement found ~2.2s/call (7x over its 300ms budget). A later end-to-end measurement in the real pipeline found ~0.15–0.18s/call — well within budget — and directly contradicts the first number. Not reconciled yet; see KNOWN_ISSUES.md for the full timeline before trusting either number blindly.
- **Language claim:** fixed as of this handover (see Known Limits below) — the README no longer claims TTS auto-detects and replies in the spoken language.

## The four animation models evaluated

This is the most valuable section here — it saves the next person a month of re-treading this ground.

1. **Wav2Lip** (current production choice) — ~7-9s video gen, GPU-accelerated, ~9.7GB combined VRAM (61% of a 16GB card). A further speedup (batched export) is blocked on a missing source checkpoint — see "What's not finished" above. Full trace: `OPTIMIZATION.md`, Part 4.
2. **MuseTalk** — closed early; per the original project guide, this environment's CUDA kernels don't support the deployed GPU. *Flagging honestly: I could not find a dedicated write-up for this in the repo* — the only trace is one passing reference to a "quality gate established for MuseTalk" inside the LivePortrait section of `OPTIMIZATION.md`. If a separate note exists elsewhere, link it here; if not, that's itself worth knowing before the next person assumes this was benchmarked the same way the other three were.
3. **LatentSync** — visual quality passed (supervisor-confirmed on Genie/Aladdin), but 15-24x slower than Wav2Lip (125-175s vs. 7-9s) and pushes combined GPU memory to 97.6% of a 16GB card (vs. Wav2Lip's 61%) — and under that VRAM pressure, gets ~3x slower again (368s for the inference step alone). Full numbers: `OPTIMIZATION.md`, Part 5.
4. **LivePortrait + JoyVASA** — the most visually ambitious option (full facial animation — head tilt, blinks, expression — not just lip region), but fails on stylized art: its human-facial-landmark-based motion vectors warp Genie's 2D artwork into visible distortion. Worth revisiting only if the project ever pivots to photorealistic human avatars. Full writeup: `OPTIMIZATION.md`.

## Known limits

- **One user at a time.** `vr_service.py` holds a single GPU semaphore; a concurrent request gets a fast `429 busy` instead of queueing behind degraded shared-GPU performance.
- **English voices only.** Whisper detects the spoken language, but Kokoro TTS always replies in American English — this was a false claim in the README, now corrected (see the language-fix commit). **Kokoro has no Arabic support at all**, which is a real gap for a project set in the world of the *One Thousand and One Nights* — a product decision for the supervisor, not something to quietly route around.
- **Video generation:** ~7-9s steady-state (native, GPU); first request after a cold start is closer to 9-15s (STT/Whisper's own first-inference warmup).
- **Research-only components:** Wav2Lip and insightface are both licensed research-only — already documented in `NOTICE.md`.
- **Guardrail Layer 3 disabled by default** — see "What's not finished" above for why that decision itself isn't fully settled.

## Where to look for more detail

- `OPTIMIZATION.md` — full latency history and all four model evaluations
- `KNOWN_ISSUES.md` — CUDA/cuDNN fixes, Docker fixes, the guardrail-latency investigation
- `NOTICE.md` — research-only license notices
- `tests/redteam/` — guardrail red-team suite (41/50) and report generation
