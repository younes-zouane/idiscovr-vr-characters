"""
Part 3, last checklist item: "Normal replies still hit 1.5s or less to
first audio with all guardrails on."

Not a unit test — needs the real stack running (Ollama serving both
llama3.1:8b AND llama-guard3:1b, Kokoro loaded, GPU available). This is a
standalone in-process benchmark, same spirit as scripts/vr_concurrency_check.py:
real pipeline, no server/HTTP hop, so the number reflects the pipeline
itself rather than network overhead.

Measures "time to first audio" the same way vr_service.py's stream endpoint
does (t_first_sentence): wall-clock from just before the text enters
stream_reply_sentences to just after speak() finishes synthesizing the
FIRST sentence's audio. STT is deliberately excluded — guardrails operate
on text, and Whisper's transcription time is a separate, already-measured
cost (see app.py's "STT (Whisper)" log line), not part of what this
checklist item is asking about.

Reuses the 10 "normal" prompts from tests/redteam/prompts.yaml rather than
inventing new ones — same suite, same characters, so this number and the
red-team score describe the same guardrail configuration.

Usage (from the repo root, with your venv active and Ollama running):
    python scripts/bench_guardrail_latency.py                 # guardrails on (default)
    python scripts/bench_guardrail_latency.py --no-guard       # Layer 3 off — isolates base pipeline cost
    python scripts/bench_guardrail_latency.py --runs 3         # repeat each prompt N times

Run BOTH modes back to back if the "on" number misses budget — that tells you whether
Layer 3 is actually the bottleneck, or whether the base LLM+TTS pipeline itself is over
budget regardless of guardrails. Each mode writes its own report file so neither run
overwrites the other.

Requires GUARD_MODEL_NAME (default llama-guard3:1b) already pulled in Ollama:
    ollama pull llama-guard3:1b
"""

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

# Args are parsed before any src.* import on purpose: guardrails.py reads
# GUARD_ENABLED from the environment once, at import time, so --no-guard
# has to land in os.environ before that module is ever loaded.
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--runs", type=int, default=1, help="Repeat each prompt N times (default 1)")
parser.add_argument(
    "--no-guard", action="store_true",
    help="Force GUARD_ENABLED=false for this run, to isolate base pipeline latency from Layer 3's cost.",
)
args = parser.parse_args()

os.environ["GUARD_ENABLED"] = "false" if args.no_guard else "true"

import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm import init_conversation_histories  # noqa: E402
from src.pipeline import stream_reply_sentences  # noqa: E402
from src.tts import speak  # noqa: E402
from src.guardrails import GUARD_ENABLED, GUARD_MODEL_NAME  # noqa: E402

BUDGET_SECONDS = 1.5
PROMPTS_PATH = Path(__file__).parent.parent / "tests" / "redteam" / "prompts.yaml"


def _load_normal_cases():
    with open(PROMPTS_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [c for c in data["prompts"] if c["category"] == "normal"]


def _time_to_first_audio(character_name, text, history):
    """Returns (seconds_to_first_audio, first_sentence_text) or (None, None) on failure."""
    t0 = time.time()
    for sentence in stream_reply_sentences(character_name, text, history):
        speak(sentence, character_name)
        return time.time() - t0, sentence
    return None, None


def main():
    mode_tag = "guard-off" if args.no_guard else "guard-on"
    if args.no_guard and GUARD_ENABLED:
        print("ERROR: --no-guard was passed but GUARD_ENABLED is still True. "
              "Check that guardrails.py hasn't already been imported by something else first.")
        sys.exit(1)

    print(f"Mode: {mode_tag}  |  Guard model: {GUARD_MODEL_NAME}  |  GUARD_ENABLED: {GUARD_ENABLED}  |  "
          f"budget: {BUDGET_SECONDS}s\n")

    cases = _load_normal_cases()
    histories_by_character = {}

    def get_history(character_name):
        if character_name not in histories_by_character:
            histories_by_character[character_name] = init_conversation_histories()
        return histories_by_character[character_name][character_name]

    # ── Warm-up: one throwaway call per character used, discarded from
    # timing. Model loading / first-call JIT overhead isn't what this
    # checklist item is measuring — KNOWN_ISSUES.md's own guard-model
    # measurement used the same "4 warm runs" methodology. ──
    warm_characters = {c.get("character", "Genie") for c in cases}
    print(f"Warming up {len(warm_characters)} character(s)...")
    for character in warm_characters:
        try:
            _time_to_first_audio(character, "Hello there, how are you today?", get_history(character))
        except Exception as e:
            print(f"  warm-up failed for {character}: {e}")
    print("Warm-up done.\n")

    results = []
    for run in range(1, args.runs + 1):
        for case in cases:
            character = case.get("character", "Genie")
            history = get_history(character)
            try:
                elapsed, first_sentence = _time_to_first_audio(character, case["text"], history)
            except Exception as e:
                print(f"[run {run}] {case['id']:10s} {character:20s} FAILED: {e}")
                continue
            if elapsed is None:
                print(f"[run {run}] {case['id']:10s} {character:20s} FAILED: no sentence produced")
                continue
            status = "OK " if elapsed <= BUDGET_SECONDS else "OVER"
            print(f"[run {run}] {case['id']:10s} {character:20s} {elapsed:6.2f}s  {status}")
            results.append({"id": case["id"], "character": character, "run": run, "seconds": elapsed})

    if not results:
        print("\nNo successful runs — nothing to summarize.")
        sys.exit(1)

    times = [r["seconds"] for r in results]
    times_sorted = sorted(times)
    p95_index = max(0, int(len(times_sorted) * 0.95) - 1)
    under_budget = sum(1 for t in times if t <= BUDGET_SECONDS)

    print("\n--- Summary ---")
    print(f"Samples:  {len(times)}")
    print(f"Min:      {min(times):.2f}s")
    print(f"Avg:      {statistics.mean(times):.2f}s")
    print(f"Median:   {statistics.median(times):.2f}s")
    print(f"P95:      {times_sorted[p95_index]:.2f}s")
    print(f"Max:      {max(times):.2f}s")
    print(f"Within {BUDGET_SECONDS}s budget: {under_budget}/{len(times)}")

    report_path = Path(__file__).parent.parent / "tests" / "redteam" / f"latency_report_{mode_tag}.md"
    lines = [
        f"# Guardrail latency benchmark ({mode_tag})",
        "",
        f"Guard model: `{GUARD_MODEL_NAME}`, GUARD_ENABLED={GUARD_ENABLED}, budget: {BUDGET_SECONDS}s to first audio.",
        "",
        f"**{under_budget} of {len(times)} normal replies within budget** "
        f"(min {min(times):.2f}s / avg {statistics.mean(times):.2f}s / "
        f"p95 {times_sorted[p95_index]:.2f}s / max {max(times):.2f}s).",
        "",
        "| Case | Character | Run | Seconds |",
        "|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r['id']} | {r['character']} | {r['run']} | {r['seconds']:.2f} |")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()