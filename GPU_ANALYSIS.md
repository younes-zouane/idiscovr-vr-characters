# GPU Sizing & Performance Analysis
**Hardware:** NVIDIA RTX 5060 Ti — 16,311 MB VRAM  
**Date:** July 2026  
**Project:** Talking AI Characters — iDISCOVR VR Internship

---

## 1. Hardware Overview

| Spec | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Ti |
| Total VRAM | 16,311 MB (~16 GB) |
| CUDA Version | 13.2 |
| Driver | 595.97 |
| Max Power | 192W |
| Idle Temperature | 33°C |

---

## 2. Model Specifications

### Brain — llama3.1:8b (selected model)
| Spec | Value |
|---|---|
| Parameters | 8.0 Billion |
| Quantization | Q4_K_M (4-bit) |
| Context window | 131,072 tokens |
| VRAM usage | ~5,000 MB |
| Embedding length | 4,096 |

### Brain — llama3.2:3b (alternative tested)
| Spec | Value |
|---|---|
| Parameters | 3.2 Billion |
| Quantization | Q4_K_M (4-bit) |
| Context window | 131,072 tokens |
| VRAM usage | ~2,000 MB |
| Embedding length | 3,072 |

### Ears — faster-whisper (medium)
| Spec | Value |
|---|---|
| Model size | medium |
| Device | CUDA (GPU) |
| Compute type | float16 |
| VRAM usage | ~1,500 MB |

### Voice — Piper TTS
| Spec | Value |
|---|---|
| Runs on | CPU (no VRAM cost) |
| Per-character models | 6 distinct .onnx voice files |

---

## 3. VRAM Budget
RTX 5060 Ti — 16,311 MB total
├── llama3.1:8b (Q4_K_M)      ~5,000 MB
├── faster-whisper medium      ~1,500 MB
├── Piper TTS                  ~    0 MB  (CPU)
├── KV cache (conversation)    ~  500 MB
├── System/Windows overhead    ~  890 MB
└── FREE headroom              ~8,421 MB
─────────────────────────────────────────
Total used (1 active user):    ~7,890 MB
Utilization:                       48%

**Conclusion:** The system uses approximately half the available VRAM, 
leaving 8+ GB of headroom for additional models, larger context, 
or concurrent users.

---

## 4. Latency Benchmark (measured)

### llama3.1:8b — tested results

| Call | STT | LLM | TTS | Total |
|---|---|---|---|---|
| Cold start (1st) | 1.55s | 43.57s | 2.74s | 47.85s |
| Warm (2nd) | 0.50s | 0.69s | 1.59s | **2.78s** |

### llama3.2:3b — tested results

| Call | STT | LLM | TTS | Total |
|---|---|---|---|---|
| Cold start (1st) | 0.75s | 6.44s | 1.73s | 8.93s |
| Warm (2nd) | 0.66s | 2.18s | 1.55s | 4.39s |
| Warm (3rd) | 0.66s | 2.26s | 1.60s | 4.52s |

### Key finding
Despite having fewer parameters, llama3.2:3b is **slower** on warm 
inference than llama3.1:8b (4.45s vs 2.78s). The 8B model is better 
optimized at Q4_K_M quantization for this hardware.

**Selected model: llama3.1:8b** — better character quality AND 
faster warm inference.

### Cold start explanation
The 43s first call for 8B (and 6.4s for 3B) is a one-time cost per 
session: Ollama loads model weights from disk into VRAM. All subsequent 
calls in the same session are warm. In a real deployment, the app would 
be kept running permanently, making cold start irrelevant in practice.

---

## 5. Context Window & Memory Implications

The llama3.1:8b model has a **131,072 token context window**.

| Metric | Value |
|---|---|
| Context limit | 131,072 tokens |
| Average tokens per turn | ~150 tokens |
| Max conversation turns before limit | ~873 turns |
| Typical VR session length | ~20-50 turns |
| Risk of hitting context limit | Essentially zero |

Each character maintains its own independent conversation history 
(separate Python list per character). This means a visitor switching 
between characters doesn't consume another character's context budget.

---

## 6. Concurrency Analysis

Ollama processes requests **sequentially by default** (one at a time). 
With 8+ GB of free VRAM headroom, the limiting factor is not memory 
but rather queue depth.

| Concurrent visitors | Behavior | User experience |
|---|---|---|
| 1 | Instant processing | ~2.78s response |
| 2-3 | Second request queues briefly | ~3-6s response |
| 4-5 | Noticeable queue | ~6-12s response |
| 6+ | Long queue | Degraded experience |

### Options to improve concurrency if needed
1. **Enable Ollama parallel mode**: set `OLLAMA_NUM_PARALLEL=2` 
   environment variable — allows 2 simultaneous LLM requests using 
   more VRAM (feasible with 8GB headroom)
2. **Switch to 3B model**: saves 3GB VRAM, same sequential behavior 
   but smaller queue times per request
3. **Multiple GPU instances**: for a full museum deployment with 10+ 
   simultaneous visitors, a second GPU would be needed

---

## 7. Recommendations

| Priority | Recommendation | Reason |
|---|---|---|
| ✅ Current | Use llama3.1:8b | Best quality + fastest warm inference |
| ✅ Current | Keep faster-whisper medium | Good accuracy, acceptable speed |
| 🔧 Optional | Enable OLLAMA_NUM_PARALLEL=2 | Improves 2-3 user concurrency |
| 🔧 Optional | Pre-warm on startup | Eliminate cold start for live demos |
| 🚀 Future | Second GPU for production | Handle 10+ museum visitors |

---

## 8. Pre-warming Fix (eliminate cold start)

Add this to `app.py` at startup to pre-warm the model before 
any visitor uses it:

```python
def prewarm_model():
    print("Pre-warming LLM model...")
    client.chat.completions.create(
        model=MODEL,
        max_tokens=1,
        messages=[{"role": "user", "content": "hi"}]
    )
    print("Model warm and ready.")

prewarm_model()  # call this once at startup, before demo.launch()
```

This sends one silent dummy request at startup, loading the model 
into VRAM so the first real visitor gets a ~2.78s response instead 
of 43s.