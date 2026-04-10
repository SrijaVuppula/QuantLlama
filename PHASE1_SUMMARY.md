# QuantLlama — Phase 1 Summary

## Project Info
- **Repo:** https://github.com/SrijaVuppula/QuantLlama
- **Course:** CSCI 4900/6900 — Foundations of Deep Learning and Generative AI, UGA Spring 2026
- **Model:** `meta-llama/Llama-3.1-8B` (base model, fp16)
- **Hardware:** NVIDIA GeForce RTX 3090 — 24 GB VRAM (CS department GPU, accessed via VS Code Remote SSH)

---

## Environment

| Component | Version |
|---|---|
| OS | Linux (RHEL-based, bash 4.4) |
| Python | 3.11.13 (at `/usr/bin/python3.11`) |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 (driver 13.0) |
| transformers | 5.5.3 |
| tokenizers | 0.22.2 |
| huggingface_hub | 1.10.1 |
| datasets | 4.8.4 |
| bitsandbytes | 0.49.2 |
| peft | 0.18.1 |
| trl | 1.0.0 |
| accelerate | 1.13.0 |
| gradio | 6.11.0 |

**Venv location:** `~/QuantLlama/QuantLlama/` (Python 3.11 virtualenv)  
**VS Code auto-activate:** configured via `.vscode/settings.json`

---

## Repo Structure

```
QuantLlama/
├── .gitignore
├── .vscode/
│   └── settings.json           # Auto-activates venv in VS Code terminal
├── README.md
├── requirements.txt
├── setup/
│   ├── install.sh              # PyTorch + all library install commands
│   └── validate_env.py         # Checks all imports + GPU — all PASS
├── phase1_baseline/
│   └── benchmark.py            # fp16 baseline benchmark script
├── phase2_quantization/
│   ├── quantize_nf4.py         # (empty — Phase 2)
│   ├── quantize_gptq.py        # (empty — Phase 2)
│   └── compare.py              # (empty — Phase 2)
├── phase3_qlora/
│   ├── finetune.py             # (empty — Phase 3)
│   └── evaluate.py             # (empty — Phase 3)
├── phase4_demo/
│   └── app.py                  # (empty — Phase 4)
└── results/
    ├── .gitkeep
    └── results.csv             # Phase 1 results written here
```

---

## Phase 1 Results — fp16 Baseline

| Metric | Value |
|---|---|
| **Perplexity (WikiText-2)** | **6.1885** |
| **Inference speed** | **36.26 tok/s** |
| **Peak VRAM** | **18.984 GB** |
| **Load time** | **256.5 sec** |

**Evaluation details:**
- 64 windows × 2048 tokens = 131,072 tokens total
- Greedy decoding (do_sample=False), 10 runs × 200 new tokens, 1 warm-up discarded
- Results saved to `results/results.csv`

**results/results.csv contents after Phase 1:**
```
config,precision,perplexity_wikitext2,speed_tokens_per_sec,peak_vram_gb,load_time_sec
meta-llama/Llama-3.1-8B | fp16 | baseline,fp16,6.1885,36.26,18.984,256.5
```

---

## Key Observations for Report

- **19 GB VRAM for fp16** — an 8B model in fp16 uses 75% of a 24 GB RTX 3090. On a consumer 8 GB GPU it wouldn't load at all. This is the core motivation for quantization.
- **Perplexity 6.19** — consistent with published Llama-3.1-8B numbers, validating the setup.
- **36 tok/s** — the speed baseline. int4 quantization (Phase 2) is expected to roughly double this.

---

## Model Access Notes

- `meta-llama/Llama-3.1-8B` — **approved** (used for all phases)
- `meta-llama/Meta-Llama-3-8B` — approved (not used)
- `meta-llama/Meta-Llama-3-8B-Instruct` — pending (will use for Phase 4 demo if approved)
- HF token name: `QuantLlama` (fine-grained, stored at `~/.cache/huggingface/token`)
- Model cached at: `~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/`

---

## What's Next — Phase 2

Phase 2 runs the same benchmark suite on two int4 quantized versions of the same model:

1. **NF4 (bitsandbytes)** — `phase2_quantization/quantize_nf4.py`
2. **GPTQ (AutoGPTQ)** — `phase2_quantization/quantize_gptq.py`

Both append rows to the same `results/results.csv`, enabling a clean 3-row comparison table.

Expected outcomes vs fp16 baseline:
- VRAM: ~55–60% reduction (from ~19 GB down to ~8–9 GB)
- Speed: ~1.5–2× improvement (from 36 tok/s up to ~55–70 tok/s)
- Perplexity: ~1–2 point increase (from 6.19 up to ~7–8)
