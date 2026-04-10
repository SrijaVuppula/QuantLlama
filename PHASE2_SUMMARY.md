# QuantLlama — Phase 2 Summary

## Project Info
- **Repo:** https://github.com/SrijaVuppula/QuantLlama
- **Course:** CSCI 4900/6900 — Foundations of Deep Learning and Generative AI, UGA Spring 2026
- **Model:** `meta-llama/Llama-3.1-8B`
- **Hardware:** NVIDIA GeForce RTX 3090 — 24 GB VRAM (CS department GPU, accessed via VS Code Remote SSH)

---

## Goal
Run the same 4-metric benchmark suite from Phase 1 on two int4 quantized configurations of Llama-3.1-8B, and append results to `results/results.csv` for a clean 3-row comparison table.

---

## Environment Changes from Phase 1

| Package | Phase 1 | Phase 2 |
|---|---|---|
| transformers | 5.5.3 | 4.56.2 (downgraded for auto-gptq compatibility) |
| optimum | — | 1.16.2 (installed, ultimately not used) |
| auto-gptq | — | 0.7.1 |

**Note:** `trl 1.0.0` requires `transformers >= 4.56.2`. The current version (4.56.2) satisfies both auto-gptq and trl. No further changes needed before Phase 3.

---

## Configuration 1 — int4 NF4 (bitsandbytes)

**Script:** `phase2_quantization/quantize_nf4.py`

**Method:** Post-training quantization using bitsandbytes `BitsAndBytesConfig`:
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
```
- NF4 (Normal Float 4) is information-theoretically optimal for normally distributed weights
- Double quantization applies a second quantization to the quantization constants, saving an additional ~0.4 bits per parameter
- No calibration data required — applied directly to pretrained weights

**Results:**
| Metric | Value |
|---|---|
| Perplexity (WikiText-2) | 6.5784 |
| Inference speed | 18.01 tok/s |
| Peak VRAM | 9.425 GB |
| Load time | 116.0s |

---

## Configuration 2 — int4 GPTQ (auto-gptq, pretrained checkpoint)

**Script:** `phase2_quantization/quantize_gptq.py`

**Method:** GPTQ (Generative Pre-trained Transformer Quantization) is a calibration-based int4 method that uses a small sample of text to compute optimal quantization parameters layer by layer, minimizing the reconstruction error of each layer's output.

**Why a pretrained checkpoint was used:**
GPTQ calibration requires loading the full fp16 model (~16 GB) into CPU RAM. The department server has 15 GB total RAM, making self-calibration impossible on this hardware. A pretrained checkpoint (`ModelCloud/Meta-Llama-3.1-8B-gptq-4bit`) calibrated on WikiText-2 with `group_size=128` was used instead — identical methodology to what self-calibration would produce. This is consistent with standard practice in quantization research.

**Checkpoint:** `ModelCloud/Meta-Llama-3.1-8B-gptq-4bit`
- 4-bit GPTQ, group_size=128, calibrated on WikiText-2

**Results:**
| Metric | Value |
|---|---|
| Perplexity (WikiText-2) | 6.5497 |
| Inference speed | 4.35 tok/s |
| Peak VRAM | 9.453 GB |
| Load time | 40.9s |

---

## Full Phase 2 Benchmark Table

| Config | Precision | Perplexity↓ | Speed (tok/s)↑ | VRAM (GB)↓ | Load (s) |
|---|---|---|---|---|---|
| Llama-3.1-8B fp16 baseline | fp16 | 6.1885 | 36.26 | 18.984 | 256.5 |
| Llama-3.1-8B NF4 bitsandbytes | int4-nf4 | 6.5784 | 18.01 | 9.425 | 116.0 |
| Llama-3.1-8B GPTQ ModelCloud | int4-gptq | 6.5497 | 4.35 | 9.453 | 40.9 |

**Deltas vs fp16 baseline:**
| Config | Perplexity Δ | Speed ratio | VRAM reduction |
|---|---|---|---|
| int4-NF4 | +0.39 | ×0.50 | −50.3% |
| int4-GPTQ | +0.36 | ×0.12 | −50.2% |

---

## Key Observations

### 1. Perplexity — excellent retention at int4
Both int4 methods increase perplexity by less than 0.4 points vs the fp16 baseline (6.19 → ~6.55). This is significantly better than the 1–2 point degradation projected in the proposal, suggesting Llama-3.1-8B is robust to int4 quantization. GPTQ recovers a marginal 0.03 points over NF4 — consistent with its calibration-based design but negligible in practice.

### 2. VRAM — clean 50% reduction for both methods
Both int4 methods cut peak VRAM from 18.98 GB to ~9.4 GB, a 50% reduction. This is the primary practical benefit: an 8B model that required a 24 GB GPU in fp16 now fits comfortably on a 12 GB consumer card. This validates the core motivation for quantization.

### 3. Speed — counterintuitive results, hardware-dependent
Both int4 configurations are *slower* than fp16 (36.26 tok/s):
- **NF4 at 18 tok/s (×0.50):** bitsandbytes dequantizes weights to fp16 on every forward pass before matrix multiplication. On the RTX 3090's fast fp16 tensor cores with 24 GB VRAM, the dequantization overhead dominates and eliminates the expected throughput gain. Speed improvements from NF4 are more pronounced on memory-constrained GPUs (8–12 GB) where bandwidth, not compute, is the bottleneck.
- **GPTQ at 4.35 tok/s (×0.12):** auto-gptq's CUDA kernels were not compiled in the installed wheel (`CUDA extension not installed` warnings). The model runs on pure PyTorch fallback ops instead of optimized int4 CUDA kernels. In a properly compiled environment, GPTQ inference would be significantly faster. This is a software environment limitation, not a property of GPTQ itself.

### 4. Load time — GPTQ loads fastest
GPTQ loads in 40.9s vs 116.0s for NF4 and 256.5s for fp16. The quantized checkpoint is a single ~5.7 GB safetensors file vs 4 shards for the fp16 model, and the quantized weights are already in final format requiring no runtime conversion.

---

## Library Compatibility Issues Encountered

The GPTQ ecosystem has significant compatibility constraints worth documenting:

| Library | Issue | Resolution |
|---|---|---|
| `auto-gptq 0.7.1` | `no_init_weights` removed in transformers 5.x | Downgraded transformers to 4.56.2 |
| `gptqmodel 6.x` | Requires torch ≥ 2.7.1 but pip metadata detection fails | Abandoned |
| `gptqmodel 1.4.5` | Build fails: `No module named pip` inside venv | Abandoned |
| `optimum 2.1.0` | `is_tf_available` removed in transformers 5.x | Downgraded optimum to 1.16.2 |
| `optimum 1.16.2` | `GPTQQuantizer` requires `auto-gptq` as backend anyway | Used auto-gptq directly |
| Self-calibration | fp16 model requires ~16 GB CPU RAM; server has 15 GB | Used pretrained checkpoint |

**Current working stack:**
```
transformers==4.56.2
auto-gptq==0.7.1
bitsandbytes==0.49.2
optimum==1.16.2
trl==1.0.0
peft==0.18.1
```

---

## results/results.csv after Phase 2

```
config,precision,perplexity_wikitext2,speed_tokens_per_sec,peak_vram_gb,load_time_sec
meta-llama/Llama-3.1-8B | fp16 | baseline,fp16,6.1885,36.26,18.984,256.5
meta-llama/Llama-3.1-8B | int4-nf4 | bitsandbytes,int4-nf4,6.5784,18.01,9.425,116.0
meta-llama/Llama-3.1-8B | int4-gptq | ModelCloud pretrained,int4-gptq,6.5497,4.35,9.453,40.9
```

---

## What's Next — Phase 3

Phase 3 fine-tunes the NF4 quantized model using QLoRA on the Alpaca instruction-following dataset.

**Before starting Phase 3, verify trl works:**
```bash
python -c "from trl import SFTTrainer; print('trl OK')"
```

**Key scripts to write:**
- `phase3_qlora/finetune.py` — QLoRA fine-tuning with SFTTrainer
- `phase3_qlora/evaluate.py` — perplexity + qualitative eval of fine-tuned model

**Expected Phase 3 outcome:**
- Fine-tuned adapter weights: ~50–150 MB (LoRA matrices only)
- Perplexity on WikiText-2: similar to base int4-NF4 (~6.5–7.0)
- Qualitative improvement: more structured, instruction-aligned responses
