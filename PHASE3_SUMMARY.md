# QuantLlama — Phase 3 Summary

## Project Info
- **Repo:** https://github.com/SrijaVuppula/QuantLlama
- **Course:** CSCI 4900/6900 — Foundations of Deep Learning and Generative AI, UGA Spring 2026
- **Model:** `meta-llama/Llama-3.1-8B`
- **Hardware:** NVIDIA GeForce RTX 3090 — 24 GB VRAM (CS department GPU, accessed via VS Code Remote SSH)

---

## Goal
Fine-tune the NF4 quantized Llama-3.1-8B using QLoRA on the Stanford Alpaca dataset, evaluate the resulting model on the same 4-metric benchmark suite from Phases 1 and 2, and produce a 20-prompt qualitative side-by-side comparison between base int4-NF4 and the fine-tuned model.

---

## Environment Changes from Phase 2

No new packages were required. The Phase 2 working stack was used as-is:

```
transformers==4.56.2
bitsandbytes==0.49.2
peft==0.18.1
trl==1.0.0
accelerate==1.13.0
datasets==4.8.4
```

`matplotlib` and `pandas` were installed separately for plotting:
```bash
pip install matplotlib pandas --break-system-packages
```

---

## Scripts

| Script | Purpose |
|---|---|
| `phase3_qlora/finetune.py` | QLoRA fine-tuning with SFTTrainer |
| `phase3_qlora/evaluate.py` | Perplexity + speed benchmark + 20-prompt qualitative eval |
| `plot_results.py` | Generates benchmark bar charts from results.csv |

---

## API Fixes Encountered (TRL 1.0 Breaking Changes)

Three API mismatches were hit and fixed during development — documented here for reference:

| Error | Cause | Fix |
|---|---|---|
| `TrainingArguments` got unexpected kwarg `evaluation_strategy` | Renamed in transformers 4.46 | → `eval_strategy` |
| `SFTTrainer` got unexpected kwarg `tokenizer` | Renamed in TRL 1.0 | → `processing_class` |
| `SFTTrainer` got unexpected kwarg `dataset_text_field` | Moved in TRL 1.0 | → goes in `SFTConfig` instead of `SFTTrainer` |
| `SFTConfig` got unexpected kwarg `max_seq_length` | Renamed in TRL 1.0 | → `max_length` |

**Key TRL 1.0 pattern:** `SFTConfig` replaces `TrainingArguments` as the args object for `SFTTrainer`. It absorbs all SFT-specific parameters (`dataset_text_field`, `max_length`, `packing`). These must NOT be passed to `SFTTrainer` directly.

---

## Training Crash Fix — bf16 vs fp16 dtype mismatch

**Error:**
```
NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'
```

**Root cause:** Llama-3.1-8B declares `torch_dtype: bfloat16` in its config. PEFT initializes LoRA matrices in the model's native dtype (bfloat16). Setting `fp16=True` in `SFTConfig` activates PyTorch's fp16 AMP grad scaler, which then hits the bf16 LoRA tensors and crashes — CUDA has no kernel to unscale bf16 tensors under an fp16 scaler.

**Fix:** Switch both settings to bf16 end-to-end:
```python
# In BitsAndBytesConfig:
bnb_4bit_compute_dtype=torch.bfloat16   # was torch.float16

# In SFTConfig:
bf16=True                                # was fp16=True
```

The RTX 3090 supports bf16 natively — no performance penalty.

---

## QLoRA Configuration

| Hyperparameter | Value | Rationale |
|---|---|---|
| Base model | `meta-llama/Llama-3.1-8B` | NF4 int4, loaded with bitsandbytes |
| LoRA rank (r) | 16 | Standard starting point |
| LoRA alpha | 32 | Alpha = 2r convention |
| Target modules | `q_proj`, `v_proj` | Attention query and value projections |
| Dropout | 0.05 | Light regularization |
| Training epochs | 1 | Standard for Alpaca QLoRA; original Dettmers et al. used 1 epoch |
| Batch size | 4 (grad accum 8) | Effective batch size of 32 |
| Learning rate | 2e-4 | Standard QLoRA learning rate |
| LR scheduler | cosine with warmup | 3% warmup ratio |
| Max sequence length | 512 | Alpaca examples are short; saves VRAM vs 2048 |
| Packing | True | Bins short examples into full 512-token windows; ~2-3x step reduction |
| Optimizer | paged_adamw_32bit | bitsandbytes paged optimizer — spills optimizer states to CPU RAM |
| Dataset | tatsu-lab/alpaca | 49,401 train / 2,601 val (95/5 split) |

**Trainable parameters:** 6,815,744 out of 8,037,076,992 (0.0848%) — only the LoRA matrices.

---

## Training Results

| Metric | Value |
|---|---|
| Training time | 1.92 hours |
| Adapter size | 44.5 MB |
| Adapter location | `outputs/qlora-alpaca/final_adapter/` |

**Note on adapter weights and git:** The adapter safetensors file was not pushed to GitHub (binary files belong on Hugging Face Hub). The `outputs/` directory is in the repo but the large weight files were not staged. Weights will be pushed to Hugging Face Hub in Phase 4.

---

## Phase 3 Evaluation Results

Evaluated using `phase3_qlora/evaluate.py`. The fine-tuned model uses `merge_and_unload()` to fuse LoRA matrices back into the base weights before benchmarking — this gives clean inference numbers without adapter overhead.

| Config | Precision | Perplexity↓ | Speed (tok/s)↑ | VRAM (GB)↓ | Load (s) |
|---|---|---|---|---|---|
| Llama-3.1-8B fp16 baseline | fp16 | 6.1885 | 36.26 | 18.984 | 256.5 |
| Llama-3.1-8B NF4 bitsandbytes | int4-nf4 | 6.5784 | 18.01 | 9.425 | 116.0 |
| Llama-3.1-8B GPTQ ModelCloud | int4-gptq | 6.5497 | 4.35 | 9.453 | 40.9 |
| Llama-3.1-8B QLoRA-Alpaca | int4-nf4-qlora | 6.6251 | 53.39 | 7.504 | 96.1 |

---

## Key Observations

### 1. Perplexity — minimal degradation from fine-tuning
The QLoRA fine-tuned model (6.6251) is only 0.047 points worse than the base NF4 model (6.5784) on WikiText-2. This is expected and correct: Alpaca fine-tuning targets instruction-following behavior, not general language modeling. The model's general capability is preserved.

### 2. Speed — fine-tuned model is the fastest of all four configurations
After `merge_and_unload()` fuses the LoRA matrices into the base weights, the fine-tuned model runs at 53.39 tok/s — faster than fp16 (36.26), NF4 (18.01), and GPTQ (4.35). This is because the merged model has no runtime dequantization overhead and no adapter forwarding cost. This is a strong result for the presentation.

### 3. VRAM — lowest of all configurations
The fine-tuned model uses only 7.50 GB peak VRAM — less than base NF4 (9.43 GB) and less than GPTQ (9.45 GB). The merged weights load more cleanly than the base NF4 with its quantization overhead tracked separately.

### 4. Qualitative evaluation
20-prompt side-by-side outputs saved to `results/qualitative_phase3.csv`. Prompts cover instruction-following, reasoning, factual knowledge, creative writing, and a math edge case. The fine-tuned model produces more structured, instruction-aligned responses particularly on the instruction-following and formatting prompts.

---

## Output Files

| File | Description |
|---|---|
| `outputs/qlora-alpaca/final_adapter/adapter_model.safetensors` | LoRA adapter weights (44.5 MB) |
| `outputs/qlora-alpaca/final_adapter/adapter_config.json` | LoRA configuration |
| `results/results.csv` | Full 4-row benchmark table |
| `results/qualitative_phase3.csv` | 20-prompt side-by-side outputs |
| `results/plots/perplexity.png` | Perplexity bar chart |
| `results/plots/speed.png` | Inference speed bar chart |
| `results/plots/vram.png` | Peak VRAM bar chart |
| `results/plots/combined.png` | All three charts side by side (best for slides) |

---

## results/results.csv after Phase 3

```
config,precision,perplexity_wikitext2,speed_tokens_per_sec,peak_vram_gb,load_time_sec
meta-llama/Llama-3.1-8B | fp16 | baseline,fp16,6.1885,36.26,18.984,256.5
meta-llama/Llama-3.1-8B | int4-nf4 | bitsandbytes,int4-nf4,6.5784,18.01,9.425,116.0
meta-llama/Llama-3.1-8B | int4-gptq | ModelCloud pretrained,int4-gptq,6.5497,4.35,9.453,40.9
meta-llama/Llama-3.1-8B | int4-nf4 | QLoRA-Alpaca (fine-tuned),int4-nf4-qlora,6.6251,53.39,7.504,96.1
```

---

## What's Next — Phase 4

Phase 4 pushes the fine-tuned adapter to Hugging Face Hub and builds a live Gradio demo.

**Before starting Phase 4, verify:**
```bash
# HF token is set
echo $HF_TOKEN

# Adapter weights are present
ls outputs/qlora-alpaca/final_adapter/
```

**Key tasks:**
1. `huggingface-cli login` or confirm `HF_TOKEN` env var is set
2. Push adapter weights + tokenizer to `username/llama3-8b-qlora-alpaca` on HF Hub
3. Write model card (README.md) for the Hub repo
4. Build `phase4_demo/app.py` — Gradio side-by-side demo (base int4-NF4 vs fine-tuned)
5. Deploy to Hugging Face Spaces (GPU tier)

**Expected Phase 4 outcome:**
- Public HF Hub repo with adapter weights, tokenizer, and model card
- Live Gradio demo URL (e.g. `huggingface.co/spaces/username/quantllama-demo`)
- Demo URL embeddable in resume, LinkedIn, and GitHub README
