# QuantLlama

**Scalable LLM Deployment: Benchmarking 4-Bit Quantization and QLoRA Fine-Tuning**

CSCI 4900/6900 — Foundations of Deep Learning and Generative AI · UGA Spring 2026 · Dr. Geng Yuan

---

## Overview

This project investigates how large language models can be compressed to run on consumer hardware
without meaningfully sacrificing quality. It benchmarks post-training quantization (NF4, GPTQ) on
`meta-llama/Llama-3.1-8B` and combines it with QLoRA fine-tuning on the Stanford Alpaca dataset.

**Model:** `meta-llama/Llama-3.1-8B` · **Hardware:** NVIDIA RTX 3090 (24 GB VRAM)

---

## Results

| Config | Precision | Perplexity ↓ | Speed (tok/s) ↑ | VRAM (GB) ↓ | Load (s) |
|---|---|---|---|---|---|
| fp16 baseline | fp16 | 6.1885 | 36.26 | 18.984 | 256.5 |
| int4 NF4 (bitsandbytes) | int4-nf4 | 6.5784 | 18.01 | 9.425 | 116.0 |
| int4 GPTQ (ModelCloud) | int4-gptq | 6.5497 | 4.35 | 9.453 | 40.9 |
| **QLoRA fine-tuned (Alpaca)** | int4-nf4-qlora | **6.6251** | **53.39** | **7.504** | 96.1 |

Key findings:
- **~50% VRAM reduction** from fp16 → int4 (18.98 GB → 9.4 GB)
- **Minimal perplexity degradation** — less than 0.4 points at int4 vs fp16
- **QLoRA fine-tuned model is fastest and most memory-efficient** after `merge_and_unload()` — 53.39 tok/s and only 7.5 GB VRAM
- Only **0.085% of parameters trained** during QLoRA (6.8M / 8.03B), producing a 44.5 MB adapter

---

## Links

| Resource | URL |
|---|---|
| 🤗 Fine-tuned model (HF Hub) | https://huggingface.co/srijayadav/llama3-8b-qlora-alpaca |
| 🚀 HF Space (demo code) | https://huggingface.co/spaces/srijayadav/quantllama-demo |
| 📊 Live Gradio demo | https://f27035229341fdab36.gradio.live *(expires 1 week from April 13, 2026)* |

---

## Repo Structure
```
QuantLlama/
├── setup/
│   ├── install.sh              # PyTorch + library install commands
│   └── validate_env.py         # Checks all imports + GPU
├── phase1_baseline/
│   └── benchmark.py            # fp16 baseline benchmark
├── phase2_quantization/
│   ├── quantize_nf4.py         # NF4 bitsandbytes quantization + benchmark
│   ├── quantize_gptq.py        # GPTQ quantization + benchmark
│   └── compare.py              # Side-by-side comparison table
├── phase3_qlora/
│   ├── finetune.py             # QLoRA fine-tuning with SFTTrainer
│   └── evaluate.py             # Perplexity + qualitative eval
├── phase4_demo/
│   ├── push_to_hub.py          # Upload adapter weights to HF Hub
│   ├── app.py                  # Gradio side-by-side demo
│   └── requirements.txt        # HF Space dependencies
├── results/
│   ├── results.csv             # Full 4-row benchmark table
│   ├── qualitative_phase3.csv  # 20-prompt side-by-side outputs
│   └── plots/                  # Bar charts (perplexity, speed, VRAM)
├── plot_results.py             # Generates benchmark bar charts
├── requirements.txt            # Full environment dependencies
├── PHASE1_SUMMARY.md
├── PHASE2_SUMMARY.md
├── PHASE3_SUMMARY.md
└── PHASE4_SUMMARY.md
```
---

## How to Reproduce

### Prerequisites
- Python 3.11
- CUDA 12.x
- GPU with at least 12 GB VRAM (24 GB recommended for fp16 baseline)
- Hugging Face account with access to `meta-llama/Llama-3.1-8B`

### Setup

```bash
git clone https://github.com/SrijaVuppula/QuantLlama.git
cd QuantLlama

python3.11 -m venv QuantLlama
source QuantLlama/bin/activate

pip install -r requirements.txt
python setup/validate_env.py   # should print all PASS
```

### Run each phase

```bash
# Phase 1 — fp16 baseline
python phase1_baseline/benchmark.py

# Phase 2 — quantization benchmarks
python phase2_quantization/quantize_nf4.py
python phase2_quantization/quantize_gptq.py

# Phase 3 — QLoRA fine-tuning (requires HF token with Llama access)
export HF_TOKEN=hf_...
python phase3_qlora/finetune.py
python phase3_qlora/evaluate.py

# Phase 4 — push to HF Hub and run demo
python phase4_demo/push_to_hub.py
python phase4_demo/app.py
```

### Load the fine-tuned model from HF Hub

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained("srijayadav/llama3-8b-qlora-alpaca")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    quantization_config=bnb_config,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, "srijayadav/llama3-8b-qlora-alpaca")
```

---

## References

- Dettmers et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS 2023. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- Frantar et al. (2022). GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)
- Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Taori et al. (2023). Alpaca: A Strong, Replicable Instruction-Following Model. Stanford CRFM.
