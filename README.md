cat > ~/QuantLlama/README.md << 'README'
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

This section walks through the entire project from scratch. Each step includes the expected output
so you can verify things are working before moving on.

---

### Step 1 — Prerequisites

Before cloning the repo, confirm you have the following:

**Hardware**
- A CUDA-capable NVIDIA GPU
- At minimum 12 GB VRAM for int4 experiments
- 24 GB VRAM recommended if you want to run the fp16 baseline (the RTX 3090 was used here)
- To check your GPU and VRAM:
```bash
nvidia-smi
```
You should see your GPU listed with its memory. Example output:
```
Cloning into 'QuantLlama'...
remote: Enumerating objects: ...
```
---

### Step 3 — Create and activate a virtual environment

We use a Python 3.11 virtualenv to isolate all dependencies. The venv is created inside the repo
directory and named `QuantLlama` (same as the repo — this is intentional).

```bash
python3.11 -m venv QuantLlama
source QuantLlama/bin/activate
```

After activation, your shell prompt should show `(QuantLlama)` at the start:
```
PyTorch ........... PASS (2.11.0+cu128)
CUDA available .... PASS (device: NVIDIA GeForce RTX 3090)
transformers ...... PASS (4.56.2)
bitsandbytes ...... PASS (0.49.2)
peft .............. PASS (0.18.1)
trl ............... PASS (1.0.0)
accelerate ........ PASS (1.13.0)
datasets .......... PASS (4.8.4)
gradio ............ PASS (6.11.0)
auto-gptq ......... PASS (0.7.1)
```
If any line shows FAIL, re-run `pip install -r requirements.txt` or check that your CUDA driver
matches the PyTorch build (CUDA 12.8).

---

### Step 5 — Set your Hugging Face token

The model download and Hub upload both require your HF token. Set it as an environment variable:

```bash
export HF_TOKEN=hf_your_token_here
```

Verify it is set:
```bash
echo $HF_TOKEN | cut -c1-8
# Expected: hf_XXXXX  (first 8 characters of your token)
```

> ⚠️ This export only lasts for the current terminal session. You will need to re-run this
> command every time you open a new terminal, before running any script that accesses the Hub.

---

### Step 6 — Phase 1: fp16 Baseline Benchmark

This script loads Llama-3.1-8B in full fp16 precision and measures perplexity on WikiText-2,
inference speed, peak VRAM, and load time. Results are written to `results/results.csv`.

```bash
python phase1_baseline/benchmark.py
```

This will take approximately **5–10 minutes** — the model is ~16 GB and must be downloaded on
first run (~10 min depending on connection), then loaded into VRAM (~4 min).

Expected output at the end:
```
=== GPTQ Results ===
Perplexity (WikiText-2): 6.5497
Inference speed:         4.35 tok/s
Peak VRAM:               9.453 GB
Load time:               40.9 sec
```
> **Why is GPTQ slower than NF4?** The auto-gptq CUDA kernels are not compiled in the pip wheel.
> The model falls back to PyTorch ops. In a properly compiled environment GPTQ would be faster.
> This is a known software environment limitation, not a property of GPTQ itself.

**View the comparison table so far:**
```bash
python phase2_quantization/compare.py
```

---

### Step 8 — Phase 3: QLoRA Fine-Tuning

This phase fine-tunes the NF4 quantized model on the Stanford Alpaca instruction-following dataset
using QLoRA. Only 0.085% of parameters are trained (the LoRA matrices). Training takes ~2 hours
on an RTX 3090.

**Fine-tuning:**
```bash
python phase3_qlora/finetune.py
```

You will see a training progress bar. Loss should decrease steadily. Example:
```
{'loss': 1.832, 'grad_norm': 0.421, 'learning_rate': 0.0002, 'epoch': 0.1}
{'loss': 1.654, 'grad_norm': 0.389, 'learning_rate': 0.00019, 'epoch': 0.2}
...
Training complete. Adapter saved to outputs/qlora-alpaca/final_adapter/
Adapter size: 44.5 MB
```
The adapter weights are saved to `outputs/qlora-alpaca/final_adapter/`. These are the only files
that need to be kept — they are ~44.5 MB, not the full model.

**Evaluation (perplexity + 20-prompt qualitative eval):**
```bash
python phase3_qlora/evaluate.py
```

Expected output:
```
=== QLoRA Fine-Tuned Results ===
Perplexity (WikiText-2): 6.6251
Inference speed:         53.39 tok/s
Peak VRAM:               7.504 GB
Load time:               96.1 sec
Qualitative results saved to results/qualitative_phase3.csv
```
**Generate benchmark bar charts:**
```bash
python plot_results.py
```

Charts saved to `results/plots/` — perplexity.png, speed.png, vram.png, combined.png.

---

### Step 9 — Phase 4: Upload to Hugging Face Hub

This pushes the fine-tuned adapter weights to your public HF Hub repository so anyone can load
the model with two lines of Python.

```bash
# Make sure HF_TOKEN is set (see Step 5)
echo $HF_TOKEN | cut -c1-8   # should print hf_XXXXX

python phase4_demo/push_to_hub.py
```

Expected output:
```
Repo ready: https://huggingface.co/srijayadav/llama3-8b-qlora-alpaca
Processing Files (2 / 2): 100%|...| 44.5MB / 44.5MB
Upload complete!
View at: https://huggingface.co/srijayadav/llama3-8b-qlora-alpaca
```
---

### Step 10 — Phase 4: Run the Gradio Demo Locally

This launches the side-by-side comparison demo in your browser. Both models (base int4-NF4 and
QLoRA fine-tuned) are loaded at startup and run in parallel on every prompt.

```bash
python phase4_demo/app.py
```

Startup takes ~4 minutes (both models must load into VRAM). You will see:
```
Loading tokenizer...
Loading base int4-NF4 model...
Base model loaded in 115.9s
Loading QLoRA fine-tuned model...
Fine-tuned model loaded in 115.5s
Both models ready.

Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://xxxxxxxxxxxx.gradio.live
```
Open the local URL in your browser. Type any instruction in the text box and click **Generate**.
Both models respond simultaneously. The metadata row under each panel shows latency, throughput
(tok/s), and peak VRAM.

> The `gradio.live` public URL is shareable with anyone for 1 week without any setup on their end.

---

### Loading the Fine-Tuned Model Directly from HF Hub

If you just want to use the fine-tuned model without running any of the above scripts, you can
load it directly from Hugging Face Hub in two steps:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Step 1 — configure 4-bit loading
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Step 2 — load tokenizer from the adapter repo
tokenizer = AutoTokenizer.from_pretrained("srijayadav/llama3-8b-qlora-alpaca")

# Step 3 — load base model in int4
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",       # requires HF access approval
    quantization_config=bnb_config,
    device_map="auto",
)

# Step 4 — attach the LoRA adapter
model = PeftModel.from_pretrained(model, "srijayadav/llama3-8b-qlora-alpaca")

# Step 5 — run inference
prompt = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\nExplain what gradient descent is in simple terms.\n\n"
    "### Response:\n"
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Common Issues and Fixes

| Error | Cause | Fix |
|---|---|---|
| `CUDA out of memory` during fp16 baseline | GPU has < 20 GB VRAM | Skip Phase 1, start from Phase 2 |
| `403 Forbidden` when pushing to HF Hub | Token is read-only | Generate a new **Write** token at huggingface.co/settings/tokens |
| `$HF_TOKEN` is empty | Token not set in this session | `export HF_TOKEN=hf_...` |
| `no_init_weights` error with auto-gptq | transformers version too new | Ensure `transformers==4.56.2` — do not upgrade |
| `NotImplementedError: _amp_foreach_non_finite_check_and_unscale_cuda` | bf16/fp16 mismatch in training | Use `bf16=True` in SFTConfig, not `fp16=True` |
| `SFTTrainer` unexpected kwarg `tokenizer` | TRL 1.0 API change | Use `processing_class=` instead of `tokenizer=` |
| GPTQ very slow (4 tok/s) | CUDA kernels not compiled | Expected — auto-gptq falls back to PyTorch ops on this install |
| Demo takes 4 min to start | Both 8B models loading into VRAM | Normal — wait for "Both models ready." before opening browser |

---

## References

- Dettmers et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS 2023. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- Frantar et al. (2022). GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)
- Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Taori et al. (2023). Alpaca: A Strong, Replicable Instruction-Following Model. Stanford CRFM.
README