# QuantLlama

**Scalable LLM Deployment: Benchmarking 4-Bit Quantization and QLoRA Fine-Tuning**

---

## Overview

This project investigates how large language models can be compressed to run on consumer hardware
without meaningfully sacrificing quality. It benchmarks post-training quantization (NF4, GPTQ) on
`meta-llama/Llama-3.1-8B` and combines it with QLoRA fine-tuning on the Stanford Alpaca dataset.

**Model:** `meta-llama/Llama-3.1-8B` | **Hardware:** NVIDIA RTX 3090 (24 GB VRAM)

---

## Results

| Config | Precision | Perplexity (lower=better) | Speed (tok/s) | VRAM (GB) | Load (s) |
|---|---|---|---|---|---|
| fp16 baseline | fp16 | 6.1885 | 36.26 | 18.984 | 256.5 |
| int4 NF4 (bitsandbytes) | int4-nf4 | 6.5784 | 18.01 | 9.425 | 116.0 |
| int4 GPTQ (ModelCloud) | int4-gptq | 6.5497 | 4.35 | 9.453 | 40.9 |
| **QLoRA fine-tuned (Alpaca)** | int4-nf4-qlora | **6.6251** | **53.39** | **7.504** | 96.1 |

Key findings:
- **~50% VRAM reduction** from fp16 to int4 (18.98 GB to 9.4 GB)
- **Minimal perplexity degradation** -- less than 0.4 points at int4 vs fp16
- **QLoRA fine-tuned model is fastest and most memory-efficient** after `merge_and_unload()` -- 53.39 tok/s and only 7.5 GB VRAM
- Only **0.085% of parameters trained** during QLoRA (6.8M / 8.03B), producing a 44.5 MB adapter

---

## Links

| Resource | URL |
|---|---|
| Fine-tuned model (HF Hub) | https://huggingface.co/srijayadav/llama3-8b-qlora-alpaca |
| HF Space (demo code) | https://huggingface.co/spaces/srijayadav/quantllama-demo |

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
├── timing.py                   # Per-stage inference timing (tokenization, forward pass, decoding)
├── plot_results.py             # Generates benchmark bar charts
└── requirements.txt            # Full environment dependencies
```

---

## How to Reproduce

This section walks through the entire project from scratch. Each step includes the expected
output so you can verify things are working before moving on.

---

### Step 1 -- Prerequisites

**Hardware**

- A CUDA-capable NVIDIA GPU
- At minimum 12 GB VRAM for int4 experiments
- 24 GB VRAM recommended for the fp16 baseline (RTX 3090 was used here)

Check your GPU and VRAM:

```bash
nvidia-smi
```

Expected output (GPU name and memory will differ):

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.x    Driver Version: 535.x    CUDA Version: 12.x            |
+-------------------------------+----------------------+----------------------+
| GPU  0   RTX 3090            | 00000000:01:00.0 Off |                  N/A |
| 24MiB /  24576MiB            |      0%      Default |                  N/A |
+-----------------------------------------------------------------------------+
```

**Software**

- Python 3.11 specifically -- bitsandbytes and auto-gptq have strict version requirements
- CUDA 12.x -- check with `nvcc --version` or from the nvidia-smi output above
- git

Check Python version:

```bash
python3.11 --version
```

Expected output:

```
Python 3.11.x
```

**Hugging Face account with Llama access**

`meta-llama/Llama-3.1-8B` is a gated model. You must request access before downloading it.

1. Create a free account at https://huggingface.co/join
2. Go to https://huggingface.co/meta-llama/Llama-3.1-8B and click **Request access**
3. Wait for approval (usually a few minutes to a few hours)
4. Once approved, go to https://huggingface.co/settings/tokens
5. Click **New token**, set type to **Write**, and copy the token (starts with `hf_`)

> **Important:** The token must be Write type. Read-only tokens will fail when pushing to Hub in Phase 4.

---

### Step 2 -- Clone the repo

```bash
git clone https://github.com/SrijaVuppula/QuantLlama.git
cd QuantLlama
```

Expected output:

```
Cloning into 'QuantLlama'...
remote: Enumerating objects: 42, done.
remote: Counting objects: 100% (42/42), done.
```

---

### Step 3 -- Create and activate a virtual environment

We use a Python 3.11 virtualenv to isolate all dependencies. The venv is created inside the repo
directory and named `QuantLlama` (same as the repo -- this is intentional).

```bash
python3.11 -m venv QuantLlama
source QuantLlama/bin/activate
```

After activation your shell prompt will show `(QuantLlama)` at the start:

```
(QuantLlama) bash-4.4$
```

Confirm the right Python is active:

```bash
which python
python --version
```

Expected output:

```
/home/youruser/QuantLlama/QuantLlama/bin/python
Python 3.11.x
```

> **Important:** Always activate the venv at the start of every new terminal session.
> The venv is NOT automatically activated when you open a new terminal.

---

### Step 4 -- Install dependencies

```bash
pip install -r requirements.txt
```

This installs PyTorch (CUDA 12.8), Hugging Face Transformers, bitsandbytes, PEFT, TRL,
auto-gptq, Gradio, and all other dependencies. Takes 5-10 minutes on first run.

> **Library compatibility note:** This project uses `transformers==4.56.2` (not the latest).
> auto-gptq 0.7.1 is incompatible with transformers 5.x. Do not upgrade either independently.

Validate the environment after installation:

```bash
python setup/validate_env.py
```

Expected output (all lines should say PASS):

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

If any line shows FAIL, re-run `pip install -r requirements.txt` or verify your CUDA driver
matches the PyTorch build (CUDA 12.8).

---

### Step 5 -- Set your Hugging Face token

Both model download and Hub upload require your HF token. Set it as an environment variable:

```bash
export HF_TOKEN=hf_your_token_here
```

Verify it is set:

```bash
echo $HF_TOKEN | cut -c1-8
```

Expected output:

```
hf_XXXXX
```

> **Important:** This export only lasts for the current terminal session. Re-run it every time
> you open a new terminal before running any script that accesses the Hub.

---

### Step 6 -- Phase 1: fp16 Baseline Benchmark

Loads Llama-3.1-8B in full fp16 precision and measures perplexity on WikiText-2, inference
speed, peak VRAM, and load time. Results are written to `results/results.csv`.

```bash
python phase1_baseline/benchmark.py
```

Takes approximately 5-10 minutes. The model (~16 GB) is downloaded on first run.

Expected output:

```
=== Phase 1 Results ===
Perplexity (WikiText-2): 6.1885
Inference speed:         36.26 tok/s
Peak VRAM:               18.984 GB
Load time:               256.5 sec
Results saved to results/results.csv
```

> **Note:** Requires ~19 GB VRAM. If you have less than 20 GB, skip to Phase 2 (needs only ~9.4 GB).

---

### Step 7 -- Phase 2: Quantization Benchmarks

Runs the same benchmark suite on two int4 quantized versions and appends results to `results/results.csv`.

**NF4 quantization (bitsandbytes) -- no calibration data needed:**

```bash
python phase2_quantization/quantize_nf4.py
```

Expected output:

```
=== NF4 Results ===
Perplexity (WikiText-2): 6.5784
Inference speed:         18.01 tok/s
Peak VRAM:               9.425 GB
Load time:               116.0 sec
```

**GPTQ quantization (calibration-based):**

```bash
python phase2_quantization/quantize_gptq.py
```

> **Note on GPTQ:** Self-calibration requires ~16 GB CPU RAM. If your machine has less, the script
> automatically uses the pretrained checkpoint `ModelCloud/Meta-Llama-3.1-8B-gptq-4bit`, calibrated
> on the same WikiText-2 corpus. This is what happened in our experiments.

Expected output:

```
=== GPTQ Results ===
Perplexity (WikiText-2): 6.5497
Inference speed:         4.35 tok/s
Peak VRAM:               9.453 GB
Load time:               40.9 sec
```

> **Why is GPTQ slow here?** The auto-gptq CUDA kernels are not compiled in the pip wheel.
> The model falls back to PyTorch ops. In a properly compiled environment GPTQ is significantly faster.
> This is a known software environment limitation, not a flaw in GPTQ itself.

Print the full 3-row comparison table:

```bash
python phase2_quantization/compare.py
```

---

### Step 8 -- Phase 3: QLoRA Fine-Tuning

Fine-tunes the NF4 quantized model on Stanford Alpaca using QLoRA. Only 0.085% of parameters
are trained (the LoRA adapter matrices). Takes approximately 2 hours on an RTX 3090.

**Run fine-tuning:**

```bash
python phase3_qlora/finetune.py
```

You will see a progress bar and training loss logged every 10 steps:

```
{'loss': 1.832, 'grad_norm': 0.421, 'learning_rate': 0.0002, 'epoch': 0.1}
{'loss': 1.654, 'grad_norm': 0.389, 'learning_rate': 0.00019, 'epoch': 0.2}
...
Training complete. Adapter saved to outputs/qlora-alpaca/final_adapter/
Adapter size: 44.5 MB
```

The adapter weights (~44.5 MB) are saved to `outputs/qlora-alpaca/final_adapter/`.
These are the only output files that matter -- not the full 16 GB model.

**Run evaluation:**

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

Charts saved to `results/plots/` -- perplexity.png, speed.png, vram.png, combined.png.

---

### Step 9 -- Phase 4: Upload Adapter Weights to Hugging Face Hub

Pushes the fine-tuned adapter weights to a public HF Hub repo so anyone can load the model
with just a few lines of Python.

```bash
# Confirm your HF token is set first
echo $HF_TOKEN | cut -c1-8

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

### Step 10 -- Phase 4: Run the Gradio Demo Locally

Launches a side-by-side comparison demo in your browser. Both models are loaded at startup
and run in parallel on every prompt.

```bash
python phase4_demo/app.py
```

Startup takes ~4 minutes. Expected output:

```
Loading tokenizer...
Loading base int4-NF4 model...
Base model loaded in 115.9s
Loading QLoRA fine-tuned model...
Fine-tuned model loaded in 115.5s
Both models ready.
* Running on local URL:  http://127.0.0.1:7860
* Running on public URL: https://xxxxxxxxxxxx.gradio.live
```

Open `http://127.0.0.1:7860` in your browser. Type any instruction and click **Generate**.
Both models respond simultaneously. The metadata row under each panel shows latency,
throughput (tok/s), and peak VRAM.

The `gradio.live` URL is shareable with anyone for 1 week with no setup on their end.

---

### Loading the Fine-Tuned Model Directly from HF Hub

To use the fine-tuned model without running the phases above:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Configure 4-bit (NF4) loading
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load tokenizer from the adapter repo
tokenizer = AutoTokenizer.from_pretrained('srijayadav/llama3-8b-qlora-alpaca')

# Load base model in int4 -- requires HF access approval for meta-llama/Llama-3.1-8B
model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-3.1-8B',
    quantization_config=bnb_config,
    device_map='auto',
)

# Attach the LoRA adapter on top of the base model
model = PeftModel.from_pretrained(model, 'srijayadav/llama3-8b-qlora-alpaca')

# Run inference using the Alpaca prompt format
prompt = (
    'Below is an instruction that describes a task. '
    'Write a response that appropriately completes the request.\n\n'
    '### Instruction:\nExplain what gradient descent is in simple terms.\n\n'
    '### Response:\n'
)
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Common Issues and Fixes

| Error | Cause | Fix |
|---|---|---|
| `CUDA out of memory` during fp16 baseline | GPU has less than 20 GB VRAM | Skip Phase 1, start from Phase 2 |
| `403 Forbidden` when pushing to HF Hub | Token is read-only | Generate a new Write token at huggingface.co/settings/tokens |
| `$HF_TOKEN` is empty | Token not set in this session | `export HF_TOKEN=hf_...` and re-run |
| `no_init_weights` error with auto-gptq | transformers version too new | Ensure `transformers==4.56.2` -- do not upgrade |
| `NotImplementedError: _amp_foreach_non_finite_check_and_unscale_cuda` | bf16/fp16 mismatch in training | Use `bf16=True` in SFTConfig, not `fp16=True` |
| `SFTTrainer` unexpected kwarg `tokenizer` | TRL 1.0 API change | Use `processing_class=` instead of `tokenizer=` |
| GPTQ runs at only 4 tok/s | CUDA kernels not compiled in pip wheel | Expected -- auto-gptq falls back to PyTorch ops on this install |
| Demo takes 4 minutes to start | Both 8B models loading into VRAM | Normal -- wait for 'Both models ready.' before opening browser |

---

## References

- Dettmers et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS 2023. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- Frantar et al. (2022). GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)
- Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Taori et al. (2023). Alpaca: A Strong, Replicable Instruction-Following Model. Stanford CRFM.
