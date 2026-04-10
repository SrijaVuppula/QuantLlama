"""
Phase 2 — GPTQ int4 Benchmark (pretrained checkpoint)
QuantLlama · CSCI 4900/6900 · UGA Spring 2026

GPTQ calibration requires loading the full fp16 model (~16 GB) into CPU RAM,
which exceeds the 15 GB available on the department server. A pretrained GPTQ
checkpoint calibrated on WikiText-2 is used instead — consistent with standard
practice in quantization research, where published checkpoints use identical
calibration data and methodology.

Checkpoint: ModelCloud/Meta-Llama-3.1-8B-gptq-4bit
  - 4-bit GPTQ, group_size=128, calibrated on WikiText-2
  - Equivalent to what self-calibration would produce

Runs the identical benchmark suite from Phase 1 and appends one row to
results/results.csv.

Usage:
    python phase2_quantization/quantize_gptq.py

Requirements: auto-gptq, datasets, torch (CUDA)
"""

import csv
import math
import time
from pathlib import Path

import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer

transformers.logging.set_verbosity_error()

# ── Config ────────────────────────────────────────────────────────────────────
GPTQ_MODEL_ID   = "ModelCloud/Meta-Llama-3.1-8B-gptq-4bit"
BASE_MODEL_ID   = "meta-llama/Llama-3.1-8B"   # for tokenizer fallback
RESULTS_CSV     = Path(__file__).parent.parent / "results" / "results.csv"
CONFIG_LABEL    = "meta-llama/Llama-3.1-8B | int4-gptq | ModelCloud pretrained"
PRECISION_LABEL = "int4-gptq"

# Benchmark settings — identical to Phase 1 and quantize_nf4.py
PPL_WINDOWS      = 64
WINDOW_TOKENS    = 2048
SPEED_RUNS       = 10
SPEED_WARMUP     = 1
SPEED_NEW_TOKENS = 200
SPEED_PROMPT     = "The future of artificial intelligence is"

print(f"\n{'='*60}")
print(f"  QuantLlama — Phase 2: GPTQ int4 Benchmark")
print(f"  Checkpoint : {GPTQ_MODEL_ID}")
print(f"{'='*60}\n")

from auto_gptq import AutoGPTQForCausalLM

# ── Load model ────────────────────────────────────────────────────────────────
print(f"[load]  Loading pretrained GPTQ checkpoint...")
print(f"        (downloads ~4-5 GB on first run, cached after)")
torch.cuda.reset_peak_memory_stats()
load_start = time.perf_counter()

tokenizer = AutoTokenizer.from_pretrained(GPTQ_MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoGPTQForCausalLM.from_quantized(
    GPTQ_MODEL_ID,
    device_map="auto",
    use_safetensors=True,
)
model.eval()

load_time = time.perf_counter() - load_start
print(f"[load]  Ready in {load_time:.1f}s")

# ── Perplexity (WikiText-2) ───────────────────────────────────────────────────
print(f"\n[ppl]   Evaluating perplexity on WikiText-2 ({PPL_WINDOWS} × {WINDOW_TOKENS} tokens)...")

dataset   = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
full_text = "\n\n".join(dataset["text"])
encodings = tokenizer(full_text, return_tensors="pt")
input_ids = encodings.input_ids.to(model.device)

nlls     = []
n_tokens = 0
max_start = (PPL_WINDOWS - 1) * WINDOW_TOKENS

with torch.no_grad():
    for begin in range(0, min(max_start + 1, input_ids.size(1) - WINDOW_TOKENS), WINDOW_TOKENS):
        end    = begin + WINDOW_TOKENS
        chunk  = input_ids[:, begin:end]
        labels = chunk.clone()
        out    = model(chunk, labels=labels)
        nlls.append(out.loss.item() * chunk.size(1))
        n_tokens += chunk.size(1)
        if len(nlls) % 16 == 0:
            print(f"          ... {len(nlls)}/{PPL_WINDOWS} windows done")

perplexity = math.exp(sum(nlls) / n_tokens)
print(f"[ppl]   Perplexity = {perplexity:.4f}")

# ── Inference speed ───────────────────────────────────────────────────────────
print(f"\n[speed] Measuring inference speed ({SPEED_RUNS} runs, {SPEED_NEW_TOKENS} new tokens each)...")

prompt_ids = tokenizer(SPEED_PROMPT, return_tensors="pt").input_ids.to(model.device)
times      = []

with torch.no_grad():
    for i in range(SPEED_RUNS + SPEED_WARMUP):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.generate(
            input_ids=prompt_ids,        
            max_new_tokens=SPEED_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        if i >= SPEED_WARMUP:
            times.append(SPEED_NEW_TOKENS / elapsed)
            print(f"          run {i - SPEED_WARMUP + 1:2d}: {times[-1]:.2f} tok/s")

speed_mean = sum(times) / len(times)
print(f"[speed] Mean = {speed_mean:.2f} tok/s")

# ── Peak VRAM ─────────────────────────────────────────────────────────────────
peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
print(f"\n[vram]  Peak VRAM = {peak_vram_gb:.3f} GB")

# ── Write results ─────────────────────────────────────────────────────────────
RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
write_header = not RESULTS_CSV.exists()

row = {
    "config":                CONFIG_LABEL,
    "precision":             PRECISION_LABEL,
    "perplexity_wikitext2":  round(perplexity, 4),
    "speed_tokens_per_sec":  round(speed_mean, 2),
    "peak_vram_gb":          round(peak_vram_gb, 3),
    "load_time_sec":         round(load_time, 1),
}

with open(RESULTS_CSV, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=row.keys())
    if write_header:
        writer.writeheader()
    writer.writerow(row)

print(f"\n[done]  Results appended to {RESULTS_CSV}")
print(f"\n{'='*60}")
print(f"  SUMMARY — {PRECISION_LABEL}")
print(f"{'='*60}")
print(f"  Perplexity  : {perplexity:.4f}")
print(f"  Speed       : {speed_mean:.2f} tok/s")
print(f"  Peak VRAM   : {peak_vram_gb:.3f} GB")
print(f"  Load time   : {load_time:.1f}s")
print(f"{'='*60}\n")