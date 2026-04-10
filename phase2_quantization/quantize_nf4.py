"""
Phase 2 — NF4 (bitsandbytes) Quantization Benchmark
QuantLlama · CSCI 4900/6900 · UGA Spring 2026

Loads meta-llama/Llama-3.1-8B in 4-bit NF4 precision using bitsandbytes,
runs the identical benchmark suite from Phase 1, and appends one row to
results/results.csv.

Usage:
    python phase2_quantization/quantize_nf4.py

Requirements: transformers, bitsandbytes, datasets, torch (CUDA)
"""

import csv
import os
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID        = "meta-llama/Llama-3.1-8B"
RESULTS_CSV     = Path(__file__).parent.parent / "results" / "results.csv"
CONFIG_LABEL    = f"{MODEL_ID} | int4-nf4 | bitsandbytes"
PRECISION_LABEL = "int4-nf4"

# Benchmark settings — identical to Phase 1
PPL_WINDOWS      = 64          # number of 2048-token windows for perplexity
WINDOW_TOKENS    = 2048
SPEED_RUNS       = 10          # generation runs for speed measurement
SPEED_WARMUP     = 1           # discard first N runs
SPEED_NEW_TOKENS = 200         # tokens generated per speed run
SPEED_PROMPT     = "The future of artificial intelligence is"

# ── NF4 quantization config ───────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # Normal Float 4 — optimal for normally distributed weights
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,     # nested quantization — shaves a bit more VRAM
)

# ── Load model ────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  QuantLlama — Phase 2: NF4 Quantization")
print(f"  Model : {MODEL_ID}")
print(f"{'='*60}\n")

torch.cuda.reset_peak_memory_stats()
load_start = time.perf_counter()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)
model.eval()

load_time = time.perf_counter() - load_start
print(f"[load]  Model loaded in {load_time:.1f}s")

# ── Perplexity (WikiText-2) ───────────────────────────────────────────────────
print(f"\n[ppl]   Evaluating perplexity on WikiText-2 ({PPL_WINDOWS} × {WINDOW_TOKENS} tokens)...")

dataset   = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
full_text = "\n\n".join(dataset["text"])
encodings = tokenizer(full_text, return_tensors="pt")
input_ids = encodings.input_ids.to(model.device)

nlls      = []
n_tokens  = 0
max_start = (PPL_WINDOWS - 1) * WINDOW_TOKENS  # stay within PPL_WINDOWS windows

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

import math
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
            prompt_ids,
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
