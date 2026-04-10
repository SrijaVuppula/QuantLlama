"""
Phase 1 — fp16 Baseline Benchmark
===================================
Loads LLaMA-3 8B in fp16 and records:
  - Perplexity on WikiText-2
  - Inference speed (tokens/sec)
  - Peak VRAM (GB)
  - Model load time (sec)

Results are saved to results/results.csv
Usage: python phase1_baseline/benchmark.py
"""

import csv, math, os, time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID        = "meta-llama/Llama-3.1-8B"
PRECISION       = "fp16"
RESULTS_CSV     = "results/results.csv"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

CONTEXT_LEN     = 2048      # tokens per perplexity window
MAX_EVAL_TOKENS = 131072    # cap at 128k tokens (~64 windows)
SPEED_PROMPT    = "Explain the concept of gradient descent in machine learning."
SPEED_NEW_TOKENS = 200
SPEED_RUNS      = 10

# ── CSV ───────────────────────────────────────────────────────────────────────
FIELDS = ["config", "precision", "perplexity_wikitext2",
          "speed_tokens_per_sec", "peak_vram_gb", "load_time_sec"]

def init_csv():
    Path("results").mkdir(exist_ok=True)
    if not Path(RESULTS_CSV).exists():
        with open(RESULTS_CSV, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()
        print(f"Created {RESULTS_CSV}")

def append_csv(row):
    with open(RESULTS_CSV, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
    print(f"Results saved to {RESULTS_CSV}")

# ── Load model ────────────────────────────────────────────────────────────────
def load_model():
    print(f"\n{'='*55}")
    print(f"  Loading {MODEL_ID} in fp16")
    print(f"{'='*55}")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    load_time = time.time() - t0
    model.eval()
    print(f"  Load time: {load_time:.1f}s")
    return model, tokenizer, load_time

# ── Perplexity ────────────────────────────────────────────────────────────────
def compute_perplexity(model, tokenizer):
    print(f"\n  Loading WikiText-2 test set...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)

    n_windows = min(input_ids.shape[1] // CONTEXT_LEN,
                    MAX_EVAL_TOKENS // CONTEXT_LEN)
    print(f"  Evaluating over {n_windows} windows of {CONTEXT_LEN} tokens...")

    total_nll = 0.0
    for i in range(n_windows):
        chunk = input_ids[:, i*CONTEXT_LEN:(i+1)*CONTEXT_LEN]
        with torch.no_grad():
            loss = model(chunk, labels=chunk).loss
        total_nll += loss.item()
        if (i+1) % 10 == 0:
            print(f"    Window {i+1}/{n_windows} — "
                  f"running ppl: {math.exp(total_nll/(i+1)):.2f}")

    ppl = round(math.exp(total_nll / n_windows), 4)
    print(f"  Perplexity (WikiText-2): {ppl}")
    return ppl

# ── Speed ─────────────────────────────────────────────────────────────────────
def benchmark_speed(model, tokenizer):
    print(f"\n  Benchmarking speed ({SPEED_RUNS} runs × {SPEED_NEW_TOKENS} tokens)...")
    inputs = tokenizer(SPEED_PROMPT, return_tensors="pt").to(DEVICE)
    speeds = []
    for run in range(SPEED_RUNS + 1):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=SPEED_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        tps = SPEED_NEW_TOKENS / (time.time() - t0)
        if run == 0:
            print(f"    Warm-up: {tps:.1f} tok/s (discarded)")
        else:
            speeds.append(tps)
            print(f"    Run {run:2d}/{SPEED_RUNS}: {tps:.1f} tok/s")

    mean_tps = round(sum(speeds) / len(speeds), 2)
    print(f"  Mean speed: {mean_tps} tok/s")
    return mean_tps

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\n  Device : {DEVICE}")
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    init_csv()
    model, tokenizer, load_time = load_model()
    ppl = compute_perplexity(model, tokenizer)
    tps = benchmark_speed(model, tokenizer)
    peak_vram = round(torch.cuda.max_memory_allocated() / 1e9, 3)

    row = {
        "config"               : f"{MODEL_ID} | fp16 | baseline",
        "precision"            : PRECISION,
        "perplexity_wikitext2" : ppl,
        "speed_tokens_per_sec" : tps,
        "peak_vram_gb"         : peak_vram,
        "load_time_sec"        : round(load_time, 2),
    }

    print(f"\n{'='*55}")
    print("  Results Summary")
    print(f"{'='*55}")
    for k, v in row.items():
        print(f"  {k:<25} {v}")

    append_csv(row)
    print(f"\n  Phase 1 complete. Next: python phase2_quantization/quantize_nf4.py")

if __name__ == "__main__":
    main()
