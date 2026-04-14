"""
timing_breakdown.py — Per-stage inference timing breakdown
Measures tokenization, model forward pass, and decoding separately
across all 4 configurations to identify the speed bottleneck.
"""

import torch
import time
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MODEL_ID   = "meta-llama/Llama-3.1-8B"
ADAPTER_ID = "srijayadav/llama3-8b-qlora-alpaca"
N_RUNS     = 5
MAX_NEW_TOKENS = 200

PROMPTS = [
    "Explain what a neural network is in simple terms.",
    "Write a Python function to compute the Fibonacci sequence.",
    "What are the main causes of climate change?",
    "Describe the process of photosynthesis.",
    "Explain the difference between supervised and unsupervised learning.",
]

def time_stages(model, tokenizer, prompt, max_new_tokens=200):
    device = next(model.parameters()).device

    # Stage 1 — Tokenization (pre-processing)
    t0 = time.perf_counter()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    t1 = time.perf_counter()
    tokenize_ms = (t1 - t0) * 1000

    # Stage 2 — Model forward pass / generation (bottleneck)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    t3 = time.perf_counter()
    generate_ms = (t3 - t2) * 1000

    # Stage 3 — Decoding (post-processing)
    t4 = time.perf_counter()
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    t5 = time.perf_counter()
    decode_ms = (t5 - t4) * 1000

    new_tokens = output_ids.shape[1] - inputs["input_ids"].shape[1]
    total_ms = tokenize_ms + generate_ms + decode_ms

    return {
        "tokenize_ms": round(tokenize_ms, 3),
        "generate_ms": round(generate_ms, 3),
        "decode_ms":   round(decode_ms, 3),
        "total_ms":    round(total_ms, 3),
        "new_tokens":  new_tokens,
        "tok_per_sec": round(new_tokens / (generate_ms / 1000), 2),
        "tokenize_pct": round(tokenize_ms / total_ms * 100, 2),
        "generate_pct": round(generate_ms / total_ms * 100, 2),
        "decode_pct":   round(decode_ms   / total_ms * 100, 2),
    }

def benchmark_config(name, model, tokenizer):
    print(f"\n{'='*60}")
    print(f"Config: {name}")
    print(f"{'='*60}")

    all_results = []
    for i, prompt in enumerate(PROMPTS):
        r = time_stages(model, tokenizer, prompt)
        all_results.append(r)
        print(f"  Prompt {i+1}: tokenize={r['tokenize_ms']:.1f}ms | "
              f"generate={r['generate_ms']:.0f}ms | "
              f"decode={r['decode_ms']:.1f}ms | "
              f"total={r['total_ms']:.0f}ms | "
              f"{r['tok_per_sec']} tok/s")

    avg = {k: round(sum(r[k] for r in all_results) / len(all_results), 3)
           for k in all_results[0]}

    print(f"\n  --- Averages across {len(PROMPTS)} prompts ---")
    print(f"  Tokenization:  {avg['tokenize_ms']:.2f} ms  ({avg['tokenize_pct']:.2f}% of total)")
    print(f"  Model forward: {avg['generate_ms']:.1f} ms  ({avg['generate_pct']:.2f}% of total)")
    print(f"  Decoding:      {avg['decode_ms']:.2f} ms  ({avg['decode_pct']:.2f}% of total)")
    print(f"  Total:         {avg['total_ms']:.1f} ms")
    print(f"  Throughput:    {avg['tok_per_sec']} tok/s")
    print(f"\n  >> BOTTLENECK: Model forward pass ({avg['generate_pct']:.1f}% of total time)")

    return {"config": name, **avg}

def main():
    results = []

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # Config 1 — Base int4-NF4
    print("\nLoading base int4-NF4 model...")
    model_nf4 = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto"
    )
    model_nf4.eval()
    results.append(benchmark_config("int4-NF4 (base)", model_nf4, tokenizer))
    del model_nf4
    torch.cuda.empty_cache()

    # Config 2 — QLoRA fine-tuned
    print("\nLoading QLoRA fine-tuned model...")
    model_ft = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto"
    )
    model_ft = PeftModel.from_pretrained(model_ft, ADAPTER_ID)
    model_ft = model_ft.merge_and_unload()
    model_ft.eval()
    results.append(benchmark_config("int4-NF4-QLoRA (fine-tuned)", model_ft, tokenizer))
    del model_ft
    torch.cuda.empty_cache()

    # Save to CSV
    csv_path = "results/timing_breakdown.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {csv_path}")
    print("\n=== SUMMARY ===")
    print(f"{'Config':<35} {'Tokenize':>12} {'Generate':>12} {'Decode':>12} {'Total':>12}")
    print("-" * 85)
    for r in results:
        print(f"{r['config']:<35} {r['tokenize_ms']:>10.2f}ms {r['generate_ms']:>10.1f}ms "
              f"{r['decode_ms']:>10.2f}ms {r['total_ms']:>10.1f}ms")

if __name__ == "__main__":
    main()
