"""
QuantLlama — Phase 3: Post-Fine-Tuning Evaluation
==================================================
Evaluates the QLoRA fine-tuned model and appends results to results.csv.

Two evaluations:
  1. Perplexity on WikiText-2 (same 64-window setup as Phases 1 & 2)
  2. Qualitative side-by-side: base int4-NF4 vs fine-tuned QLoRA on 20 prompts

Run AFTER finetune.py has saved the adapter to ./outputs/qlora-alpaca/final_adapter/

Usage:
    python evaluate.py

Output:
    - Perplexity + speed row appended to ../results/results.csv
    - Qualitative outputs saved to ../results/qualitative_phase3.csv
"""

import os
import csv
import time
import math
import threading
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ── Config ────────────────────────────────────────────────────────────────────

BASE_MODEL   = "meta-llama/Llama-3.1-8B"
ADAPTER_PATH = "./outputs/qlora-alpaca/final_adapter"
RESULTS_CSV  = "../results/results.csv"
QUAL_CSV     = "../results/qualitative_phase3.csv"
HF_TOKEN     = os.environ.get("HF_TOKEN")

# Perplexity eval settings — identical to Phase 1/2 for comparability
WIKITEXT_WINDOWS = 64
WINDOW_SIZE      = 2048
STRIDE           = 2048

# Speed benchmark settings — identical to Phase 1/2
SPEED_RUNS       = 10
SPEED_TOKENS     = 200

# ── 20 qualitative prompts (same set used in Phase 2) ─────────────────────────
# Mix of: instruction following, reasoning, factual, creative, edge cases

QUAL_PROMPTS = [
    # Instruction-following (should improve most after Alpaca fine-tuning)
    "Write a Python function that reverses a string without using slicing.",
    "Explain the difference between RAM and ROM in simple terms.",
    "Summarize the French Revolution in exactly three sentences.",
    "List five tips for improving time management as a college student.",
    "Write a professional email declining a job offer politely.",

    # Reasoning
    "If all roses are flowers and some flowers fade quickly, do all roses fade quickly?",
    "A train leaves Chicago at 60 mph. Another leaves New York at 80 mph toward Chicago. The cities are 800 miles apart. When do they meet?",
    "What are the ethical implications of AI systems making medical diagnoses?",
    "Why might a company choose a relational database over a NoSQL database?",
    "Explain why quicksort is generally faster than bubble sort in practice.",

    # Factual knowledge
    "What is the capital of Australia and why do people often get it wrong?",
    "How does HTTPS ensure a secure connection between a browser and server?",
    "What is the difference between machine learning and deep learning?",
    "Who wrote 'Pride and Prejudice' and when was it published?",
    "What causes the northern lights (aurora borealis)?",

    # Creative / open-ended
    "Write a haiku about a neural network that has lost its training data.",
    "Describe a future where large language models run entirely on smartphones.",
    "Write the opening sentence of a thriller novel set inside a data center.",
    "What would a conversation between Isaac Newton and a modern AI researcher sound like?",

    # Edge case (where quantization sometimes degrades)
    "Solve step by step: what is 17 × 23 + 144 ÷ 12?",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_bnb_model(model_id, adapter_path=None, token=None):
    """Load a model in NF4 int4, optionally with a LoRA adapter on top."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=token,
        trust_remote_code=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()    # fuse LoRA into weights for cleaner inference
    model.eval()
    return model, tokenizer


def compute_perplexity(model, tokenizer, num_windows=WIKITEXT_WINDOWS, window_size=WINDOW_SIZE):
    """
    Compute perplexity on WikiText-2 using a sliding window approach.
    Identical methodology to Phase 1 and Phase 2 for fair comparison.
    """
    print("  Loading WikiText-2 test split...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(model.device)

    total_tokens = input_ids.size(1)
    nlls = []
    count = 0

    with torch.no_grad():
        for begin in range(0, total_tokens - window_size, STRIDE):
            if count >= num_windows:
                break
            end = begin + window_size
            chunk = input_ids[:, begin:end]
            labels = chunk.clone()
            outputs = model(input_ids=chunk, labels=labels)
            nlls.append(outputs.loss.item())
            count += 1
            if count % 16 == 0:
                print(f"    [{count}/{num_windows} windows] running PPL: "
                      f"{math.exp(sum(nlls) / len(nlls)):.4f}")

    ppl = math.exp(sum(nlls) / len(nlls))
    return ppl


def measure_speed(model, tokenizer, num_runs=SPEED_RUNS, new_tokens=SPEED_TOKENS):
    """Measure inference speed in tokens/sec. Matches Phase 1/2 methodology."""
    prompt = (
        "Below is an instruction that describes a task. Write a response "
        "that appropriately completes the request.\n\n"
        "### Instruction:\nExplain what gradient descent is.\n\n"
        "### Response:\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    times = []

    with torch.no_grad():
        for i in range(num_runs + 1):           # +1 warm-up
            torch.cuda.synchronize()
            t0 = time.time()
            _ = model.generate(
                **inputs,
                max_new_tokens=new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            torch.cuda.synchronize()
            elapsed = time.time() - t0
            if i > 0:                           # discard warm-up run
                times.append(new_tokens / elapsed)

    return sum(times) / len(times)


def generate_response(model, tokenizer, instruction, max_new_tokens=256):
    """Generate a response for a single instruction-following prompt."""
    prompt = (
        "Below is an instruction that describes a task. Write a response "
        "that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        "### Response:\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only the newly generated tokens
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("QuantLlama Phase 3 — Evaluation")
    print("=" * 60)

    if not os.path.exists(ADAPTER_PATH):
        raise FileNotFoundError(
            f"Adapter not found at {ADAPTER_PATH}. "
            "Run finetune.py first."
        )

    results = {}

    # ── A. Evaluate fine-tuned QLoRA model ────────────────────────────────────
    print("\n[A] Loading fine-tuned QLoRA model (base NF4 + adapter)...")
    torch.cuda.reset_peak_memory_stats()
    t_load = time.time()
    ft_model, ft_tokenizer = load_bnb_model(BASE_MODEL, ADAPTER_PATH, HF_TOKEN)
    load_time_ft = time.time() - t_load
    vram_ft = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Loaded in {load_time_ft:.1f}s | VRAM: {vram_ft:.2f} GB")

    print("\n  Computing perplexity (fine-tuned)...")
    ppl_ft = compute_perplexity(ft_model, ft_tokenizer)
    print(f"  Perplexity (fine-tuned): {ppl_ft:.4f}")

    print("\n  Measuring inference speed (fine-tuned)...")
    speed_ft = measure_speed(ft_model, ft_tokenizer)
    print(f"  Speed (fine-tuned): {speed_ft:.2f} tok/s")

    results["finetuned"] = {
        "config": "meta-llama/Llama-3.1-8B | int4-nf4 | QLoRA-Alpaca (fine-tuned)",
        "precision": "int4-nf4-qlora",
        "perplexity_wikitext2": round(ppl_ft, 4),
        "speed_tokens_per_sec": round(speed_ft, 2),
        "peak_vram_gb": round(vram_ft, 3),
        "load_time_sec": round(load_time_ft, 1),
    }

    # ── B. Qualitative evaluation — base NF4 vs fine-tuned ────────────────────
    print("\n[B] Qualitative evaluation — 20 prompts...")
    print("    Loading base NF4 model for side-by-side comparison...")

    torch.cuda.reset_peak_memory_stats()
    base_model, base_tokenizer = load_bnb_model(BASE_MODEL, None, HF_TOKEN)

    qual_rows = []
    for i, prompt in enumerate(QUAL_PROMPTS, 1):
        print(f"  [{i:02d}/20] {prompt[:60]}...")

        # Run both models (sequentially to avoid VRAM pressure)
        base_resp = generate_response(base_model, base_tokenizer, prompt)
        ft_resp   = generate_response(ft_model,   ft_tokenizer,   prompt)

        qual_rows.append({
            "prompt_id": i,
            "prompt": prompt,
            "base_int4_nf4_response": base_resp,
            "finetuned_qlora_response": ft_resp,
        })

    # Save qualitative results (ensure results/ dir exists)
    os.makedirs(os.path.dirname(os.path.abspath(RESULTS_CSV)), exist_ok=True)
    with open(QUAL_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["prompt_id", "prompt", "base_int4_nf4_response", "finetuned_qlora_response"],
        )
        writer.writeheader()
        writer.writerows(qual_rows)
    print(f"\n  Qualitative results saved to {QUAL_CSV}")

    # ── C. Append to results.csv ───────────────────────────────────────────────
    print("\n[C] Appending results to results.csv...")
    fieldnames = [
        "config", "precision", "perplexity_wikitext2",
        "speed_tokens_per_sec", "peak_vram_gb", "load_time_sec",
    ]
    csv_exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not csv_exists:
            writer.writeheader()
        writer.writerow(results["finetuned"])
    print(f"  Written to {RESULTS_CSV}")

    # ── D. Print final comparison table ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 3 RESULTS")
    print("=" * 60)
    print(f"{'Config':<45} {'PPL':>7} {'tok/s':>7} {'VRAM':>6}")
    print("-" * 70)

    # Load all rows from CSV for full comparison
    with open(RESULTS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            short = row["config"].split("|")[-1].strip()[:44]
            ppl   = row["perplexity_wikitext2"]
            spd   = row["speed_tokens_per_sec"]
            vram  = row["peak_vram_gb"]
            print(f"{short:<45} {ppl:>7} {spd:>7} {vram:>6}")
    print("=" * 60)


if __name__ == "__main__":
    main()