"""
Phase 2 — Comparison Table + Qualitative Evaluation
QuantLlama · CSCI 4900/6900 · UGA Spring 2026

Reads results/results.csv and prints a formatted benchmark comparison table
(fp16 vs int4-nf4 vs int4-gptq). Also runs a 20-prompt qualitative evaluation
across all loaded configurations and saves outputs to results/qualitative_eval.csv.

Usage:
    # Print comparison table only (no GPU required):
    python phase2_quantization/compare.py --table-only

    # Full qualitative eval (requires GPU, models loaded):
    python phase2_quantization/compare.py

Requirements: transformers, bitsandbytes, auto-gptq, torch (CUDA)
"""

import argparse
import csv
import time
from pathlib import Path

import torch

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID            = "meta-llama/Llama-3.1-8B"
GPTQ_CHECKPOINT_DIR = Path(__file__).parent / "gptq_checkpoint"
RESULTS_CSV         = Path(__file__).parent.parent / "results" / "results.csv"
QUAL_CSV            = Path(__file__).parent.parent / "results" / "qualitative_eval.csv"

# 20 evaluation prompts — mix of reasoning, factual, creative, and simple tasks
EVAL_PROMPTS = [
    # Complex reasoning (quantization tends to degrade these)
    "Explain step by step how to solve the Tower of Hanoi problem with 3 disks.",
    "If I have a 12-gallon jug and a 5-gallon jug, how do I measure exactly 6 gallons?",
    "A train leaves Station A at 60 mph. Another leaves Station B at 80 mph toward it. They are 280 miles apart. When do they meet?",
    "What are the logical flaws in this argument: 'All birds can fly. Penguins are birds. Therefore penguins can fly.'",
    "Describe the chain of events that would follow if the Moon suddenly disappeared.",

    # Technical / factual (usually robust to quantization)
    "What is the difference between supervised and unsupervised learning?",
    "Explain how a transformer attention mechanism works in simple terms.",
    "What is the Central Limit Theorem and why does it matter in statistics?",
    "What are the key differences between Python lists and tuples?",
    "How does gradient descent find the minimum of a loss function?",

    # Creative / open-ended (moderate sensitivity)
    "Write a short poem about the night sky.",
    "Describe a futuristic city in the year 2150.",
    "Write the opening paragraph of a mystery novel set in a lighthouse.",
    "Invent a new sport that can be played in zero gravity.",
    "Write a haiku about debugging code.",

    # Instruction-following (should improve post-QLoRA, good to baseline here)
    "List 5 tips for staying focused while working from home.",
    "Summarize the French Revolution in 3 sentences.",
    "Give me a recipe for a simple pasta dish with 5 ingredients.",
    "Translate the following to formal academic language: 'The experiment didn't work out.'",
    "What should I pack for a 3-day hiking trip?",
]

MAX_NEW_TOKENS = 256
TEMPERATURE    = 0.0   # greedy, for reproducibility

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--table-only",
    action="store_true",
    help="Print comparison table from results.csv only. No GPU or models needed.",
)
args = parser.parse_args()

# ── Helper: print comparison table ───────────────────────────────────────────
def print_comparison_table():
    if not RESULTS_CSV.exists():
        print(f"[error] {RESULTS_CSV} not found. Run Phase 1 and Phase 2 scripts first.")
        return

    with open(RESULTS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("[error] results.csv is empty.")
        return

    # Order: fp16 → nf4 → gptq
    order = {"fp16": 0, "int4-nf4": 1, "int4-gptq": 2}
    rows.sort(key=lambda r: order.get(r["precision"], 99))

    # Find fp16 row for delta calculations
    fp16 = next((r for r in rows if r["precision"] == "fp16"), None)

    col_w = [30, 10, 12, 14, 12, 12]
    headers = ["Config", "Precision", "Perplexity↓", "Speed (tok/s)↑", "VRAM (GB)↓", "Load (s)"]

    sep  = "─" * (sum(col_w) + len(col_w) * 3 + 1)
    fmt  = "│ " + " │ ".join(f"{{:<{w}}}" for w in col_w) + " │"

    print(f"\n{'='*70}")
    print("  QuantLlama — Phase 2 Benchmark Comparison")
    print(f"{'='*70}")
    print(sep)
    print(fmt.format(*headers))
    print(sep)

    for r in rows:
        label = r["config"].split("|")[0].strip()[:col_w[0]]
        vals  = [
            label,
            r["precision"],
            r["perplexity_wikitext2"],
            r["speed_tokens_per_sec"],
            r["peak_vram_gb"],
            r["load_time_sec"],
        ]
        print(fmt.format(*[str(v) for v in vals]))

    print(sep)

    # Delta summary vs fp16
    if fp16 and len(rows) > 1:
        print("\n  Deltas vs fp16 baseline:")
        for r in rows:
            if r["precision"] == "fp16":
                continue
            try:
                ppl_delta   = float(r["perplexity_wikitext2"]) - float(fp16["perplexity_wikitext2"])
                speed_ratio = float(r["speed_tokens_per_sec"])  / float(fp16["speed_tokens_per_sec"])
                vram_ratio  = float(r["peak_vram_gb"])           / float(fp16["peak_vram_gb"])
                print(f"    {r['precision']:<12}  "
                      f"perplexity +{ppl_delta:+.4f}   "
                      f"speed ×{speed_ratio:.2f}   "
                      f"VRAM ×{vram_ratio:.2f} ({(1-vram_ratio)*100:.0f}% reduction)")
            except (ValueError, ZeroDivisionError):
                pass

    print(f"\n  Source: {RESULTS_CSV}\n")


# ── Table-only mode ───────────────────────────────────────────────────────────
if args.table_only:
    print_comparison_table()
    raise SystemExit(0)

# ── Full qualitative eval mode ────────────────────────────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

configs = {}

print("\n[load]  Loading fp16 model...")
tok_fp16 = AutoTokenizer.from_pretrained(MODEL_ID)
mdl_fp16 = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map="auto"
)
mdl_fp16.eval()
configs["fp16"] = (mdl_fp16, tok_fp16)

print("[load]  Loading NF4 model...")
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
tok_nf4 = AutoTokenizer.from_pretrained(MODEL_ID)
mdl_nf4 = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_cfg, device_map="auto", torch_dtype=torch.float16
)
mdl_nf4.eval()
configs["int4-nf4"] = (mdl_nf4, tok_nf4)

if GPTQ_CHECKPOINT_DIR.exists():
    print("[load]  Loading GPTQ model...")
    from auto_gptq import AutoGPTQForCausalLM
    tok_gptq = AutoTokenizer.from_pretrained(str(GPTQ_CHECKPOINT_DIR))
    mdl_gptq = AutoGPTQForCausalLM.from_quantized(
        str(GPTQ_CHECKPOINT_DIR), device_map="auto", use_safetensors=True
    )
    mdl_gptq.eval()
    configs["int4-gptq"] = (mdl_gptq, tok_gptq)
else:
    print(f"[warn]  No GPTQ checkpoint at {GPTQ_CHECKPOINT_DIR} — skipping GPTQ column.")

# ── Run qualitative eval ──────────────────────────────────────────────────────
print(f"\n[qual]  Running {len(EVAL_PROMPTS)} prompts × {len(configs)} configs...\n")

QUAL_CSV.parent.mkdir(parents=True, exist_ok=True)
fieldnames = ["prompt_id", "prompt"] + list(configs.keys())

results = []

for i, prompt in enumerate(EVAL_PROMPTS):
    row = {"prompt_id": i + 1, "prompt": prompt}
    print(f"  Prompt {i+1:2d}/{len(EVAL_PROMPTS)}: {prompt[:60]}...")

    for cfg_name, (model, tokenizer) in configs.items():
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            out_ids = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Decode only the generated portion
        generated = tokenizer.decode(
            out_ids[0, input_ids.size(1):], skip_special_tokens=True
        ).strip()
        row[cfg_name] = generated
        print(f"    [{cfg_name}] {generated[:80]}...")

    results.append(row)

with open(QUAL_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"\n[done]  Qualitative results saved to {QUAL_CSV}")

# Print the benchmark table at the end for convenience
print_comparison_table()
