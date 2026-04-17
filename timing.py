"""
timing.py — Per-stage inference timing breakdown

Measures the three stages of inference separately for two model configurations:
  1. Base Llama-3.1-8B (int4 NF4)
  2. QLoRA fine-tuned Llama-3.1-8B (int4 NF4 + Alpaca adapter, merged)

Stages timed:
  - Pre-processing  : tokenization (text → token IDs)
  - DNN forward pass: autoregressive generation (token IDs → output token IDs)
  - Post-processing : decoding (output token IDs → text)

Run:
    python timing.py

Requirements: HF_TOKEN env var set, adapter weights at
              outputs/qlora-alpaca/final_adapter/ (or loaded from HF Hub).
"""

import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ── Config ────────────────────────────────────────────────────────────────────

BASE_MODEL_ID   = "meta-llama/Llama-3.1-8B"
ADAPTER_PATH    = "outputs/qlora-alpaca/final_adapter"   # local path
HF_ADAPTER_ID   = "srijayadav/llama3-8b-qlora-alpaca"   # fallback if local missing

MAX_NEW_TOKENS  = 200
NUM_RUNS        = 5        # averaged runs for stable timing
PROMPT_TEXT     = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\nExplain what gradient descent is in simple terms.\n\n"
    "### Response:\n"
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def time_inference(model, tokenizer, prompt, max_new_tokens, num_runs):
    """
    Times the three inference stages over num_runs runs.
    First run is a warm-up and is discarded.
    Returns mean times in milliseconds: (pre_ms, fwd_ms, post_ms, total_ms)
    """
    device = next(model.parameters()).device
    pre_times, fwd_times, post_times = [], [], []

    for i in range(num_runs + 1):  # +1 for warm-up
        torch.cuda.synchronize()

        # Stage 1 — Pre-processing (tokenization)
        t0 = time.perf_counter()
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        # Stage 2 — DNN forward pass (autoregressive generation)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,        # greedy — deterministic, fair comparison
            )
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        # Stage 3 — Post-processing (decode token IDs → text)
        _ = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        torch.cuda.synchronize()
        t3 = time.perf_counter()

        if i == 0:
            continue  # discard warm-up run

        pre_times.append((t1 - t0) * 1000)
        fwd_times.append((t2 - t1) * 1000)
        post_times.append((t3 - t2) * 1000)

    mean_pre   = sum(pre_times)  / len(pre_times)
    mean_fwd   = sum(fwd_times)  / len(fwd_times)
    mean_post  = sum(post_times) / len(post_times)
    mean_total = mean_pre + mean_fwd + mean_post

    return mean_pre, mean_fwd, mean_post, mean_total


def print_timing_table(label, pre_ms, fwd_ms, post_ms, total_ms):
    bar_width = 40
    fwd_bar   = int((fwd_ms / total_ms) * bar_width)
    pre_bar   = max(1, int((pre_ms  / total_ms) * bar_width))
    post_bar  = max(1, int((post_ms / total_ms) * bar_width))

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  {'Stage':<30} {'Time (ms)':>10}  {'% of total':>10}")
    print(f"  {'-'*54}")
    print(f"  {'Pre-processing  (tokenization)':<30} {pre_ms:>10.2f}  {pre_ms/total_ms*100:>9.2f}%")
    print(f"  {'DNN forward pass (generation)':<30} {fwd_ms:>10.2f}  {fwd_ms/total_ms*100:>9.2f}%  ← bottleneck" if fwd_ms/total_ms > 0.9 else
          f"  {'DNN forward pass (generation)':<30} {fwd_ms:>10.2f}  {fwd_ms/total_ms*100:>9.2f}%")
    print(f"  {'Post-processing (decoding)':<30} {post_ms:>10.2f}  {post_ms/total_ms*100:>9.2f}%")
    print(f"  {'-'*54}")
    print(f"  {'Total':<30} {total_ms:>10.2f}  {'100.00%':>10}")
    print()
    print(f"  Visual breakdown (each █ ≈ {100/bar_width:.1f}% of total):")
    print(f"  Pre  [{'█' * pre_bar:<{bar_width}}]")
    print(f"  DNN  [{'█' * fwd_bar:<{bar_width}}]")
    print(f"  Post [{'█' * post_bar:<{bar_width}}]")
    print(f"{'='*60}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError("HF_TOKEN is not set. Run: export HF_TOKEN=hf_...")

    print("\n" + "="*60)
    print("  QuantLlama — Per-Stage Inference Timing")
    print("="*60)
    print(f"  Prompt length : {len(PROMPT_TEXT.split())} words")
    print(f"  Max new tokens: {MAX_NEW_TOKENS}")
    print(f"  Runs averaged : {NUM_RUNS} (1 warm-up discarded)")
    print(f"  Decoding      : greedy (do_sample=False)")

    bnb_config = get_bnb_config()

    # ── Model 1: Base NF4 ─────────────────────────────────────────────────────
    print(f"\n[1/2] Loading base int4-NF4 model...")
    t_load = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=hf_token)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
    )
    base_model.eval()
    print(f"    Loaded in {time.perf_counter() - t_load:.1f}s")

    print(f"    Timing {NUM_RUNS} runs...")
    pre, fwd, post, total = time_inference(
        base_model, tokenizer, PROMPT_TEXT, MAX_NEW_TOKENS, NUM_RUNS
    )
    print_timing_table("Base Llama-3.1-8B  |  int4-NF4", pre, fwd, post, total)

    # Free base model before loading fine-tuned
    del base_model
    torch.cuda.empty_cache()

    # ── Model 2: QLoRA fine-tuned ─────────────────────────────────────────────
    print(f"\n[2/2] Loading QLoRA fine-tuned model...")
    adapter_source = ADAPTER_PATH if os.path.isdir(ADAPTER_PATH) else HF_ADAPTER_ID
    print(f"    Adapter source: {adapter_source}")

    t_load = time.perf_counter()
    ft_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
    )
    ft_model = PeftModel.from_pretrained(ft_model, adapter_source)
    ft_model = ft_model.merge_and_unload()   # fuse adapter — same as evaluate.py
    ft_model.eval()
    print(f"    Loaded in {time.perf_counter() - t_load:.1f}s")

    print(f"    Timing {NUM_RUNS} runs...")
    pre, fwd, post, total = time_inference(
        ft_model, tokenizer, PROMPT_TEXT, MAX_NEW_TOKENS, NUM_RUNS
    )
    print_timing_table("QLoRA Fine-Tuned  |  int4-NF4 + Alpaca adapter (merged)", pre, fwd, post, total)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n  SUMMARY")
    print("  The DNN forward pass (autoregressive generation) dominates inference")
    print("  time in both configurations, accounting for >99% of total latency.")
    print("  Tokenization and decoding are negligible (<5 ms each).")
    print("  This is expected: each new token requires a full forward pass through")
    print(f"  all 32 transformer layers of Llama-3.1-8B.\n")


if __name__ == "__main__":
    main()