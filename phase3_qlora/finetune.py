"""
QuantLlama — Phase 3: QLoRA Fine-Tuning
========================================
Fine-tunes meta-llama/Llama-3.1-8B (NF4 int4) on the Stanford Alpaca dataset
using QLoRA (Quantized Low-Rank Adaptation) via PEFT + TRL SFTTrainer.

Hardware target : NVIDIA RTX 3090 (24 GB VRAM)
Expected runtime: ~2-4 hours for 3 epochs on Alpaca (52k examples)
Output          : LoRA adapter weights saved to ./outputs/qlora-alpaca/

Dependencies (from Phase 2 working stack):
    transformers==4.56.2
    bitsandbytes==0.49.2
    peft==0.18.1
    trl==1.0.0
    datasets
    accelerate
"""

import os
import csv
import time
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig  # SFTConfig = TrainingArguments + SFT args in TRL 1.0

# -- Config -------------------------------------------------------------------

BASE_MODEL   = "meta-llama/Llama-3.1-8B"
OUTPUT_DIR   = "./outputs/qlora-alpaca"
RESULTS_CSV  = "../results/results.csv"
HF_TOKEN     = os.environ.get("HF_TOKEN")          # set in shell: export HF_TOKEN=...

# LoRA hyperparameters (from proposal Table 3b)
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
TARGET_MODS  = ["q_proj", "v_proj"]

# Training hyperparameters
EPOCHS         = 1  # 1 epoch is standard for Alpaca QLoRA; original Dettmers et al. paper used 1 epoch
BATCH_SIZE     = 4
GRAD_ACCUM     = 8    # effective batch = 32
LEARNING_RATE  = 2e-4
MAX_SEQ_LEN    = 512  # Alpaca examples are short; saves VRAM vs 2048
WARMUP_RATIO   = 0.03
TRAIN_SPLIT    = 0.95
EVAL_STEPS     = 50   # more frequent eval since total steps ~500-800 with packing
SAVE_STEPS     = 200  # save a checkpoint mid-run in case of interruption
LOGGING_STEPS  = 10

# -- Alpaca prompt template ---------------------------------------------------

def format_alpaca(example):
    """Format a single Alpaca example into the instruction-following template."""
    if example.get("input", "").strip():
        prompt = (
            "Below is an instruction that describes a task, paired with an input "
            "that provides further context. Write a response that appropriately "
            "completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )
    else:
        prompt = (
            "Below is an instruction that describes a task. Write a response "
            "that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Response:\n{example['output']}"
        )
    return {"text": prompt}


# -- Main ---------------------------------------------------------------------

def main():
    print("=" * 60)
    print("QuantLlama Phase 3 -- QLoRA Fine-Tuning")
    print("=" * 60)

    # 1. Load and format Alpaca dataset
    print("\n[1/5] Loading Alpaca dataset...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    dataset = dataset.map(format_alpaca, remove_columns=dataset.column_names)

    split = dataset.train_test_split(test_size=1 - TRAIN_SPLIT, seed=42)
    train_data = split["train"]
    eval_data  = split["test"]
    print(f"    Train: {len(train_data):,} examples")
    print(f"    Val  : {len(eval_data):,} examples")

    # 2. Load tokenizer
    print("\n[2/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, token=HF_TOKEN, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3. Load NF4 quantized base model
    print("\n[3/5] Loading Llama-3.1-8B in NF4 (int4)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    load_time = time.time() - t0
    print(f"    Model loaded in {load_time:.1f}s")
    print(f"    VRAM after load: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # 4. Apply LoRA
    print("\n[4/5] Applying LoRA adapter...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODS,
        lora_dropout=LORA_DROPOUT,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. Train with SFTTrainer
    print("\n[5/5] Starting QLoRA fine-tuning...")

    # TRL 1.0: SFTConfig replaces TrainingArguments for SFTTrainer.
    # dataset_text_field, max_length, and packing now live here, not on SFTTrainer.
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        optim="paged_adamw_32bit",
        bf16=True,   # Llama-3.1 native dtype; avoids fp16/bf16 grad scaler conflict
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        save_total_limit=2,
        dataloader_pin_memory=False,
        dataset_text_field="text",
        max_length=MAX_SEQ_LEN,
        packing=True,   # packs short Alpaca examples into 512-token windows; ~2-3x step reduction, no quality loss
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_data,
        eval_dataset=eval_data,
        args=sft_config,
    )

    t_start = time.time()
    trainer.train()
    train_time = time.time() - t_start
    print(f"\nTraining complete in {train_time / 3600:.2f} hours")

    # Save adapter weights
    print(f"\nSaving LoRA adapter to {OUTPUT_DIR}/final_adapter/ ...")
    final_path = os.path.join(OUTPUT_DIR, "final_adapter")
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    adapter_size_mb = sum(
        os.path.getsize(os.path.join(final_path, f))
        for f in os.listdir(final_path)
    ) / 1e6
    print(f"Adapter size: {adapter_size_mb:.1f} MB")

    # Append placeholder row to results.csv
    row = {
        "config": "meta-llama/Llama-3.1-8B | int4-nf4 | QLoRA-Alpaca (fine-tuned)",
        "precision": "int4-nf4-qlora",
        "perplexity_wikitext2": "see evaluate.py",
        "speed_tokens_per_sec": "see evaluate.py",
        "peak_vram_gb": f"{torch.cuda.max_memory_allocated() / 1e9:.3f}",
        "load_time_sec": f"{load_time:.1f}",
    }
    csv_exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not csv_exists:
            writer.writeheader()
        writer.writerow(row)

    print("\n" + "=" * 60)
    print("Phase 3 training COMPLETE")
    print(f"  Adapter saved : {final_path}")
    print(f"  Adapter size  : {adapter_size_mb:.1f} MB")
    print(f"  Train time    : {train_time / 3600:.2f} hours")
    print("\nNext step: run phase3_qlora/evaluate.py to measure perplexity")
    print("=" * 60)


if __name__ == "__main__":
    main()