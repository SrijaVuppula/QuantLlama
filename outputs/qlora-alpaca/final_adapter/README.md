---
base_model: meta-llama/Llama-3.1-8B
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:meta-llama/Llama-3.1-8B
- lora
- qlora
- sft
- transformers
- trl
- quantization
- instruction-tuning
license: llama3
language:
- en
datasets:
- tatsu-lab/alpaca
---

# Llama-3.1-8B QLoRA Alpaca — QuantLlama

QLoRA fine-tuned adapter for `meta-llama/Llama-3.1-8B` on the Stanford Alpaca instruction-following dataset. Trained as part of a benchmarking study on 4-bit quantization and parameter-efficient fine-tuning for CSCI 4900/6900 at the University of Georgia.

This is a **LoRA adapter only** — load it on top of the base model quantized to int4 (NF4) via bitsandbytes.

## Benchmark Results

All configs evaluated on WikiText-2 perplexity, inference speed, and peak VRAM on an NVIDIA RTX 3090 (24 GB).

| Config | Precision | Perplexity ↓ | Speed (tok/s) ↑ | VRAM (GB) ↓ |
|---|---|---|---|---|
| Llama-3.1-8B fp16 baseline | fp16 | 6.1885 | 36.26 | 18.984 |
| Llama-3.1-8B NF4 (bitsandbytes) | int4-nf4 | 6.5784 | 18.01 | 9.425 |
| Llama-3.1-8B GPTQ (ModelCloud) | int4-gptq | 6.5497 | 4.35 | 9.453 |
| **Llama-3.1-8B QLoRA-Alpaca (this model)** | int4-nf4-qlora | **6.6251** | **53.39** | **7.504** |

The fine-tuned model is the **fastest and most memory-efficient** of all four configurations after `merge_and_unload()` fuses the LoRA matrices into the base weights.

## How to Use

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

base_model_id = "meta-llama/Llama-3.1-8B"
adapter_id = "srijayadav/llama3-8b-qlora-alpaca"

# Load base model in int4
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(adapter_id)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

# Load LoRA adapter on top
model = PeftModel.from_pretrained(model, adapter_id)

# Inference
prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nExplain what gradient descent is in simple terms.\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training Details

| Parameter | Value |
|---|---|
| Base model | meta-llama/Llama-3.1-8B |
| Dataset | tatsu-lab/alpaca (49,401 train / 2,601 val) |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| Target modules | q_proj, v_proj |
| Dropout | 0.05 |
| Epochs | 1 |
| Batch size | 4 (grad accum 8, effective 32) |
| Learning rate | 2e-4 |
| LR scheduler | cosine with 3% warmup |
| Max sequence length | 512 |
| Packing | True |
| Optimizer | paged_adamw_32bit |
| Compute dtype | bfloat16 |
| Training time | ~1.92 hours |
| Adapter size | 44.5 MB |
| Hardware | NVIDIA RTX 3090 24 GB |
| Trainable params | 6,815,744 / 8,037,076,992 (0.085%) |

## Citation

```bibtex
@article{dettmers2023qlora,
  title={QLoRA: Efficient Finetuning of Quantized LLMs},
  author={Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  journal={NeurIPS},
  year={2023}
}

@misc{taori2023alpaca,
  title={Alpaca: A Strong, Replicable Instruction-Following Model},
  author={Taori, Rohan and others},
  year={2023},
  publisher={Stanford Center for Research on Foundation Models}
}
```
