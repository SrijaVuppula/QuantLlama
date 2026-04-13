# QuantLlama — Phase 4 Summary

## Project Info
- **Repo:** https://github.com/SrijaVuppula/QuantLlama
- **Course:** CSCI 4900/6900 — Foundations of Deep Learning and Generative AI, UGA Spring 2026
- **Model:** `meta-llama/Llama-3.1-8B`
- **Hardware:** NVIDIA GeForce RTX 3090 — 24 GB VRAM (CS department GPU, accessed via VS Code Remote SSH)

---

## Goal
Push the Phase 3 QLoRA adapter weights to Hugging Face Hub, write a model card, build a live Gradio side-by-side demo comparing base int4-NF4 vs the fine-tuned model, and deploy it as a Hugging Face Space.

---

## Deliverables

| Artifact | Status | URL |
|---|---|---|
| HF Model repo (adapter weights + model card) | ✅ Live | https://huggingface.co/srijayadav/llama3-8b-qlora-alpaca |
| HF Space (app code) | ✅ Live | https://huggingface.co/spaces/srijayadav/quantllama-demo |
| Live Gradio demo (gradio.live, 1-week link) | ✅ Running | https://f27035229341fdab36.gradio.live |

---

## Phase 4a — Hugging Face Hub Upload

### Script: `phase4_demo/push_to_hub.py`

Uploads all files from `outputs/qlora-alpaca/final_adapter/` to the public HF Hub repo
`srijayadav/llama3-8b-qlora-alpaca` in a single `upload_folder` call.

**Files uploaded:**
| File | Description |
|---|---|
| `adapter_model.safetensors` | LoRA adapter weights (44.5 MB) |
| `adapter_config.json` | LoRA configuration (rank, alpha, target modules, etc.) |
| `tokenizer.json` | Llama-3.1 tokenizer (17.2 MB) |
| `tokenizer_config.json` | Tokenizer config |
| `special_tokens_map.json` | Special token definitions |
| `README.md` | Model card |

**Issues encountered:**
| Error | Cause | Fix |
|---|---|---|
| `403 Forbidden` on `create_repo` | HF token was read-only | Generated new write-access token at huggingface.co/settings/tokens |
| `$HF_TOKEN` empty on shell start | Token not persisted across sessions | `export HF_TOKEN=hf_...` at session start |

---

## Phase 4b — Model Card

The default PEFT-generated `README.md` was a stub full of `[More Information Needed]` placeholders.
Replaced with a proper model card including YAML frontmatter, full 4-row benchmark table, complete
usage code, training details table, and BibTeX citations for QLoRA and Alpaca papers.

---

## Phase 4c — Gradio Demo

### Script: `phase4_demo/app.py`

Side-by-side Gradio interface comparing base int4-NF4 vs QLoRA fine-tuned model.

**Architecture:**
- Both models loaded at startup (base NF4 + fine-tuned with `merge_and_unload()`)
- Inference runs in parallel via `threading.Thread`
- Per-generation metadata: latency (s), throughput (tok/s), peak VRAM (GB)

**UI components:**
- Instruction text area
- Max new tokens slider (50–500, default 200)
- Temperature slider (0.0–1.5, default 0.7)
- Two output panels side by side (base vs fine-tuned)
- Metadata row under each panel
- 5 example prompts

**Startup output (verified working):**
Loading tokenizer...
Loading base int4-NF4 model...
Base model loaded in 115.9s
Loading QLoRA fine-tuned model...
Fine-tuned model loaded in 115.5s
Both models ready.

Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://f27035229341fdab36.gradio.live
**Warnings (harmless):**
- `FutureWarning: _check_is_size will be removed` — bitsandbytes internal, no impact
- `UserWarning: Merge lora module to 4-bit linear may get different generations due to rounding errors` — expected with merge_and_unload on quantized model

---

## Phase 4d — HF Spaces Deployment

Space created at `srijayadav/quantllama-demo` using `create_repo(..., repo_type='space', space_sdk='gradio')`.
`app.py` and `requirements.txt` uploaded via `api.upload_file()`.

**GPU tier upgrade — not completed:**
Attempted to set hardware to `t4-small` via `api.request_space_hardware()`. Got `402 Payment Required` —
HF requires pre-paid credits for GPU Spaces. Decision: use the department GPU + `share=True`
gradio.live URL for the presentation instead. The Space page remains live with the code visible.

---

## Live Demo Test Results

Tested with prompt: `"explain neural networks"`, max_tokens=50, temperature=0.7

| Panel | Time | Speed | VRAM |
|---|---|---|---|
| Base int4-NF4 | 29.54s | 6.8 tok/s | 11.68 GB |
| QLoRA fine-tuned | 29.61s | 6.8 tok/s | 11.68 GB |

**Key qualitative observation:**
The base model bled into a second `### Instruction:` block mid-generation — classic behavior of a
non-instruction-tuned model. The fine-tuned model produced clean structured paragraphs with no
format bleed. Strong slide moment for the presentation.

---

## results/results.csv — Final (All 4 Phases)
config,precision,perplexity_wikitext2,speed_tokens_per_sec,peak_vram_gb,load_time_sec
meta-llama/Llama-3.1-8B | fp16 | baseline,fp16,6.1885,36.26,18.984,256.5
meta-llama/Llama-3.1-8B | int4-nf4 | bitsandbytes,int4-nf4,6.5784,18.01,9.425,116.0
meta-llama/Llama-3.1-8B | int4-gptq | ModelCloud pretrained,int4-gptq,6.5497,4.35,9.453,40.9
meta-llama/Llama-3.1-8B | int4-nf4 | QLoRA-Alpaca (fine-tuned),int4-nf4-qlora,6.6251,53.39,7.504,96.1
---

## Repo Structure After Phase 4
QuantLlama/
├── phase4_demo/
│   ├── push_to_hub.py          # Uploads adapter weights to HF Hub
│   ├── app.py                  # Gradio side-by-side demo
│   └── requirements.txt        # Space dependencies
└── results/
├── results.csv
├── qualitative_phase3.csv
└── plots/
├── perplexity.png
├── speed.png
├── vram.png
└── combined.png
