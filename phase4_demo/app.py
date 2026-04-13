import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import threading
import time

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL_ID  = "meta-llama/Llama-3.1-8B"
ADAPTER_ID     = "srijayadav/llama3-8b-qlora-alpaca"
ALPACA_PROMPT  = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)

# ── Load models at startup ────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_ID)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("Loading base int4-NF4 model...")
t0 = time.time()
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)
base_model.eval()
base_load_time = time.time() - t0
print(f"Base model loaded in {base_load_time:.1f}s")

print("Loading QLoRA fine-tuned model...")
t0 = time.time()
ft_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)
ft_model = PeftModel.from_pretrained(ft_model, ADAPTER_ID)
ft_model = ft_model.merge_and_unload()
ft_model.eval()
ft_load_time = time.time() - t0
print(f"Fine-tuned model loaded in {ft_load_time:.1f}s")
print("Both models ready.")

# ── Inference helper ──────────────────────────────────────────────────────────
def generate(model, prompt, max_new_tokens, temperature):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0
    vram_gb = torch.cuda.max_memory_allocated() / 1e9
    new_tokens = output_ids.shape[1] - inputs["input_ids"].shape[1]
    tok_per_sec = new_tokens / elapsed if elapsed > 0 else 0
    response = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    meta = f"⏱ {elapsed:.2f}s · {tok_per_sec:.1f} tok/s · {vram_gb:.2f} GB VRAM"
    return response.strip(), meta

# ── Gradio inference function ─────────────────────────────────────────────────
def run_both(instruction, max_new_tokens, temperature):
    if not instruction.strip():
        return "Please enter an instruction.", "", "", ""

    prompt = ALPACA_PROMPT.format(instruction=instruction.strip())

    results = {}

    def run_base():
        results["base"] = generate(base_model, prompt, max_new_tokens, temperature)

    def run_ft():
        results["ft"] = generate(ft_model, prompt, max_new_tokens, temperature)

    t_base = threading.Thread(target=run_base)
    t_ft   = threading.Thread(target=run_ft)
    t_base.start(); t_ft.start()
    t_base.join();  t_ft.join()

    base_out, base_meta = results["base"]
    ft_out,   ft_meta   = results["ft"]
    return base_out, base_meta, ft_out, ft_meta

# ── Gradio UI ─────────────────────────────────────────────────────────────────
EXAMPLES = [
    ["Explain gradient descent in simple terms."],
    ["Write a Python function that reverses a string."],
    ["What are three benefits of exercise?"],
    ["Summarize the plot of Romeo and Juliet in two sentences."],
    ["Give me a recipe for a simple pasta dish."],
]

with gr.Blocks(title="QuantLlama — QLoRA Demo") as demo:
    gr.Markdown(
        """
# QuantLlama — Llama-3.1-8B: Base int4-NF4 vs QLoRA Fine-Tuned
**CSCI 4900/6900 · University of Georgia · Spring 2026**

Side-by-side comparison of the base Llama-3.1-8B quantized to int4 (NF4) vs the same model
fine-tuned with QLoRA on the Stanford Alpaca dataset. Both run in 4-bit on a single GPU.

| Config | Perplexity ↓ | Speed ↑ | VRAM ↓ |
|---|---|---|---|
| Base int4-NF4 | 6.5784 | 18.01 tok/s | 9.43 GB |
| QLoRA fine-tuned | 6.6251 | 53.39 tok/s | 7.50 GB |
        """
    )

    with gr.Row():
        instruction = gr.Textbox(
            label="Instruction",
            placeholder="Enter an instruction, e.g. 'Explain gradient descent in simple terms.'",
            lines=3,
        )

    with gr.Row():
        max_new_tokens = gr.Slider(50, 500, value=200, step=50, label="Max new tokens")
        temperature    = gr.Slider(0.0, 1.5, value=0.7, step=0.1, label="Temperature")

    generate_btn = gr.Button("Generate ▶", variant="primary")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🔵 Base int4-NF4")
            base_output = gr.Textbox(label="Output", lines=12, interactive=False)
            base_meta   = gr.Textbox(label="Latency / throughput / VRAM", interactive=False)
        with gr.Column():
            gr.Markdown("### 🟢 QLoRA Fine-Tuned (Alpaca)")
            ft_output = gr.Textbox(label="Output", lines=12, interactive=False)
            ft_meta   = gr.Textbox(label="Latency / throughput / VRAM", interactive=False)

    gr.Examples(
        examples=EXAMPLES,
        inputs=instruction,
        label="Example prompts",
    )

    generate_btn.click(
        fn=run_both,
        inputs=[instruction, max_new_tokens, temperature],
        outputs=[base_output, base_meta, ft_output, ft_meta],
    )

if __name__ == "__main__":
    demo.launch(share=True)
