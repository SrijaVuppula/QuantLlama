"""
QuantLlama — Plot Benchmark Results
=====================================
Reads results/results.csv and saves three publication-quality bar charts:
    results/plots/perplexity.png
    results/plots/speed.png
    results/plots/vram.png

Usage:
    python plot_results.py

Dependencies: matplotlib, pandas (both standard in the venv)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Load results ──────────────────────────────────────────────────────────────

CSV_PATH  = "results/results.csv"
PLOT_DIR  = "results/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# Drop any placeholder rows from the echo command
df = df[~df["perplexity_wikitext2"].astype(str).str.contains("see evaluate")]
df = df.reset_index(drop=True)

# Short labels for x-axis
LABELS = ["fp16\nbaseline", "int4\nNF4", "int4\nGPTQ", "QLoRA\nfine-tuned"]
COLORS = ["#185FA5", "#639922", "#BA7517", "#533AB7"]
EDGE   = ["#0C447C", "#3B6D11", "#854F0B", "#3C3489"]

# ── Shared style ──────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        12,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "axes.grid.axis":   "y",
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "figure.dpi":       150,
})

def save_bar(values, ylabel, title, filename, ymin, ymax, higher_better=False):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    bars = ax.bar(
        LABELS, values,
        color=COLORS, edgecolor=EDGE, linewidth=0.8, width=0.5
    )

    # Value labels on top of each bar
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (ymax - ymin) * 0.015,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
            color="#333333"
        )

    # Highlight best bar with a star annotation
    best_idx = values.index(min(values)) if not higher_better else values.index(max(values))
    best_bar = bars[best_idx]
    ax.annotate(
        "best",
        xy=(best_bar.get_x() + best_bar.get_width() / 2, best_bar.get_height()),
        xytext=(0, 18), textcoords="offset points",
        ha="center", fontsize=9, color=EDGE[best_idx],
        arrowprops=dict(arrowstyle="-", color=EDGE[best_idx], lw=0.8)
    )

    ax.set_ylim(ymin, ymax)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.tick_params(axis="x", length=0, pad=8)

    fig.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 1. Perplexity ─────────────────────────────────────────────────────────────

ppl_values = df["perplexity_wikitext2"].astype(float).tolist()
save_bar(
    values       = ppl_values,
    ylabel       = "Perplexity on WikiText-2  (lower = better)",
    title        = "Perplexity: fp16 vs int4 NF4 vs int4 GPTQ vs QLoRA fine-tuned",
    filename     = "perplexity.png",
    ymin         = 5.8,
    ymax         = 7.1,
    higher_better= False,
)

# ── 2. Inference speed ────────────────────────────────────────────────────────

speed_values = df["speed_tokens_per_sec"].astype(float).tolist()
save_bar(
    values       = speed_values,
    ylabel       = "Tokens / second  (higher = better)",
    title        = "Inference speed: fp16 vs int4 NF4 vs int4 GPTQ vs QLoRA fine-tuned",
    filename     = "speed.png",
    ymin         = 0,
    ymax         = 65,
    higher_better= True,
)

# ── 3. Peak VRAM ──────────────────────────────────────────────────────────────

vram_values = df["peak_vram_gb"].astype(float).tolist()
save_bar(
    values       = vram_values,
    ylabel       = "Peak VRAM  GB  (lower = better)",
    title        = "Peak VRAM usage: fp16 vs int4 NF4 vs int4 GPTQ vs QLoRA fine-tuned",
    filename     = "vram.png",
    ymin         = 0,
    ymax         = 22,
    higher_better= False,
)

# ── 4. Combined 3-panel figure (good for slides / report) ─────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

datasets = [
    (ppl_values,   "Perplexity (lower = better)",   5.8, 7.1,  False),
    (speed_values, "Tokens / sec (higher = better)", 0,  65,   True),
    (vram_values,  "Peak VRAM GB (lower = better)",  0,  22,   False),
]
subtitles = ["Perplexity", "Inference speed", "Peak VRAM"]

for ax, (vals, ylabel, ymin, ymax, hi), subtitle in zip(axes, datasets, subtitles):
    bars = ax.bar(LABELS, vals, color=COLORS, edgecolor=EDGE, linewidth=0.8, width=0.55)

    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (ymax - ymin) * 0.015,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold", color="#333333"
        )

    best_idx = vals.index(min(vals)) if not hi else vals.index(max(vals))
    best_bar = bars[best_idx]
    ax.annotate(
        "best",
        xy=(best_bar.get_x() + best_bar.get_width() / 2, best_bar.get_height()),
        xytext=(0, 16), textcoords="offset points",
        ha="center", fontsize=8, color=EDGE[best_idx],
        arrowprops=dict(arrowstyle="-", color=EDGE[best_idx], lw=0.8)
    )

    ax.set_ylim(ymin, ymax)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(subtitle, fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", labelsize=9, length=0, pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

# Shared legend at bottom
legend_patches = [
    mpatches.Patch(color=COLORS[i], label=lbl)
    for i, lbl in enumerate(["fp16 baseline", "int4 NF4", "int4 GPTQ", "QLoRA fine-tuned"])
]
fig.legend(
    handles=legend_patches, loc="lower center", ncol=4,
    fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.06)
)

fig.suptitle(
    "QuantLlama — Llama-3.1-8B Quantization Benchmark",
    fontsize=14, fontweight="bold", y=1.02
)
fig.tight_layout()
combined_path = os.path.join(PLOT_DIR, "combined.png")
fig.savefig(combined_path, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {combined_path}")

print("\nAll plots saved to results/plots/")
print("  perplexity.png  — individual perplexity chart")
print("  speed.png       — individual speed chart")
print("  vram.png        — individual VRAM chart")
print("  combined.png    — all three side by side (best for slides)")