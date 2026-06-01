#!/usr/bin/env python3
"""Generate missing AL plots from al_results.json."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results" / "al"
JSON_PATH = RESULTS_DIR / "al_results.json"

with open(JSON_PATH) as f:
    data = json.load(f)

config = data["config"]
full_baseline = data["full_data_baseline"]
results = data["results"]

SCENES = ["z1", "z2", "e1", "e2"]
STRATEGIES = list(results["z1"].keys())

COLORS = {
    "Random":         "#888888",
    "Entropy":        "#1f77b4",
    "Margin":         "#ff7f0e",
    "LeastConf":      "#2ca02c",
    "CoreSet":        "#d62728",
    "BADGE":          "#9467bd",
    "RX-Guided":      "#8c564b",
    "Unc+KernelKMeans": "#e377c2",
}
LINESTYLES = {
    "Random":         "--",
    "Entropy":        "-",
    "Margin":         "-",
    "LeastConf":      "-",
    "CoreSet":        "-",
    "BADGE":          "-",
    "RX-Guided":      "-",
    "Unc+KernelKMeans": "-",
}

# ── 1. Aggregated AUC + AP (mean across 4 scenes) ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Aggregated Active Learning Performance (Mean over 4 Scenes)", fontsize=13, fontweight="bold")

for ax, metric, ylabel in [
    (axes[0], "auc", "ROC-AUC"),
    (axes[1], "ap",  "Average Precision"),
]:
    for strat in STRATEGIES:
        xs = results[SCENES[0]][strat]["x"]
        means_per_scene = []
        for scene in SCENES:
            means_per_scene.append(results[scene][strat][f"{metric}_mean"])
        agg_mean = np.mean(means_per_scene, axis=0)

        ax.plot(xs, agg_mean,
                label=strat,
                color=COLORS.get(strat, None),
                linestyle=LINESTYLES.get(strat, "-"),
                linewidth=2, marker="o", markersize=3)

    # full-data baseline (mean across scenes)
    baseline_val = np.mean([full_baseline[metric][s] for s in SCENES])
    ax.axhline(baseline_val, color="black", linestyle=":", linewidth=1.5,
               label=f"Full-data ({baseline_val:.4f})")

    ax.set_xlabel("Labeled Budget (# pixels)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f"Aggregated {ylabel}", fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out = RESULTS_DIR / "al_aggregated.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# ── 2. Final bar chart (AUC at max budget, per scene + mean) ──────────────────
fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=False)
fig.suptitle("Final AUC at Max Budget (600 pixels labeled)", fontsize=13, fontweight="bold")

scene_labels = SCENES + ["Mean"]
for idx, (ax, scene_label) in enumerate(zip(axes, scene_labels)):
    vals = []
    for strat in STRATEGIES:
        if scene_label == "Mean":
            v = np.mean([results[s][strat]["auc_mean"][-1] for s in SCENES])
        else:
            v = results[scene_label][strat]["auc_mean"][-1]
        vals.append(v)

    colors = [COLORS.get(s, "#333333") for s in STRATEGIES]
    bars = ax.bar(range(len(STRATEGIES)), vals, color=colors, edgecolor="white", linewidth=0.5)

    # baseline
    if scene_label == "Mean":
        bl = np.mean([full_baseline["auc"][s] for s in SCENES])
    else:
        bl = full_baseline["auc"][scene_label]
    ax.axhline(bl, color="black", linestyle=":", linewidth=1.5, label=f"Full ({bl:.4f})")

    ax.set_xticks(range(len(STRATEGIES)))
    ax.set_xticklabels(STRATEGIES, rotation=45, ha="right", fontsize=8)
    ax.set_title(scene_label.upper() if scene_label != "Mean" else "Mean", fontsize=11, fontweight="bold")
    ax.set_ylabel("AUC" if idx == 0 else "", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)
    # annotate bars
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{v:.4f}", ha="center", va="bottom", fontsize=6, rotation=90)

plt.tight_layout()
out = RESULTS_DIR / "al_final_bar.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# ── 3. Label efficiency (budget to reach 90/95/99% of full-data AUC) ─────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Label Efficiency — Budget to Reach X% of Full-Data AUC", fontsize=13, fontweight="bold")

thresholds = [0.90, 0.95, 0.99]
for ax, thresh in zip(axes, thresholds):
    budgets = {}
    for strat in STRATEGIES:
        scene_budgets = []
        for scene in SCENES:
            bl = full_baseline["auc"][scene]
            target = thresh * bl
            xs = results[scene][strat]["x"]
            means = results[scene][strat]["auc_mean"]
            reached = None
            for x, m in zip(xs, means):
                if m >= target:
                    reached = x
                    break
            scene_budgets.append(reached if reached is not None else config["MAX_BUDGET"] + 50)
        budgets[strat] = np.mean(scene_budgets)

    sorted_strats = sorted(budgets, key=lambda s: budgets[s])
    vals = [budgets[s] for s in sorted_strats]
    colors = [COLORS.get(s, "#333333") for s in sorted_strats]
    bars = ax.bar(range(len(sorted_strats)), vals, color=colors, edgecolor="white")
    ax.set_xticks(range(len(sorted_strats)))
    ax.set_xticklabels(sorted_strats, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean Budget (pixels)", fontsize=10)
    ax.set_title(f"Reach {int(thresh*100)}% of Full-Data AUC", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, v in zip(bars, vals):
        label = f"{int(v)}" if v <= config["MAX_BUDGET"] else "N/A"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                label, ha="center", va="bottom", fontsize=8)

plt.tight_layout()
out = RESULTS_DIR / "al_label_efficiency.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# ── 4. Novel strategies vs baselines (RX-Guided + Unc+KernelKMeans vs Random + Entropy) ──
novel = ["RX-Guided", "Unc+KernelKMeans"]
baselines_cmp = ["Random", "Entropy", "CoreSet", "BADGE"]
highlight = novel + baselines_cmp

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("Novel Strategies vs Baselines — AUC Learning Curves per Scene", fontsize=13, fontweight="bold")

for row, metric, ylabel in [(0, "auc", "ROC-AUC"), (1, "ap", "Average Precision")]:
    for col, scene in enumerate(SCENES):
        ax = axes[row][col]
        for strat in highlight:
            xs = results[scene][strat]["x"]
            means = results[scene][strat][f"{metric}_mean"]
            stds = results[scene][strat][f"{metric}_std"]
            lw = 2.5 if strat in novel else 1.5
            alpha_fill = 0.15 if strat in novel else 0.08
            label = f"★ {strat}" if strat in novel else strat
            line, = ax.plot(xs, means,
                            label=label,
                            color=COLORS.get(strat, None),
                            linestyle=LINESTYLES.get(strat, "-"),
                            linewidth=lw, marker="o", markersize=3)
            ax.fill_between(xs,
                            np.array(means) - np.array(stds),
                            np.array(means) + np.array(stds),
                            color=line.get_color(), alpha=alpha_fill)

        bl = full_baseline[metric][scene]
        ax.axhline(bl, color="black", linestyle=":", linewidth=1.2,
                   label=f"Full ({bl:.4f})")
        ax.set_title(f"{ylabel} — {scene.upper()}", fontsize=10)
        ax.set_xlabel("Budget", fontsize=9)
        ax.set_ylabel(ylabel if col == 0 else "", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
out = RESULTS_DIR / "al_novel_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

print("\nAll plots generated successfully.")
