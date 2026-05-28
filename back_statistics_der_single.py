import os
import re
import math
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# =========================================================
# Configuration
# =========================================================
beamshape    = 'rec'
sample_sizes = [100, 200, 300, 400, 500]   # adjust to however many rounds you ran

AXIS_LABEL_FONT_SIZE = 18
TITLE_FONT_SIZE      = 20
LEGEND_FONT_SIZE     = 15
TICK_FONT_SIZE       = 14
ANNOTATE_FONT_SIZE   = 10
TIMESTAMP            = datetime.now().strftime("%Y%m%d_%H%M%S")

# =========================================================
# Strategy definitions
# (key, curve_dir, seed_name, color, marker, label)
# Comment out any row to drop that method from all plots.
# =========================================================
base_dir = os.path.dirname(os.path.abspath(__file__))

STRATEGIES = [
    (
        "random",
        os.path.join(base_dir, "training_curve_random"),
        f"{beamshape}_1",
        "blue", "s", "Random baseline",
    ),
    (
        "sddal",
        os.path.join(base_dir, "training_curve_sddal"),
        f"{beamshape}_1",
        "green", "o", "SDDAL (full retrain)",
    ),
    (
        "replay",
        os.path.join(base_dir, "training_curve_cl"),
        f"{beamshape}_1_cl",
        "orange", "^", "SDDAL-Replay",
    ),
    (
        "der",
        os.path.join(base_dir, "training_curve_der"),
        f"{beamshape}_1_der",
        "red", "D", "SDDAL-DER",
    ),
]

# =========================================================
# Metric regex patterns
# =========================================================
pattern_dict = {
    "MAE":  re.compile(r"Mean\s+MAE\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
    "SSIM": re.compile(r"Mean\s+SSIM\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
    "FRCM": re.compile(r"Mean\s+FRCM\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
}

metric_info = {
    "MAE":  {"ylabel": "MAE",  "save_name": f"der_single_MAE_{TIMESTAMP}.png"},
    "SSIM": {"ylabel": "SSIM", "save_name": f"der_single_SSIM_{TIMESTAMP}.png"},
    "FRCM": {"ylabel": "FRCM", "save_name": f"der_single_FRCM_{TIMESTAMP}.png"},
}


def extract_metric(text, metric_name):
    m = pattern_dict[metric_name].search(text)
    if m is None:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    return v if math.isfinite(v) else None


# =========================================================
# Load evaluation results
# =========================================================
all_results = {}   # key -> {"MAE": [...], "SSIM": [...], "FRCM": [...]}

print("=" * 60)
print("Scanning evaluation.txt files...")
print("=" * 60)

for key, curve_dir, seed_name, color, marker, label in STRATEGIES:
    if not os.path.isdir(curve_dir):
        print(f"[SKIP] Folder not found for '{key}': {curve_dir}")
        continue

    seed_dir = os.path.join(curve_dir, seed_name)
    if not os.path.isdir(seed_dir):
        print(f"[SKIP] Seed folder not found: {seed_dir}")
        continue

    all_results[key] = {"MAE": [], "SSIM": [], "FRCM": []}
    print(f"\nStrategy: {label}  ({seed_name})")

    for n in sample_sizes:
        exp_name  = f"{seed_name}_{n}"
        eval_path = os.path.join(seed_dir, exp_name, "evaluation.txt")

        if not os.path.isfile(eval_path):
            print(f"  [SKIP] missing: {eval_path}")
            for m in ("MAE", "SSIM", "FRCM"):
                all_results[key][m].append(None)
            continue

        with open(eval_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        mae  = extract_metric(text, "MAE")
        ssim = extract_metric(text, "SSIM")
        frcm = extract_metric(text, "FRCM")
        all_results[key]["MAE"].append(mae)
        all_results[key]["SSIM"].append(ssim)
        all_results[key]["FRCM"].append(frcm)
        print(f"  {exp_name}: MAE={mae}, SSIM={ssim}, "
              f"FRCM={'None' if frcm is None else f'{frcm:.6f}'}")

print("\nFinished scanning.")

# =========================================================
# Plot — one figure per metric
# =========================================================
for metric_name, info in metric_info.items():
    # Check if any strategy has valid data for this metric
    has_any = any(
        any(v is not None and math.isfinite(v) for v in all_results[k][metric_name])
        for k in all_results
    )
    if not has_any:
        print(f"\nSkip {metric_name}: no valid data found.")
        continue

    plt.figure(figsize=(10, 6))

    for key, _, seed_name, color, marker, label in STRATEGIES:
        if key not in all_results:
            continue
        y = all_results[key][metric_name]
        x_valid = [n for n, v in zip(sample_sizes, y) if v is not None]
        y_valid = [v for v in y if v is not None]
        if not x_valid:
            continue

        plt.plot(
            x_valid, y_valid,
            marker=marker, markersize=6,
            linewidth=2.2, color=color,
            alpha=0.9, label=label,
        )
        for x, yv in zip(x_valid, y_valid):
            plt.text(
                x, yv, f"{yv:.4f}",
                ha="center", va="bottom",
                fontsize=ANNOTATE_FONT_SIZE, color=color,
            )

    plt.xlabel("Number of scanner-acquired samples", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel(info["ylabel"], fontsize=AXIS_LABEL_FONT_SIZE)
    plt.title(f"{metric_name} vs. samples — seed 1 comparison", fontsize=TITLE_FONT_SIZE)
    plt.xticks(sample_sizes, fontsize=TICK_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()

    save_path = os.path.join(base_dir, info["save_name"])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved: {save_path}")

print("\nDone.")
