import os
import re
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datetime import datetime

# =========================================================
# Configuration
# =========================================================
beamshape    = 'rec'
sample_sizes = [100, 200, 300, 400, 500]   # adjust to however many rounds you ran

L2_BETA   = [0.001, 0.01, 0.1] # [0.1, 0.3, 0.5, 0.7, 0.9]  # alpha values to compare

AXIS_LABEL_FONT_SIZE = 18
TITLE_FONT_SIZE      = 20
LEGEND_FONT_SIZE     = 13
TICK_FONT_SIZE       = 14
TIMESTAMP            = datetime.now().strftime("%Y%m%d_%H%M%S")

# =========================================================
# Strategy definitions
# (key, curve_dir, seed_name, color, marker, label)
# Comment out any row to drop that method from all plots.
# =========================================================
base_dir = os.path.dirname(os.path.abspath(__file__))

# Non-DER reference methods
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
        "blue", "^", "SDDAL-Replay",
    ),
]

# DER alpha sweep — generated from DER_ALPHAS, folder name: rec_1_der_01, rec_1_der_03, ...
_der_cmap   = cm.get_cmap("Reds")
_der_colors = [_der_cmap(0.35 + 0.55 * i / (len(L2_BETA) - 1)) for i in range(len(L2_BETA))]

for _i, _a in enumerate(L2_BETA):
    _suffix = str(_a) #.replace(".", "")          # 0.1 → "01", 0.3 → "03", etc.
    _seed   = f"{beamshape}_1_l2_{_suffix}"    # e.g. rec_1_der_01
    STRATEGIES.append((
        f"l2_{_suffix}",
        os.path.join(base_dir, "training_curve_l2"),
        _seed,
        _der_colors[_i], "D", f"L2 (selective) β={_a}",
    ))

# =========================================================
# Metric regex patterns
# =========================================================
pattern_dict = {
    "MAE":  re.compile(r"Mean\s+MAE\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
    "SSIM": re.compile(r"Mean\s+SSIM\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
    "FRCM": re.compile(r"Mean\s+FRCM\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
}

metric_info = {
    "MAE":  {"ylabel": "MAE",  "save_name": f"reg_single_MAE_{TIMESTAMP}.png"},
    "SSIM": {"ylabel": "SSIM", "save_name": f"reg_single_SSIM_{TIMESTAMP}.png"},
    "FRCM": {"ylabel": "FRCM", "save_name": f"reg_single_FRCM_{TIMESTAMP}.png"},
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
# Console table — exact values for all methods
# =========================================================
for metric_name in metric_info:
    has_any = any(
        any(v is not None and math.isfinite(v) for v in all_results[k][metric_name])
        for k in all_results
    )
    if not has_any:
        continue

    col_w = 12
    header_label = f"{'Method':<22}"
    header_cols  = "".join(f"{n:>{col_w}}" for n in sample_sizes)
    print(f"\n{'=' * 80}")
    print(f"{metric_name} — seed 1")
    print(f"{'=' * 80}")
    print(header_label + header_cols)
    print("-" * 80)

    for key, _, seed_name, color, marker, label in STRATEGIES:
        if key not in all_results:
            continue
        row = f"{label:<22}"
        for v in all_results[key][metric_name]:
            if v is None or not math.isfinite(v):
                row += f"{'—':>{col_w}}"
            else:
                row += f"{v:>{col_w}.4f}"
        print(row)

# =========================================================
# Plot — one figure per metric
# =========================================================
for metric_name, info in metric_info.items():
    has_any = any(
        any(v is not None and math.isfinite(v) for v in all_results[k][metric_name])
        for k in all_results
    )
    if not has_any:
        print(f"\nSkip {metric_name}: no valid data found.")
        continue

    plt.figure(figsize=(11, 6))

    for key, _, seed_name, color, marker, label in STRATEGIES:
        if key not in all_results:
            continue
        y = all_results[key][metric_name]
        x_valid = [n for n, v in zip(sample_sizes, y) if v is not None]
        y_valid = [v for v in y if v is not None]
        if not x_valid:
            continue

        # DER lines are thinner and slightly transparent to avoid clutter
        is_der = key.startswith("der_")
        plt.plot(
            x_valid, y_valid,
            marker=marker,
            markersize=5 if is_der else 7,
            linewidth=1.6 if is_der else 2.4,
            color=color,
            alpha=0.75 if is_der else 0.95,
            linestyle="--" if is_der else "-",
            label=label,
        )

    plt.xlabel("Number of scanner-acquired samples", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel(info["ylabel"], fontsize=AXIS_LABEL_FONT_SIZE)
    plt.title(f"{metric_name} vs. samples — seed 1, L2 BETA sweep", fontsize=TITLE_FONT_SIZE)
    plt.xticks(sample_sizes, fontsize=TICK_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=LEGEND_FONT_SIZE, loc="best")
    plt.tight_layout()

    save_path = os.path.join(base_dir, info["save_name"])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved: {save_path}")

print("\nDone.")
