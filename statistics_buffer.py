"""
Buffer size ablation plot: compare replay with buffer_size = 250 / 300 / 350.
Evaluation range: 100–500 AL samples (5 seeds each).

All three buffer sizes live in the same directory (training_curve_cl).
Seed folders are named rec_1_250, rec_1_300, rec_1_350, rec_2_250, etc.

Expected directory layout:
  training_curve_cl/
    rec_1_250/
      rec_1_250_100/evaluation.txt
      rec_1_250_200/evaluation.txt
      ...
    rec_1_300/
      rec_1_300_100/evaluation.txt
      ...
    rec_1_350/ ...
    rec_2_250/ ...
    rec_2_300/ ...
    rec_2_350/ ...
    ...
"""

import math
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from itertools import combinations
from scipy.stats import ttest_ind


# =========================================================
# CONFIGURE
# =========================================================
beamshape = "rec"

# All buffer experiments live in the same directory.
TC_DIR = "training_curve_cl"

# (label, buffer_suffix, color)
# seed folders will be: rec_1_{buffer_suffix}, rec_2_{buffer_suffix}, ...
BUFFER_CONFIGS = [
    ("Buffer=250", "250", "#5aafe0"),
    ("Buffer=300", "", "#1f77b4"),
    ("Buffer=350", "350", "#0a3d62"),
]

N_SEEDS = 5
sample_sizes = [100, 200, 300, 400, 500]
# =========================================================

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
SHOW_STD_BAND   = True   # set False for clean mean-only lines
PRINT_P_TABLE   = True

AXIS_LABEL_FONT_SIZE  = 18
TITLE_FONT_SIZE       = 20
LEGEND_FONT_SIZE      = 15
TICK_FONT_SIZE        = 14
P_VALUE_FONT_SIZE     = 11
P_VALUE_OFFSET_RATIO  = 0.025

base_dir  = os.path.dirname(os.path.abspath(__file__))
tc_path   = os.path.join(base_dir, TC_DIR)

if not os.path.isdir(tc_path):
    raise FileNotFoundError(f"Base directory not found: {tc_path}")

pattern_dict = {
    "MAE":  re.compile(r"Mean\s+MAE\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
    "SSIM": re.compile(r"Mean\s+SSIM\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
}
METRICS = list(pattern_dict.keys())

# Build strategy list: all point to the same tc_path, differ only by seed names
STRATEGIES = [
    (
        suffix,                                                          # key
        tc_path,                                                         # directory
        color,                                                           # color
        label,                                                           # label
        [f"{beamshape}_{i}" + (f"_{suffix}" if suffix else "") for i in range(1, N_SEEDS + 1)] #[f"{beamshape}_{i}_{suffix}" for i in range(1, N_SEEDS + 1)],  # seed_names
    )
    for label, suffix, color in BUFFER_CONFIGS
]


# =========================================================
# Helpers
# =========================================================
def is_finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def extract_metric(text, metric):
    m = pattern_dict[metric].search(text)
    if m is None:
        return None
    try:
        v = float(m.group(1))
    except Exception:
        return None
    return v if math.isfinite(v) else None


def safe_mean(vals):
    valid = [v for v in vals if is_finite(v)]
    return float(np.mean(valid)) if valid else np.nan


def fmt_p(p):
    if p is None or not math.isfinite(p):
        return "nan"
    return f"{p:.3g}"


# =========================================================
# Load data
# =========================================================
print("=" * 60)
print("Scanning evaluation files...")
print("=" * 60)

# all_data[key][seed_name][metric] = [val_at_100, val_at_200, ...]
all_data = {}
missing, invalid = [], []

for key, strategy_dir, _, label, seed_names in STRATEGIES:
    print(f"\n{label}  ({strategy_dir})")
    all_data[key] = {}

    for seed_name in seed_names:
        seed_dir = os.path.join(strategy_dir, seed_name)
        if not os.path.isdir(seed_dir):
            raise FileNotFoundError(f"Seed folder not found: {seed_dir}")

        all_data[key][seed_name] = {m: [] for m in METRICS}

        for n in sample_sizes:
            eval_path = os.path.join(seed_dir, f"{seed_name}_{n}", "evaluation.txt")

            if not os.path.isfile(eval_path):
                missing.append(eval_path)
                print(f"  [SKIP] missing: {eval_path}")
                for m in METRICS:
                    all_data[key][seed_name][m].append(None)
                continue

            with open(eval_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

            extracted = {m: extract_metric(text, m) for m in METRICS}

            if any(extracted[m] is None for m in METRICS):
                invalid.append(eval_path)
                print(f"  [SKIP] invalid: {eval_path}")
                for m in METRICS:
                    all_data[key][seed_name][m].append(None)
                continue

            for m in METRICS:
                all_data[key][seed_name][m].append(extracted[m])
            print(f"  {seed_name} n={n:>4}  " +
                  "  ".join(f"{m}={extracted[m]:.6f}" for m in METRICS))

if missing:
    print("\nMissing files:\n" + "\n".join(missing))
if invalid:
    print("\nInvalid files:\n" + "\n".join(invalid))


# =========================================================
# Pairwise p-values (Welch's t-test)
# =========================================================
_active = [(key, label) for key, _, _, label, _ in STRATEGIES]
PAIRS = [
    (a_key, b_key, f"{a_lab} vs {b_lab}")
    for (a_key, a_lab), (b_key, b_lab) in combinations(_active, 2)
]
pvalues = {(a, b): {m: [] for m in METRICS} for a, b, _ in PAIRS}

for metric in METRICS:
    for key_a, key_b, _ in PAIRS:
        seeds_a = list(all_data[key_a].keys())
        seeds_b = list(all_data[key_b].keys())
        for idx in range(len(sample_sizes)):
            ga = [all_data[key_a][s][metric][idx]
                  for s in seeds_a if is_finite(all_data[key_a][s][metric][idx])]
            gb = [all_data[key_b][s][metric][idx]
                  for s in seeds_b if is_finite(all_data[key_b][s][metric][idx])]
            if len(ga) >= 2 and len(gb) >= 2:
                _, p = ttest_ind(ga, gb, equal_var=False, nan_policy="omit")
                pvalues[(key_a, key_b)][metric].append(
                    p if (p is not None and math.isfinite(p)) else np.nan)
            else:
                pvalues[(key_a, key_b)][metric].append(np.nan)


# =========================================================
# Print p-value table
# =========================================================
if PRINT_P_TABLE:
    for metric in METRICS:
        print(f"\n{'=' * 70}")
        print(f"P-VALUES (Welch's t-test) — {metric}")
        print("=" * 70)
        for key_a, key_b, pair_label in PAIRS:
            print(f"\n  {pair_label}")
            print(f"  {'n':>6} | {'mean_A':>10} | {'mean_B':>10} | {'p':>8}")
            print("  " + "-" * 42)
            for idx, n in enumerate(sample_sizes):
                ma = safe_mean([all_data[key_a][s][metric][idx]
                                for s in all_data[key_a]])
                mb = safe_mean([all_data[key_b][s][metric][idx]
                                for s in all_data[key_b]])
                p  = pvalues[(key_a, key_b)][metric][idx]
                print(f"  {n:>6} | {ma:>10.6f} | {mb:>10.6f} | {fmt_p(p):>8}")


# =========================================================
# Plot: mean (± std band) for each buffer size
# =========================================================
for metric in METRICS:
    plt.figure(figsize=(9, 6))

    mean_curves = {}
    for key, _, color, label, seed_names in STRATEGIES:
        means, stds = [], []
        for idx in range(len(sample_sizes)):
            vals = [all_data[key][s][metric][idx]
                    for s in seed_names if is_finite(all_data[key][s][metric][idx])]
            if vals:
                arr = np.array(vals, dtype=float)
                means.append(float(np.mean(arr)))
                stds.append(float(np.std(arr)))
            else:
                means.append(np.nan)
                stds.append(np.nan)

        x   = np.array(sample_sizes, dtype=float)
        mu  = np.array(means, dtype=float)
        sig = np.array(stds,  dtype=float)
        mean_curves[key] = mu

        valid = np.isfinite(mu)
        if not np.any(valid):
            continue

        if SHOW_STD_BAND:
            plt.fill_between(x, mu - sig, mu + sig,
                             where=valid, interpolate=True,
                             color=color, alpha=0.15)

        plt.plot(x[valid], mu[valid],
                 marker="o", markersize=6, linewidth=2.4,
                 color=color, label=f"{label} mean" + (" ± std" if SHOW_STD_BAND else ""))

    plt.xlabel("Number of AL samples", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel(metric, fontsize=AXIS_LABEL_FONT_SIZE)
    plt.title(f"Buffer size ablation — {metric}", fontsize=TITLE_FONT_SIZE)
    plt.xticks(sample_sizes, fontsize=TICK_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()
    plt.draw()

    # Annotate p-values for every pair
    y_min, y_max = plt.gca().get_ylim()
    offset = P_VALUE_OFFSET_RATIO * (y_max - y_min)
    level_count = {}  # idx -> how many annotations already placed

    for key_a, key_b, pair_label in PAIRS:
        pvs = pvalues[(key_a, key_b)][metric]
        for idx, n in enumerate(sample_sizes):
            p = pvs[idx]
            if not (isinstance(p, float) and math.isfinite(p)):
                continue
            ma = mean_curves.get(key_a, np.full(len(sample_sizes), np.nan))[idx]
            mb = mean_curves.get(key_b, np.full(len(sample_sizes), np.nan))[idx]
            y_candidates = [v for v in (ma, mb) if math.isfinite(v)]
            if not y_candidates:
                continue
            level = level_count.get(idx, 0)
            plt.text(n, max(y_candidates) + offset * (1 + level),
                     f"p={fmt_p(p)}",
                     ha="center", va="bottom",
                     fontsize=P_VALUE_FONT_SIZE, color="gray")
            level_count[idx] = level + 1

    save_path = os.path.join(base_dir, f"buffer_ablation_{metric}_{TIMESTAMP}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved: {save_path}")

print("\nDone.")
