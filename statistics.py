import os
import re
import math
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import combinations

from scipy.stats import ttest_ind

# =========================================================
# 0) Global display control
# =========================================================
SHOW_P_VALUES_ON_PLOT = True
beamshape = 'rec'

P_VALUE_FONT_SIZE = 12
P_VALUE_VERTICAL_OFFSET_RATIO = 0.025

PRINT_P_VALUE_TABLE = True

AXIS_LABEL_FONT_SIZE = 18
TITLE_FONT_SIZE = 20
LEGEND_FONT_SIZE = 15
TICK_FONT_SIZE = 14

SEED_MARKERS = ["o", "s", "^", "D", "v"]
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# =========================================================
# 1) Path settings
# =========================================================
base_dir = os.path.dirname(os.path.abspath(__file__))

#baseline_dir = os.path.join(base_dir, "training_curve_random")
compare_1    = os.path.join(base_dir, "training_curve_sddal")
compare_2       = os.path.join(base_dir, "training_curve_cl")


# Compatibility fallback for old random folder name
# if not os.path.isdir(baseline_dir):
#     alt = os.path.join(base_dir, "training_curve_random_syncUNet_varseed")
#     if os.path.isdir(alt):
#         baseline_dir = alt

# Directory validation runs after STRATEGIES so only active entries are checked.

# =========================================================
# 2) Basic configuration
# =========================================================
sample_sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

# Seed names per strategy (cl folder uses "_cl" suffix)
seed_names    = [beamshape + "_1", beamshape + "_2", beamshape + "_3", beamshape + "_4", beamshape + "_5"]
seed_names_cl = [beamshape + "_1", beamshape + "_2", beamshape + "_3", beamshape + "_4", beamshape + "_5"]

metric_info = {
    "MAE": {
        "ylabel": "MAE",
        "raw_save_name": f"all_curves_MAE_{TIMESTAMP}.png",
        "mean_save_name": f"mean_std_curves_MAE_{TIMESTAMP}.png",
    },
    "SSIM": {
        "ylabel": "SSIM",
        "raw_save_name": f"all_curves_SSIM_{TIMESTAMP}.png",
        "mean_save_name": f"mean_std_curves_SSIM_{TIMESTAMP}.png",
    },
    "FRCM": {
        "ylabel": "FRCM",
        "raw_save_name": f"all_curves_FRCM_{TIMESTAMP}.png",
        "mean_save_name": f"mean_std_curves_FRCM_{TIMESTAMP}.png",
    },
}

# Strategy config: (key, directory, color, label, marker, seed_list)
# Comment out any line here to drop that strategy from all plots and comparisons.
STRATEGIES = [
    #("baseline", baseline_dir, "blue",   "Baseline (random)", "s", seed_names),
    ("sddal_baseline",    compare_1,   "green",  "SDDAL_baseline",             "o", seed_names),
    ("sddal_replay", compare_2,      "blue", "SDDAL_replay",          "^", seed_names_cl),
]

# Validate directories only for active strategies.
for _key, _path, *_ in STRATEGIES:
    if not os.path.isdir(_path):
        raise FileNotFoundError(f"Folder not found for '{_key}': {_path}")

# =========================================================
# 3) Regular expressions
# =========================================================
pattern_dict = {
    "MAE":  re.compile(r"Mean\s+MAE\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
    "SSIM": re.compile(r"Mean\s+SSIM\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
    "FRCM": re.compile(r"Mean\s+FRCM\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
}

# =========================================================
# 4) Data structure
# =========================================================
all_results = {key: {} for key, *_ in STRATEGIES}

def get_seeds(key):
    for s_key, _, _, _, _, seeds in STRATEGIES:
        if s_key == key:
            return seeds
    return seed_names

invalid_eval_paths = []
missing_eval_paths = []

# =========================================================
# 5) Pairwise p-value storage
#    pvalue_results[(key_a, key_b)][metric] = list of p-values
#    PAIRS is auto-generated from active STRATEGIES — no manual edits needed.
# =========================================================
_active = [(key, label) for key, _, _, label, _, _ in STRATEGIES]
PAIRS = [
    (a_key, b_key, f"{a_label} vs {b_label}")
    for (a_key, a_label), (b_key, b_label) in combinations(_active, 2)
]
pvalue_results = {(a, b): {} for a, b, _ in PAIRS}


def is_finite_number(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def extract_metric(text, metric_name):
    match = pattern_dict[metric_name].search(text)
    if match is None:
        return None
    try:
        value = float(match.group(1))
    except Exception:
        return None
    return value if math.isfinite(value) else None


def format_p_value(p):
    if p is None or (isinstance(p, float) and not math.isfinite(p)):
        return "nan"
    return f"{p:.3g}"


def safe_mean(values):
    valid = [v for v in values if is_finite_number(v)]
    return float(np.mean(valid)) if valid else np.nan


def compute_metric_exists(metric_name):
    for key, *_ in STRATEGIES:
        for seed_name in all_results[key]:
            if any(is_finite_number(v) for v in all_results[key][seed_name][metric_name]):
                return True
    return False

def strategy_seed_names(key):
    return get_seeds(key)


# =========================================================
# Scan evaluation files
# =========================================================
print("=" * 60)
print("Scanning evaluation.txt files...")
print("=" * 60)

for key, strategy_dir, color, label, marker, s_seeds in STRATEGIES:
    print(f"\nStrategy: {label}")
    all_results[key] = {}

    for seed_name in s_seeds:
        seed_dir = os.path.join(strategy_dir, seed_name)
        if not os.path.isdir(seed_dir):
            raise FileNotFoundError(f"Seed folder not found: {seed_dir}")

        all_results[key][seed_name] = {"MAE": [], "SSIM": [], "FRCM": []}

        print(f"  Reading {seed_name} ...")

        for n in sample_sizes:
            exp_name = f"{seed_name}_{n}"
            exp_dir  = os.path.join(seed_dir, exp_name)
            eval_path = os.path.join(exp_dir, "evaluation.txt")

            if not os.path.isfile(eval_path):
                missing_eval_paths.append(eval_path)
                print(f"    [SKIP] missing: {eval_path}")
                for m in ("MAE", "SSIM", "FRCM"):
                    all_results[key][seed_name][m].append(None)
                continue

            with open(eval_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

            mae  = extract_metric(text, "MAE")
            ssim = extract_metric(text, "SSIM")
            frcm = extract_metric(text, "FRCM")

            if mae is None or ssim is None:
                invalid_eval_paths.append(eval_path)
                print(f"    [SKIP] invalid data: {eval_path}")
                for m in ("MAE", "SSIM", "FRCM"):
                    all_results[key][seed_name][m].append(None)
                continue

            all_results[key][seed_name]["MAE"].append(mae)
            all_results[key][seed_name]["SSIM"].append(ssim)
            all_results[key][seed_name]["FRCM"].append(frcm)
            print(
                f"    {exp_name}: MAE={mae:.6f}, SSIM={ssim:.6f}, "
                f"FRCM={'None' if frcm is None else f'{frcm:.6f}'}"
            )

print("\nFinished scanning all evaluation files.")

if missing_eval_paths:
    print("\n" + "=" * 60 + "\nMissing evaluation.txt files:")
    for p in missing_eval_paths:
        print(p)

if invalid_eval_paths:
    print("\n" + "=" * 60 + "\nFiles with invalid metric data:")
    for p in invalid_eval_paths:
        print(p)

# =========================================================
# Compute pairwise p-values (Welch's t-test)
# =========================================================
for metric_name in metric_info:
    for key_a, key_b, _ in PAIRS:
        seeds_a = get_seeds(key_a)
        seeds_b = get_seeds(key_b)
        raw_p_values = []
        for idx in range(len(sample_sizes)):
            group_a = [
                all_results[key_a][s][metric_name][idx]
                for s in seeds_a
                if is_finite_number(all_results[key_a][s][metric_name][idx])
            ]
            group_b = [
                all_results[key_b][s][metric_name][idx]
                for s in seeds_b
                if is_finite_number(all_results[key_b][s][metric_name][idx])
            ]

            if len(group_a) >= 2 and len(group_b) >= 2:
                _, p = ttest_ind(group_a, group_b, equal_var=False, nan_policy="omit")
                raw_p_values.append(p if (p is not None and math.isfinite(p)) else np.nan)
            else:
                raw_p_values.append(np.nan)

        pvalue_results[(key_a, key_b)][metric_name] = raw_p_values

# =========================================================
# Print p-value table
# =========================================================
if PRINT_P_VALUE_TABLE:
    strategy_label = {key: label for key, _, _, label, _, _ in STRATEGIES}

    for metric_name in metric_info:
        print(f"\n{'=' * 80}")
        print(f"P-VALUES (Welch's t-test) — {metric_name}")
        print("=" * 80)

        for key_a, key_b, pair_label in PAIRS:
            print(f"\n  {pair_label}")
            header = f"  {'n':>6} | {'mean_A':>12} | {'mean_B':>12} | {'p':>10}"
            print(header)
            print("  " + "-" * (len(header) - 2))

            for idx, n in enumerate(sample_sizes):
                vals_a = [all_results[key_a][s][metric_name][idx] for s in get_seeds(key_a)]
                vals_b = [all_results[key_b][s][metric_name][idx] for s in get_seeds(key_b)]
                ma = safe_mean(vals_a)
                mb = safe_mean(vals_b)
                p  = pvalue_results[(key_a, key_b)][metric_name][idx]

                ma_s = "nan" if not math.isfinite(ma) else f"{ma:.6f}"
                mb_s = "nan" if not math.isfinite(mb) else f"{mb:.6f}"
                p_s  = format_p_value(p)

                print(f"  {n:>6} | {ma_s:>12} | {mb_s:>12} | {p_s:>10}")

# =========================================================
# Plot raw curves (one per seed)
# =========================================================
for metric_name, info in metric_info.items():
    if not compute_metric_exists(metric_name):
        print(f"\nSkip {metric_name}: no valid data found.")
        continue

    plt.figure(figsize=(10, 6))

    for key, _, color, label, _, s_seeds in STRATEGIES:
        for i, seed_name in enumerate(s_seeds):
            y = all_results[key][seed_name][metric_name]
            x_valid = [x for x, v in zip(sample_sizes, y) if is_finite_number(v)]
            y_valid = [v for v in y if is_finite_number(v)]
            if not x_valid:
                continue
            plt.plot(
                x_valid, y_valid,
                marker=SEED_MARKERS[i % len(SEED_MARKERS)],
                markersize=6, linewidth=2.2, color=color, alpha=0.9,
                label=f"{label} - {seed_name}",
            )

    plt.xlabel("Number of training samples", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel(info["ylabel"], fontsize=AXIS_LABEL_FONT_SIZE)
    plt.title(f"{metric_name} vs. Number of training samples", fontsize=TITLE_FONT_SIZE)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.xticks(fontsize=TICK_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)
    plt.tight_layout()

    save_path = os.path.join(base_dir, info["raw_save_name"])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved raw curve figure: {save_path}")

# =========================================================
# Plot mean ± std curves
# Annotate SDDAL vs SDDAL-CL p-values on the plot
# =========================================================
for metric_name, info in metric_info.items():
    if not compute_metric_exists(metric_name):
        continue

    plt.figure(figsize=(10, 6))

    stored_mean_curves = {}

    for key, _, color, label, marker, s_seeds in STRATEGIES:
        mean_values, std_values = [], []

        for idx in range(len(sample_sizes)):
            point_values = [
                all_results[key][s][metric_name][idx]
                for s in s_seeds
                if is_finite_number(all_results[key][s][metric_name][idx])
            ]
            if not point_values:
                mean_values.append(np.nan)
                std_values.append(np.nan)
            else:
                arr = np.array(point_values, dtype=float)
                mean_values.append(float(np.mean(arr)))
                std_values.append(float(np.std(arr)))

        x_arr    = np.array(sample_sizes, dtype=float)
        mean_arr = np.array(mean_values,  dtype=float)
        std_arr  = np.array(std_values,   dtype=float)

        stored_mean_curves[key] = mean_arr.copy()

        valid_mask = np.isfinite(mean_arr)
        if not np.any(valid_mask):
            print(f"Warning: no valid mean/std points for {metric_name} in {key}, skip.")
            continue

        plt.fill_between(
            x_arr,
            mean_arr - std_arr,
            mean_arr + std_arr,
            where=valid_mask, interpolate=True,
            color=color, alpha=0.18,
        )
        plt.plot(
            x_arr[valid_mask], mean_arr[valid_mask],
            marker=marker, markersize=6, linewidth=2.4,
            color=color, alpha=0.9, label=f"{label} mean ± std",
        )

    plt.xlabel("Number of training samples", fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel(info["ylabel"], fontsize=AXIS_LABEL_FONT_SIZE)
    plt.title(f"{metric_name} mean curve with std region", fontsize=TITLE_FONT_SIZE)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.xticks(fontsize=TICK_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)
    plt.tight_layout()
    plt.draw()

    if SHOW_P_VALUES_ON_PLOT and PAIRS:
        # Annotate p-values for the last active pair in STRATEGIES.
        ann_key_a, ann_key_b, _ = PAIRS[-1]
        if (ann_key_a, ann_key_b) in pvalue_results:
            y_min, y_max = plt.gca().get_ylim()
            text_offset = P_VALUE_VERTICAL_OFFSET_RATIO * (y_max - y_min)

            pvals  = pvalue_results[(ann_key_a, ann_key_b)][metric_name]
            mean_a = stored_mean_curves.get(ann_key_a, np.full(len(sample_sizes), np.nan))
            mean_b = stored_mean_curves.get(ann_key_b, np.full(len(sample_sizes), np.nan))

            for idx, n in enumerate(sample_sizes):
                p = pvals[idx]
                if not (isinstance(p, float) and math.isfinite(p)):
                    continue

                y_candidates = []
                if math.isfinite(mean_a[idx]):
                    y_candidates.append(mean_a[idx])
                if math.isfinite(mean_b[idx]):
                    y_candidates.append(mean_b[idx])
                if not y_candidates:
                    continue

                plt.text(
                    n, max(y_candidates) + text_offset,
                    f"p={format_p_value(p)}",
                    ha="center", va="bottom", fontsize=P_VALUE_FONT_SIZE,
                    color="gray",
                )

    save_path = os.path.join(base_dir, info["mean_save_name"])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved mean±std figure: {save_path}")

print("\nAll figures have been generated successfully.")
