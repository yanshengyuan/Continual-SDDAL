import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# =========================================================
# 0) Configuration
# =========================================================
AXIS_LABEL_FONT_SIZE = 18
TITLE_FONT_SIZE      = 20
LEGEND_FONT_SIZE     = 15
TICK_FONT_SIZE       = 14
ANNOT_FONT_SIZE      = 12

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# =========================================================
# 1) Strategy config
#    (key, csv_filename, color, label)
#    Comment out any line to drop that method from all plots.
# =========================================================
STRATEGIES = [
    ("sddal",           "sddal.csv",          "green",     "SDDAL"),
    ("sddal_cl_replay", "sddal_cl_replay.csv", "steelblue", "SDDAL-CL (replay)"),
    # ("sddal_cl",      "sddal_cl.csv",        "orange",    "SDDAL-CL"),
]

# =========================================================
# 2) Load CSVs
# =========================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
time_dir = os.path.join(base_dir, "time")

data = {}
for key, filename, color, label in STRATEGIES:
    csv_path = os.path.join(time_dir, filename)
    if not os.path.isfile(csv_path):
        print(f"[SKIP] {label}: file not found at {csv_path}")
        continue
    df = pd.read_csv(csv_path)
    required = {"dataset_size", "cumul_trainer_s", "per_round_trainer_s","cumul_scanner_s", "per_round_scanner_s"}
    if not required.issubset(df.columns):
        print(f"[SKIP] {label}: missing columns in {csv_path} (need {required})")
        continue
    data[key] = df
    print(f"Loaded {label}: {len(df)} rounds")

# =========================================================
# 3) Plot 1 — cumulative trainer time
#    Annotation: final value in hours (2 decimal) at end of line
# =========================================================
fig, ax = plt.subplots(figsize=(10, 6))

for key, _, color, label in STRATEGIES:
    if key not in data:
        continue
    df = data[key]
    x = df["dataset_size"].values
    y = df["cumul_scanner_s"].values #y = df["cumul_trainer_s"].values

    ax.plot(x, y, marker="o", markersize=5, linewidth=2.2,
            color=color, alpha=0.9, label=label)

    # Annotate final value in hours at the end of the line
    x_end, y_end = x[-1], y[-1]
    ax.annotate(
        f"{y_end / 3600:.2f}h",
        xy=(x_end, y_end),
        xytext=(6, 0), textcoords="offset points",
        fontsize=ANNOT_FONT_SIZE, color=color,
        va="center",
    )

ax.set_xlabel("Dataset size (# training samples)", fontsize=AXIS_LABEL_FONT_SIZE)
ax.set_ylabel("Cumulative Scanner time (s)", fontsize=AXIS_LABEL_FONT_SIZE)
ax.set_title("Cumulative scanner time", fontsize=TITLE_FONT_SIZE)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=LEGEND_FONT_SIZE)
ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
plt.tight_layout()
save_path = os.path.join(base_dir, f"time_cumul_scanner_{TIMESTAMP}.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {save_path}")

# =========================================================
# 4) Plot 2 — per-round trainer time
#    + horizontal dashed line at mean, annotated in seconds (0 decimal)
# =========================================================
fig, ax = plt.subplots(figsize=(10, 6))

for key, _, color, label in STRATEGIES:
    if key not in data:
        continue
    df = data[key]
    x = df["dataset_size"].values
    y = df["per_round_scanner_s"].values
    mean_s = float(np.mean(y))

    # Individual line
    ax.plot(x, y, marker="o", markersize=5, linewidth=2.2,
            color=color, alpha=0.9, label=label)

    # Mean dashed line
    ax.hlines(mean_s, x[0], x[-1],
              colors=color, linestyles="--", linewidth=1.8)

    # Annotate mean at the right end of the dashed line
    ax.annotate(
        f"mean: {mean_s:.0f}s",
        xy=(x[-1], mean_s),
        xytext=(6, 0), textcoords="offset points",
        fontsize=ANNOT_FONT_SIZE, color=color,
        va="center",
    )

ax.set_xlabel("Dataset size (# training samples)", fontsize=AXIS_LABEL_FONT_SIZE)
ax.set_ylabel("Scanner time per round (s)", fontsize=AXIS_LABEL_FONT_SIZE)
ax.set_title("Per-round scanner time", fontsize=TITLE_FONT_SIZE)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=LEGEND_FONT_SIZE)
ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
plt.tight_layout()
save_path = os.path.join(base_dir, f"time_per_round_scanner_{TIMESTAMP}.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {save_path}")

print("\nAll timing figures generated.")
