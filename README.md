# SDDAL-CL: Simulation-Driven Differentiable Active Learning with Continual Learning

This repository contains the implementation of SDDAL-CL, a continual learning extension of the Simulation-Driven Differentiable Active Learning (SDDAL) framework for phase retrieval in Selective Laser Melting (SLM) machines. The work is part of a master's thesis at Eindhoven University of Technology (TU/e).

## Attribution

The original SDDAL framework — including the differentiable simulation pipeline, QuantUNetT scanner architecture, MMD diversity loss, and active query strategy — was developed by **Shengyuan Yan** ([@yanshengyuan](https://github.com/yanshengyuan)).

This repository extends Shengyuan's work by integrating **continual learning** (experience replay) into the SDDAL scanner training loop, replacing the round-by-round cold-start retraining with an incremental warm-start approach. The CL-specific orchestration scripts are the contributions of this thesis work.

## Background

Phase retrieval in SLM machines requires recovering the wavefront phase from intensity measurements. SDDAL frames this as an active learning problem: a scanner model selects the most informative beam configurations (parameterized by Zernike coefficients) to label, and a downstream model is trained on the collected data.

**SDDAL-CL** introduces experience replay into the scanner training loop, keeping training cost constant across rounds instead of growing linearly with the accumulated dataset.

## Repository Structure

```
.
├── SDDAL_InShaPe_randinit_syncUNet_ReparamCKL/   # Baseline SDDAL pipeline
│   ├── SDDAL.sh                                   # Main orchestration script
│   ├── Initializer.py                             # Generates initial training set
│   ├── Trainer.py                                 # Full cold-start training each round
│   ├── Scanner.py                                 # Active query via uncertainty + diversity
│   ├── QuantUNetT_model.py                        # Scanner model (quantile U-Net)
│   ├── unetT_model.py                             # Evaluation model
│   └── M290_MachineSimu_GPU/                      # SLM optical simulation
│
├── SDDAL_InShaPe_cl/                              # SDDAL-CL pipeline (continual learning)
│   ├── SDDAL_ting.sh                              # CL orchestration script
│   ├── SDDAL.sh                                   # Baseline script (reference copy)
│   ├── ContinualTrainer.py                        # Warm-start training with experience replay
│   ├── replay_buffer.py                           # Reservoir-sampling replay buffer
│   └── (shared files same as baseline above)
│
├── training_curve_sddal/                          # Evaluation pipeline for SDDAL-collected data
│   ├── TrainSet_curve_sddal.sh                    # Orchestrates retrain at fixed dataset sizes
│   ├── retrain.py                                 # Trains UNetT from scratch on collected data
│   └── FRCM.py                                    # Computes MAE / SSIM / FRCM metrics
│
├── training_curve_cl/                             # Evaluation pipeline for SDDAL-CL-collected data
│   └── (same structure as training_curve_sddal)
│
├── initial_sets/                                  # Fixed initial datasets for reproducible cold start
├── time/                                          # Timing CSVs, one per method (manually populated)
│   ├── sddal.csv                                  # timing_log.csv from a representative SDDAL run
│   ├── sddal_cl_replay.csv                        # timing_log.csv from a representative SDDAL-CL (replay) run
│   └── sddal_cl.csv                              # (planned) timing_log.csv from full SDDAL-CL
├── statistics.py                                  # Plots MAE/SSIM/FRCM learning curves + p-values
└── statistics_time.py                             # Plots training time comparison
```

## Experiment Folder Structure

Each run produces an experiment folder named `Design_<beamshape>_<seed>/` (baseline) or `Design_<beamshape>_<seed>_cl/` (CL). These are created by copying the template folder `Design_<beamshape>/` (e.g. `Design_rec/`), which ships with the repository and contains only the fixed test set. All training data and model checkpoints are excluded from version control via `.gitignore`.

```
Design_rec_1/                          # Example: baseline, beamshape=rec, seed=1
├── test_set/                          # Fixed held-out test set (copied from template)
│   ├── intensity/
│   │   ├── img/                       # Intensity images (.png)
│   │   └── npy/                       # Intensity arrays (.npy)
│   └── phase/
│       ├── img/                       # Phase images (.png)
│       └── npy/                       # Phase arrays (.npy)
│
├── training_set/                      # Accumulated active learning data (grows each round)
│   ├── intensity/
│   │   ├── img/                       # intensity_<round>_<j>.png
│   │   └── npy/                       # intensity_<round>_<j>.npy
│   ├── phase/
│   │   ├── img/                       # phase_<round>_<j>.png
│   │   └── npy/                       # phase_<round>_<j>.npy
│   ├── zernikes/                      # zernikes_<round>_<j>.npy  (optimized coefficients)
│   ├── init_zernikes/                 # zernikes_<round>_<j>.npy  (scanner initialization)
│   ├── zernikes_init.npy              # Zernike reference for warm-start initialization
│   └── zernikes_hist_round<N>.npy/png # Coefficient distribution histogram per round
│
├── models/
│   ├── QuantUNetT_rec.pth.tar         # Latest scanner checkpoint
│   └── replay_buffer.pkl              # (CL only) Serialized replay buffer state
│
├── latest_uncertainty/                # Scanner debug outputs (uncertainty maps)
└── timing_log.csv                     # Per-round timing: round, dataset_size, wall_clock_s,
                                       #   cumul_trainer_s, per_round_trainer_s,
                                       #   cumul_scanner_s, per_round_scanner_s
```

The CL variant (`Design_rec_1_cl/`) has the same layout with the addition of `models/replay_buffer.pkl`.

The template `Design_rec/` (and equivalents for other beam shapes) must be present in the working directory before running either pipeline. It provides the pre-populated `test_set/` and the empty directory skeleton that the scripts expect.

## Models

### QuantUNetT (Scanner)

A 5-level encoder / 4-level decoder U-Net that outputs three channels: `(low, mu, high)`, representing the lower quantile, median, and upper quantile of the predicted phase. Trained with a combined quantile (pinball) + MSE loss. Used exclusively as the **scanner** which is never used for final evaluation.

### UNetT (Evaluator)

A standard U-Net variant trained from scratch on the accumulated dataset **after** active learning completes. Used exclusively for **evaluation**, decoupling data quality from scanner quality.

## Active Learning Query Strategy

`Scanner.py` optimizes a batch of Zernike coefficients by minimizing a combined loss:

- **UtilityLoss**: maximizes the prediction interval `(high − low)` and selects configurations where the scanner is most uncertain.
- **MMDUniformLoss**: penalizes deviation from a uniform distribution over `[−1.5, 1.5]` via Maximum Mean Discrepancy and promotes diversity in the selected batch.

The simulator (`M290_MachineSimu_GPU`) renders intensity images from the optimized coefficients, which become the new labeled samples.

## Pipelines

### Baseline SDDAL (`SDDAL_InShaPe_randinit_syncUNet_ReparamCKL/SDDAL.sh`)

Each round:

1. **Scanner.py** selects a new batch of samples (default 5).
2. **Trainer.py** retrains QuantUNetT **from scratch** on all accumulated data (cold-start, 15 epochs).

Training cost grows linearly with the number of rounds.

```bash
# Example: seed=1, 80 rounds (400 scanner-acquired samples), init_size=100
bash SDDAL.sh rec 0.0002 100 true false 1 80 0 5 1 false 1 123
```

### SDDAL-CL (`SDDAL_InShaPe_cl/SDDAL_ting.sh`)

Round 1: identical cold-start via `Trainer.py` (15 epochs), same initialization as baseline.

Rounds 2+:

1. **Scanner.py** selects a new batch (same as baseline).
2. **ContinualTrainer.py** updates QuantUNetT with **warm-start + experience replay** (10 epochs):
   - Loads weights from the previous checkpoint (no weight reset).
   - Combines only the new samples with a fixed-size replay buffer (reservoir sampling).
   - Training cost is constant: `O(new_samples + buffer_size)` per round.
   - Saves updated model and buffer to disk.

```bash
# Example: same settings as above + buffer_size=500
bash SDDAL_ting.sh rec 0.0002 100 true false 1 80 0 5 1 false 1 123 500
```

#### SDDAL.sh / SDDAL_ting.sh arguments

| #   | Argument        | Description                                     |
| --- | --------------- | ----------------------------------------------- |
| 1   | `beamshape`     | Beam shape config (e.g. `rec`, `chair`, `ring`) |
| 2   | `lr`            | Learning rate                                   |
| 3   | `init_size`     | Number of samples in the initial set            |
| 4   | `init_only`     | `true` to exit after initialization             |
| 5   | `scan_only`     | `true` to skip training and run scanner only    |
| 6   | `retrain_freq`  | Train every N rounds                            |
| 7   | `end_round`     | Final active learning round                     |
| 8   | `start_round`   | Resume from this round (0 = fresh start)        |
| 9   | `scanner_batch` | Samples selected per scanner round              |
| 10  | `gpu`           | GPU device index                                |
| 11  | `fix_init`      | `true` to load init set from `initial_sets/`    |
| 12  | `seed`          | Master random seed                              |
| 13  | `init_seed`     | Seed for model weight initialization            |
| 14  | `buffer_size`   | _(SDDAL_ting.sh only)_ Replay buffer capacity   |

Output is written to `Design_<beamshape>_<seed>/` (baseline) or `Design_<beamshape>_<seed>_cl/` (CL). A `timing_log.csv` is produced per run with per-round and cumulative trainer/scanner times.

## Evaluation Pipeline

The evaluation pipeline is intentionally separated from the scanner training loop to give a fair comparison of **data quality** independent of scanner model quality.

For each method (SDDAL / SDDAL-CL), at fixed dataset sizes (e.g. 200, 300, …, 1100 samples):

1. **`retrain.py`** trains a fresh UNetT from scratch on the accumulated dataset (15 epochs).
2. **`FRCM.py`** computes MAE, SSIM, and FRCM on the held-out test set.
3. Results are written to `evaluation.txt` inside each experiment folder.

To launch the evaluation pipeline for one seed:

```bash
# Inside training_curve_sddal/ or training_curve_cl/
bash TrainSet_curve_sddal.sh rec 0.0002 <path_to_collected_data> <seed>
```

## Analysis Scripts

| Script               | Purpose                                                                                                                                         |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `statistics.py`      | Learning curves (MAE / SSIM / FRCM vs dataset size) for SDDAL and SDDAL-CL; mean ± std across seeds; Welch's t-test p-values between strategies |
| `statistics_time.py` | Training time comparison between methods. Reads one `timing_log.csv` per method from the `time/` folder. Produces two plots: (1) cumulative trainer time with the final value annotated in hours; (2) per-round trainer time with a horizontal dashed line and annotation for the mean across all rounds (in seconds). |

### Timing analysis

Training time is determined by dataset size and method, not by the random seed, so a single representative run per method is sufficient. To add a method to the timing plots, copy its `timing_log.csv` into the `time/` folder under the corresponding filename defined in `STRATEGIES` at the top of `statistics_time.py`, then run the script. Commenting out a line in `STRATEGIES` removes that method from all plots.

## Reproducibility

- `init.pth.tar` stores the initial model weights so all seeds share the same random initialization.
- `init_seed` controls weight initialization; `seed` controls the scanner's sampling sequence.
- `fix_init=true` loads pre-generated initial datasets from `initial_sets/` so the starting dataset is identical across methods.

## Dependencies

- Python 3.10
- PyTorch
- NumPy, SciPy, Matplotlib
- CUDA-capable GPU (the scanner monitors free VRAM and waits for ≥ 5200 MB before querying)
