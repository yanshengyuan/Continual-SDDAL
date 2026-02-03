#!/bin/bash

# ==========================================
# Usage:
#   bash Neural_Experimental_Design.sh <beamshape> <lr> <start_round> <end_round> <gpu> <scanner_batch_size> <init_size> <init_only?> <retrain_frequency> <scan_only?>
#
# Example:
#   regular scheme: bash SDDAL.sh rec 0.0002 100 false 1 580 0 5 1 false
#   scan-only scheme: run the following sequence to do 1000initial+1000SDDAL
#                   bash SDDAL.sh rec 0.0002 1000 true 9999 9999 0 9999 9999
#                   bash SDDAL.sh rec 9999 9999 false 1 200 0 5 9999 true
# ==========================================

beamshape=${1:-chair}
lr=${2:-0.0002}
init_size=${3:-100}
init_only=${4:-false}
start_round=${5:-1}
end_round=${6:-580}
gpu=${7:-0}
scanner_batch_size=${8:-5}
retrain_freq=${9:-1}
scan_only=${10:-false}

trainer_batch_size=2  # fixed for Trainer.py

echo "========================================="
echo " Neural Experimental Design (NED) Pipeline"
echo " Beamshape            : ${beamshape}"
echo " Learning rate        : ${lr}"
echo " Rounds               : ${start_round} → ${end_round}"
echo " GPU                  : ${gpu}"
echo " Trainer batch size   : ${trainer_batch_size}"
echo " Scanner batch size   : ${scanner_batch_size}"
echo " Initial set size     : ${init_size}"
echo " Retrain frequency    : ${retrain_freq}"
echo " Scan only?           : ${scan_only}"
echo " Init only?           : ${init_only}"
echo "========================================="

# --- Handle init_only=true separately ---
if [ "${init_only}" = true ]; then
    echo "------------------------------"
    echo "  init_only=true → Generate initial training set and train once"
    echo "------------------------------"

    echo "  Running Initializer.py..."
    python3 Initializer.py \
        --beamshape ${beamshape} \
        --gpu ${gpu} \
        --init_size ${init_size} \
        --vis_path Design_${beamshape}

    echo "  Training model on initial set..."
    python3 Trainer.py \
        --train_data Design_${beamshape} \
        --epochs 15 \
        --batch_size ${trainer_batch_size} \
        --gpu ${gpu} \
        --lr ${lr} \
        --step_size 2 \
        --seed 123 \
        --pth_name Design_${beamshape}/models/QuantUNetT_${beamshape}

    echo "------------------------------"
    echo "  init_only pipeline finished."
    echo "------------------------------"
    exit 0
fi

# --- Regular behavior (init_only=false) ---
# Run Initializer.py only when starting from round 1 AND scan_only=false
if [ "${scan_only}" = true ]; then
    echo "------------------------------"
    echo "  scan_only=true → Skipping Initializer.py"
    echo "------------------------------"

elif [ "${start_round}" -eq 1 ]; then
    echo "------------------------------"
    echo "  Running Initializer.py"
    echo "------------------------------"

    python3 Initializer.py \
        --beamshape ${beamshape} \
        --gpu ${gpu} \
        --init_size ${init_size} \
        --vis_path Design_${beamshape}

else
    echo "------------------------------"
    echo "  Skipping Initializer.py (resuming from round ${start_round})"
    echo "------------------------------"
fi

# Loop over rounds
for ((round_sampling=${start_round}; round_sampling<=${end_round}; round_sampling++))
do
    echo "------------------------------"
    echo "  Starting Round ${round_sampling}"
    echo "------------------------------"
	
	# Re-train only when round index matches frequency
	# --- If scan_only=true → skip training completely
    if [ "${scan_only}" = false ]; then
		if (( (round_sampling-1) % retrain_freq == 0 )); then
			echo "------------------------------"
			echo "  Re-training model at round ${round_sampling}"
			echo "  (Training happens every ${retrain_freq} scans)"
			echo "------------------------------"

			python3 Trainer.py \
				--train_data Design_${beamshape} \
				--epochs 15 \
				--batch_size ${trainer_batch_size} \
				--gpu ${gpu} \
				--lr ${lr} \
				--step_size 2 \
				--seed 123 \
				--pth_name Design_${beamshape}/models/QuantUNetT_${beamshape}
		else
			echo "  Skipping training at this round (waiting for next frequency point)"
		fi
	else
		echo "  scan_only=true → training skipped."
		
	fi
	
    # Run Scanner every round (adds new samples to the dataset)
    python3 Scanner.py \
        --beamshape ${beamshape} \
        --gpu ${gpu} \
        --batch_size ${scanner_batch_size} \
        --pth_name QuantUNetT_${beamshape} \
        --round_sampling ${round_sampling} \
        --vis_path Design_${beamshape}

done

# Zernike coefficients statistics
echo "========================================="
echo "   Zernike coefficient statistics in progress…"
echo "========================================="
python3 zernike_statistics.py --beamshape ${beamshape} --init_size ${init_size}

echo "========================================="
echo "   SDDAL (Simulation-Driven Differentiable Active Learning) Pipeline Completed Successfully!"
echo "========================================="
