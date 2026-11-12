#!/bin/bash
#SBATCH --job-name=skytrain
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=5G
#SBATCH --time=00:55:00
#SBATCH --array=31-68,71-74
#SBATCH --output=log-skytrain-%A_%a.out

# Train the per-order telluric (sky) model with tellurics_training.py, then build
# its sky mask with calculate_skymask.py (all False for orders bluer than 5015 A).
#
# Each SLURM array task handles one echelle order. Orders 69 and 70 are skipped
# (blacklisted in wrap_up_everything.py).
#
# Usage (from the repository root, with the Python environment activated):
#   DATA_DIR=/path/to/data sbatch skytrain.sh     # all orders, as a job array
#   DATA_DIR=/path/to/data bash skytrain.sh 73    # a single order
#
# Input:  ${DATA_DIR}/newprod_order{N}_full-template.pt, the batches newprod_order{N}_full_{k}.pt
#         and newprod_order{N}_full-param.pt
# Output: skymodel/sky_order{N}_full.pt and skymask/newprod_order{N}_full-skymask.pt (+ .png),
#         the inputs of merge_orders.py

set -euo pipefail

ORDER="${1:-${SLURM_ARRAY_TASK_ID:?pass an order number or submit with sbatch}}"
DATA_DIR="${DATA_DIR:?set DATA_DIR to the directory with the prepared NEID data}"
DATASET="newprod_order"
SUFFIX="full"
OUT_DIR="skymodel"

mkdir -p "$OUT_DIR"
skyfile="${OUT_DIR}/sky_order${ORDER}_${SUFFIX}.pt"

# --clobber resumes from $skyfile if it already exists.
common=(--order "$ORDER" --when "$SUFFIX" --dir "${DATA_DIR%/}/"
        --latents 1 --batch_number 1 --shuffle --clobber --verbose)

# Stage 1: warm-up at a high learning rate with the consistency loss.
./tellurics_training.py "${common[@]}" --batch_size 1000 --rate 2e-2 --iteration 1000 \
    --consistency "$DATASET" "$skyfile"

# Stage 2: continue from the stage-1 model at a lower learning rate.
./tellurics_training.py "${common[@]}" --batch_size 3000 --rate 1e-3 --iteration 3000 \
    "$DATASET" "$skyfile"

# Sky mask: the pixels where merge_orders.py divides out the telluric lines.
./calculate_skymask.py "$skyfile" --dir "${DATA_DIR%/}/" --data "$DATASET" --when "$SUFFIX" --plot
