#!/bin/bash

DATASETS=(
    aime24
    aime25
    amc23
    olympiadbench
    math500
)

METHODS=(
    greedy
    bon
    greedy_pass_n
    bon_majority
    dn_bon
    dn_bon_updated
)

mkdir -p logs

for DATASET in "${DATASETS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        JOB_NAME="qwen_1_5b_${DATASET}_${METHOD}"
        OUT_FILE="logs/out_${DATASET}_${METHOD}.log"
        ERR_FILE="logs/err_${DATASET}_${METHOD}.log"

        sbatch \
            --job-name="$JOB_NAME" \
            --output="$OUT_FILE" \
            --error="$ERR_FILE" \
            run_single_experiment.slurm "$DATASET" "$METHOD"
    done
done