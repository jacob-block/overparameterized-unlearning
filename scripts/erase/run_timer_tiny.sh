#!/bin/bash


DATASET="Tiny"
RETAIN_PCT="01"
PCT_COLOR="01"
hyperparam_str="retain${RETAIN_PCT}"

DATA_PATH="" #FILL IN

echo ""
echo "Running timer"
echo "Saving results to saved-results/erase/${DATASET}/pct-color-${PCT_COLOR}/timer/${hyperparam_str}"
echo ""

python3 -u timer.py \
    --data-path $DATA_PATH \
    --init-model-dir "base-models/erase/${DATASET}/pct-color-${PCT_COLOR}" \
    --save-dir "saved-results/erase/${DATASET}/pct-color-${PCT_COLOR}/timer/${hyperparam_str}" \
    --retain-access-pct $RETAIN_PCT \
    --batch-size 512
