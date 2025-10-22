#!/bin/bash

DATASET="CIFAR"
EPOCHS="5"
RETAIN_PCT="001"
PCT_COLOR="01"
hyperparam_str="epochs${EPOCHS}-retain${RETAIN_PCT}"

echo ""
echo "Running grid search from config file: configs/erase/${hyperparam_str}.json"
echo "Saving results to saved-results/erase/${DATASET}/${hyperparam_str}"
echo ""

ibrun python3 main.py grid-search \
    --experiment erase \
    --init-model-dir "base-models/erase/${DATASET}/pct-color-${PCT_COLOR}/" \
    --batch-size 256 \
    --grid-config "configs/erase/${DATASET}/${hyperparam_str}.json" \
    --save-dir "saved-results/erase/${DATASET}/pct-color-${PCT_COLOR}/${hyperparam_str}" \
    --mpi