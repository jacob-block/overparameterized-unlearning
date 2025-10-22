#!/bin/bash

DATASET="CIFAR"
EPOCHS="5"
RETAIN_PCT="001"
PCT_FORGET="01"
hyperparam_str="epochs${EPOCHS}-retain${RETAIN_PCT}"

echo ""
echo "Running grid search for collapse experiment from config file: configs/collapse/${hyperparam_str}.json"
echo "Saving results to saved-results/collapse/${DATASET}/pct-forget-${PCT_FORGET}/${hyperparam_str}"
echo ""

ibrun python3 main.py grid-search \
    --experiment collapse \
    --init-model-dir "base-models/collapse/${DATASET}/pct-forget-${PCT_FORGET}" \
    --data-path "./data" \
    --batch-size 256 \
    --grid-config "configs/collapse/${DATASET}/${hyperparam_str}.json" \
    --save-dir "saved-results/collapse/${DATASET}/pct-forget-${PCT_FORGET}/${hyperparam_str}" \
    --mpi
