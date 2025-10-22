#!/bin/bash

DATASET="CIFAR"

python main.py plot-results \
    --experiment erase \
    --init-model-dir base-models/erase/${DATASET}/pct-color-01 \
    --unlearned-dir saved-results/erase/${DATASET}/pct-color-01/epochs5-retain01