#!/bin/bash

EPOCHS=1000

python3 main.py plot-results \
    --experiment sin \
    --init-model-dir base-models/sin \
    --unlearned-dir saved-results/sin/unlearned/${EPOCHS}epochs \
    --seed 1 \
    --gen-legend
