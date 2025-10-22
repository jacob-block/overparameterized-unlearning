#!/bin/bash


EPOCHS=1000

ibrun python3 main.py grid-search \
    --experiment sin \
    --init-model-dir base-models/sin \
    --grid-config configs/sin/grid_${EPOCHS}epochs.json \
    --save-dir saved-results/sin/unlearned/${EPOCHS}epochs \
    --mpi