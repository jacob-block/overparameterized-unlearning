#!/bin/bash


EPOCH_VALS=(10 100 1000)

for EPOCHS in "${EPOCH_VALS[@]}"; do
python3 main.py parse-results \
    --experiment sin \
    --save-dir saved-models/sin/unlearned/${EPOCHS}epochs
done
