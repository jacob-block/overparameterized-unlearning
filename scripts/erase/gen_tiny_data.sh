#!/bin/bash

python3 -u main.py generate-data \
    --experiment erase \
    --pct-color 0.001 \
    --dataset TinyImageNet \
    --data-path "${SCRATCH}/data" \
    --batch-size 512 \
    --num-workers 32 \
    --out-dir "base-models/erase/Tiny/pct-color-001" \
    --epochs-initial 100 \
    --epochs-gt 100 \
    --color-start-epoch 10 \
    --seed-start 1 \
    --seed-end 6 \
    --verbose
