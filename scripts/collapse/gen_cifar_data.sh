#!/bin/bash

python3 -u main.py generate-data \
    --experiment collapse \
    --pct-forget .01 \
    --dataset CIFAR-10 \
    --batch-size 256 \
    --data-path ${SCRATCH}/data \
    --out-dir base-models/collapse/CIFAR/pct-forget-01 \
    --epochs-warm-start 20 \
    --epochs-initial 50 \
    --epochs-gt 50 \
    --seed-start 1 \
    --seed-end 6 \
    --verbose