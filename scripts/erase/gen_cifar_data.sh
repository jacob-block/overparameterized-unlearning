#!/bin/bash


python3 -u main.py generate-data \
    --experiment erase \
    --pct-color .001 \
    --dataset CIFAR-10 \
    --batch-size 256 \
    --out-dir base-models/erase/CIFAR/pct-color-001 \
    --epochs-initial 100 \
    --epochs-gt 100 \
    --seed-start 1 \
    --seed-end 6 \
    --verbose
