#!/bin/bash

OUT_DIR=saved-models/sin/base

python3 main.py generate-data \
    --experiment sin \
    --out-dir $OUT_DIR \
    --seed-start 1 \
    --seed-end 11 \
    --lr 1e-3 \
    --epochs-initial 100000 \
    --epochs-gt 50000 \
    --device cuda \
    --net-width 300 \
    --num-samples-r 50 \
    --num-samples-f 5 \
    --x-min -15.708 \
    --x-max 15.708 \
    --num-test-pts 1000 \
    --verbose
