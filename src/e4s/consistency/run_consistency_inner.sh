#!/bin/bash
set -e

src/scripts/configure_setup.sh
source venv/bin/activate

python src/e4s/consistency/prepare_datasets.py

TRAIN=data/consistency/train
MODEL_DIR=data/consistency/model
RESULTS_BASE=data/results/consistency
mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_DIR/vectorizer.pickle" ]; then
    echo "Skipping model training (already exists)"
else
    python src/e4s/consistency/pan23-verif-baseline-cngdist.py \
        --train \
        --model_dir="$MODEL_DIR" \
        -p="$TRAIN" \
        -t="$TRAIN" \
        -num_iterations=100
fi

for TEST_DIR in data/consistency/*/test; do
    NAME=$(basename $(dirname "$TEST_DIR"))
    OUT_DIR="$RESULTS_BASE/$NAME"

    if [ -f "$OUT_DIR/out.json" ]; then
        echo "Skipping $NAME (already evaluated)"
        continue
    fi

    mkdir -p "$OUT_DIR"

    python src/e4s/consistency/pan23-verif-baseline-cngdist.py \
        --model_dir="$MODEL_DIR" \
        -i="$TEST_DIR" \
        -num_iterations=100 \
        -o="$OUT_DIR"

    python src/e4s/consistency/evaluator.py \
        -i "$TEST_DIR" \
        -a "$OUT_DIR" \
        -o "$OUT_DIR"
done

python src/e4s/consistency/fill_results_table.py
