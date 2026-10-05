#!/bin/bash
set -e

EXPERIMENT="adherence"
CHECKPOINT="colbert-ir/colbertv2.0"
DOC_MAXLEN=512
NBITS=8
POOL_SIZES=(1 10 25 50 75 100 150 200 300 400 500 750 1000)

DATASET_TEST="data/dataset_test.jsonl"
SIMULATIONS_DIR="data/simulations"
OUTPUT_BASE="data/adherence"

run_dataset() {
    local name="$1"
    local data_dir="$2"
    local index_name="$3"

    echo "  Indexing ${name}..."
    python3 src/e4s/adherence/index_simple.py \
        --collection "${data_dir}/documents.tsv" \
        --index_name "$index_name" \
        --experiment "$EXPERIMENT" \
        --checkpoint "$CHECKPOINT" \
        --doc_maxlen "$DOC_MAXLEN" \
        --nbits "$NBITS"

    mkdir -p "${data_dir}/runs" "${data_dir}/results"

    for pool_size in "${POOL_SIZES[@]}"; do
        echo "  Pool size ${pool_size}..."
        python3 src/e4s/adherence/search_conversations.py \
            --index_name "$index_name" \
            --queries "${data_dir}/queries.tsv" \
            --qrels "${data_dir}/qrels.txt" \
            --collection "${data_dir}/documents.tsv" \
            --output "${data_dir}/runs/run_pool_${pool_size}.txt" \
            --experiment "$EXPERIMENT" \
            --pool_size "$pool_size" \
            --k 1000
        python3 src/e4s/adherence/evaluate.py \
            --qrels "${data_dir}/qrels.txt" \
            --run "${data_dir}/runs/run_pool_${pool_size}.txt" \
            --output "${data_dir}/results/results_pool_${pool_size}.txt"
    done
}

echo "=========================================="
echo "GROUND TRUTH"
echo "=========================================="
GT_DIR="${OUTPUT_BASE}/ground_truth"
if [ -d "${GT_DIR}/results" ]; then
    echo "Skipping ground_truth (already evaluated)"
else
    python3 src/e4s/adherence/prepare_retrieval.py \
        --input "$DATASET_TEST" \
        --output_dir "$GT_DIR"
    run_dataset "ground_truth" "$GT_DIR" "adherence_ground_truth"
fi

DATASET_ARGS=("--dataset" "${GT_DIR}/results:Ground Truth")

echo ""
echo "=========================================="
echo "SIMULATIONS"
echo "=========================================="
for sim_file in ${SIMULATIONS_DIR}/*.jsonl; do
    [ -f "$sim_file" ] || continue
    fname=$(basename "$sim_file" .jsonl)
    model_id="${fname#simulations_}"
    safe_id=$(echo "$model_id" | tr '/.' '_')

    echo ""
    echo "Processing: ${model_id}"
    sim_dir="${OUTPUT_BASE}/${safe_id}"

    if [ -d "${sim_dir}/results" ]; then
        echo "Skipping ${model_id} (already evaluated)"
        DATASET_ARGS+=("--dataset" "${sim_dir}/results:${model_id}")
        continue
    fi

    python3 src/e4s/adherence/prepare_retrieval.py \
        --input "$sim_file" \
        --output_dir "$sim_dir" \
        --personas "$DATASET_TEST"
    run_dataset "$model_id" "$sim_dir" "adherence_${safe_id}"
    DATASET_ARGS+=("--dataset" "${sim_dir}/results:${model_id}")
done

echo ""
echo "=========================================="
echo "COMPARISON"
echo "=========================================="
python3 src/e4s/adherence/compare_datasets.py \
    --base_dir "${OUTPUT_BASE}" \
    --pool_sizes "${POOL_SIZES[@]}" \
    --output "${OUTPUT_BASE}/comparison.csv"

python3 src/e4s/adherence/plot.py \
    --csv "${OUTPUT_BASE}/comparison.csv" \
    --output "${OUTPUT_BASE}/plot.pdf"

python3 src/e4s/adherence/similarity.py \
    --csv "${OUTPUT_BASE}/comparison.csv" \
    --reference "Ground Truth" \
    --output "${OUTPUT_BASE}/similarity.csv"

python3 src/e4s/adherence/table_adherence.py \
    --similarity "${OUTPUT_BASE}/similarity.csv" \
    --reference "Ground Truth" \
    --output "${OUTPUT_BASE}/table_adherence.tex"

echo ""
echo "Done. Results in ${OUTPUT_BASE}/"
