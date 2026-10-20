#!/bin/bash
#SBATCH --job-name="e4s-emo-orig"
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=16G
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

source secrets

singularity run --disable-cache --nv \
    --bind "$HF_HOME" \
    --pwd "$PWD" \
    $SIF \
    /bin/bash -c "src/scripts/configure_setup.sh && source venv/bin/activate && \
        python3 src/e4s/emotionality_personachat/evaluate.py \
        --gt_path data/original_e4s/personachat.jsonl \
        --simulations_dir data/original_e4s/simulations \
        --output_dir data/original_e4s/emotionality/results \
        --neutral"
