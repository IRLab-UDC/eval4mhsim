#!/bin/bash
#SBATCH --job-name="e4s-emo"
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --nodelist=morgoth
#SBATCH --mem-per-cpu=16G
#SBATCH -o /mnt/experiments/nlp/eliseo/eval4sim_mh/logs/%x-%j.out
#SBATCH -e /mnt/experiments/nlp/eliseo/eval4sim_mh/logs/%x-%j.err

SIF="/mnt/experiments/slurm/singularity-containers/eliseo/cuda-eliseo.sif"

source secrets

singularity run --disable-cache --nv \
    --bind /mnt:/mnt \
    --pwd "$PWD" \
    $SIF \
    /bin/bash -c "src/scripts/configure_setup.sh && source venv/bin/activate && \
        python3 src/e4s/emotionality/evaluate.py \
        --dataset_test data/dataset_test.jsonl \
        --simulations_dir data/simulations \
        --output_dir data/results/emotionality \
        --neutral"
