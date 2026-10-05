#!/bin/bash
#SBATCH --job-name="e4s-nat"
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --nodelist=morgoth
#SBATCH --mem-per-cpu=64G
#SBATCH -o /mnt/experiments/nlp/eliseo/eval4sim_mh/logs/%x-%j.out
#SBATCH -e /mnt/experiments/nlp/eliseo/eval4sim_mh/logs/%x-%j.err

SIF="/mnt/experiments/slurm/singularity-containers/eliseo/cuda-eliseo.sif"

source secrets

singularity run --disable-cache --nv \
    --bind /mnt:/mnt \
    --pwd "$PWD" \
    $SIF \
    /bin/bash -c "src/scripts/configure_setup.sh && source venv/bin/activate && \
        python src/e4s/naturalness/dnli_evaluator.py --batch && \
        python src/e4s/naturalness/create_overall_summary.py data/results/naturalness/ && \
        python src/e4s/naturalness/fill_results_table.py"

