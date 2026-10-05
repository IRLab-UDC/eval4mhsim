#!/bin/bash
#SBATCH --job-name="e4s-con"
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --nodelist=morgoth
#SBATCH --mem-per-cpu=16G
#SBATCH -o /mnt/experiments/nlp/eliseo/eval4sim_mh/logs/%x-%j.out
#SBATCH -e /mnt/experiments/nlp/eliseo/eval4sim_mh/logs/%x-%j.err

SIF="/mnt/experiments/slurm/singularity-containers/eliseo/cuda-eliseo.sif"

source secrets

singularity run --disable-cache \
    --bind /mnt:/mnt \
    --pwd "$PWD" \
    $SIF \
    bash src/e4s/consistency/run_consistency_inner.sh