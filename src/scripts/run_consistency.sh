#!/bin/bash
#SBATCH --job-name="e4s-con"
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16G
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

source secrets

singularity run --disable-cache \
    --bind "$HF_HOME" \
    --pwd "$PWD" \
    $SIF \
    bash src/e4s/consistency/run_consistency_inner.sh