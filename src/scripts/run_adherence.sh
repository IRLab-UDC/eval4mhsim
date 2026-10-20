#!/bin/bash
#SBATCH --job-name="e4s-adh"
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
        bash src/e4s/adherence/run_pipeline.sh"
