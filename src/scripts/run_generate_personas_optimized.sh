#!/bin/bash
#SBATCH --job-name="e4s-gen-op"
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --mem-per-cpu=64G
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

source secrets

singularity run --disable-cache --nv \
    --bind "$HF_HOME" \
    --pwd "$PWD" \
    $SIF \
    /bin/bash -c "src/scripts/configure_setup.sh && source venv/bin/activate && \
        python src/persona_descriptions/generate_personas_power.py"
