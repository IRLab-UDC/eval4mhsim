#!/bin/bash
#SBATCH --job-name="e4s-gen-op"
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
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
        python src/persona_descriptions/generate_personas_power.py"
