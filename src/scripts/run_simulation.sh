#!/bin/bash
#SBATCH --job-name="e4s-sim"
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --nodelist=morgoth
#SBATCH --mem-per-cpu=64G
#SBATCH -o /mnt/experiments/nlp/eliseo/eval4sim_mh/logs/%x-%j.out
#SBATCH -e /mnt/experiments/nlp/eliseo/eval4sim_mh/logs/%x-%j.err

SIF="/mnt/experiments/slurm/singularity-containers/eliseo/cuda-eliseo.sif"

source secrets

mkdir -p data/simulations

MODELS=(
    "google/gemma-3-4b-it"
    "google/gemma-3-12b-it"
    "google/gemma-3-27b-it"
)

declare -A SCENARIOS
# SCENARIOS["ZS"]="--persona-source none --few-shot-k 0"
# SCENARIOS["ICL"]="--persona-source none --few-shot-k 10 --few-shot-source same_user"
# SCENARIOS["ICLR"]="--persona-source none --few-shot-k 10 --few-shot-source random_user"
# SCENARIOS["ZS-P"]="--persona-source default --few-shot-k 0"
# SCENARIOS["ICL-P"]="--persona-source default --few-shot-k 10 --few-shot-source same_user"
# SCENARIOS["ICLR-P"]="--persona-source default --few-shot-k 10 --few-shot-source random_user"
# SCENARIOS["ZS-PO"]="--persona-source optimized --few-shot-k 0"
# SCENARIOS["ICL-PO"]="--persona-source optimized --few-shot-k 10 --few-shot-source same_user"
# SCENARIOS["ICLR-PO"]="--persona-source optimized --few-shot-k 10 --few-shot-source random_user"
# SCENARIOS["ZS-POW"]="--persona-source optimized --personas-optimized data/personas_pow.jsonl --persona-tag pow --few-shot-k 0"
# SCENARIOS["ICL-POW"]="--persona-source optimized --personas-optimized data/personas_pow.jsonl --persona-tag pow --few-shot-k 10 --few-shot-source same_user"
# SCENARIOS["ICLR-POW"]="--persona-source optimized --personas-optimized data/personas_pow.jsonl --persona-tag pow --few-shot-k 10 --few-shot-source random_user"
# SCENARIOS["ZS-POWE"]="--persona-source optimized --personas-optimized data/personas_powe.jsonl --persona-tag powe --few-shot-k 0"
# SCENARIOS["ICL-POWE"]="--persona-source optimized --personas-optimized data/personas_powe.jsonl --persona-tag powe --few-shot-k 10 --few-shot-source same_user"
# SCENARIOS["ICLR-POWE"]="--persona-source optimized --personas-optimized data/personas_powe.jsonl --persona-tag powe --few-shot-k 10 --few-shot-source random_user"
SCENARIOS["ZS-POWER"]="--persona-source optimized --personas-optimized data/personas_power.jsonl --persona-tag power --few-shot-k 0"
SCENARIOS["ICL-POWER"]="--persona-source optimized --personas-optimized data/personas_power.jsonl --persona-tag power --few-shot-k 10 --few-shot-source same_user"
SCENARIOS["ICLR-POWER"]="--persona-source optimized --personas-optimized data/personas_power.jsonl --persona-tag power --few-shot-k 10 --few-shot-source random_user"

for model in "${MODELS[@]}"; do
    for label in "${!SCENARIOS[@]}"; do
        singularity run --disable-cache --nv \
            --bind /mnt:/mnt \
            --pwd "$PWD" \
            $SIF \
            /bin/bash -c "src/scripts/configure_setup.sh && source venv/bin/activate && \
                python src/persona_simulation/simulate.py --model $model ${SCENARIOS[$label]}"
    done
done
