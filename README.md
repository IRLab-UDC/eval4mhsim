# Emotional Fidelity Beyond Persona Consistency in LLM-Based User Modeling

Code and data for the paper:

> **Emotional Fidelity Beyond Persona Consistency in LLM-Based User Modeling**
> Eliseo Bao, Anxo Pérez, Javier Parapar · *Under review*

---

<p align="center">
  <img src="assets/overview.png" alt="Overview of Eval4MHSim: a simulator conditions on real user activity to generate a simulated reply, which is compared against the corresponding real reply across four dimensions." width="850">
</p>

## Overview

Persona consistency and turn-level coherence do not guarantee that a simulator reproduces the affective characteristics of the population it simulates. We study this for real Reddit users with self-reported depression severity.

- **Eval4MHSim** extends Eval4Sim with *emotionality*, a fourth dimension next to *adherence*, *consistency* and *naturalness*. It measures the alignment between the Ekman emotion distributions of pooled real and simulated replies through the Jensen-Shannon divergence.
- **POWER** (*Persona Optimized on Writing and Emotionality Refined*) is a generator–critic pipeline that turns behavioral profiles and user writing samples into affectively grounded personas. It is compared with a direct profile-based persona (**P**).

We build personas for 116 eRisk users who completed the BDI-II. On 63 of them (785 test replies) we evaluate 27 configurations: 3 Gemma 3 sizes × 3 persona settings (none, P, POWER) × 3 conditioning strategies (ZS, ICL, ICLR).

### Key Findings

- Same-user in-context examples are the strongest driver of alignment with the users' real replies.
- POWER with same-user examples gives the best overall score (12B ICL-POWER, 0.983), and a 12B model outperforms all 27B configurations.
- Adding a persona lowers emotionality in 17 of 18 matched comparisons.
- Zero-shot POWER personas at 4B and 12B reach near-top adherence but the lowest emotionality.

---

## Repository Structure

```
src/
  dataset_gathering/      # Dataset collection and preprocessing
  persona_descriptions/   # Persona prompt construction and generation
  persona_simulation/     # Simulation (simulate.py, vLLM-based)
  e4s/
    adherence/            # ColBERT-based adherence evaluation
    consistency/          # Authorship verification (TF-IDF, PAN metrics)
    naturalness/          # Dialogue NLI naturalness evaluation
    emotionality/         # Ekman emotion classification and JSD scoring
    overall/              # Aggregate score computation and tables
  scripts/                # SLURM job scripts for each pipeline stage
```

---

## Data

Users and BDI-II questionnaires come from the [eRisk](https://erisk.irlab.org/) depression severity task (CLEF 2019–2021). **eRisk data is not publicly available**; access requires registration with the eRisk organizers. Comments and threads are collected with the Reddit API up to each user's last eRisk writing.

Of 170 eRisk users, 116 have a RedditMetis profile, 95 keep at least three training comments, and 63 have test replies.

The full P and POWER personas and their BDI-II labels are available under a data use agreement (contact [eliseo.bao@udc.es](mailto:eliseo.bao@udc.es)). [`data/public_sample/`](data/public_sample/) holds three anonymized users with both personas (`personas_p.jsonl`, `personas_power.jsonl`, one `{"id", "system_prompt"}` per line) and their BDI-II answers, score and severity (`golden.jsonl`).

---

## Setup

**Requirements:** Python 3.10+, CUDA-capable GPU(s), Singularity (for SLURM execution).

```bash
# 1. Configure environment variables (HF token, cache paths)
cp secrets_example secrets
# Edit secrets with your HF_HOME path, HF_TOKEN and Reddit API credentials

# 2. Create virtualenv and install dependencies
bash src/scripts/configure_setup.sh
```

Dependencies are listed in `requirements.txt` (scikit-learn, pandas, matplotlib, ColBERT, FAISS, sentence-transformers, vLLM).

---

## Pipeline

Steps must be run in order. Each step has a corresponding SLURM script in `src/scripts/`. For non-SLURM environments, extract the Python command from the script body.

### 1. Dataset preparation

```bash
python src/dataset_gathering/fetch_activity.py
python src/dataset_gathering/build_dataset.py
python src/dataset_gathering/clean_dataset.py
python src/dataset_gathering/build_cutoffs.py
```

### 2. Persona construction

Fetch RedditMetis behavioral profiles, then build personas under each of the two strategies evaluated in the paper: **P** (direct profile-based persona) and **POWER** (generator–critic pipeline grounded in behavioral profile, writing samples, and explicit affective instructions):

<p align="center">
  <img src="assets/power_pipeline.png" alt="Overview of the POWER persona construction pipeline: a generator drafts a persona from user evidence, a critic scores it on behavioral accuracy, writing style, and emotionality, and feedback drives iterative refinement." width="850">
</p>

```bash
python src/persona_descriptions/fetch_redditmetis.py
python src/persona_descriptions/fetch_sources.py

# P: direct profile-based persona
python src/persona_descriptions/build_prompts_p.py

# POWER: generator-critic pipeline (requires GPU; SLURM: run_generate_personas_optimized.sh)
python src/persona_descriptions/build_prompts_power.py
python src/persona_descriptions/generate_personas_power.py
```

**P** uses only the raw RedditMetis profile (via Llama 3.3 70B). **POWER** additionally conditions on up to 20 writing samples per generator iteration, runs three generator iterations with critic feedback from Phi-4 along behavioral, stylistic, and affective dimensions, and explicitly estimates the user's Ekman emotion tendencies.

### 3. Simulation

Runs all 27 configurations (3 model sizes × 3 persona settings × 3 conditioning strategies). Edit `src/scripts/run_simulation.sh` to select which scenarios to run.

```bash
# SLURM
sbatch src/scripts/run_simulation.sh

# Direct (example: ICL-POWER with Gemma 3 12B)
python src/persona_simulation/simulate.py \
    --model google/gemma-3-12b-it \
    --persona-source optimized \
    --personas-optimized data/personas_power.jsonl \
    --persona-tag power \
    --few-shot-k 10 \
    --few-shot-source same_user

# Direct (example: ZS, no persona, no examples)
python src/persona_simulation/simulate.py \
    --model google/gemma-3-12b-it \
    --persona-source none \
    --few-shot-k 0
```

Conditioning strategies: **ZS** (zero-shot, no demonstrations), **ICL** (`--few-shot-source same_user`), **ICLR** (`--few-shot-source random_user`, a control). Simulations are written to `data/simulations/` with filenames encoding model, temperature, persona variant, and few-shot configuration.

### 4. Evaluation

Run each dimension independently; results are written to `data/results/`.

```bash
# Adherence (ColBERT retrieval)
sbatch src/scripts/run_adherence.sh

# Consistency (authorship verification)
sbatch src/scripts/run_consistency.sh

# Naturalness (Dialogue NLI)
sbatch src/scripts/run_naturalness.sh

# Emotionality (Ekman JSD)
sbatch src/scripts/run_emotionality.sh
# Direct:
# python src/e4s/emotionality/evaluate.py \
#     --dataset_test data/dataset_test.jsonl \
#     --simulations_dir data/simulations \
#     --output_dir data/results/emotionality \
#     --neutral
```

### 5. Aggregate scores and tables

Restrict the outputs to the evaluated users, then build the overall table:

```bash
python src/e4s/overall/filter_results.py --users users.jsonl --output data/results_filtered
python src/e4s/overall/table_overall.py \
    --adherence data/results_filtered/adherence/similarity.csv \
    --consistency data/results_filtered/consistency/ \
    --naturalness data/results_filtered/naturalness/overall_summary.json \
    --emotionality data/results_filtered/emotionality/similarity.csv \
    --output data/results_filtered/overall.tex
```

---

## Emotionality Generalization (Eval4Sim)

To reproduce the cross-domain emotionality analysis against the PersonaChat reference (three Gemma 3 and three Qwen3 sizes), run the emotionality evaluation on simulations from the original Eval4Sim experiments, placing them under `data/original_e4s/`:

```bash
python src/e4s/emotionality/evaluate.py --simulations_dir data/original_e4s/simulations ...
python src/e4s/emotionality/plot.py
```

---

## License

Code and the public sample are released under the [MIT License](LICENSE).

## Citation

Citation will be added upon publication.
