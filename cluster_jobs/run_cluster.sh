#!/bin/bash
#SBATCH --partition=gpu_branco # lab partition — 30-day limit
#SBATCH --output=logs/JALpipeline_%j.out
#SBATCH --error=logs/JALpipeline_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G

set -euo pipefail

echo "RUNNING ON: $(hostname)"

# Set these paths for your cluster mount points.
export JAL2_DATA_ROOT="/ceph/branco/Jasmine_Laurence/Experimental_Data"
export JAL2_DLC_CONFIG="/ceph/branco/Jasmine_Laurence/DLC/DLC_220424_JAL6_7_inc/JAL_NPX1-Jasmine-2023-03-22/config.yaml"

module purge
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate JAL2pipeline

# Run from repository root.
REPO_ROOT="$HOME/repos/JAL2_subgoal_pipeline"
cd "$REPO_ROOT"
mkdir -p logs

python cluster_jobs/run_process_postprocess_cluster.py
