#!/bin/bash
#SBATCH --job-name=basaran_galactica
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=basaran_galactica_%j.log
#SBATCH --mail-user=first.last@kit.edu
#SBATCH --ntasks=1
#SBATCH --time=07:00:00
#SBATCH --gres=gpu:4

module load devel/python/3.8.6_intel_19.1
module load devel/cuda/11.6

source /path/to/venv/bin/activate

MODEL=facebook/galactica-120b MODEL_HALF_PRECISION=true MODEL_TRUST_REMOTE_CODE=true PORT=10127 python -m basaran
