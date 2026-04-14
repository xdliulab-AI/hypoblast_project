#!/usr/bin/env bash

#SBATCH -J scPoli
#SBATCH -o scPoli.%j.out
#SBATCH -e scPoli.%j.err
#SBATCH -p v100,v100-af,a40-tmp,a40-quad,a100-40g
#SBATCH -q gpu                        # Select QOS
#SBATCH -c 32                         # 64 CPU cores for batch processing
#SBATCH --gres=gpu:1                  # Single A100 GPU
#SBATCH --mem=400G                    # Realistic memory allocation


source /home/liuxiaodongLab/jiangjing/miniconda3/etc/profile.d/conda.sh
conda activate /storage2/liuxiaodongLab/jiangjing/miniconda3/envs/agent_new
module load gcc/11.2.0

# Run label transfer
python 03_leiden_resolution.py
