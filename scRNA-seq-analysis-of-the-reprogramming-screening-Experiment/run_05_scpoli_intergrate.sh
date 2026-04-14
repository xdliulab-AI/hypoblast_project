#!/usr/bin/env bash

#SBATCH -J scPoli
#SBATCH -o scPoli.%j.out
#SBATCH -e scPoli.%j.err
#SBATCH -p v100,v100-af,a40-tmp,a40-quad,a100-40g
#SBATCH -q gpu                        # Select QOS
#SBATCH -c 32                         # 64 CPU cores for batch processing
#SBATCH --gres=gpu:1                  # Single A100 GPU
#SBATCH --mem=800G                    # Realistic memory allocation


source /home/liuxiaodongLab/jiangjing/miniconda3/etc/profile.d/conda.sh
conda activate /storage2/liuxiaodongLab/jiangjing/miniconda3/envs/agent_new
module load gcc/11.2.0

# Run label transfer
python 02_scpoli_intergrate.py \
	--input_path /storage2/liuxiaodongLab/jiangjing/Projects/XueyingFan/PD_XueyingFan/20260310_hypoblast_dataset/output_01/data_with_predictions.h5ad \
	--cell_type_key lineage_pred  \
        --n_top_genes 2000 \
	--output_dir ../output_emb200
