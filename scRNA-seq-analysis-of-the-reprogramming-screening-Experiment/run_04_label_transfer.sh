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
python 01_label_transfer.py \
	--query_path /storage2/liuxiaodongLab/jiangjing/Projects/XueyingFan/PD_XueyingFan/20260310_hypoblast_dataset/code/adata_qc_filtered.h5ad  \
	--model_dir /storage2/liuxiaodongLab/jiangjing/Projects/XueyingFan/PD_XueyingFan/20260127_hypoblast_100sample_anno/model/ \
	--reference_path /storage2/liuxiaodongLab/jiangjing/Projects/XueyingFan/PD_XueyingFan/20260127_hypoblast_100sample_anno/model/adata.h5ad \
	--cell_type_key lineage \
	--output_dir ../output_01 \
	--chunk_size 100000

