#!/usr/bin/env bash

#SBATCH -J scPoli
#SBATCH -o scPoli.%j.out
#SBATCH -e scPoli.%j.err
#SBATCH -p v100,v100-af,a40-tmp,a40-quad,a100-40g
#SBATCH -q gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=800G    


source /home/liuxiaodongLab/jiangjing/miniconda3/etc/profile.d/conda.sh
conda activate /storage2/liuxiaodongLab/jiangjing/miniconda3/envs/agent_new

# Run label transfer
python label_transfer_color_v3.py \
      --file_path /storage2/liuxiaodongLab/jiangjing/Projects/XueyingFan/PD_XueyingFan/20260127_hypoblast_100sample_anno/code/adata_qc_with_groups.h5ad \
      --figures_folder /storage2/liuxiaodongLab/jiangjing/Projects/XueyingFan/PD_XueyingFan/20260209_hypoblast_99sample_anno/label_transfer \
      --custom_model_dir /storage2/liuxiaodongLab/jiangjing/Projects/XueyingFan/PD_XueyingFan/20260127_hypoblast_100sample_anno/model \
      --custom_adata_path /storage2/liuxiaodongLab/jiangjing/Projects/XueyingFan/PD_XueyingFan/20260127_hypoblast_100sample_anno/model/adata.h5ad

echo "Label transfer completed"
