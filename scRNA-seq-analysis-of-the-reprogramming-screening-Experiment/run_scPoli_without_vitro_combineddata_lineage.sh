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
module load gcc/11.2.0

# Run label transfer
python scPoli_without_invitro_combineddata_examine_count_lineage.py  --n_top_genes 2000 \
	--model_dir /storage2/liuxiaodongLab/jiangjing/Projects/XueyingFan/PD_XueyingFan/20260209_hypoblast_99sample_anno/model \
        --data_path /storage2/liuxiaodongLab/jiangjing/Projects/XueyingFan/PD_XueyingFan/20260209_hypoblast_99sample_anno/output/adata_qc_with_groups_label_hvg.h5ad
