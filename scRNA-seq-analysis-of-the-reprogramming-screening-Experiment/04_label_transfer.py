#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
scPoli标签传递脚本
用于将参考数据集的细胞类型标签传递给查询数据集
"""

import os
import gc
import torch
import argparse
import warnings
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import issparse, csr_matrix, vstack
from scarches.models.scpoli import scPoli
import time
import psutil
import multiprocessing as mp

# 环境优化与配置
warnings.filterwarnings("ignore")
torch.set_num_threads(mp.cpu_count())
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"GPU可用: {torch.cuda.get_device_name(0)}")

sc.settings.verbosity = 0
sc.set_figure_params(dpi=80, facecolor='white', frameon=False)


# ================= 性能监控工具 =================

class Timer:
    def __init__(self, name="步骤"):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        print(f"\n⏱️ 开始: {self.name}...")
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        print(f"✅ {self.name} 完成，耗时: {self.interval:.2f} 秒 ({self.interval/60:.2f} 分钟)")


def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem_gb = process.memory_info().rss / 1024**3
    print(f"📊 当前内存使用: {mem_gb:.2f} GB")
    return mem_gb


def cleanup_memory(*args):
    """清理内存中的变量"""
    for var in args:
        del var
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print_memory_usage()


# ================= 核心工具函数 =================

def align_genes(query_adata, source_adata, chunk_size=200000):
    """
    将query数据对齐到source基因空间（缺失基因填充0）
    """
    print(f"--- [Step] 启动基因对齐: {query_adata.n_obs} 细胞 ---")

    # 确保稀疏矩阵格式
    if not issparse(query_adata.X):
        query_adata.X = csr_matrix(query_adata.X)
    query_adata.X = query_adata.X.astype(np.float32)

    # 获取共同基因（不区分大小写）
    source_genes_lower = {gene.lower(): gene for gene in source_adata.var_names}
    query_genes_lower = {gene.lower(): gene for gene in query_adata.var_names}
    common_lower = set(source_genes_lower.keys()).intersection(set(query_genes_lower.keys()))
    common_genes = [source_genes_lower[gl] for gl in common_lower]

    print(f"找到共有基因: {len(common_genes)} 个")
    print(f"源数据基因总数: {len(source_adata.var_names)} 个")

    # 建立索引映射
    source_gene_dict = {gene: i for i, gene in enumerate(source_adata.var_names)}

    # 找到query中对应的基因索引
    query_gene_idx = []
    for source_gene in common_genes:
        source_lower = source_gene.lower()
        for i, qg in enumerate(query_adata.var_names):
            if qg.lower() == source_lower:
                query_gene_idx.append(i)
                break

    source_gene_idx = [source_gene_dict[g] for g in common_genes]

    # 分块对齐
    n_cells, n_genes_source = query_adata.shape[0], source_adata.shape[1]
    chunks = []
    for i in range(0, n_cells, chunk_size):
        end_idx = min(i + chunk_size, n_cells)
        print(f"  处理细胞 {i} - {end_idx}...")
        chunk_data = query_adata.X[i:end_idx, query_gene_idx]
        chunk_extended = csr_matrix((end_idx - i, n_genes_source), dtype=np.float32)
        chunk_extended[:, source_gene_idx] = chunk_data
        chunks.append(chunk_extended)

    print("合并对齐后的数据...")
    return vstack(chunks)


def run_label_transfer(query_adata, source_adata, ref_model, output_dir, file_path, cell_type_key):
    """
    执行 scPoli 标签传递全流程
    """
    base_name = os.path.basename(file_path).split('.')[0]
    figures_folder = os.path.join(output_dir, "figures")
    os.makedirs(figures_folder, exist_ok=True)

    print(f"\n检查基因数匹配情况:")
    print(f"  Query数据基因数: {query_adata.shape[1]}")
    print(f"  Source数据基因数: {source_adata.shape[1]}")

    # 1. 初始化 Query 模型
    print("--- [Step] 初始化 Query 模型 ---")
    scpoli_query = scPoli.load_query_data(
        adata=query_adata,
        reference_model=ref_model,
        labeled_indices=[],
    )

    # 2. 迁移训练
    print("--- [Step] 开始迁移训练 ---")
    scpoli_query.train(
        n_epochs=50,
        pretraining_epochs=40,
        eta=5,
        unlabeled_prototype_training=False
    )

    # 3. 检查并转换数据格式
    print("--- [Step] 准备预测数据 ---")
    if issparse(query_adata.X):
        print("检测到稀疏矩阵，转换为稠密矩阵...")
        query_adata_dense = query_adata.copy()
        query_adata_dense.X = query_adata_dense.X.toarray().astype(np.float32)
    else:
        query_adata_dense = query_adata

    # 4. 预测
    print("--- [Step] 提取预测结果 ---")
    scpoli_query.model.eval()
    with torch.no_grad():
        results = scpoli_query.classify(query_adata_dense, scale_uncertainties=True)
        preds = results[cell_type_key]["preds"]
        uncert = results[cell_type_key]["uncert"]

        q_latent = scpoli_query.get_latent(query_adata_dense, mean=True)
        adata_latent = sc.AnnData(X=q_latent.astype(np.float32), obs=query_adata.obs.copy())
        adata_latent.obs[f'{cell_type_key}_pred'] = preds
        adata_latent.obs[f'{cell_type_key}_uncert'] = uncert

    # 5. UMAP可视化
    print("--- [Step] 生成UMAP图 ---")
    sc.pp.neighbors(adata_latent, random_state=42)
    sc.tl.umap(adata_latent, random_state=42)

    sc.pl.umap(adata_latent, color=f'{cell_type_key}_pred', show=False, frameon=False)
    plt.savefig(os.path.join(figures_folder, f"{base_name}_prediction_umap.png"),
                dpi=80, bbox_inches='tight')
    plt.close()

    # 清理临时变量
    if 'query_adata_dense' in locals() and query_adata_dense is not query_adata:
        del query_adata_dense
    gc.collect()

    return adata_latent, q_latent, preds, uncert, scpoli_query


# ================= 主程序 =================

def main():
    parser = argparse.ArgumentParser(description='scPoli标签传递 - 将参考标签传递给查询数据')
    
    parser.add_argument('--query_path', type=str, required=True, 
                        help='查询数据路径（待标注的数据）')
    parser.add_argument('--reference_path', type=str, required=True, 
                        help='参考数据路径（已标注的源数据）')
    parser.add_argument('--model_dir', type=str, required=True, 
                        help='预训练模型目录')
    parser.add_argument('--cell_type_key', type=str, default='lineage', 
                        help='源数据中的细胞类型列名')
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='输出目录')
    parser.add_argument('--chunk_size', type=int, default=200000, 
                        help='基因对齐的分块大小')
    
    args = parser.parse_args()

    # 设置输出目录
    if args.output_dir is None:
        base_name = os.path.basename(args.query_path).split('.')[0]
        args.output_dir = f"./label_transfer_{base_name}"

    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("scPoli 标签传递")
    print("=" * 60)
    print(f"查询数据: {args.query_path}")
    print(f"参考数据: {args.reference_path}")
    print(f"模型目录: {args.model_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"细胞类型键: {args.cell_type_key}")
    print("=" * 60)

    total_timer = Timer("标签传递完整流程")
    with total_timer:

        # 加载数据
        with Timer("数据加载"):
            print_memory_usage()
            
            # 加载查询数据
            print("加载查询数据...")
            query_adata = sc.read_h5ad(args.query_path)
            if 'orig.ident' not in query_adata.obs.columns:
                query_adata.obs['orig.ident'] = os.path.basename(args.query_path).split('.')[0]
            
            print(f"查询数据维度: {query_adata.shape}")
            
            # 加载参考数据
            print("加载参考数据...")
            ref_adata = sc.read_h5ad(args.reference_path)
            if not issparse(ref_adata.X):
                ref_adata.X = csr_matrix(ref_adata.X)
            ref_adata.X = ref_adata.X.astype(np.float32)
            
            print(f"参考数据维度: {ref_adata.shape}")

            # 加载模型
            print("加载预训练模型...")
            ref_model = scPoli.load(args.model_dir, adata=ref_adata)

        # 基因对齐
        with Timer("基因对齐"):
            print("将查询数据对齐到参考数据基因空间...")
            aligned_X = align_genes(
                query_adata,
                ref_adata,
                chunk_size=args.chunk_size
            )

            query_aligned = sc.AnnData(
                X=aligned_X,
                obs=query_adata.obs.copy(),
                var=ref_adata.var.copy(),
                dtype=np.float32
            )

            # 添加临时细胞类型标签
            query_aligned.obs[args.cell_type_key] = ref_adata.obs[args.cell_type_key].iloc[0]

            print(f"对齐后数据维度: {query_aligned.shape}")
            print(f"基因数: {query_aligned.shape[1]} (应与模型期望的{ref_adata.shape[1]}一致)")

            # 清理不再需要的变量
            del aligned_X
            gc.collect()

        # 运行标签传递
        with Timer("标签传递"):
            adata_latent, q_latent, preds, uncert, scpoli_query = run_label_transfer(
                query_aligned, ref_adata, ref_model, args.output_dir, args.query_path, args.cell_type_key
            )

            # 清理不再需要的变量
            del query_aligned, ref_adata, ref_model, scpoli_query
            gc.collect()

        # 保存结果
        with Timer("保存结果"):
            # 将预测结果添加到原始数据
            query_adata.obs[f'{args.cell_type_key}_pred'] = preds
            query_adata.obs[f'{args.cell_type_key}_uncert'] = uncert
            query_adata.obsm["X_scpoli_transfer"] = q_latent

            # 清理不再需要的变量
            del q_latent, preds, uncert
            gc.collect()

            # 保存带有标签传递结果的数据
            output_file = os.path.join(args.output_dir, "data_with_predictions.h5ad")
            query_adata.write_h5ad(output_file, compression='gzip')
            print(f"✅ 标签传递结果已保存: {output_file}")
            
            # 同时保存一个精简版（只包含obs和obsm）
            minimal_adata = sc.AnnData(
                obs=query_adata.obs.copy(),
                obsm={'X_scpoli_transfer': query_adata.obsm['X_scpoli_transfer'].copy()}
            )
            minimal_file = os.path.join(args.output_dir, "predictions_only.h5ad")
            minimal_adata.write_h5ad(minimal_file, compression='gzip')
            print(f"✅ 精简预测结果已保存: {minimal_file}")

        # 清理
        del query_adata
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print_memory_usage()

    # 打印统计信息
    print("\n" + "="*60)
    print("🎉 标签传递完成！")
    print("="*60)
    print(f"总耗时: {total_timer.interval:.2f} 秒 ({total_timer.interval/60:.2f} 分钟)")
    print(f"输出目录: {args.output_dir}")
    
    # 输出文件大小
    if os.path.exists(output_file):
        size_gb = os.path.getsize(output_file) / 1024**3
        print(f"输出文件大小: {size_gb:.2f} GB")


if __name__ == "__main__":
    main()
