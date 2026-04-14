#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
scPoli整合分析与下游分析脚本（最终优化版）
流程：HVG训练 -> 完整数据嵌入 -> 完整数据下游分析
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
from scipy.sparse import issparse
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


# ================= 整合分析类 =================

class ScPoliIntegration:
    def __init__(self, seed=42, cell_type_key="lineage_pred", n_top_genes=2000, output_dir=None):
        self.seed = seed
        self.cell_type_key = cell_type_key
        self.n_top_genes = n_top_genes
        self.output_dir = output_dir or os.path.join(os.getcwd(), "integration_results")

        # 创建所有必要的目录
        self.figures_dir = os.path.join(self.output_dir, "figures")
        self.models_dir = os.path.join(self.output_dir, "models")
        self.h5ad_dir = os.path.join(self.output_dir, "h5ad_results")

        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.h5ad_dir, exist_ok=True)

        print(f"📁 输出目录: {self.output_dir}")
        print(f"📁 图表目录: {self.figures_dir}")
        print(f"📁 模型目录: {self.models_dir}")
        print(f"📁 h5ad目录: {self.h5ad_dir}")

    def preprocess_with_hvg(self, adata):
        """预处理并选择高变基因（仅用于模型训练）"""
        with Timer("数据预处理和HVG选择"):
            adata_tmp = adata.copy()
            sc.pp.normalize_total(adata_tmp, target_sum=1e4)
            sc.pp.log1p(adata_tmp)

            sc.pp.highly_variable_genes(
                adata_tmp,
                n_top_genes=self.n_top_genes,
                flavor="cell_ranger",
                batch_key="orig.ident",
                subset=False
            )

            hvg_mask = adata_tmp.var['highly_variable'].values
            adata_hvg = adata[:, hvg_mask].copy()

            if "counts" not in adata_hvg.layers:
                if issparse(adata_hvg.X):
                    adata_hvg.layers["counts"] = adata_hvg.X.copy()
                else:
                    adata_hvg.layers["counts"] = adata_hvg.X.copy()

            # 保存HVG基因信息
            self.hvg_genes = adata_hvg.var_names.tolist()

            # 保存HVG数据（可选，用于调试）
            hvg_output = os.path.join(self.h5ad_dir, "hvg_training_data.h5ad")
            adata_hvg.write_h5ad(hvg_output, compression='gzip')
            print(f"✅ HVG训练数据已保存: {hvg_output}")

            # 清理临时变量
            del adata_tmp
            gc.collect()

            print(f"HVG数据维度: {adata_hvg.shape}")
            print(f"HVG基因数: {adata_hvg.n_vars} (原始基因数的 {adata_hvg.n_vars/adata.n_vars*100:.1f}%)")

        return adata_hvg

    def train_integration_model(self, adata_hvg):
        """训练整合模型（仅在HVG数据上）"""
        with Timer("整合模型训练"):
            condition_key = "orig.ident"

            # 确保数据类型正确
            adata_hvg.obs[condition_key] = adata_hvg.obs[condition_key].astype(str)
            adata_hvg.obs[self.cell_type_key] = adata_hvg.obs[self.cell_type_key].astype(str)

            print(f"批次条件: {condition_key}")
            print(f"细胞类型键: {self.cell_type_key}")
            print(f"唯一批次: {adata_hvg.obs[condition_key].unique()}")
            print(f"唯一细胞类型: {adata_hvg.obs[self.cell_type_key].unique()[:5]}...")

            early_stopping_kwargs = {
                "early_stopping_metric": "val_prototype_loss",
                "mode": "min",
                "threshold": 0,
                "patience": 20,
                "reduce_lr": True,
                "lr_patience": 13,
                "lr_factor": 0.1,
            }

            scpoli_model = scPoli(
                adata=adata_hvg,
                condition_keys=condition_key,
                cell_type_keys=self.cell_type_key,
                embedding_dims=200,
                recon_loss='nb'
            )

            scpoli_model.train(
                n_epochs=50,
                pretraining_epochs=40,
                early_stopping_kwargs=early_stopping_kwargs,
                eta=5
            )

            # 保存模型
            self.save_model(scpoli_model)

        return scpoli_model

    def save_model(self, model):
        """保存scPoli模型到磁盘"""
        with Timer("保存模型"):
            model_path = os.path.join(self.models_dir, "scpoli_model.pkl")
            model.save(model_path, overwrite=True)

            # 保存模型基因列表
            if hasattr(self, 'hvg_genes'):
                gene_list_path = os.path.join(self.models_dir, "model_genes.txt")
                with open(gene_list_path, 'w') as f:
                    f.write('\n'.join(self.hvg_genes))
                print(f"🧬 模型基因列表已保存: {gene_list_path}")

            print(f"✅ 模型已保存: {model_path}")

    def get_embeddings_for_full_data(self, model, adata_full, batch_size=100000):
        """为完整数据分批获取scPoli嵌入"""
        with Timer("为完整数据生成scPoli嵌入"):
            model.model.eval()
            
            n_cells = adata_full.n_obs
            n_batches = (n_cells + batch_size - 1) // batch_size
            
            print(f"总细胞数: {n_cells}, 批次大小: {batch_size}, 总批次数: {n_batches}")
            
            all_latent = []
            for i in range(n_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, n_cells)
                
                print(f"  处理批次 {i+1}/{n_batches}: 细胞 {start}-{end}")
                
                # 提取批次数据并只保留HVG基因
                batch = adata_full[start:end, self.hvg_genes].copy()
                
                # 转换为稠密矩阵（模型需要）
                if issparse(batch.X):
                    batch.X = batch.X.toarray().astype(np.float32)
                
                # 获取嵌入
                with torch.no_grad():
                    latent = model.get_latent(batch, mean=True)
                all_latent.append(latent)
                
                # 清理
                del batch
                if i % 5 == 4:  # 每5批清理一次
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # 合并所有批次的嵌入
            full_latent = np.vstack(all_latent)
            adata_full.obsm["scPoli"] = full_latent
            
            print(f"✅ 完整数据scPoli嵌入生成完成，维度: {full_latent.shape}")
            
            # 保存嵌入向量
            embedding_df = pd.DataFrame(
                full_latent,
                index=adata_full.obs_names,
                columns=[f"scPoli_{i+1}" for i in range(full_latent.shape[1])]
            )
            embedding_path = os.path.join(self.h5ad_dir, "scpoli_embeddings_full.csv")
            embedding_df.to_csv(embedding_path)
            print(f"✅ scPoli嵌入向量已保存: {embedding_path}")
            
        return adata_full

    def run_downstream_on_full_data(self, adata_full):
        """在完整数据的scPoli嵌入上进行下游分析"""
        with Timer("完整数据的下游分析"):
            
            print("1. 计算邻居图...")
            sc.pp.neighbors(
                adata_full, 
                use_rep="scPoli", 
                random_state=42
            )

            print("2. 计算UMAP...")
            sc.tl.umap(
                adata_full, 
                random_state=42
            )

            print("3. Leiden聚类...")
            sc.tl.leiden(
                adata_full, 
                resolution=1, 
                random_state=42, 
                key_added="leiden_scpoli_1"
            )

            n_clusters = adata_full.obs['leiden_scpoli_1'].nunique()
            print(f"✅ 找到 {n_clusters} 个聚类")
            
        return adata_full

    def visualize(self, adata_full):
        """可视化（基于完整数据的UMAP）"""
        with Timer("可视化"):
            
            # UMAP按批次着色
            print("生成批次UMAP...")
            fig, ax = plt.subplots(figsize=(10, 8))
            sc.pl.umap(adata_full, color="orig.ident", ax=ax, show=False, size=1)
            plt.savefig(os.path.join(self.figures_dir, "umap_batch_full.png"),
                       dpi=150, bbox_inches='tight')
            plt.close()

            # UMAP按细胞类型着色
            print("生成细胞类型UMAP...")
            fig, ax = plt.subplots(figsize=(10, 8))
            sc.pl.umap(adata_full, color=self.cell_type_key, ax=ax, show=False, size=1)
            plt.savefig(os.path.join(self.figures_dir, "umap_celltype_full.png"),
                       dpi=150, bbox_inches='tight')
            plt.close()

            # UMAP按Leiden聚类着色
            print("生成聚类UMAP...")
            fig, ax = plt.subplots(figsize=(10, 8))
            sc.pl.umap(adata_full, color="leiden_scpoli_1", ax=ax, show=False, size=1)
            plt.savefig(os.path.join(self.figures_dir, "umap_leiden_full.png"),
                       dpi=150, bbox_inches='tight')
            plt.close()

            print(f"✅ 所有图表已保存至: {self.figures_dir}")


# ================= 主程序 =================

def main():
    parser = argparse.ArgumentParser(description='scPoli整合分析与下游分析')

    parser.add_argument('--input_path', type=str, required=True,
                        help='输入数据路径（应包含标签预测结果）')
    parser.add_argument('--cell_type_key', type=str, default='lineage_pred',
                        help='细胞类型列名（通常是预测结果的列名）')
    parser.add_argument('--n_top_genes', type=int, default=2000,
                        help='高变基因数')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录')
    parser.add_argument('--batch_size', type=int, default=100000,
                        help='生成嵌入时的批次大小')

    args = parser.parse_args()

    # 设置输出目录
    if args.output_dir is None:
        base_name = os.path.basename(args.input_path).split('.')[0]
        args.output_dir = f"./integration_{base_name}"

    print("=" * 60)
    print("scPoli整合分析与下游分析（最终优化版）")
    print("=" * 60)
    print(f"输入数据: {args.input_path}")
    print(f"细胞类型键: {args.cell_type_key}")
    print(f"高变基因数: {args.n_top_genes}")
    print(f"输出目录: {args.output_dir}")
    print(f"批次大小: {args.batch_size}")
    print("=" * 60)

    total_timer = Timer("完整分析流程")
    with total_timer:

        # 1. 加载数据
        with Timer("数据加载"):
            print_memory_usage()
            adata = sc.read_h5ad(args.input_path)

            # 确保必要的列存在
            if 'orig.ident' not in adata.obs.columns:
                adata.obs['orig.ident'] = 'sample'
                print("⚠️ 'orig.ident'列不存在，使用默认值'sample'")

            if args.cell_type_key not in adata.obs.columns:
                possible_keys = [col for col in adata.obs.columns if 'lineage' in col or 'cell_type' in col or 'pred' in col]
                if possible_keys:
                    args.cell_type_key = possible_keys[0]
                    print(f"使用 '{args.cell_type_key}' 作为细胞类型列")
                else:
                    raise KeyError(f"找不到合适的细胞类型列")

            print(f"数据维度: {adata.shape}")
            print(f"细胞数: {adata.n_obs}, 基因数: {adata.n_vars}")

        # 2. 初始化整合器
        integrator = ScPoliIntegration(
            seed=42,
            cell_type_key=args.cell_type_key,
            n_top_genes=args.n_top_genes,
            output_dir=args.output_dir
        )

        # 3. HVG预处理（用于模型训练）
        adata_hvg = integrator.preprocess_with_hvg(adata)

        # 4. 训练整合模型（仅在HVG数据上）
        scpoli_model = integrator.train_integration_model(adata_hvg)

        # 5. 为完整数据生成scPoli嵌入（分批处理）
        adata = integrator.get_embeddings_for_full_data(
            scpoli_model, adata, batch_size=args.batch_size
        )

        # 6. 在完整数据的嵌入上进行下游分析（邻居、UMAP、聚类）
        adata = integrator.run_downstream_on_full_data(adata)

        # 7. 可视化
        integrator.visualize(adata)

        # 8. 保存最终结果
        with Timer("保存最终结果"):
            final_path = os.path.join(integrator.h5ad_dir, "final_complete.h5ad")
            
            adata.write_h5ad(final_path, compression='gzip')
            print(f"✅ 最终结果已保存: {final_path}")

        # 9. 清理
        del adata_hvg, scpoli_model, adata
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print_memory_usage()

    # 打印统计信息
    print("\n" + "="*60)
    print("🎉 分析完成！")
    print("="*60)
    print(f"总耗时: {total_timer.interval:.2f} 秒 ({total_timer.interval/60:.2f} 分钟)")
    print(f"输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
