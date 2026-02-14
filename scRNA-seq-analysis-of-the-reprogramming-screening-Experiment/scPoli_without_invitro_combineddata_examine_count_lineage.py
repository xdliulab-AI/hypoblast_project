import os
import argparse
import anndata as ad
import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scvi
from sklearn.metrics import classification_report
from scarches.models.scpoli import scPoli
from scarches.dataset.trvae.data_handling import remove_sparsity

import warnings
warnings.filterwarnings('ignore')

class scPoliIntegration:
    def __init__(self, seed=42, cell_type_key="lineage_pred", n_top_genes=2000, model_dir=None):
        """初始化scPoli整合分析类"""
        self.seed = seed
        self.cell_type_key = cell_type_key
        self.n_top_genes = n_top_genes
        self.model_dir = model_dir
        self.setup_environment()

    def setup_environment(self):
        """设置绘图和计算环境"""
        sc.settings.set_figure_params(dpi=100, frameon=False, figsize=(4, 4))
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['figure.figsize'] = (4, 4)
        sc.settings.seed = self.seed
        print(f"环境设置完成 - 细胞类型键: {self.cell_type_key}, 高变基因数: {self.n_top_genes}")

    def setup_model_directory(self):
        """设置模型保存目录"""
        if self.model_dir is None:
            # 默认保存到当前目录下的model文件夹
            self.model_dir = os.path.join(os.getcwd(), "scpoli_model")

        # 创建目录（如果不存在）
        os.makedirs(self.model_dir, exist_ok=True)
        print(f"模型将保存到: {self.model_dir}")
        return self.model_dir

    def check_data_quality(self, adata, dataset_name=""):
        """
        检查数据质量
        """
        print(f"\n=== 检查 {dataset_name} 数据质量 ===")
        print(f"数据维度: {adata.n_obs} 细胞, {adata.n_vars} 基因")
        print(f"X矩阵类型: {type(adata.X)}")
        print(f"X矩阵数据类型: {adata.X.dtype}")

        # 检查稀疏矩阵
        if hasattr(adata.X, 'toarray'):
            print("X矩阵是稀疏矩阵")
            sample_data = adata.X[:5, :5].toarray()
        else:
            sample_data = adata.X[:5, :5]
            print("X矩阵是稠密矩阵")

        print(f"数据样本 (前5x5):\n{sample_data}")

        # 检查关键列是否存在
        required_columns = ['orig.ident', self.cell_type_key]
        for col in required_columns:
            if col in adata.obs.columns:
                print(f"{col} 存在，唯一值数量: {adata.obs[col].nunique()}")
            else:
                print(f"❌ 警告: {col} 不存在于数据中")

    def ensure_numeric_data(self, adata):
        """
        确保数据是数值类型
        """
        print("确保数据为数值类型...")

        # 处理X矩阵
        if hasattr(adata.X, 'toarray'):
            # 稀疏矩阵转稠密并确保float32
            adata.X = adata.X.toarray().astype(np.float32)
        else:
            # 确保是float32
            adata.X = adata.X.astype(np.float32)

        print(f"X矩阵数据类型已设置为: {adata.X.dtype}")

        # 处理layers中的counts
        if "counts" in adata.layers:
            if hasattr(adata.layers["counts"], 'toarray'):
                adata.layers["counts"] = adata.layers["counts"].toarray().astype(np.int32)
            else:
                adata.layers["counts"] = adata.layers["counts"].astype(np.int32)
            print(f"counts层数据类型: {adata.layers['counts'].dtype}")

        return adata

    def fix_duplicate_index(self, adata, dataset_name=""):
        """修复重复的索引"""
        print(f"检查 {dataset_name} 的索引唯一性...")

        if adata.obs.index.duplicated().any():
            duplicate_count = adata.obs.index.duplicated().sum()
            print(f"发现 {duplicate_count} 个重复索引，正在修复...")

            new_index = []
            count_dict = {}

            for original_idx in adata.obs.index:
                if original_idx not in count_dict:
                    count_dict[original_idx] = 0
                    new_index.append(original_idx)
                else:
                    count_dict[original_idx] += 1
                    new_index.append(f"{original_idx}_dup{count_dict[original_idx]}")

            adata.obs.index = new_index
            print("重复索引修复完成")
        else:
            print(f"{dataset_name} 索引唯一性检查通过")

        return adata

    def clean_data_types(self, adata):
        """清理数据类型"""
        print("清理数据类型...")

        # 定义数值列
        numeric_columns = ['nCount_RNA', 'nFeature_RNA', 'percent.mt']

        for col in numeric_columns:
            if col in adata.obs.columns:
                print(f"处理列: {col}")
                print(f"  原始类型: {adata.obs[col].dtype}")

                if adata.obs[col].dtype == 'object':
                    try:
                        adata.obs[col] = pd.to_numeric(adata.obs[col], errors='coerce')
                        print(f"  转换为数值类型: {adata.obs[col].dtype}")
                    except Exception as e:
                        print(f"  转换失败: {e}")
                        adata.obs.drop(columns=[col], inplace=True)

        # 确保分类变量是字符串类型
        categorical_columns = ['orig.ident', self.cell_type_key]
        for col in categorical_columns:
            if col in adata.obs.columns:
                adata.obs[col] = adata.obs[col].astype(str)
                print(f"  将 {col} 转换为字符串类型")

        return adata

    def load_and_validate_data(self, data_path):
        """加载和验证数据"""
        print("加载数据...")

        # 加载主要数据
        print(f"加载主要数据: {data_path}")
        adata = sc.read_h5ad(data_path)
        print(f"主要数据维度: {adata.n_obs} 细胞, {adata.n_vars} 基因")

        # 检查数据质量
        self.check_data_quality(adata, "主要数据")

        # 修复重复索引
        adata = self.fix_duplicate_index(adata, "主要数据")

        # 清理数据类型
        adata = self.clean_data_types(adata)

        # 确保数值数据
        adata = self.ensure_numeric_data(adata)

        return adata

    def preprocess_data(self, adata):
        """数据预处理"""
        print("数据预处理...")

        # 确保模型目录已设置
        self.setup_model_directory()

        # 使用临时副本进行高变基因选择
        print("使用临时副本进行高变基因选择...")
        adata_tmp = adata.copy()

        # 确保临时数据是数值类型
        adata_tmp = self.ensure_numeric_data(adata_tmp)

        # 对临时数据进行标准化和log转换
        print("对临时数据进行标准化...")
        sc.pp.normalize_total(adata_tmp, target_sum=1e4)
        sc.pp.log1p(adata_tmp)

        # 选择高变基因
        print("选择高变基因...")
        sc.pp.highly_variable_genes(
            adata_tmp,
            n_top_genes=self.n_top_genes,
            flavor="cell_ranger",
            batch_key="orig.ident",
            subset=False
        )

        # 使用高变基因筛选原始数据
        print(f"使用 {self.n_top_genes} 个高变基因筛选数据...")
        hvg_mask = adata_tmp.var['highly_variable'].values
        adata_hvg = adata[:, hvg_mask].copy()

        # 将高变基因信息添加到原始数据
        adata.var['highly_variable'] = hvg_mask

        # 准备用于scPoli训练的数据
        print("准备scPoli训练数据...")

        # 确保数据是浮点数类型
        adata_hvg = self.ensure_numeric_data(adata_hvg)

        # 确保counts层存在且为整数
        if "counts" not in adata_hvg.layers:
            # 创建counts层（假设X矩阵包含原始counts）
            if np.issubdtype(adata_hvg.X.dtype, np.floating):
                # 如果是浮点数，转换为整数（四舍五入）
                counts_data = np.round(adata_hvg.X).astype(np.int32)
            else:
                counts_data = adata_hvg.X.astype(np.int32)
            adata_hvg.layers["counts"] = counts_data
            print("创建了counts层")
        else:
            # 确保counts层是整数类型
            if hasattr(adata_hvg.layers["counts"], 'toarray'):
                counts_data = adata_hvg.layers["counts"].toarray().astype(np.int32)
            else:
                counts_data = adata_hvg.layers["counts"].astype(np.int32)
            adata_hvg.layers["counts"] = counts_data

        print(f"高变基因数据维度: {adata_hvg.shape}")
        print(f"高变基因数量: {adata_hvg.n_vars}")
        print(f"X矩阵数据类型: {adata_hvg.X.dtype}")
        print(f"counts层数据类型: {adata_hvg.layers['counts'].dtype}")

        return adata_hvg

    def train_scpoli_model(self, adata):
        """训练scPoli模型"""
        # 设置模型目录
        model_dir = self.setup_model_directory()
        model_subdir = os.path.join(model_dir, f'scpoli_model_{self.cell_type_key}_hvg{self.n_top_genes}')

        print("\n=== scPoli 模型训练 ===")

        # 数据验证
        print("验证训练数据...")
        print(f"数据形状: {adata.shape}")
        print(f"X矩阵数据类型: {adata.X.dtype}")

        # 检查数据范围
        if hasattr(adata.X, 'toarray'):
            x_data = adata.X.toarray()
        else:
            x_data = adata.X

        print(f"数据范围: [{x_data.min():.2f}, {x_data.max():.2f}]")
        print(f"数据均值: {x_data.mean():.2f}")

        # 设置参数
        condition_key = "orig.ident"
        cell_type_key = self.cell_type_key

        print(f"条件变量: {condition_key}")
        print(f"细胞类型变量: {cell_type_key}")
        print(f"条件分布:\n{adata.obs[condition_key].value_counts()}")
        print(f"细胞类型分布:\n{adata.obs[cell_type_key].value_counts()}")

        # 确保分类变量是字符串类型
        adata.obs[condition_key] = adata.obs[condition_key].astype(str)
        adata.obs[cell_type_key] = adata.obs[cell_type_key].astype(str)

        # 训练参数
        early_stopping_kwargs = {
            "early_stopping_metric": "val_prototype_loss",
            "mode": "min",
            "threshold": 0,
            "patience": 20,
            "reduce_lr": True,
            "lr_patience": 13,
            "lr_factor": 0.1,
        }

        # 训练scPoli模型
        print("初始化scPoli模型...")

        try:
            scpoli_model = scPoli(
                adata=adata,
                condition_keys=condition_key,
                cell_type_keys=cell_type_key,
                embedding_dims=50,
                recon_loss='nb'
            )

            print("开始训练scPoli模型...")
            scpoli_model.train(
                n_epochs=50,
                pretraining_epochs=40,
                early_stopping_kwargs=early_stopping_kwargs,
                eta=5
            )

            print("✅ scPoli模型训练完成")

        except Exception as e:
            print(f"❌ scPoli模型训练失败: {e}")
            raise

        return scpoli_model, model_subdir

    def save_model_and_results(self, scpoli_model, model_dir, adata):
        """保存模型和结果"""
        print("保存模型和结果...")

        # 确保数据类型正确
        adata = self.clean_data_types(adata)
        adata = self.ensure_numeric_data(adata)

        # 保存模型
        try:
            print(f"保存模型到: {model_dir}")
            scpoli_model.save(model_dir, overwrite=True, save_anndata=True)
            print(f"✅ 模型已保存至 {model_dir}")
        except Exception as e:
            print(f"❌ 保存错误: {e}")
            # 尝试其他保存方式
            try:
                import pickle
                model_path = os.path.join(model_dir, "scpoli_model.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(scpoli_model, f)
                print(f"✅ 模型已通过pickle保存至 {model_path}")
            except Exception as e2:
                print(f"❌ pickle保存也失败: {e2}")

        return scpoli_model

    def get_latent_representation(self, scpoli_model, adata):
        """获取潜在表示"""
        print("获取scPoli潜在表示...")

        try:
            scpoli_model.model.eval()

            # 确保数据格式正确
            if hasattr(adata.X, 'toarray'):
                adata.X = adata.X.toarray().astype(np.float32)

            data_latent_source = scpoli_model.get_latent(adata, mean=True)
            adata.obsm["scPoli"] = data_latent_source

            print(f"潜在表示维度: {data_latent_source.shape}")
            print("✅ 潜在表示获取成功")

        except Exception as e:
            print(f"❌ 获取潜在表示失败: {e}")
            raise

        return adata

    def perform_clustering_analysis(self, adata):
        """执行降维聚类分析"""
        print("执行降维聚类分析...")
        
        # 计算邻居图
        print("计算邻居图...")
        sc.pp.neighbors(adata, use_rep="scPoli", random_state=self.seed)
        
        # UMAP降维
        print("计算UMAP...")
        sc.tl.umap(adata, random_state=self.seed)
        
        # Leiden聚类
        print("执行Leiden聚类...")
        sc.tl.leiden(adata, random_state=self.seed)
        
        # 可选：尝试不同的分辨率参数进行聚类
        print("尝试不同的聚类分辨率...")
        for res in [0.4, 0.6, 0.8, 1.0]:
            cluster_key = f'leiden_res_{res}'
            sc.tl.leiden(adata, resolution=res, key_added=cluster_key, random_state=self.seed)
            print(f"  {cluster_key}: {adata.obs[cluster_key].nunique()} 个聚类")
        
        print("✅ 降维聚类分析完成")
        return adata

    def save_integrated_data(self, adata):
        """保存整合后的数据（包含所有聚类结果）"""
        print("保存整合后的数据...")
        
        # 确保模型目录已设置
        self.setup_model_directory()
        
        # 确保数据类型正确
        adata = self.clean_data_types(adata)
        adata = self.ensure_numeric_data(adata)
        
        # 构建输出路径
        output_path = os.path.join(self.model_dir, f"scpoli_integrated_{self.cell_type_key}_hvg{self.n_top_genes}.h5ad")
        
        try:
            # 保存包含所有分析结果的h5ad文件
            adata.write_h5ad(output_path)
            print(f"✅ 整合数据已保存至: {output_path}")
            print(f"   包含以下分析结果:")
            print(f"   - 潜在表示 (scPoli): {adata.obsm['scPoli'].shape}")
            print(f"   - UMAP坐标: {adata.obsm['X_umap'].shape}")
            print(f"   - Leiden聚类结果: {list(adata.obs.columns[adata.obs.columns.str.startswith('leiden')])}")
            
        except Exception as e:
            print(f"❌ 保存整合数据时出错: {e}")
            # 尝试保存到当前目录
            fallback_path = f'scpoli_integrated_{self.cell_type_key}_hvg{self.n_top_genes}.h5ad'
            try:
                adata.write_h5ad(fallback_path)
                print(f"✅ 整合数据已保存到当前目录: {fallback_path}")
            except Exception as e2:
                print(f"❌ 整合数据保存失败: {e2}")
        
        return output_path

    def visualize_results(self, adata):
        """可视化结果"""
        print("基于scPoli的聚类可视化...")

        # 确保模型目录已设置
        self.setup_model_directory()

        # 绘制UMAP图 - orig.ident
        plt.figure(figsize=(8, 6))
        sc.pl.umap(adata, color="orig.ident", frameon=False, show=False,
                  title=f"Batch (orig.ident) - {self.n_top_genes} HVGs", size=20)
        plt.tight_layout()
        plot_path = os.path.join(self.model_dir, f"scpoli_orig_ident_{self.cell_type_key}_hvg{self.n_top_genes}.pdf")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 批次UMAP图保存至: {plot_path}")

        # 绘制UMAP图 - lineage_pred
        plt.figure(figsize=(8, 6))
        sc.pl.umap(adata, color=self.cell_type_key, frameon=False, show=False,
                  title=f"Cell Type ({self.cell_type_key}) - {self.n_top_genes} HVGs", size=20)
        plt.tight_layout()
        plot_path = os.path.join(self.model_dir, f"scpoli_{self.cell_type_key}_hvg{self.n_top_genes}.pdf")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 细胞类型UMAP图保存至: {plot_path}")

        # 绘制UMAP图 - leiden聚类 (默认分辨率)
        plt.figure(figsize=(8, 6))
        sc.pl.umap(adata, color="leiden", frameon=False, show=False,
                  title=f"Leiden Clusters - {self.n_top_genes} HVGs", size=20)
        plt.tight_layout()
        plot_path = os.path.join(self.model_dir, f"scpoli_leiden_{self.cell_type_key}_hvg{self.n_top_genes}.pdf")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Leiden聚类UMAP图保存至: {plot_path}")

        # 绘制不同分辨率的聚类结果
        leiden_cols = [col for col in adata.obs.columns if col.startswith('leiden_res_')]
        for col in leiden_cols:
            plt.figure(figsize=(8, 6))
            sc.pl.umap(adata, color=col, frameon=False, show=False,
                      title=f"Leiden {col} - {self.n_top_genes} HVGs", size=20)
            plt.tight_layout()
            plot_path = os.path.join(self.model_dir, f"scpoli_{col}_{self.cell_type_key}_hvg{self.n_top_genes}.pdf")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ {col} UMAP图保存至: {plot_path}")

    def run_full_analysis(self, data_path):
        """运行完整分析流程"""
        try:
            print("=== 开始完整scPoli分析流程 ===")

            # 1. 加载数据
            print("\n步骤1: 加载和验证数据")
            adata = self.load_and_validate_data(data_path)

            # 2. 数据预处理
            print("\n步骤2: 数据预处理")
            adata_hvg = self.preprocess_data(adata)

            # 3. 训练scPoli模型
            print("\n步骤3: 训练scPoli模型")
            scpoli_model, model_dir = self.train_scpoli_model(adata_hvg)

            # 4. 保存模型
            print("\n步骤4: 保存模型")
            scpoli_model = self.save_model_and_results(scpoli_model, model_dir, adata_hvg)

            # 5. 获取潜在表示
            print("\n步骤5: 获取潜在表示")
            adata = self.get_latent_representation(scpoli_model, adata_hvg)

            # 6. 执行降维聚类分析
            print("\n步骤6: 执行降维聚类分析")
            adata = self.perform_clustering_analysis(adata)

            # 7. 保存整合后的数据（包含所有聚类结果）
            print("\n步骤7: 保存整合数据")
            output_path = self.save_integrated_data(adata)

            # 8. 可视化结果
            print("\n步骤8: 可视化结果")
            self.visualize_results(adata)

            print("\n🎉 ✅ scPoli分析完成！")
            print(f"模型和结果保存在: {self.model_dir}")
            print(f"最终h5ad文件包含:")
            print(f"  - scPoli潜在表示")
            print(f"  - UMAP降维坐标") 
            print(f"  - Leiden聚类结果（多种分辨率）")
            print(f"  - 所有原始观测数据和变量数据")
            print(f"文件位置: {output_path}")

        except Exception as e:
            print(f"\n❌ 分析过程中出现错误: {e}")
            import traceback
            print(f"详细错误信息:\n{traceback.format_exc()}")
            raise


def main():
    """主函数，支持命令行参数"""
    parser = argparse.ArgumentParser(description='scPoli整合分析 - 专注于lineage_pred细胞类型')
    parser.add_argument('--data_path', type=str, required=True,
                       help='输入数据路径 (H5AD格式，必需)')
    parser.add_argument('--n_top_genes', type=int, choices=[2000, 4000],
                       default=2000, help='高变基因数量: 2000 或 4000 (默认: 2000)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子 (默认: 42)')
    parser.add_argument('--model_dir', type=str, default=None,
                       help='模型保存目录 (默认: 当前目录下的scpoli_model文件夹)')

    args = parser.parse_args()

    print("=" * 60)
    print("scPoli整合分析 - lineage_pred细胞类型")
    print("=" * 60)
    print(f"数据路径: {args.data_path}")
    print(f"细胞类型键: lineage_pred")
    print(f"高变基因数: {args.n_top_genes}")
    print(f"随机种子: {args.seed}")
    print(f"模型目录: {args.model_dir if args.model_dir else '当前目录下的scpoli_model文件夹'}")
    print("=" * 60)

    # 验证数据文件是否存在
    if not os.path.exists(args.data_path):
        print(f"❌ 错误: 数据文件不存在: {args.data_path}")
        return

    # 创建分析器实例
    analyzer = scPoliIntegration(
        seed=args.seed,
        cell_type_key="lineage_pred",  # 固定使用lineage_pred
        n_top_genes=args.n_top_genes,
        model_dir=args.model_dir
    )

    # 运行分析
    analyzer.run_full_analysis(args.data_path)


# 运行分析
if __name__ == "__main__":
    main()
