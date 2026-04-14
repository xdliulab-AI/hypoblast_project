#!/usr/bin/env python3
"""
单细胞数据降维和聚类脚本
功能：读取h5ad文件，进行PCA、UMAP降维，并使用多个分辨率进行Leiden聚类
"""

import scanpy as sc
import warnings
warnings.filterwarnings('ignore')

# 设置scanpy的参数
sc.settings.verbosity = 3  # 设置输出信息的详细程度
sc.settings.set_figure_params(dpi=80, facecolor='white')

def main():
    # 输入文件路径
    input_file = "/storage2/liuxiaodongLab/jiangjing/Projects/XueyingFan/PD_XueyingFan/20260310_hypoblast_dataset/output_02/h5ad_results/final_complete.h5ad"

    print("="*60)
    print("开始单细胞数据分析流程")
    print("="*60)

    # 读取数据
    print(f"\n[1/5] 读取数据文件: {input_file}")
    try:
        adata = sc.read_h5ad(input_file)
        print(f"   成功读取数据:")
        print(f"   细胞数: {adata.n_obs}")
        print(f"   基因数: {adata.n_vars}")
    except Exception as e:
        print(f"   读取文件失败: {e}")
        return

    # 检查数据是否已经过预处理
    print("\n[2/5] 检查数据预处理状态")
    if 'highly_variable' not in adata.var.columns:
        print("   未找到highly_variable基因标记，建议先进行预处理")
        print("   继续运行，但结果可能不理想")

    # PCA降维
    print("\n[3/5] 进行PCA降维...")
    try:
        sc.tl.pca(adata)
        print(f"   PCA完成，方差解释率: {adata.uns['pca']['variance_ratio'][:5]}")
    except Exception as e:
        print(f"   PCA计算失败: {e}")
        return

    # 计算邻居图
    print("\n[4/5] 计算邻居图 (使用前30个PCs)...")
    try:
        sc.pp.neighbors(adata, n_pcs=30)
        print("   邻居图计算完成")
    except Exception as e:
        print(f"   邻居图计算失败: {e}")
        return

    # UMAP降维
    print("\n[5/5] 进行UMAP降维...")
    try:
        sc.tl.umap(adata)
        print("   UMAP降维完成")
    except Exception as e:
        print(f"   UMAP计算失败: {e}")
        return

    # 使用不同分辨率进行Leiden聚类
    print("\n" + "="*60)
    print("开始Leiden聚类 (多个分辨率)")
    print("="*60)

    resolutions = [0.6]

    for res in resolutions:
        print(f"\n  使用分辨率 {res} 进行聚类...")
        try:
            # 使用不同的resolution运行Leiden聚类
            sc.tl.leiden(adata, resolution=res, key_added=f'leiden_res{res}')

            # 获取聚类结果
            n_clusters = len(adata.obs[f'leiden_res{res}'].unique())
            print(f"    分辨率 {res}: 得到 {n_clusters} 个簇")

            # 可以选择保存UMAP图
            sc.pl.umap(adata, color=f'leiden_res{res}',
                      title=f'UMAP (Leiden res={res}, {n_clusters} clusters)',
                      save=f'_leiden_res{res}.png')

        except Exception as e:
            print(f"    分辨率 {res} 聚类失败: {e}")
            continue

    # 显示所有聚类结果的统计
    print("\n" + "="*60)
    print("聚类结果统计")
    print("="*60)

    leiden_cols = [col for col in adata.obs.columns if col.startswith('leiden_res')]
    if leiden_cols:
        for col in leiden_cols:
            res_value = col.replace('leiden_res', '')
            n_clusters = len(adata.obs[col].unique())
            print(f"  分辨率 {res_value}: {n_clusters} 个簇")
    else:
        print("  未找到Leiden聚类结果")

    # 保存结果
    print("\n" + "="*60)
    print("保存结果")
    print("="*60)

    output_file = input_file.replace('.h5ad', '_clustered.h5ad')
    try:
        adata.write(output_file)
        print(f"  结果已保存到: {output_file}")
    except Exception as e:
        print(f"  保存文件失败: {e}")

    # 显示数据概览
    print("\n" + "="*60)
    print("最终数据概览")
    print("="*60)
    print(adata)
    print("\n可用的聚类结果列:")
    for col in adata.obs.columns:
        if col.startswith('leiden'):
            print(f"  - {col}")

    print("\n脚本运行完成!")

if __name__ == "__main__":
    main()
