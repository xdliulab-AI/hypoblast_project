#!/usr/bin/env python
# coding: utf-8

import os
import scanpy as sc
import numpy as np
import pandas as pd
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Label Transfer module unavailable: {e}")
    TORCH_AVAILABLE = False
    torch = None
import matplotlib.pyplot as plt
try:
    from scarches.models.scpoli import scPoli
    SCPOLI_AVAILABLE = True
except (ImportError, AttributeError, ValueError) as e:
    error_msg = str(e)
    if "numpy" in error_msg.lower() and "dtypes" in error_msg.lower():
        print(f"⚠️ Label Transfer module unavailable: NumPy compatibility issue with advanced ML packages. Basic scanpy analysis available.")
    else:
        print(f"Warning: scPoli not available: {e}")
    SCPOLI_AVAILABLE = False
    scPoli = None
from sklearn.metrics import classification_report
import warnings
import gc

# Suppress all warnings
warnings.filterwarnings("ignore")


def check_preprocessing(adata):
    """
    Check if the dataset is preprocessed by verifying the presence of key attributes.
    
    Parameters:
    - adata: AnnData object to check.
    
    Returns:
    - True if preprocessed, False otherwise.
    """
    required_attributes = {
        "layers": ["counts", "logcounts"],
        "obsm": ["X_pca"],
        "var": ["highly_variable"]
    }
    missing_attributes = []

    # Check layers
    for layer in required_attributes["layers"]:
        if layer not in adata.layers:
            missing_attributes.append(f"layers['{layer}']")
    
    # Check obsm
    for key in required_attributes["obsm"]:
        if key not in adata.obsm:
            missing_attributes.append(f"obsm['{key}']")
    
    # Check var
    for key in required_attributes["var"]:
        if key not in adata.var:
            missing_attributes.append(f"var['{key}']")
    
    if missing_attributes:
        print(f"Dataset is not fully preprocessed. Missing attributes: {', '.join(missing_attributes)}")
        return False
    
    print("Dataset is preprocessed and ready for label transfer.")
    return True


def load_model(model_dir, adata_path, model_type):
    """
    Load the scPoli model and reference dataset from the same directory.
    
    Parameters:
    - model_dir: Directory containing both the pre-trained scPoli model and reference data.
    - adata_path: Path to the reference AnnData file (can be None if using default).
    - model_type: Type of model ("lineage" or "cell_type").
    
    Returns:
    - source_adata: Reference AnnData object.
    - enhanced_scpoli_model: Loaded scPoli model.
    - cell_type_key: Key for cell type annotations (e.g., "lineage" or "reanno").
    """
    
    # Check if scPoli is available
    if not SCPOLI_AVAILABLE:
        raise ImportError("scPoli package is not available. Please install scarches with: pip install scarches")
    
    try:
        # Check if GPU is available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # If adata_path is not provided or doesn't exist, look for it in model_dir
        if adata_path is None or not os.path.exists(adata_path):
            # Look for adata.h5ad in the model directory
            adata_path = os.path.join(model_dir, "adata.h5ad")
            print(f"Looking for reference data in model directory: {adata_path}")
        
        # Load the reference dataset
        if not os.path.exists(adata_path):
            raise FileNotFoundError(f"Reference AnnData file not found: {adata_path}")
        source_adata = sc.read_h5ad(adata_path)
        print(f"Reference dataset loaded successfully from: {adata_path}")
        
        # Ensure observation names are unique
        if not source_adata.obs_names.is_unique:
            print("Observation names are not unique. Making them unique...")
            source_adata.obs_names_make_unique()
        
        # Construct the model file path
        model_file_name = "model_params.pt"  # Correct model file name
        model_path = os.path.join(model_dir, model_file_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load the scPoli model
        map_location = torch.device(device)  # Use GPU if available, otherwise CPU
        enhanced_scpoli_model = scPoli.load(model_dir, adata=source_adata, map_location=map_location)
        
        # Set the appropriate cell type key based on model type
        if model_type == "lineage":
            cell_type_key = "lineage"
        elif model_type == "cell_type":
            cell_type_key = "reanno"
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        print(f"{model_type.capitalize()} model loaded successfully from {model_dir}.")
        return source_adata, enhanced_scpoli_model, cell_type_key
    
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None


def remove_sparsity(adata):
    """Remove sparsity from AnnData object with memory optimization."""
    if isinstance(adata.X, np.ndarray):
        return adata
    
    # Convert to dense array with float32 to save memory
    print("Converting sparse matrix to dense with float32 optimization...")
    adata.X = adata.X.toarray().astype(np.float32)
    return adata


def align_genes(query_adata, source_adata):
    """
    Align genes between query and reference datasets with improved tracking.
    
    Parameters:
    - query_adata: Query AnnData object.
    - source_adata: Reference AnnData object.
    
    Returns:
    - Aligned query AnnData object with tracking identifiers.
    """
    # Store original query information for tracking
    original_query_cell_ids = query_adata.obs_names.copy()
    original_cell_count = query_adata.n_obs
    print(f"Original query cell count: {original_cell_count}")
    
    # Add unique identifiers to query_adata for precise tracking
    query_adata.obs['is_original_query'] = True
    query_adata.obs['original_query_id'] = range(len(query_adata))
    
    # Reorganize query dataset to match genes in the reference dataset
    all_genes = source_adata.var_names
    missing_genes = all_genes.difference(query_adata.var_names)
    missing_data = np.zeros((query_adata.shape[0], len(missing_genes)), dtype=np.float32)  # Use float32
    
    query_adata_df = pd.DataFrame(query_adata.X, columns=query_adata.var_names, index=query_adata.obs_names)
    missing_df = pd.DataFrame(missing_data, columns=missing_genes, index=query_adata.obs_names)
    query_adata_combined_df = pd.concat([query_adata_df, missing_df], axis=1)[all_genes]
    
    # Create extended AnnData with float32 optimization
    query_adata_extended = sc.AnnData(
        X=query_adata_combined_df.values.astype(np.float32),  # Ensure float32
        obs=query_adata.obs.copy(),  # This preserves is_original_query and original_query_id
        var=pd.DataFrame(index=all_genes),
        layers={'counts': query_adata_combined_df.values.astype(np.float32)}  # Ensure float32
    )
    
    
    # Check if 'features' column exists in the original query_adata.var before copying
    if 'features' in query_adata.var.columns:
        # If features column exists, reindex and copy it
        query_adata_extended.var['features'] = query_adata.var.reindex(all_genes)['features']
        print("Copied existing 'features' column from original query data")
    else:
        # If features column doesn't exist, check if source_adata has it and copy from there
        if 'features' in source_adata.var.columns:
            query_adata_extended.var['features'] = source_adata.var['features'].copy()
            print("Copied 'features' column from source_adata")
        else:
            # If neither has features column, create a default one using gene names
            query_adata_extended.var['features'] = query_adata_extended.var_names
            print("Created 'features' column using gene names as default")
    
    print(f"Extended query data shape: {query_adata_extended.shape}")
    print(f"is_original_query column exists: {'is_original_query' in query_adata_extended.obs.columns}")
    
    return query_adata_extended


def label_transfer(query_file, figures_folder, model_type="lineage", 
                  custom_model_dir=None, custom_adata_path=None,
                  progress_callback=None):
    """
    Perform label transfer on a single preprocessed .h5ad file using the specified model.
    
    Parameters:
    - query_file: Path to the preprocessed query dataset (.h5ad file).
    - figures_folder: Path to save output figures and files.
    - model_type: Type of model to use ("lineage" or "cell_type").
    - custom_model_dir: Optional custom path to the model directory.
    - custom_adata_path: Optional custom path to the reference dataset.
    - progress_callback: Optional callback for tracking training progress.
    
    Returns:
    - A summary of results (classification report, saved file paths, etc.).
    """
    
    # Check if torch and scPoli are available
    if not TORCH_AVAILABLE:
        error_msg = "⚠️ Label Transfer module unavailable: PyTorch compatibility issue. Please check torch installation."
        print(f"Error: {error_msg}")
        return {
            "status": "error", 
            "message": error_msg,
            "query_file": query_file
        }
    
    if not SCPOLI_AVAILABLE:
        error_msg = "⚠️ Advanced label transfer (scPoli) unavailable due to package compatibility issues. This is a known issue with NumPy/JAX versions. Basic scanpy-based cell annotation is still available through other analysis tools."
        print(f"Error: {error_msg}")
        return {
            "status": "error",
            "message": error_msg,
            "query_file": query_file,
            "note": "Try using 'Preprocess Data' and 'Data Visualization' for basic cell type analysis, or consider using an older version of the analysis environment."
        }
    
    # Define model directories and paths
    if custom_model_dir and custom_adata_path:
        model_dir = custom_model_dir
        adata_path = custom_adata_path
        print(f"Using custom model directory: {model_dir}")
        print(f"Using custom reference dataset: {adata_path}")
    else:
        # Use default paths based on model_type (both model and data in same directory)
        if model_type == "lineage":
            model_dir = './knowledge_base/models/enhanced_reference_model_lineage_2ndround/'
            adata_path = None  # Will be auto-detected in model_dir
        elif model_type == "cell_type":
            model_dir = './knowledge_base/models/enhanced_reference_model_reanno_2ndround/'
            adata_path = None  # Will be auto-detected in model_dir
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        print(f"Using default model directory for {model_type}: {model_dir}")
        print(f"Reference dataset will be auto-detected in model directory")
    
    # Initialize progress status
    if progress_callback:
        progress_callback.status_text.text("Loading model and reference dataset...")
        progress_callback.progress_bar.progress(0.05)  # 5% progress
    
    # Load the model
    try:
        source_adata, enhanced_scpoli_model, cell_type_key = load_model(model_dir, adata_path, model_type)
        if source_adata is None or enhanced_scpoli_model is None:
            raise ValueError("Failed to load the reference model or dataset.")
    except ImportError as e:
        # Handle scPoli not available error specifically
        error_msg = f"scPoli dependency error: {str(e)}"
        print(f"Error: {error_msg}")
        return {
            "status": "error",
            "message": error_msg,
            "query_file": query_file
        }
    
    # Update progress
    if progress_callback:
        progress_callback.status_text.text("Model loaded successfully. Loading query dataset...")
        progress_callback.progress_bar.progress(0.1)  # 10% progress
    
    # Create the figures directory if it doesn't exist
    os.makedirs(figures_folder, exist_ok=True)
    print(f"Outputs will be saved to: {figures_folder}")
    
    # Load the query dataset
    try:
        query_adata = sc.read_h5ad(query_file)
        file_name = os.path.basename(query_file)  # Extract just the file name
        print(f"Processing {file_name}...")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {query_file}")
    
    # Update progress
    if progress_callback:
        progress_callback.status_text.text("Query dataset loaded. Checking preprocessing...")
        progress_callback.progress_bar.progress(0.15)  # 15% progress
    
    # Check if the dataset is preprocessed
    if not check_preprocessing(query_adata):
        raise ValueError("Dataset is not preprocessed. Please preprocess the dataset first.")
    
    # Update progress
    if progress_callback:
        progress_callback.status_text.text("Preprocessing verified. Aligning genes...")
        progress_callback.progress_bar.progress(0.2)  # 20% progress
    
    # Store original query information for validation
    original_query_cell_ids = query_adata.obs_names.copy()
    original_cell_count = query_adata.n_obs
    
    # Handle missing 'orig.ident' column
    if 'orig.ident' not in query_adata.obs.columns:
        # Create orig.ident from file basename (without extension)
        base_name = os.path.splitext(file_name)[0]
        # Remove common prefixes to get a cleaner name
        for prefix in ['corrected_processed_', 'processed_', 'preprocessed_']:
            if base_name.startswith(prefix):
                base_name = base_name[len(prefix):]
                break
        query_adata.obs['orig.ident'] = base_name
        print(f"Created 'orig.ident' column from filename: {base_name}")
    
    # Get query name from orig.ident (should be unique)
    unique_values = set(query_adata.obs["orig.ident"])
    if len(unique_values) == 1:
        query_name = unique_values.pop()
    else:
        query_name = 'query'  # fallback for multiple values
    print(f"Query name: {query_name}")
    
    # Check overlap in source_adata using the dynamic query_name
    if 'orig.ident' in source_adata.obs.columns:
        source_overlap_mask = source_adata.obs['orig.ident'] == query_name
        source_overlap_count = source_overlap_mask.sum()
        source_overlap_cells = source_adata.obs_names[source_overlap_mask]
        print(f"{query_name} cells in source data: {source_overlap_count}")
        if source_overlap_count > 0:
            print(f"{query_name} cell ID examples in source data: {source_overlap_cells[:5].tolist()}")
        
        # Check for identical cell IDs
        overlapping_ids = set(original_query_cell_ids) & set(source_overlap_cells)
        print(f"Number of identical cell IDs: {len(overlapping_ids)}")
    else:
        print("No orig.ident column in source data for overlap checking")
    
    # Remove sparsity and optimize data types for memory efficiency
    query_adata = remove_sparsity(query_adata)
    
    # Convert to float32 to reduce memory usage by half
    print("Converting data types to float32 for memory optimization...")
    if hasattr(query_adata.X, 'dtype') and query_adata.X.dtype == np.float64:
        query_adata.X = query_adata.X.astype(np.float32)
        print("✅ Converted X matrix from float64 to float32")
    
    # Convert layers to float32 as well
    for layer_name in query_adata.layers:
        if hasattr(query_adata.layers[layer_name], 'dtype') and query_adata.layers[layer_name].dtype == np.float64:
            query_adata.layers[layer_name] = query_adata.layers[layer_name].astype(np.float32)
            print(f"✅ Converted {layer_name} layer from float64 to float32")
    
    # Align genes (this function now adds tracking identifiers)
    query_adata_extended = align_genes(query_adata, source_adata)
    if cell_type_key in source_adata.obs.columns:
        source_cell_types = source_adata.obs[cell_type_key].unique()
        if len(source_cell_types) > 0:
            # 选择第一个细胞类型作为占位符
            placeholder_cell_type = source_cell_types[0]
            query_adata_extended.obs[cell_type_key] = placeholder_cell_type
            print(f"设置查询数据细胞类型为占位符: {placeholder_cell_type}")
        else:
            # 备用方案：创建一个简单的占位符
            query_adata_extended.obs[cell_type_key] = "Query_Cell"
            print("设置查询数据细胞类型为: Query_Cell")
    else:
        # 如果源数据没有细胞类型列，创建一个
        query_adata_extended.obs[cell_type_key] = "Query_Cell"
        print("创建查询数据细胞类型列: Query_Cell")

    # 标记这些是查询数据
    query_adata_extended.obs['data_type'] = 'query'

    if 'orig.ident' not in query_adata_extended.obs.columns:
        query_adata_extended.obs['orig.ident'] = query_name
    
    # Update progress
    if progress_callback:
        progress_callback.status_text.text("Genes aligned. Initializing scPoli model...")
        progress_callback.progress_bar.progress(0.25)  # 25% progress
    
    # Label transfer to query dataset
    print("Initializing scPoli query model...")
    scpoli_query = scPoli.load_query_data(
        adata=query_adata_extended,
        reference_model=enhanced_scpoli_model,
        labeled_indices=[],
    )
    
    # Update progress before training
    if progress_callback:
        progress_callback.status_text.text("Starting model training (this takes several minutes)...")
        progress_callback.progress_bar.progress(0.3)  # 30% progress
    
    # Aggressive memory cleanup before intensive operations
    print("Performing garbage collection before training...")
    gc.collect()
    
    # Override standard output to capture progress
    import sys
    from io import StringIO
    import re
    
    # Create a custom output interceptor class
    class ProgressCaptureOutput(StringIO):
        def __init__(self, progress_callback):
            super().__init__()
            self.progress_callback = progress_callback
            self.last_percentage = 0
        
        def write(self, text):
            super().write(text)
            # Check if this is a progress update line
            if '|' in text and '%' in text:
                try:
                    # Extract percentage using regex
                    match = re.search(r'\|\s*(\d+\.\d+)%', text)
                    if match:
                        percentage = float(match.group(1)) / 100.0
                        self.last_percentage = percentage
                        
                        # Extract loss values if available
                        loss_match = re.search(r'val_loss:\s*(\d+\.\d+)', text)
                        loss_value = loss_match.group(1) if loss_match else "N/A"
                        
                        # Update progress bar and status
                        self.progress_callback.progress_bar.progress(0.3 + 0.4 * percentage)  # Scale to 30-70% range
                        self.progress_callback.status_text.text(
                            f"Training progress: {percentage*100:.1f}% - Loss: {loss_value}\n{text}"
                        )
                except Exception as e:
                    # If parsing fails, just display the text
                    self.progress_callback.status_text.text(f"Training in progress...\n{text}")
            return
    
    # If we have a progress callback, capture stdout to monitor progress
    if progress_callback:
        # Create a capture output object
        capture_out = ProgressCaptureOutput(progress_callback)
        
        # Save original stdout
        original_stdout = sys.stdout
        
        try:
            # Redirect stdout to our capture object
            sys.stdout = capture_out
            
            # Train the model (output will be captured)
            scpoli_query.train(
                n_epochs=50,
                pretraining_epochs=40,
                eta=5
            )
        finally:
            # Restore original stdout
            sys.stdout = original_stdout
            
            # Get the final percentage
            final_percentage = capture_out.last_percentage
            
            # Update progress status
            progress_callback.status_text.text(f"Training completed at {final_percentage*100:.1f}%. Performing label transfer...")
            progress_callback.progress_bar.progress(0.7)  # 70% progress
    else:
        # No progress callback, just train normally
        scpoli_query.train(
            n_epochs=50,
            pretraining_epochs=40,
            eta=5
        )
    
    # Ensure data is float32 for classification (memory optimization)
    print("Ensuring float32 data type before classification...")
    query_adata_extended.X = query_adata_extended.X.astype(np.float32)
    
    # Also optimize any obsm arrays that might consume memory
    for key in query_adata_extended.obsm.keys():
        if hasattr(query_adata_extended.obsm[key], 'dtype') and query_adata_extended.obsm[key].dtype == np.float64:
            query_adata_extended.obsm[key] = query_adata_extended.obsm[key].astype(np.float32)
            print(f"✅ Converted obsm['{key}'] from float64 to float32")
    
    # Update progress
    if progress_callback:
        progress_callback.status_text.text("Performing classification...")
        progress_callback.progress_bar.progress(0.75)  # 75% progress
    
    # Memory cleanup before classification (the memory-intensive step)
    print("Performing garbage collection before classification...")
    gc.collect()
    
    # Perform label transfer
    results_dict = scpoli_query.classify(query_adata_extended, scale_uncertainties=True)
    preds = results_dict[cell_type_key]["preds"]
    uncert = results_dict[cell_type_key]["uncert"]
    
    # Update progress
    if progress_callback:
        progress_callback.status_text.text("Generating visualizations...")
        progress_callback.progress_bar.progress(0.8)  # 80% progress
    
    # Evaluate classification performance
    classification_df = pd.DataFrame(
        classification_report(
            y_true=query_adata_extended.obs[cell_type_key],
            y_pred=preds,
            output_dict=True,
        )
    ).transpose()
    print(classification_df)
    
    # Get latent representations with enhanced tracking
    scpoli_query.model.eval()
    
    print("Applying workaround for PyTorch/SciPy compatibility issue...")
    def ensure_dense_for_latent(adata):
        """确保数据是密集数组以兼容 get_latent"""
        if hasattr(adata.X, 'toarray'):
            print("Converting sparse matrix to dense for latent extraction...")
            adata.X = adata.X.toarray().astype(np.float32)
        return adata

    # 对源数据和查询数据都应用修复
    source_adata_fixed = ensure_dense_for_latent(source_adata.copy())
    query_adata_extended_fixed = ensure_dense_for_latent(query_adata_extended.copy())

    # 使用修复后的数据进行 latent 提取
    print("Getting latent representations with compatibility fix...")
    data_latent_source = scpoli_query.get_latent(source_adata_fixed, mean=True)
    adata_latent_source = sc.AnnData(data_latent_source.astype(np.float32))
    adata_latent_source.obs = source_adata.obs.copy()

    # 同样修复查询数据的 latent 提取
    data_latent = scpoli_query.get_latent(query_adata_extended_fixed, mean=True)
    adata_latent = sc.AnnData(data_latent.astype(np.float32))
    adata_latent.obs = query_adata_extended.obs.copy()
    
    adata_latent.obs[f'{cell_type_key}_pred'] = preds.tolist()
    adata_latent.obs[f'{cell_type_key}_uncert'] = uncert.tolist()
    adata_latent.obs['classifier_outcome'] = (
        adata_latent.obs[f'{cell_type_key}_pred'] == adata_latent.obs[cell_type_key]
    )
    
    # Get prototypes
    labeled_prototypes = scpoli_query.get_prototypes_info()
    labeled_prototypes.obs['study'] = 'labeled prototype'
    labeled_prototypes.obs['is_original_query'] = False
    labeled_prototypes.obs['original_query_id'] = -2  # Use -2 to mark labeled prototypes
    
    unlabeled_prototypes = scpoli_query.get_prototypes_info(prototype_set='unlabeled')
    unlabeled_prototypes.obs['study'] = 'unlabeled prototype'
    unlabeled_prototypes.obs['is_original_query'] = False
    unlabeled_prototypes.obs['original_query_id'] = -3  # Use -3 to mark unlabeled prototypes
    
    # Update progress
    if progress_callback:
        progress_callback.status_text.text("Generating UMAP embeddings...")
        progress_callback.progress_bar.progress(0.85)  # 85% progress
    
    # Combine AnnDatas with prototypes
    adata_latent_full = adata_latent_source.concatenate(
        [adata_latent, labeled_prototypes, unlabeled_prototypes],
        batch_key='query'
    )
    
    print(f"\n=== Post-concatenation data analysis ===")
    print(f"Total data size: {adata_latent_full.n_obs}")
    print(f"Query identifier distribution: {adata_latent_full.obs['query'].value_counts()}")
    print(f"is_original_query distribution: {adata_latent_full.obs['is_original_query'].value_counts()}")
    
    # Set predictions to NaN for reference data (query='0')
    adata_latent_full.obs[f'{cell_type_key}_pred'][adata_latent_full.obs['query'].isin(['0'])] = np.nan
    
    # Compute UMAP and neighbors
    sc.pp.neighbors(adata_latent_full, n_neighbors=15)
    sc.tl.umap(adata_latent_full)
    
    # Get AnnData without prototypes for cleaner visualization
    adata_no_prototypes = adata_latent_full[adata_latent_full.obs['query'].isin(['0', '1'])]
    
    # CRITICAL FIX: Compute UMAP directly on query data (matching working script approach)
    print("Computing UMAP directly on query latent data...")
    sc.pp.neighbors(adata_latent, n_neighbors=15)
    sc.tl.leiden(adata_latent, resolution=0.5)  # Add leiden clustering like working script
    sc.tl.umap(adata_latent)
    print(f"✅ Computed UMAP directly on query data: {adata_latent.obsm['X_umap'].shape}")
    
    # 新增：存储scPoli生成的UMAP到查询数据中
    print("\n=== Storing scPoli-generated UMAP coordinates ===")
    
    # 1. 首先从adata_latent中获取UMAP坐标
    query_latent_umap = adata_latent.obsm['X_umap']
    print(f"UMAP coordinates shape from scPoli: {query_latent_umap.shape}")
    
    # 2. 确保我们有正确的细胞ID映射
    # 将UMAP坐标与原始查询数据对齐
    query_cell_ids = query_adata_extended.obs_names
    print(f"Query cell IDs count: {len(query_cell_ids)}")
    
    # 3. 将UMAP坐标添加到原始查询数据中
    # 确保使用正确的键名，避免覆盖现有的UMAP
    umap_key = f'X_umap_scpoli_{model_type}'
    
    # 检查是否需要创建对齐的数组
    if len(query_adata.obs_names) == query_latent_umap.shape[0]:
        # 细胞数量匹配，直接存储
        query_adata.obsm[umap_key] = query_latent_umap
        print(f"✅ Directly stored scPoli UMAP to query_adata.obsm['{umap_key}']")
    else:
        # 细胞数量不匹配，需要创建对齐的数组
        print(f"⚠️ Cell count mismatch: query_adata has {len(query_adata.obs_names)} cells, UMAP has {query_latent_umap.shape[0]} cells")
        print("Attempting to align UMAP coordinates by cell IDs...")
        
        # 创建全为NaN的数组，然后填充匹配的细胞
        aligned_umap = np.full((len(query_adata.obs_names), 2), np.nan)
        
        # 找到匹配的细胞索引
        matching_indices = []
        for i, cell_id in enumerate(query_adata.obs_names):
            if cell_id in query_adata_extended.obs_names:
                # 找到在扩展数据中的位置
                j = list(query_adata_extended.obs_names).index(cell_id)
                if j < query_latent_umap.shape[0]:
                    aligned_umap[i] = query_latent_umap[j]
                    matching_indices.append((i, j))
        
        query_adata.obsm[umap_key] = aligned_umap
        print(f"✅ Aligned and stored scPoli UMAP to query_adata.obsm['{umap_key}']")
        print(f"Matched {len(matching_indices)} cells out of {len(query_adata.obs_names)}")
    
    # 验证存储的UMAP
    if umap_key in query_adata.obsm:
        stored_umap = query_adata.obsm[umap_key]
        print(f"✅ Successfully stored UMAP coordinates: shape={stored_umap.shape}")
        print(f"UMAP range: X: [{stored_umap[:, 0].min():.3f}, {stored_umap[:, 0].max():.3f}], "
              f"Y: [{stored_umap[:, 1].min():.3f}, {stored_umap[:, 1].max():.3f}]")
        
        # 检查NaN值
        nan_count = np.isnan(stored_umap).any(axis=1).sum()
        if nan_count > 0:
            print(f"⚠️ UMAP contains {nan_count} cells with NaN values")
    else:
        print("❌ Failed to store UMAP coordinates")
    
    # Update progress
    if progress_callback:
        progress_callback.status_text.text("Saving UMAP plots...")
        progress_callback.progress_bar.progress(0.9)  # 90% progress
    
    # Plot and save UMAP plots with ordered categorical data
    print("Generating ordered UMAP plots...")
    base_filename = os.path.splitext(file_name)[0]

    # Generate plots for both full data and query-only data
    for data_obj, data_name in [(adata_no_prototypes, "full"), (adata_latent, "query")]:
        
        # Set up ordered categorical data for predictions
        if f'{cell_type_key}_pred' in data_obj.obs.columns:
            print(f"  Creating {model_type} prediction UMAP plot for {data_name} data...")
            
            # Convert to categorical and set up ordering
            data_obj.obs[f'{cell_type_key}_pred'] = data_obj.obs[f'{cell_type_key}_pred'].astype('category')
            present_categories = data_obj.obs[f'{cell_type_key}_pred'].cat.categories.tolist()
            
            if model_type == "lineage":
                # 删除自定义颜色部分，直接使用scanpy默认颜色
                # 设置类别顺序（如果需要）
                data_obj.obs[f'{cell_type_key}_pred'] = data_obj.obs[f'{cell_type_key}_pred'].cat.set_categories(
                    sorted(present_categories)  # 或者保持原有顺序，但不设置自定义颜色
                )
                
                # 绘图时不指定palette参数，让scanpy使用默认颜色
                sc.pl.umap(
                    data_obj,
                    color=f'{cell_type_key}_pred',
                    show=False,  # 移除palette参数
                    frameon=False
                )
                plot_file = os.path.join(figures_folder, f"{base_filename}_{data_name}_lineage_pred.pdf")  # 可以去掉"ordered"后缀
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"    Saved lineage plot: {plot_file}")
                
            elif model_type == "cell_type":
                # 同样移除自定义颜色设置
                data_obj.obs[f'{cell_type_key}_pred'] = data_obj.obs[f'{cell_type_key}_pred'].cat.set_categories(
                    sorted(present_categories)  # 使用字母顺序或保持原样
                )

                # 绘图时不指定颜色
                sc.pl.umap(
                    data_obj,
                    color=f'{cell_type_key}_pred',
                    show=False,
                    frameon=False
                )
                plot_file = os.path.join(figures_folder, f"{base_filename}_{data_name}_reanno_pred.pdf")
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"    Saved reanno plot: {plot_file}")
    
    # Generate additional standard plots
    umap_dataset_file = os.path.join(figures_folder, f"{base_filename}_scPoli_{model_type}_dataset.pdf")
    umap_uncert_file = os.path.join(figures_folder, f"{base_filename}_scPoli_{model_type}_uncert.pdf")
    
    # Plot datasets
    sc.pl.umap(
        adata_no_prototypes,
        color='orig.ident',
        show=False,
        frameon=False
    )
    plt.savefig(umap_dataset_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot uncertainty
    sc.pl.umap(
        adata_no_prototypes,
        color=f'{cell_type_key}_uncert',
        show=False,
        frameon=False,
        cmap='magma',
        vmax=1
    )
    plt.savefig(umap_uncert_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 新增：使用scPoli UMAP生成额外的可视化
    if umap_key in query_adata.obsm:
        print("\n=== Generating additional plots with scPoli UMAP ===")
        
        # 创建临时AnnData用于绘制scPoli UMAP
        adata_scpoli_umap = query_adata.copy()
        
        # 添加scPoli UMAP坐标
        adata_scpoli_umap.obsm['X_umap'] = query_adata.obsm[umap_key]
        
        # 绘制scPoli UMAP的细胞类型预测
        scpoli_umap_file = os.path.join(figures_folder, f"{base_filename}_scPoli_{model_type}_umap_pred.pdf")
        
        if f'{cell_type_key}_pred' in adata_scpoli_umap.obs.columns:
            # 设置类别顺序
            adata_scpoli_umap.obs[f'{cell_type_key}_pred'] = adata_scpoli_umap.obs[f'{cell_type_key}_pred'].astype('category')
            sc.pl.umap(
                adata_scpoli_umap,
                color=f'{cell_type_key}_pred',
                show=False,
                frameon=False
            )
            plt.savefig(scpoli_umap_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved scPoli UMAP plot: {scpoli_umap_file}")
    
    # Update progress
    if progress_callback:
        progress_callback.status_text.text("Saving results...")
        progress_callback.progress_bar.progress(0.95)  # 95% progress
    
    # Enhanced result filtering - get only true original query cells
    print(f"\n=== Precise filtering of original query cells ===")
    
    # === CRITICAL SECTION: Properly extract and save results (Fixed based on working script) ===
    print(f"\n=== Extracting results for original query cells ===")
    
    def safe_boolean_indexing(adata, mask_series, description=""):
        """安全地使用布尔索引，处理 NA 值和其他边界情况"""
        print(f"{description}:")
        print(f"  Original mask shape: {mask_series.shape}")
        print(f"  Original mask dtype: {mask_series.dtype}")
        print(f"  NA values in mask: {mask_series.isna().sum()}")
        print(f"  True values in mask: {mask_series.sum() if mask_series.dtype != 'object' else 'N/A'}")
        
        # 处理 NA 值
        if mask_series.isna().any():
            print("  ⚠️ Mask contains NA values, converting to False")
            mask_clean = mask_series.fillna(False)
        else:
            mask_clean = mask_series
            
        # 确保是布尔类型
        if mask_clean.dtype != 'bool':
            print(f"  ⚠️ Mask is not boolean type ({mask_clean.dtype}), converting to bool")
            mask_clean = mask_clean.astype(bool)
        
        print(f"  Clean mask - True values: {mask_clean.sum()}")
        
        # 安全索引
        try:
            result = adata[mask_clean]
            print(f"  ✅ Successfully indexed {result.n_obs} cells")
            return result
        except Exception as e:
            print(f"  ❌ Indexing failed: {e}")
            # 备用方法：使用位置索引
            print("  Trying alternative indexing method...")
            valid_indices = mask_clean[mask_clean].index
            result = adata[valid_indices]
            print(f"  ✅ Alternative method indexed {result.n_obs} cells")
            return result

    # 使用安全索引方法
    original_query_mask = adata_latent_full.obs['is_original_query'] == True
    full_latent_correct = safe_boolean_indexing(
        adata_latent_full, 
        original_query_mask,
        "Extracting original query cells using is_original_query"
    )

    # 验证我们得到了正确数量的细胞
    if full_latent_correct.n_obs == original_cell_count:
        print(f"✅ Success! Got correct cell count: {full_latent_correct.n_obs}")
    else:
        print(f"❌ Cell count mismatch. Expected: {original_cell_count}, Got: {full_latent_correct.n_obs}")
        print("This might be expected if some cells were filtered during processing")

    # Extract key results and save back to original query data
    print("Transferring results back to original query data...")
    
    # Transfer predictions to original adata using the method from working script (lines 341-348)
    # Use adata_latent (before concatenation) instead of full_latent_correct (after concatenation with modified IDs)
    print("Transferring annotation predictions to original adata")
    matching_indices = adata_latent.obs.index.intersection(query_adata.obs.index)
    
    # Initialize columns with NaN
    if f'{cell_type_key}_pred' not in query_adata.obs.columns:
        query_adata.obs[f'{cell_type_key}_pred'] = np.nan
    if f'{cell_type_key}_uncert' not in query_adata.obs.columns:
        query_adata.obs[f'{cell_type_key}_uncert'] = np.nan
    
    # Transfer results using the matching indices (this should work since adata_latent has original cell IDs)    
    
    matching_indices = adata_latent.obs.index.intersection(query_adata.obs.index)
    pred_values = adata_latent.obs.loc[matching_indices, f'{cell_type_key}_pred'].astype(str).values
    uncert_values = adata_latent.obs.loc[matching_indices, f'{cell_type_key}_uncert'].values

    # 直接创建新列，避免分类数据类型冲突
    query_adata.obs[f'{cell_type_key}_pred'] = ""  # 先创建空字符串列
    query_adata.obs[f'{cell_type_key}_uncert'] = np.nan  # 创建NaN列

    # 填充数据
    query_adata.obs.loc[matching_indices, f'{cell_type_key}_pred'] = pred_values
    query_adata.obs.loc[matching_indices, f'{cell_type_key}_uncert'] = uncert_values

    print(f"Found {len(matching_indices)} matching cell IDs for result transfer")
    print(f"Unique predictions transferred: {query_adata.obs[f'{cell_type_key}_pred'].unique()}")
    
    # Set up categorical ordering in the original query data for consistency
    if f'{cell_type_key}_pred' in query_adata.obs.columns:
        print(f"Setting up categorical ordering for {cell_type_key}_pred in original data...")
        query_adata.obs[f'{cell_type_key}_pred'] = query_adata.obs[f'{cell_type_key}_pred'].astype('category')
        present_categories = query_adata.obs[f'{cell_type_key}_pred'].cat.categories.tolist()
        
        if model_type == "lineage":
            # 使用字母顺序排序
            ordered_present = sorted(present_categories)
            query_adata.obs[f'{cell_type_key}_pred'] = query_adata.obs[f'{cell_type_key}_pred'].cat.set_categories(ordered_present)
            print(f"  Applied alphabetical ordering for lineage: {len(ordered_present)} categories")
            
        elif model_type == "cell_type":
            # 使用字母顺序排序
            ordered_present = sorted(present_categories)
            query_adata.obs[f'{cell_type_key}_pred'] = query_adata.obs[f'{cell_type_key}_pred'].cat.set_categories(ordered_present)
            print(f"  Applied alphabetical ordering for reanno: {len(ordered_present)} categories")
    
    print(f"Transferred {len([col for col in query_adata.obs.columns if col.endswith('_pred') or col.endswith('_uncert')])} result columns")
    
    # 确保UMAP坐标已经正确存储
    print("Verifying UMAP coordinate storage...")
    if umap_key in query_adata.obsm:
        print(f"✅ UMAP coordinates already stored in obsm['{umap_key}']")
    else:
        print("⚠️ UMAP coordinates not found, attempting to store them now...")
        # 再次尝试存储UMAP
        if 'X_umap' in adata_latent.obsm:
            query_adata.obsm[umap_key] = adata_latent.obsm['X_umap']
            print(f"✅ Stored UMAP coordinates from adata_latent to obsm['{umap_key}']")
    
    # Verify UMAP was properly added
    if umap_key in query_adata.obsm:
        print(f"✅ Successfully stored scPoli UMAP coordinates for {query_adata.obsm[umap_key].shape[0]} cells")
        # 添加UMAP键到结果中以便跟踪
        umap_keys_in_data = [key for key in query_adata.obsm.keys() if 'umap' in key.lower()]
        print(f"All UMAP keys in data: {umap_keys_in_data}")
    else:
        print("⚠️ UMAP coordinate storage failed")
    
    # Clean data types before saving
    print("Cleaning data for saving...")
    for col in query_adata.obs.columns:
        if col.endswith('_pred'):
            query_adata.obs[col] = query_adata.obs[col].astype('category')
        elif col.endswith('_uncert'):
            query_adata.obs[col] = pd.to_numeric(query_adata.obs[col], errors='coerce')
    
    # 新增：创建输出文件名
    # 生成新的输出文件名
    output_filename = f"{base_filename}_scpoli_{model_type}.h5ad"
    output_path = os.path.join(figures_folder, output_filename)
    
    print(f"\n=== Saving results to new file ===")
    print(f"Output file: {output_path}")
    
    # Clean data before saving to prevent string conversion errors
    print("Cleaning data for HDF5 compatibility...")
    
    # Fix potential issues with raw.var column names
    if query_adata.raw is not None:
        if '_index' in query_adata.raw.var.columns:
            query_adata.raw.var.rename(columns={'_index': 'index'}, inplace=True)
    else:
        print("query_adata.raw is None. Using query_adata.var instead.")
    
    # Clean obs columns to ensure HDF5 compatibility
    try:
        for col in query_adata.obs.columns:
            # Convert any mixed types to strings where needed
            if query_adata.obs[col].dtype == 'object':
                # Check if column contains mixed types
                unique_types = set(type(x) for x in query_adata.obs[col].dropna())
                if len(unique_types) > 1:
                    print(f"Converting mixed-type column '{col}' to string")
                    query_adata.obs[col] = query_adata.obs[col].astype(str)
            
            # Handle boolean columns that might cause issues
            elif query_adata.obs[col].dtype == 'bool':
                # Convert boolean to categorical for better HDF5 compatibility
                query_adata.obs[col] = pd.Categorical(query_adata.obs[col])
            
            # Ensure numeric columns are properly typed
            elif col.endswith('_pred') or col.endswith('_uncert'):
                if col.endswith('_pred'):
                    # Prediction columns should be categorical
                    query_adata.obs[col] = pd.Categorical(query_adata.obs[col])
                elif col.endswith('_uncert'):
                    # Uncertainty columns should be float
                    query_adata.obs[col] = pd.to_numeric(query_adata.obs[col], errors='coerce')
        
        # Clean var columns
        for col in query_adata.var.columns:
            if query_adata.var[col].dtype == 'object':
                unique_types = set(type(x) for x in query_adata.var[col].dropna())
                if len(unique_types) > 1:
                    print(f"Converting mixed-type var column '{col}' to string")
                    query_adata.var[col] = query_adata.var[col].astype(str)
        
        # Ensure obsm arrays are properly typed
        for key in query_adata.obsm.keys():
            if not isinstance(query_adata.obsm[key], np.ndarray):
                query_adata.obsm[key] = np.array(query_adata.obsm[key], dtype=np.float32)
            elif query_adata.obsm[key].dtype not in [np.float32, np.float64, np.int32, np.int64]:
                query_adata.obsm[key] = query_adata.obsm[key].astype(np.float32)
        
        print("✅ Data cleaning completed")
        
        print("\n" + "="*60)
        print("CELL TYPE PREDICTION STATISTICS")
        print("="*60)

        # 按样本统计
        if 'orig.ident' in query_adata.obs.columns:
            sample_stats = pd.crosstab(
                query_adata.obs['orig.ident'],
                query_adata.obs[f'{cell_type_key}_pred'],
                normalize='index'
            ) * 100

            print("\nCell Type Distribution by Sample (%):")
            print(sample_stats.round(2))

            # 保存详细统计到CSV
            stats_filename = os.path.join(figures_folder, f"{base_filename}_{model_type}_celltype_stats.csv")
            sample_stats.round(2).to_csv(stats_filename)
            print(f"Detailed statistics saved to: {stats_filename}")

            # 显示每个样本的前5个主要细胞类型
            print("\nTop cell types per sample:")
            for sample in sample_stats.index:
                top_types = sample_stats.loc[sample].nlargest(5)
                print(f"  {sample}: {dict(top_types)}")
        else:
            # 整体统计
            overall_stats = query_adata.obs[f'{cell_type_key}_pred'].value_counts(normalize=True) * 100
            print("\nOverall Cell Type Distribution (%):")
            print(overall_stats.round(2))

            # 保存整体统计
            stats_filename = os.path.join(figures_folder, f"{base_filename}_{model_type}_overall_stats.csv")
            overall_stats.round(2).to_csv(stats_filename)
            print(f"Overall statistics saved to: {stats_filename}")

        # 显示预测质量信息
        if f'{cell_type_key}_uncert' in query_adata.obs.columns:
            uncertainty_stats = query_adata.obs[f'{cell_type_key}_uncert'].describe()
            print(f"\nUncertainty Statistics:")
            print(f"  Mean: {uncertainty_stats['mean']:.3f}")
            print(f"  Std:  {uncertainty_stats['std']:.3f}")
            print(f"  Min:  {uncertainty_stats['min']:.3f}")
            print(f"  Max:  {uncertainty_stats['max']:.3f}")

            # 高置信度预测的比例（不确定度 < 0.5）
            high_confidence = (query_adata.obs[f'{cell_type_key}_uncert'] < 0.5).sum() / len(query_adata) * 100
            print(f"  High confidence predictions (<0.5): {high_confidence:.1f}%")

        print("="*60)

        
    except Exception as clean_error:
        print(f"Warning during data cleaning: {str(clean_error)}")
    
    # Export metadata with annotations to CSV for download (do this first to ensure it's available)
    base_filename = os.path.splitext(os.path.basename(query_file))[0]
    metadata_filename = f"{base_filename}_annotated_metadata.csv"
    metadata_path = os.path.join(figures_folder, metadata_filename)
    
    try:
        query_adata.obs.to_csv(metadata_path)
        print(f"✅ Metadata with annotations exported to: {metadata_path}")
    except Exception as meta_error:
        print(f"⚠️  Warning: Could not export metadata CSV: {str(meta_error)}")
        metadata_path = None
    
    # 新增：保存到新的h5ad文件
    try:
        print(f"\nSaving results to new h5ad file: {output_path}")
        query_adata.write_h5ad(output_path)
        print(f"✅ Saved scPoli results with UMAP to: {output_path}")
        
        # 同时更新原始文件（可选）
        try:
            query_adata.write_h5ad(filename=query_file)
            print(f"✅ Also updated original query file: {query_file}")
        except Exception as update_error:
            print(f"⚠️  Could not update original file: {str(update_error)}")
            
    except Exception as save_error:
        print(f"❌ Error saving to {output_path}: {str(save_error)}")
        
        # Try saving with compression disabled as fallback
        try:
            print("Attempting to save without compression...")
            query_adata.write_h5ad(filename=output_path, compression=None)
            print(f"✅ Saved scPoli results with UMAP (uncompressed): {output_path}")
        except Exception as fallback_error:
            print(f"❌ Fallback save also failed: {str(fallback_error)}")
            
            # Save essential results to CSV as last resort
            backup_csv = output_path.replace('.h5ad', '_backup_results.csv')
            try:
                backup_data = query_adata.obs[[col for col in query_adata.obs.columns 
                                              if col.endswith('_pred') or col.endswith('_uncert')]]
                backup_data.to_csv(backup_csv)
                print(f"⚠️  Saved essential results to CSV backup: {backup_csv}")
            except Exception as csv_error:
                print(f"❌ Even CSV backup failed: {str(csv_error)}")
                raise save_error  # Re-raise original error
    
    # Save additional CSV results if we have precise filtering
    if 'full_latent' in locals():
        output_csv = os.path.join(figures_folder, f'{base_filename}_scPoli_{model_type}_results.csv')
        full_latent.to_csv(output_csv, index=True)
        print(f"Saved clean query results to {output_csv}")
        print(f"Saved {len(full_latent)} rows (original query cells only, no source contamination)")
    
    # Complete the progress display
    if progress_callback:
        progress_callback.status_text.text("Label transfer completed successfully!")
        progress_callback.progress_bar.progress(1.0)  # 100% progress
    
    # Collect generated plot files
    plot_files = {
        "datasets": umap_dataset_file,
        "uncertainty": umap_uncert_file
    }
    
    # Add ordered prediction plots based on model type
    if model_type == "lineage":
        plot_files.update({
            "lineage_full_ordered": os.path.join(figures_folder, f"{base_filename}_full_lineage_pred_ordered.pdf"),
            "lineage_query_ordered": os.path.join(figures_folder, f"{base_filename}_query_lineage_pred_ordered.pdf")
        })
    elif model_type == "cell_type":
        plot_files.update({
            "reanno_full_ordered": os.path.join(figures_folder, f"{base_filename}_full_reanno_pred_ordered.pdf"),
            "reanno_query_ordered": os.path.join(figures_folder, f"{base_filename}_query_reanno_pred_ordered.pdf")
        })
    
    # 添加scPoli UMAP图
    if 'scpoli_umap_file' in locals():
        plot_files["scpoli_umap"] = scpoli_umap_file
    
    # Return enhanced result dictionary
    results = {
        "query_file": query_file,
        "output_file": output_path,  # 新增：输出文件路径
        "model_type": model_type,
        "cell_type_key": cell_type_key,
        "figures_folder": figures_folder,
        "umap_plots": plot_files,
        "metadata_csv": metadata_path,
        "query_dataset_shape": query_adata.shape,
        "source_dataset_shape": source_adata.shape,
        "original_cell_count": original_cell_count,
        "num_predictions": len(preds),
        "unique_predictions": len(set(preds)),
        "precise_filtering": 'full_latent' in locals(),
        "filtered_cell_count": len(full_latent) if 'full_latent' in locals() else "N/A",
        "ordered_visualization": True,
        "lineage_colors_used": False,  
        "reanno_ordering_used": False, 
        "default_colors_used": True,    
        "alphabetical_ordering_used": True,
        "scpoli_umap_stored": umap_key in query_adata.obsm,  # 新增：标记UMAP是否已存储
        "scpoli_umap_key": umap_key if umap_key in query_adata.obsm else None  # 新增：UMAP键名
    }
        
    return results


def main():
    """
    Main function for command-line execution of label transfer.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Label Transfer using scPoli')
    
    # Required arguments
    parser.add_argument('--file_path', '--query_file', type=str, required=True,
                        help='Path to the preprocessed query dataset (.h5ad file)')
    parser.add_argument('--figures_folder', '--output_dir', type=str, required=True,
                        help='Path to save output figures and files')
    
    # Optional arguments with defaults
    parser.add_argument('--model_type', type=str, default='lineage',
                        choices=['lineage', 'cell_type'],
                        help='Type of model to use ("lineage" or "cell_type")')
    parser.add_argument('--custom_model_dir', '--model_dir', type=str, default=None,
                        help='Optional custom path to the model directory')
    parser.add_argument('--custom_adata_path', '--reference_path', type=str, default=None,
                        help='Optional custom path to the reference dataset')
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        print(f"Starting label transfer task...")
        print(f"Query file: {args.file_path}")
        print(f"Output folder: {args.figures_folder}")
        print(f"Model type: {args.model_type}")
        
        # Call the label transfer function
        results = label_transfer(
            query_file=args.file_path,
            figures_folder=args.figures_folder,
            model_type=args.model_type,
            custom_model_dir=args.custom_model_dir,
            custom_adata_path=args.custom_adata_path,
            progress_callback=None  # No progress callback for command-line usage
        )
        
        # Print summary of results
        print("\n" + "="*50)
        print("LABEL TRANSFER COMPLETED SUCCESSFULLY")
        print("="*50)
        print(f"Query file: {results['query_file']}")
        print(f"Output file with scPoli UMAP: {results['output_file']}")
        print(f"Model type: {results['model_type']}")
        print(f"Cell type key: {results['cell_type_key']}")
        print(f"Output folder: {results['figures_folder']}")
        print(f"Query dataset shape: {results['query_dataset_shape']}")
        print(f"Source dataset shape: {results['source_dataset_shape']}")
        print(f"Number of predictions: {results['num_predictions']}")
        print(f"Unique predictions: {results['unique_predictions']}")
        print(f"scPoli UMAP stored: {results['scpoli_umap_stored']}")
        if results['scpoli_umap_stored']:
            print(f"scPoli UMAP key: {results['scpoli_umap_key']}")
        
        print(f"\nGenerated UMAP plots:")
        for plot_type, plot_path in results['umap_plots'].items():
            print(f"  - {plot_type}: {plot_path}")
        
        print(f"\nResults saved to new file: {results['output_file']}")
        print("="*50)
        
    except Exception as e:
        print(f"Error during label transfer: {str(e)}")
        import traceback
        print(f"Full traceback:\n{traceback.format_exc()}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
