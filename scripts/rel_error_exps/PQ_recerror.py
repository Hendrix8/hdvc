#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate reconstruction error for PQ models from CSV results.
Loads models from experiment folders and computes MSE between original and reconstructed vectors.
"""

import numpy as np
import faiss
import pandas as pd
import sys
import argparse
from pathlib import Path

# =====================
# === Import helpers ===
# =====================
module_path = '/home/cpanourg/projects/2-hdvc/'
if module_path not in sys.path:
    sys.path.append(module_path)

from src.utils import load_dataset




# =============================
# === Reconstruction error ===
# =============================
def calculate_reconstruction_error(pq, test_data, max_samples=10000):
    """
    Calculate reconstruction error (MSE) for PQ model.
    
    Args:
        pq: FAISS ProductQuantizer
        test_data: numpy array of test vectors (nb, dim)
        max_samples: Maximum number of samples to use for error calculation
    
    Returns:
        dict with 'mse_mean', 'mse_std', 'mse_max', 'mse_min'
    """
    # Sample subset if too large
    nb = len(test_data)
    n_sample = min(max_samples, nb)
    
    if n_sample < nb:
        # indices = np.random.choice(nb, n_sample, replace=False)
        indices = np.arange(n_sample) 
        test_sample = test_data[indices]
    else:
        test_sample = test_data
    
    print(f"  Computing codes for {len(test_sample)} vectors...")
    # Encode vectors to codes
    # try:
    codes = pq.compute_codes(test_sample)
    # except TypeError:
    #     codes = np.zeros((len(test_sample), pq.code_size), dtype='uint8')
    #     pq.compute_codes(test_sample, codes)
    
    print(f"  Reconstructing vectors from codes...")
    # Reconstruct vectors from codes using FAISS IndexPQ's sa_decode method
    # This is the standard and most reliable way to decode PQ codes
    # reconstructed = np.zeros((len(test_sample), pq.d), dtype=np.float32)
    
    # Create a temporary IndexPQ to use sa_decode
    temp_index = faiss.IndexPQ(pq.d, pq.M, pq.nbits)
    temp_index.pq = pq
    
    # Decode codes to reconstruct vectors
    # sa_decode expects codes in the same format as sa_encode produces
    reconstructed = temp_index.sa_decode(codes)
    
    # Calculate MSE (Mean Squared Error)
    mse_per_vector = np.mean((test_sample - reconstructed) ** 2, axis=1)
    
    return {
        'mse_mean': float(np.mean(mse_per_vector)),
        'mse_std': float(np.std(mse_per_vector)),
        'mse_max': float(np.max(mse_per_vector)),
        'mse_min': float(np.min(mse_per_vector)),
        'n_samples': len(test_sample),
    }


# =============================
# === Main processing ===
# =============================
def process_csv(csv_path, dataset_path, max_samples=10000, 
                sample_db=None, dim=None):
    """
    Process CSV file and add reconstruction error column.
    
    Args:
        csv_path: Path to CSV file with experiment results
        dataset_path: Path to the test/database dataset file
        max_samples: Maximum samples to use for reconstruction error calculation
        sample_db: Number of database vectors to load (None = use all or max_samples)
        dim: Override dimension (None = infer from model)
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    print(f"📂 Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if 'experiment_folder' not in df.columns:
        raise ValueError("CSV must contain 'experiment_folder' column")
    
    if 'dataset' not in df.columns:
        raise ValueError("CSV must contain 'dataset' column")
    
    # Check if reconstruction error columns already exist
    rec_error_cols = ['rec_error_mse_mean', 'rec_error_mse_std', 'rec_error_mse_max', 'rec_error_mse_min']
    has_rec_error = all(col in df.columns for col in rec_error_cols)
    
    if not has_rec_error:
        # Initialize columns
        for col in rec_error_cols:
            df[col] = np.nan
    
    print(f"Found {len(df)} rows to process")
    
    # Process each row
    for idx, row in df.iterrows():
        experiment_folder = Path(row['experiment_folder'])
        dataset_name = row['dataset']
        model_path = experiment_folder / 'pq_model.index'
        
        print(f"\n[{idx+1}/{len(df)}] Processing: {dataset_name} - {experiment_folder.name}")
        
        # Check if already computed
        if has_rec_error and not pd.isna(row.get('rec_error_mse_mean', np.nan)):
            print(f"  ⏭️  Skipping (already computed)")
            continue
        
        # Check if model exists
        if not model_path.exists():
            print(f"  ⚠️  Model not found: {model_path}")
            continue
        
        # Load model
        print(f"  📥 Loading model: {model_path}")
        try:
            pq_index = faiss.read_index(str(model_path))
            pq = pq_index.pq  # Extract ProductQuantizer
            print(f"  ✅ Model loaded: dim={pq.d}, M={pq.M}, nbits={pq.nbits}")
        except Exception as e:
            print(f"  ❌ Error loading model: {e}")
            continue
        
        # Use provided dataset path and infer dimension from model
        dataset_dim = dim if dim is not None else pq.d
        print(f"  📂 Loading test data: {dataset_path}")
        
        # Load test data
        try:
            # Determine how many samples to load
            load_size = sample_db if sample_db is not None else max_samples
            test_data, _ = load_dataset(
                dataset_path, 
                None, 
                dataset_dim,
                db_chunk_size=load_size,
                qr_chunk_size=None
            )
            
            # Verify dimension matches
            if test_data.shape[1] != pq.d:
                print(f"  ⚠️  Dimension mismatch: data={test_data.shape[1]}, model={pq.d}")
                # Try to use model dimension
                if pq.d != dataset_dim:
                    print(f"  ⚠️  Using model dimension {pq.d} instead")
            
            print(f"  ✅ Loaded {len(test_data)} test vectors")
        except Exception as e:
            print(f"  ❌ Error loading test data: {e}")
            continue
        
        # Calculate reconstruction error
        try:
            rec_error = calculate_reconstruction_error(pq, test_data, max_samples=max_samples)
            
            # Update dataframe
            df.at[idx, 'rec_error_mse_mean'] = rec_error['mse_mean']
            df.at[idx, 'rec_error_mse_std'] = rec_error['mse_std']
            df.at[idx, 'rec_error_mse_max'] = rec_error['mse_max']
            df.at[idx, 'rec_error_mse_min'] = rec_error['mse_min']
            
            print(f"  ✅ Reconstruction error: MSE={rec_error['mse_mean']:.6f} ± {rec_error['mse_std']:.6f}")
        except Exception as e:
            print(f"  ❌ Error calculating reconstruction error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save updated CSV
    print(f"\n💾 Saving updated CSV to {csv_path}")
    df.to_csv(csv_path, index=False)
    print(f"✅ Done!")


# ==================================
# === Main function with defaults ===
# ==================================
def main():
    """
    Main function with default arguments.
    Modify these defaults to run without command-line arguments.
    """
    # ============================================
    # DEFAULT CONFIGURATION - Modify these values
    # ============================================
    DEFAULT_CSV_PATH = "/mnthdd/cpanourg/2-hdvc/results/relerr/gist_PQ_adc_vs_exact_eval.csv"
    DEFAULT_DATASET_PATH = "/mnthdd/cpanourg/2-hdvc/data/gist/gist_base.fvecs"
    DEFAULT_MAX_SAMPLES = 1_000_000
    DEFAULT_SAMPLE_DB = None  # None = use max_samples
    DEFAULT_DIM = None  # None = infer from model
    
    # ============================================
    # Parse command-line arguments (optional)
    # ============================================
    parser = argparse.ArgumentParser(
        description="Calculate reconstruction error for PQ models from CSV results."
    )
    
    parser.add_argument(
        "--csv_path",
        type=str,
        default=DEFAULT_CSV_PATH,
        help=f"Path to CSV file with experiment results (default: {DEFAULT_CSV_PATH})"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help=f"Path to test/database dataset file (default: {DEFAULT_DATASET_PATH})"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help=f"Maximum number of samples to use for reconstruction error calculation (default: {DEFAULT_MAX_SAMPLES})"
    )
    parser.add_argument(
        "--sample_db",
        type=int,
        default=DEFAULT_SAMPLE_DB,
        help="Number of database vectors to load (default: max_samples)"
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=DEFAULT_DIM,
        help="Override dimension (default: infer from model)"
    )
    
    args = parser.parse_args()
    
    # ============================================
    # Run processing
    # ============================================
    print("="*70)
    print("PQ Reconstruction Error Calculator")
    print("="*70)
    print(f"CSV Path: {args.csv_path}")
    print(f"Dataset Path: {args.dataset_path}")
    print(f"Max Samples: {args.max_samples}")
    print(f"Sample DB: {args.sample_db if args.sample_db else 'max_samples'}")
    print(f"Dimension Override: {args.dim if args.dim else 'auto (from model)'}")
    print("="*70)
    print()
    
    process_csv(
        csv_path=args.csv_path,
        dataset_path=args.dataset_path,
        max_samples=args.max_samples,
        sample_db=args.sample_db,
        dim=args.dim
    )


# ==================================
# === Command-line entry point ===
# ==================================
if __name__ == "__main__":
    main()

