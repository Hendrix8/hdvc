"""
Download MS MARCO v2.1 embeddings from Hugging Face and convert to .npy format
compatible with msmarco.py script.

Dataset: https://huggingface.co/datasets/Cohere/msmarco-v2.1-embed-english-v3

Usage:
    python download_msmarco_hf.py --output-dir /mnthdd/cpanourg/2-hdvc/data/msmarco
    python download_msmarco_hf.py --output-dir /mnthdd/cpanourg/2-hdvc/data/msmarco --shard-size 1000000
"""

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' library not found. Install it with: pip install datasets")
    sys.exit(1)

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
except ImportError:
    print("Error: 'pyarrow' library not found. Install it with: pip install pyarrow")
    sys.exit(1)


VECTOR_COL = "emb"  # Column name for embeddings in the dataset
DEFAULT_SHARD_SIZE = 1_000_000  # Number of vectors per .npy shard file


def arrow_col_to_numpy(col: pa.Array, known_dim: int | None = None):
    """
    Convert Arrow column to numpy float32 matrix (B, d)
    Supports:
      - list<float32|float64>
      - large_list<float32|float64>
      - fixed_size_list<float32|float64>
    """
    col = col.combine_chunks()
    t = col.type

    # --- fixed_size_list ---
    if pa.types.is_fixed_size_list(t) and (pa.types.is_float32(t.value_type) or pa.types.is_float64(t.value_type)):
        d = t.list_size
        if pa.types.is_float64(t.value_type):
            col = pc.cast(col, pa.fixed_size_list(pa.float32(), d))
        flat = col.values.to_numpy()
        arr = np.asarray(flat.reshape(-1, d), dtype=np.float32, order="C")
        return arr, d

    # --- list / large_list ---
    if (pa.types.is_list(t) or pa.types.is_large_list(t)) and \
       (pa.types.is_float32(t.value_type) or pa.types.is_float64(t.value_type)):
        # cast double->float32 if needed
        if pa.types.is_float64(t.value_type):
            if pa.types.is_list(t):
                col = pc.cast(col, pa.list_(pa.float32()))
            else:  # large_list
                col = pc.cast(col, pa.large_list(pa.float32()))
        py = col.to_pylist()
        d = len(py[0]) if known_dim is None else known_dim
        if any(len(x) != d for x in py):
            raise ValueError("Inconsistent dims detected in list<...> column")
        arr = np.asarray(py, dtype=np.float32, order="C")
        return arr, d

    raise TypeError(f"Column type must be (large_)list<float32|float64>, got {t}")


def download_and_convert_hf_dataset(output_dir: str, shard_size: int = DEFAULT_SHARD_SIZE, 
                                     streaming: bool = True, max_rows: int | None = None):
    """
    Download MS MARCO v2.1 embeddings from Hugging Face and save as .npy shards.
    
    Args:
        output_dir: Directory to save .npy shard files
        shard_size: Number of vectors per .npy file
        streaming: Use streaming mode (recommended for large datasets)
        max_rows: Maximum number of rows to download (None = all)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    passages_dir = output_path / "passages_npy"
    passages_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading MS MARCO v2.1 embeddings from Hugging Face...")
    print(f"Output directory: {passages_dir}")
    print(f"Shard size: {shard_size:,} vectors per file")
    print(f"Streaming mode: {streaming}")
    if max_rows:
        print(f"Max rows: {max_rows:,}")
    print()
    
    # Load dataset in streaming mode
    try:
        dataset = load_dataset(
            "Cohere/msmarco-v2.1-embed-english-v3",
            "passages",
            split="train",
            streaming=streaming
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying without streaming mode...")
        streaming = False
        dataset = load_dataset(
            "Cohere/msmarco-v2.1-embed-english-v3",
            "passages",
            split="train"
        )
    
    # Process dataset
    current_shard = []
    shard_idx = 0
    total_vectors = 0
    dim = None
    
    pbar = tqdm(desc="Processing embeddings", unit="vec")
    
    for row in dataset:
        if max_rows and total_vectors >= max_rows:
            break
        
        # Extract embedding
        emb = row[VECTOR_COL]
        
        # Convert to numpy if needed
        if isinstance(emb, list):
            emb_array = np.array(emb, dtype=np.float32)
        elif isinstance(emb, np.ndarray):
            emb_array = emb.astype(np.float32)
        elif isinstance(emb, pa.Array):
            # Handle Arrow array
            emb_array, d = arrow_col_to_numpy(emb, known_dim=dim)
            if dim is None:
                dim = d
        else:
            raise TypeError(f"Unexpected embedding type: {type(emb)}")
        
        # Check dimension consistency
        if dim is None:
            dim = emb_array.shape[0]
        elif emb_array.shape[0] != dim:
            raise ValueError(f"Dimension mismatch: expected {dim}, got {emb_array.shape[0]}")
        
        current_shard.append(emb_array)
        total_vectors += 1
        pbar.update(1)
        
        # Save shard when it reaches shard_size
        if len(current_shard) >= shard_size:
            shard_array = np.array(current_shard, dtype=np.float32)
            shard_path = passages_dir / f"shard_{shard_idx:05d}.npy"
            np.save(shard_path, shard_array)
            print(f"\nSaved shard {shard_idx}: {shard_path} (shape: {shard_array.shape})")
            current_shard = []
            shard_idx += 1
    
    # Save remaining vectors in final shard
    if current_shard:
        shard_array = np.array(current_shard, dtype=np.float32)
        shard_path = passages_dir / f"shard_{shard_idx:05d}.npy"
        np.save(shard_path, shard_array)
        print(f"\nSaved final shard {shard_idx}: {shard_path} (shape: {shard_array.shape})")
    
    pbar.close()
    
    print(f"\n{'='*60}")
    print(f"Download complete!")
    print(f"Total vectors: {total_vectors:,}")
    print(f"Dimension: {dim}")
    print(f"Number of shards: {shard_idx + 1}")
    print(f"Output directory: {passages_dir}")
    print(f"{'='*60}")
    print(f"\nNext step: Update msmarco.py with:")
    print(f"  NPY_DIR = '{passages_dir}'")


def download_from_parquet_files(parquet_dir: str, output_dir: str, 
                                shard_size: int = DEFAULT_SHARD_SIZE):
    """
    Alternative method: If you've already downloaded parquet files locally,
    convert them to .npy shards.
    
    Args:
        parquet_dir: Directory containing parquet files
        output_dir: Directory to save .npy shard files
        shard_size: Number of vectors per .npy file
    """
    from glob import glob
    
    parquet_path = Path(parquet_dir)
    output_path = Path(output_dir)
    passages_dir = output_path / "passages_npy"
    passages_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all parquet files
    parquet_files = sorted(glob(str(parquet_path / "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")
    
    print(f"Found {len(parquet_files)} parquet files")
    print(f"Output directory: {passages_dir}")
    print(f"Shard size: {shard_size:,} vectors per file")
    print()
    
    current_shard = []
    shard_idx = 0
    total_vectors = 0
    dim = None
    
    pbar = tqdm(total=None, desc="Processing parquet files", unit="vec")
    
    for parquet_file in parquet_files:
        pf = pq.ParquetFile(parquet_file)
        schema = pf.schema_arrow
        
        if VECTOR_COL not in schema.names:
            print(f"Warning: '{VECTOR_COL}' column not found in {parquet_file}, skipping")
            continue
        
        num_rgs = pf.num_row_groups
        
        for rg in range(num_rgs):
            tbl = pf.read_row_group(rg, columns=[VECTOR_COL])
            arr_np, d = arrow_col_to_numpy(tbl.column(0), known_dim=dim)
            
            if dim is None:
                dim = d
            elif d != dim:
                raise ValueError(f"Dim mismatch: {d} vs existing {dim} in {parquet_file} rg={rg}")
            
            # Process vectors in batches
            for i in range(arr_np.shape[0]):
                current_shard.append(arr_np[i])
                total_vectors += 1
                pbar.update(1)
                
                # Save shard when it reaches shard_size
                if len(current_shard) >= shard_size:
                    shard_array = np.array(current_shard, dtype=np.float32)
                    shard_path = passages_dir / f"shard_{shard_idx:05d}.npy"
                    np.save(shard_path, shard_array)
                    print(f"\nSaved shard {shard_idx}: {shard_path} (shape: {shard_array.shape})")
                    current_shard = []
                    shard_idx += 1
    
    # Save remaining vectors in final shard
    if current_shard:
        shard_array = np.array(current_shard, dtype=np.float32)
        shard_path = passages_dir / f"shard_{shard_idx:05d}.npy"
        np.save(shard_path, shard_array)
        print(f"\nSaved final shard {shard_idx}: {shard_path} (shape: {shard_array.shape})")
    
    pbar.close()
    
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"Total vectors: {total_vectors:,}")
    print(f"Dimension: {dim}")
    print(f"Number of shards: {shard_idx + 1}")
    print(f"Output directory: {passages_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Download MS MARCO v2.1 embeddings from Hugging Face and convert to .npy format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download from Hugging Face (streaming mode, recommended)
  python download_msmarco_hf.py --output-dir /mnthdd/cpanourg/2-hdvc/data/msmarco
  
  # Download with custom shard size
  python download_msmarco_hf.py --output-dir /mnthdd/cpanourg/2-hdvc/data/msmarco --shard-size 2000000
  
  # Download limited number of vectors (for testing)
  python download_msmarco_hf.py --output-dir /mnthdd/cpanourg/2-hdvc/data/msmarco --max-rows 10000000
  
  # Convert from local parquet files
  python download_msmarco_hf.py --parquet-dir /path/to/parquet/files --output-dir /mnthdd/cpanourg/2-hdvc/data/msmarco
        """
    )
    
    parser.add_argument('--output-dir', type=str,
                       default='/mnthdd/cpanourg/2-hdvc/data/msmarco',
                       help='Output directory for .npy shard files')
    parser.add_argument('--parquet-dir', type=str, default=None,
                       help='If provided, convert from local parquet files instead of downloading from Hugging Face')
    parser.add_argument('--shard-size', type=int, default=DEFAULT_SHARD_SIZE,
                       help=f'Number of vectors per .npy file (default: {DEFAULT_SHARD_SIZE:,})')
    parser.add_argument('--max-rows', type=int, default=None,
                       help='Maximum number of rows to download (for testing)')
    parser.add_argument('--no-streaming', action='store_true',
                       help='Disable streaming mode (loads entire dataset into memory). By default, streaming is enabled.')
    
    args = parser.parse_args()
    
    # Check if parquet directory was provided and exists
    use_parquet = args.parquet_dir is not None and os.path.exists(args.parquet_dir)
    
    if use_parquet:
        # Convert from local parquet files
        print(f"Using local parquet files from: {args.parquet_dir}")
        download_from_parquet_files(
            parquet_dir=args.parquet_dir,
            output_dir=args.output_dir,
            shard_size=args.shard_size
        )
    else:
        # Download from Hugging Face
        if args.parquet_dir is not None:
            print(f"Warning: Parquet directory '{args.parquet_dir}' does not exist. Downloading from Hugging Face instead.")
        print(f"Downloading from Hugging Face...")
        download_and_convert_hf_dataset(
            output_dir=args.output_dir,
            shard_size=args.shard_size,
            streaming=not args.no_streaming,  # streaming=True by default, False if --no-streaming is set
            max_rows=args.max_rows
        )


if __name__ == "__main__":
    main()
