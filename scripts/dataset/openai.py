# save as parquet_to_fvecs_openai5m.py
# Usage:
#   python parquet_to_fvecs_openai5m.py /path/to/openai_large_5m  \
#       --base-out openai5m_base.fvecs --query-out openai5m_query.fvecs
#
# Notes:
#   - Vector column name is fixed as `emb`
#   - Supports resume from checkpoint (recover based on existing .fvecs file size)
#   - By default writes full dataset (5M base + full queries), can use --base-limit / --query-limit to export only first N records
#   - Read in chunks by row-group, FixedSizeList<float32> uses zero-copy reshape; List<float32> uses to_pylist()
 
import os
import sys
import argparse
from glob import glob
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import pyarrow.compute as pc  # NEW
 
VECTOR_COL = "emb"          # Column name: fixed as emb according to your data
BLOCK = 100_000             # Number of records to write each time (I/O block size, adjust based on machine)
ROWGROUP_LIMIT = None       # For debugging: limit number of row-groups to read per file
 
def infer_written_and_dim(path: str):
    """Infer number of written records and dimension from fvecs file size; returns (0, None) if file doesn't exist"""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return 0, None
    with open(path, "rb") as f:
        # Dimension of first record
        dim = np.fromfile(f, dtype=np.int32, count=1)
        if dim.size == 0:
            return 0, None
        dim = int(dim[0])
    rec_bytes = 4 + 4 * dim
    return os.path.getsize(path) // rec_bytes, dim
 
 
def arrow_col_to_numpy(col: pa.Array, known_dim: int | None):
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
 
def write_block_fvecs_interleaved(fout, block: np.ndarray, dim: int):
    """
    Correct fvecs writing method: write one by one (each record: int32(dim) + float32[dim])
    """
    # Python loop is not a bottleneck for I/O, safe and reliable
    for i in range(block.shape[0]):
        np.array([dim], dtype=np.int32).tofile(fout)
        block[i].astype(np.float32, copy=False).tofile(fout)
 
def slice_openai5m_to_fvecs(
    parquet_dir: str,
    base_out: str,
    train_out: str,
    query_out: str,
    base_size: int,
    train_size: int,
    query_size: int,
):
    """
    Stream over shuffle_train-*.parquet files and slice into base/train/query .fvecs files:
      - base:  [0, base_size)
      - train: [base_size, base_size + train_size)
      - query: [base_size + train_size, base_size + train_size + query_size)
    All three are taken from the shuffled training set.
    """
    train_files = sorted(glob(os.path.join(parquet_dir, "shuffle_train-*.parquet*")))
    if not train_files:
        print(f"[ERROR] No shuffle_train-*.parquet files found under {parquet_dir}")
        sys.exit(1)

    if base_size < 0 or train_size < 0 or query_size < 0:
        raise ValueError("Sizes must be non-negative.")

    total_needed = base_size + train_size + query_size
    if total_needed == 0:
        print("[WARN] All sizes are zero; nothing to do.")
        return

    # Open files conditionally
    fb = open(base_out, "wb") if base_size > 0 else None
    ft = open(train_out, "wb") if train_size > 0 else None
    fq = open(query_out, "wb") if query_size > 0 else None

    dim = None
    offset = 0  # global row index across all train shards

    pbar = tqdm(total=total_needed, desc="slicing OpenAI5M parquet -> fvecs", unit="vec")

    for fp in train_files:
        if offset >= total_needed:
            break

        pf = pq.ParquetFile(fp)
        num_rgs = pf.num_row_groups
        if ROWGROUP_LIMIT is not None:
            num_rgs = min(num_rgs, ROWGROUP_LIMIT)

        for rg in range(num_rgs):
            if offset >= total_needed:
                break

            tbl = pf.read_row_group(rg, columns=[VECTOR_COL])
            arr_np, d = arrow_col_to_numpy(tbl.column(0), dim)

            if dim is None:
                dim = d
            elif d != dim:
                raise ValueError(f"Dim mismatch: {d} vs existing {dim} in {fp} rg={rg}")

            n_i = arr_np.shape[0]
            file_start = offset
            file_end = offset + n_i  # [file_start, file_end)

            # 1) base segment [0, base_size)
            if base_size > 0 and fb is not None:
                seg_s = max(file_start, 0)
                seg_e = min(file_end, base_size)
                if seg_s < seg_e:
                    ls = seg_s - file_start
                    le = seg_e - file_start
                    write_block_fvecs_interleaved(fb, arr_np[ls:le], dim)
                    pbar.update(le - ls)

            # 2) train segment [base_size, base_size + train_size)
            train_start = base_size
            train_end = base_size + train_size
            if train_size > 0 and ft is not None:
                seg_s = max(file_start, train_start)
                seg_e = min(file_end, train_end)
                if seg_s < seg_e:
                    ls = seg_s - file_start
                    le = seg_e - file_start
                    write_block_fvecs_interleaved(ft, arr_np[ls:le], dim)
                    pbar.update(le - ls)

            # 3) query segment [base_size + train_size, total_needed)
            query_start = base_size + train_size
            query_end = total_needed
            if query_size > 0 and fq is not None:
                seg_s = max(file_start, query_start)
                seg_e = min(file_end, query_end)
                if seg_s < seg_e:
                    ls = seg_s - file_start
                    le = seg_e - file_start
                    write_block_fvecs_interleaved(fq, arr_np[ls:le], dim)
                    pbar.update(le - ls)

            offset += n_i

    if fb is not None:
        fb.close()
    if ft is not None:
        ft.close()
    if fq is not None:
        fq.close()

    pbar.close()
    print("[OK] Done slicing OpenAI5M.")
    if base_size > 0:
        print(f"  base:   first {base_size:,} -> {base_out}")
    if train_size > 0:
        print(f"  train:  next  {train_size:,} -> {train_out}")
    if query_size > 0:
        print(f"  query:  next  {query_size:,} -> {query_out} (after {base_size + train_size:,})")


def main():
    ap = argparse.ArgumentParser(
        description="Parquet (emb) → base/train/query fvecs converter (OpenAI-5M)"
    )
    ap.add_argument(
        "--parquet_dir",
        type=str,
        default="/mnthdd/cpanourg/2-hdvc/data/openai/openai_large_5m",
        help="Directory containing shuffle_train-*-of-10.parquet files from OpenAI5M.",
    )
    ap.add_argument(
        "--base-out",
        type=str,
        default="/mnthdd/cpanourg/2-hdvc/data/openai/openai5m_base.fvecs",
        help="Output fvecs for base vectors",
    )
    ap.add_argument(
        "--train-out",
        type=str,
        default="/mnthdd/cpanourg/2-hdvc/data/openai/openai5m_train.fvecs",
        help="Output fvecs for train vectors",
    )
    ap.add_argument(
        "--query-out",
        type=str,
        default="/mnthdd/cpanourg/2-hdvc/data/openai/openai5m_query.fvecs",
        help="Output fvecs for query vectors",
    )
    ap.add_argument(
        "--base-size",
        type=int,
        default=1_000_000,
        help="Number of base vectors to export (default: 1M)",
    )
    ap.add_argument(
        "--train-size",
        type=int,
        default=1_000_000,
        help="Number of train vectors to export after base (default: 1M)",
    )
    ap.add_argument(
        "--query-size",
        type=int,
        default=10_000,
        help="Number of query vectors to export after base+train (default: 10k)",
    )
    args = ap.parse_args()

    slice_openai5m_to_fvecs(
        parquet_dir=args.parquet_dir,
        base_out=args.base_out,
        train_out=args.train_out,
        query_out=args.query_out,
        base_size=args.base_size,
        train_size=args.train_size,
        query_size=args.query_size,
    )


if __name__ == "__main__":
    main()
 