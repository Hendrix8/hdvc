import os
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
 
# ====== Configuration: only modify here ======
NPY_DIR      = "/mnthdd/cpanourg/2-hdvc/data/msmarco/passages_npy"  # .npy shard directory

# Main outputs
BASE_OUT     = "/mnthdd/cpanourg/2-hdvc/data/msmarco/base.fvecs"
TRAIN_OUT    = "/mnthdd/cpanourg/2-hdvc/data/msmarco/train.fvecs"
QUERY_OUT    = "/mnthdd/cpanourg/2-hdvc/data/msmarco/query.fvecs"

# Optional: additional output of first 5M train vectors (after query segment)
TRAIN_5M_OUT = "/mnthdd/cpanourg/2-hdvc/data/msmarco/msmarco1M_train5M.fvecs"

BASE_SIZE    = 1_000_000
TRAIN_SIZE   = 1_000_000
QUERY_SIZE   = 10_000
TRAIN5M_SIZE = 5_000_000  # additional 5M train (optional)
BLOCK        = 200_000    # Number of rows to write per block (adjust based on machine I/O)
# =================================

START_TRAIN = BASE_SIZE
START_QUERY = BASE_SIZE + TRAIN_SIZE
END_QUERY   = START_QUERY + QUERY_SIZE
END_TRAIN5M = END_QUERY + TRAIN5M_SIZE  # Additional output of first 5M train (optional)
 
def write_one(fout, vec, dim):
    """fvecs: each record is int32(dim) + float32[dim], written one by one"""
    np.array([dim], dtype=np.int32).tofile(fout)
    vec.astype(np.float32, copy=False).tofile(fout)

def write_range_interleaved(fout, arr, start, end, dim):
    """Write arr[start:end] to fvecs one by one (chunked, low memory)"""
    n = end - start
    cur = 0
    while cur < n:
        step = min(BLOCK, n - cur)
        block = arr[start + cur : start + cur + step]
        # Ensure float32 + C contiguous
        if block.dtype != np.float32 or not block.flags["C_CONTIGUOUS"]:
            block = np.asarray(block, dtype=np.float32, order="C")
        for i in range(block.shape[0]):
            write_one(fout, block[i], dim)
        cur += step
 
def main(
    make_base: bool = True,
    make_train: bool = True,
    make_query: bool = True,
    make_train5m: bool = True,
):
    if not (make_base or make_train or make_query or make_train5m):
        raise SystemExit("Nothing to do: all outputs disabled.")

    files = sorted(glob(os.path.join(NPY_DIR, "*.npy")))
    if not files:
        raise SystemExit(f"No .npy files found under {NPY_DIR}")
 
    # Open output files (overwrite mode), only for enabled outputs
    fb = open(BASE_OUT, "wb") if make_base else None
    ft = open(TRAIN_OUT, "wb") if make_train else None
    fq = open(QUERY_OUT, "wb") if make_query else None
    ftt = open(TRAIN_5M_OUT, "wb") if make_train5m else None  # Additional output of first 5M train (optional)

    dim = None
    offset = 0  # Starting row number of current shard in global context

    # Decide how many vectors we need based on which outputs are enabled
    need_total = 0
    if make_base:
        need_total = max(need_total, BASE_SIZE)
    if make_train:
        need_total = max(need_total, START_QUERY)
    if make_query:
        need_total = max(need_total, END_QUERY)
    if make_train5m:
        need_total = max(need_total, END_TRAIN5M)

    if need_total == 0:
        raise SystemExit("Computed need_total == 0; nothing to process.")
 
    pbar = tqdm(total=need_total, desc="slicing npy -> fvecs", unit="vec")
 
    for fp in files:
        if offset >= need_total:
            break
 
        arr = np.load(fp, mmap_mode="r")  # (n_i, d), float32/float64
        n_i, d = arr.shape
        if dim is None:
            dim = int(d)
        elif d != dim:
            raise ValueError(f"Dimension mismatch in {fp}: got {d}, expected {dim}")
 
        file_start = offset
        file_end   = offset + n_i  # Half-open interval [file_start, file_end)

        # 1) Write base segment [0, BASE_SIZE)
        if make_base and fb is not None:
            seg_s = max(file_start, 0)
            seg_e = min(file_end, BASE_SIZE)
            if seg_s < seg_e:
                ls = seg_s - file_start
                le = seg_e - file_start
                write_range_interleaved(fb, arr, ls, le, dim)
                pbar.update(le - ls)

        # 2) Write train segment [START_TRAIN, START_QUERY)
        if make_train and ft is not None:
            seg_s = max(file_start, START_TRAIN)
            seg_e = min(file_end, START_QUERY)
            if seg_s < seg_e:
                ls = seg_s - file_start
                le = seg_e - file_start
                write_range_interleaved(ft, arr, ls, le, dim)
                pbar.update(le - ls)

        # 3) Write query segment [START_QUERY, END_QUERY)
        if make_query and fq is not None:
            seg_s = max(file_start, START_QUERY)
            seg_e = min(file_end, END_QUERY)
            if seg_s < seg_e:
                ls = seg_s - file_start
                le = seg_e - file_start
                write_range_interleaved(fq, arr, ls, le, dim)
                pbar.update(le - ls)

        # 4) Additional output of first 5M train (optional)
        if make_train5m and ftt is not None:
            seg_s = max(file_start, END_QUERY)
            seg_e = min(file_end, END_TRAIN5M)
            if seg_s < seg_e:
                ls = seg_s - file_start
                le = seg_e - file_start
                write_range_interleaved(ftt, arr, ls, le, dim)
                pbar.update(le - ls)
 
        offset += n_i
 
    if fb is not None:
        fb.close()
    if ft is not None:
        ft.close()
    if fq is not None:
        fq.close()
    if ftt is not None:
        ftt.close()  # Additional output of first 5M train (optional)
    pbar.close()

    # Brief report
    print(f"[OK] Done.")
    if make_base:
        print(f"  base:   first {BASE_SIZE:,} -> {BASE_OUT}")
    if make_train:
        print(f"  train:  next  {TRAIN_SIZE:,} -> {TRAIN_OUT}")
    if make_query:
        print(f"  query:  next  {QUERY_SIZE:,} -> {QUERY_OUT} (after {BASE_SIZE + TRAIN_SIZE:,})")
    if make_train5m:
        print(f"  train5M (optional): next {TRAIN5M_SIZE:,} -> {TRAIN_5M_OUT} (after {END_QUERY:,})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Slice MS MARCO embeddings shards into FAISS .fvecs files."
    )
    parser.add_argument("--no-base", default=False, action="store_true", help="Do not write base.fvecs")
    parser.add_argument("--no-train", default=False, action="store_true", help="Do not write train.fvecs")
    parser.add_argument("--no-query", default=False, action="store_true", help="Do not write query.fvecs")
    parser.add_argument(
        "--no-train5m",
        default=True,
        action="store_true",
        help="Do not write the extra 5M train vectors file.",
    )
    args = parser.parse_args()

    main(
        make_base=not args.no_base,
        make_train=not args.no_train,
        make_query=not args.no_query,
        make_train5m=not args.no_train5m,
    )
 