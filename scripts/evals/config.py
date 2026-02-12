"""
Dataset configuration: maps dataset name to (dataset_path, query_path).
Paths point to database/test vectors and query vectors (not train/learn).
"""

DATA_ROOT = "/data/cpanourg/2-hdvc/data"

DATASET_CONFIG = {
    "bigann": (
        f"{DATA_ROOT}/bigann/SIFT1M/bigann_base.bvecs",
        f"{DATA_ROOT}/bigann/SIFT1M/bigann_query.bvecs",
    ),
    "gist": (
        f"{DATA_ROOT}/gist/gist_base.fvecs",
        f"{DATA_ROOT}/gist/gist_query.fvecs",
    ),
    "msmarco": (
        f"{DATA_ROOT}/msmarco/base1m.fvecs",
        f"{DATA_ROOT}/msmarco/query10k.fvecs",
    ),
    "openai": (
        f"{DATA_ROOT}/openai/openai_base1m.fvecs",
        f"{DATA_ROOT}/openai/openai_query10k.fvecs",
    ),
    "deep": (
        f"{DATA_ROOT}/deep1b/dataset/fvecs/test_1m.fvecs",
        f"{DATA_ROOT}/deep1b/dataset/fvecs/query_10k.fvecs",
    ),
}
