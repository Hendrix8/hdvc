import numpy as np 
import sys
import time
import csv
import os
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
import torch
from pathlib import Path
import hydra
from omegaconf import OmegaConf
from datetime import datetime
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

module_path = '/home/cpanourg/projects/2-hdvc/'

if module_path not in sys.path:
    sys.path.append(module_path)

from src.utils import read_fvecs, write_fvecs
from src.utils import append_or_create_csv


# Imports experiments (necessary to register experiments)
from lib.Qinco.qinco.qinco_tasks import QincoConvertTask, QincoEvalTask, QincoTrainTask
from lib.Qinco.qinco.search.search_tasks import (
    BuildIndexTask,
    EncodeDBTask,
    IVFTrainTask,
    SearchTask,
    TrainPairwiseDecoderTask,
)
# from lib.Qinco

root = 'data'
data_fp = f'/{root}/cpanourg/2-hdvc/data/' 
temp_fp = f'/{root}/cpanourg/2-hdvc/temp/' 
results_fp = f'/{root}/cpanourg/2-hdvc/results/'

dataset_name = 'deep'

if dataset_name == 'gist':
    db = np.array(read_fvecs(f'{data_fp}gist/gist_base.fvecs')).astype(np.float32)
    qr = np.array(read_fvecs(f'{data_fp}gist/gist_query.fvecs')).astype(np.float32)
    
elif dataset_name == 'deep':
    db = np.fromfile(f'{data_fp}deep1b/dataset/deep1b-96-1m.bin', dtype=np.float32).reshape(1_000_000, -1)
    qr = np.fromfile(f'{data_fp}deep1b/queries/queries-hard10p-deep1b-len96-1000.bin', np.float32).reshape(1000, -1)
    
    db_fp = f'{temp_fp}db_set_{dataset_name}.fvecs'
    qr_fp = f'{temp_fp}qr_set_{dataset_name}.fvecs'
    
    write_fvecs(db_fp, db)
    write_fvecs(qr_fp, qr)


nb = db.shape[0]  # Number of database vectors
nq = qr.shape[0]  # Number of query vectors
dim = db.shape[1]  # Dimensionality of the vectors

print(f"Database shape: {db.shape}")
print(f"Query shape: {qr.shape}")
print(f"Dimension: {dim}")

dim = db.shape[1]
nbits = dim 

import numpy as np

def check_znorm(dataset, name="dataset", sample_size=1000):
    """Check approximate z-normalization on a random sample of a large time series dataset."""
    n_sample = min(sample_size, len(dataset))
    idx = np.random.choice(len(dataset), size=n_sample, replace=False)
    X_sample = dataset[idx]

    # Compute per-series mean and std
    means = X_sample.mean(axis=1)
    stds = X_sample.std(axis=1)

    # Aggregate summary stats
    mean_of_means = means.mean()
    mean_of_stds = stds.mean()
    std_of_means = means.std()
    std_of_stds = stds.std()

    print(f"\n{name}")
    print(f"Sampled {n_sample} / {len(dataset)} series")
    print(f"Average mean across sampled series: {mean_of_means:.5f} ± {std_of_means:.5f}")
    print(f"Average std  across sampled series: {mean_of_stds:.5f} ± {std_of_stds:.5f}")

    return mean_of_means, mean_of_stds


check_znorm(db, name="db", sample_size=10000)
check_znorm(qr, name="queries", sample_size=10000)



scaler = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0)
db = scaler.fit_transform(db).squeeze()
qr = scaler.fit_transform(qr).squeeze()

check_znorm(db, name="db", sample_size=10000)
check_znorm(qr, name="qr", sample_size=10000)


# --- Configuration ---
sampling = 'random'
train_ratio = 0.8   # portion for training
val_ratio = 0.1     # portion for validation
test_ratio = 0.1    # portion for testing
assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."

# --- Shuffle the dataset ---
if sampling == 'random':
    perm = np.random.permutation(len(db))
    db = db[perm]  # shuffle along first dim
else:
    print("⚠️ Sampling is not random — keeping original order.")

# --- Compute split indices ---
n_total = len(db)
n_train = int(train_ratio * n_total)
n_val = int(val_ratio * n_total)
n_test = n_total - n_train - n_val  # ensures total matches exactly

# --- Slice into splits ---
train_set = db[:n_train]
val_set = db[n_train:n_train + n_val]
test_set = db[n_train + n_val:]

# --- Display summary ---
print(f"Train set size: {len(train_set)}")
print(f"Validation set size: {len(val_set)}")
print(f"Test set size: {len(test_set)}")

train_val_set_fp = f"{temp_fp}{dataset_name}_train_ratio{train_ratio}.fvecs"
write_fvecs(train_val_set_fp, np.concatenate([train_set,val_set]))



cfg = OmegaConf.load("/home/cpanourg/projects/2-hdvc/lib/Qinco/config/qinco_cfg.yaml")


EXPERIMENTS = {
    "train": QincoTrainTask,
    "eval_valset": QincoTrainTask,
    "eval": QincoEvalTask,
    "eval_time": QincoEvalTask,
    "convert": QincoConvertTask,
    "ivf_centroids": IVFTrainTask,
    "encode": EncodeDBTask,
    "build_index": BuildIndexTask,
    "train_pairwise_decoder": TrainPairwiseDecoderTask,
    "search": SearchTask,
}


# Get current datetime
now = datetime.now()

# Format as string: year_month_day_hour_minute_second
datetime_str = now.strftime("%Y_%m_%d_%H_%M_%S")

print(datetime_str)


cfg.task = 'train'
cfg.output = f'/{root}/cpanourg/2-hdvc/results/qinco2/qinco_weights_{datetime_str}.pt'

# cfg.L = 16 # num of resblocks in each step
# cfg.dh = 384

# # ----------------------------------------------------------------
# # these values can be set as 0 to disable these components 
# cfg.de = 384 # embedding dimension (if 0 dimension is the same as data)
# cfg.A = 16 # num of fast pre-selected candidates (if 0 no beam search)
# cfg.B = 32 # size of beam search (if 0 then no pre-selected candidates) 
# # ----------------------------------------------------------------

# cfg.M = 8 # number of codebooks
# cfg.K = 256 # codebook size 
# cfg.ivf_K = 1048576

# cfg.epochs = 70 # note that it runs along with the patience (=10 by default)

cfg.L = 16        # paper default
cfg.dh = 384      # paper default
cfg.de = 384      # paper default
cfg.M = 8         # paper default
cfg.K = 256       # paper default
cfg.A = 16        # paper default
cfg.B = 32        # paper default
cfg.ivf_K = 1048576  # paper default (1M centroids)
cfg.epochs = 70   # paper default
cfg.optimizer = "adamw"
cfg.lr = 8e-4
cfg.wd = 0.1
cfg.grad_clip = 0.1
cfg.batch = 1024

cfg.db = db_fp
cfg.trainset = train_val_set_fp 
cfg.ds.valset = n_val

expe = EXPERIMENTS[cfg.task](cfg)

expe.accelerator.print(f"====================== RUNNING TASK {cfg.task}")
expe.run()
expe.accelerator.print("Task done")
expe.accelerator.end_training()  # Destroy process group
