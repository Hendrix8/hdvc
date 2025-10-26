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

# sampling for training 
sampling = 'random'
train_ratio = 0.001 
val_ratio = 0.2

n_samples = int(train_ratio * db.shape[0])
n_val_samples = int(n_samples * val_ratio)

train_set_fp = f'{temp_fp}{dataset_name}_train_ratio{train_ratio}.fvecs'
train_set = db[np.random.choice(db.shape[0], size=n_samples, replace=False)]

print(f"Train set size: {n_samples}")
print(f'Validation set size: {n_val_samples}')

write_fvecs(train_set_fp, train_set)

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

cfg.L = 16
cfg.de = 384
cfg.dh = 384

cfg.A = 16
cfg.B = 32
cfg.M = 8
cfg.K = 256
cfg.ivf_K = 1048576

cfg.epochs = 10 # note that it runs along with the patience (=10 by default)

cfg.db = db_fp
cfg.trainset = train_set_fp 
cfg.ds.valset = n_val_samples

print(f"Checking epochs setting: {cfg.epochs}")  # Add this to verify the value

expe = EXPERIMENTS[cfg.task](cfg)

expe.accelerator.print(f"====================== RUNNING TASK {cfg.task}")
expe.run()
expe.accelerator.print("Task done")
expe.accelerator.end_training()  # Destroy process group
