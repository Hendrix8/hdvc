import numpy as np 
import faiss
import sys
import time
import csv
import os
module_path = '/home/cpanourg/projects/2-hdvc/'

if module_path not in sys.path:
    sys.path.append(module_path)

from src.utils import read_fvecs
from src.utils import append_or_create_csv

dataset_name = 'GIST'
sampling_method = 'random'  # 'random'
train_size_ratio = 0.1  # Ratio of the database to use for training if using sampling
# Loading the GIST dataset
db = np.array(read_fvecs('/data/cpanourg/2-hdvc/data/gist/gist_base.fvecs'))
qr = np.array(read_fvecs('/data/cpanourg/2-hdvc/data/gist/gist_query.fvecs'))

nb = db.shape[0]  # Number of database vectors
nq = qr.shape[0]  # Number of query vectors
dim = db.shape[1]  # Dimensionality of the vectors

# Step 1: Define the number of centroids for each subquantizer
n_centroids = 256  # Number of centroids for each subquantizer
n_subquantizers = 8  # Number of subquantizers for the PQ


start = time.time()
opq = faiss.OPQMatrix(dim, n_subquantizers)
pq = faiss.IndexPQ(dim, n_subquantizers, n_centroids)

# If you have a separate training set, you should use that instead of the full database.
if sampling_method == 'random':
    num_samples = min(int(train_size_ratio * nb), db.shape[0])  # Number of samples for training
    training_data = db[np.random.choice(db.shape[0], num_samples, replace=False)]

# Perform training
opq.train(training_data)
db_opq = opq.apply(db)
qr_opq = opq.apply(qr)

pq.train(db_opq)

# Step 4: Add the database vectors to the PQ index
pq.add(db_opq)

end = time.time()
opq_train_add_time = np.round(end - start, 2)
print(f"Time to train and add to OPQ index: {opq_train_add_time} seconds")


start = time.time()
# Step 7: Compute approximate PQ distances
D_pq, I_pq = pq.search(qr_opq, nb)  # Approximate distances (PQ) to all DB vectors
end = time.time()   
opq_dist_time = np.round(end - start, 2)
print(f"Time to compute OPQ distances: {opq_dist_time} seconds")


start = time.time()
# Step 5: Calculate exact Euclidean distances using the flat index
index_flat = faiss.IndexFlatL2(dim)  # Exact L2 index
index_flat.add(db_opq)
end = time.time()

l2_add_time = np.round(end - start, 2)
print(f"Time to add to Flat index: {l2_add_time} seconds")

start = time.time()
# Step 6: Compute the distances (all distances between query and database)
D_exact, I_exact = index_flat.search(qr_opq, nb)  # Exact distances (Euclidean) to all DB vectors
end = time.time()

l2_dist_time = np.round(end - start, 2)
print(f"Time to compute exact distances: {l2_dist_time} seconds")

# Step 8: Compute relative error matrix
# Define a small epsilon value to replace zero distances
epsilon = 1e-6

# Replace zeros in D_exact with epsilon
D_exact_safe = np.where(D_exact == 0, epsilon, D_exact)

# Now compute the relative error
rel_error_matrix = np.abs(D_exact_safe - D_pq) / D_exact_safe

rel_error_mean = np.mean(rel_error_matrix)
rel_error_std = np.std(rel_error_matrix)


# Example usage
header = ['method', 'dataset', 'mean_rel_error', 'std_rel_error', 'qr_size', 'db_size', 'train_ratio', 'sample_method', 'train_add_time', 'dist_time', 'subspaces', 'centroids']
rows_to_append = [['OPQ', dataset_name, rel_error_mean, rel_error_std, nq, nb, train_size_ratio, sampling_method, opq_train_add_time, opq_dist_time, n_subquantizers, n_centroids]]

# Specify the CSV file name
file_name = '/data/cpanourg/2-hdvc/results/rel_error.csv'

# Call the function to append or create the file
append_or_create_csv(file_name, header, rows_to_append)
