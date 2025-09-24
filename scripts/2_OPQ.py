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


# Loading the GIST dataset
db = np.array(read_fvecs('/data/cpanourg/2-hdvc/data/gist/gist_base.fvecs'))
qr = np.array(read_fvecs('/data/cpanourg/2-hdvc/data/gist/gist_query.fvecs'))


dataset_name = 'GIST'
sampling_method = 'random'  # 'random'
train_size_ratio = 0.1  # Ratio of the database to use for training if using sampling
nb = db.shape[0]  # Number of database vectors
nq = qr.shape[0]  # Number of query vectors
k = 100 # Number of nearest neighbors to search for
dim = db.shape[1]  # Dimensionality of the vectors
nbits = 8 # 2^nbits is the number of centroids for each subquantizer
n_subquantizers = 8   # Number of subquantizers for the PQ


start = time.time()
opq = faiss.OPQMatrix(dim, n_subquantizers)
pq = faiss.IndexPQ(dim, n_subquantizers, nbits)

index = faiss.IndexPreTransform(opq, pq)

# If you have a separate training set, you should use that instead of the full database.
if sampling_method == 'random':
    num_samples = min(int(train_size_ratio * nb), nb)  # Number of samples for training
    training_data = db[np.random.choice(nb, num_samples, replace=False)]

# Perform training
index.train(training_data)
index.add(db)

end = time.time()
opq_train_add_time = np.round(end - start, 2)
print(f"Time to train and add to OPQ index: {opq_train_add_time} seconds")


start = time.time()
# Step 7: Compute approximate PQ distances
D_pq, I_pq = index.search(qr, k)  # Approximate distances (PQ) to all DB vectors
end = time.time()   
opq_dist_time = np.round(end - start, 2)
print(f"Time to compute OPQ distances: {opq_dist_time} seconds")


start = time.time()
# Step 5: Calculate exact Euclidean distances using the flat index
index_flat = faiss.IndexFlatL2(dim)  # Exact L2 index
index_flat.add(db)
end = time.time()

l2_add_time = np.round(end - start, 2)
print(f"Time to add to Flat index: {l2_add_time} seconds")

start = time.time()
# Step 6: Compute the distances (all distances between query and database)
D_exact, I_exact = index_flat.search(qr, k)  # Exact distances (Euclidean) to all DB vectors
end = time.time()

l2_dist_time = np.round(end - start, 2)
print(f"Time to compute exact distances: {l2_dist_time} seconds")

# Step 8: Compute relative error matrix

# D_exact_euclid = np.sqrt(np.maximum(D_exact, 0))
# D_pq_euclid = np.sqrt(np.maximum(D_pq, 0))

# epsilon = 1e-9

# # Replace zeros in D_exact with epsilon
# D_exact_safe = np.where(D_exact_euclid == 0, epsilon, D_exact_euclid)

# # Now compute the relative error
# rel_error_matrix = np.abs(D_exact_safe - D_pq_euclid) / D_exact_safe

# rel_error_mean = np.mean(rel_error_matrix)
# rel_error_std = np.std(rel_error_matrix)


# # ---------------------
# # Recall@k
# def recall_at_k(I_exact, I_approx, k):
#     nq = I_exact.shape[0]
#     hits = 0
#     for i in range(nq):
#         hits += len(set(I_exact[i, :k]) & set(I_approx[i, :k]))
#     return hits / (nq * k)

# recall1 = recall_at_k(I_exact, I_pq, 1)
# recall10 = recall_at_k(I_exact, I_pq, 10)
# recall100 = recall_at_k(I_exact, I_pq, 100)

# print(f"Recall@1: {recall1:.4f}")
# print(f"Recall@10: {recall10:.4f}")
# print(f"Recall@100: {recall100:.4f}")
# # ---------------------

def recall_at_k(I_true, I_test, k):
    n_queries = I_true.shape[0]
    recall = 0
    for i in range(n_queries):
        # True neighbors (I_true[i]) should be compared against the test neighbors (I_test[i])
        # Ensure you're comparing sets of indices in the top k results
        recall += len(set(I_true[i, :k]) & set(I_test[i, :k])) / k
    return recall / n_queries


recallk = recall_at_k(I_exact, I_pq, k)

print(f"Recall@{k}: {recallk:.4f}")



# Example usage
# header = ['method', 'dataset', 'mean_rel_error', 'std_rel_error', 'qr_size', 'db_size', 'train_ratio', 'sample_method', 'train_add_time', 'dist_time', 'subspaces', 'centroids']
# rows_to_append = [['OPQ', dataset_name, rel_error_mean, rel_error_std, nq, nb, train_size_ratio, sampling_method, opq_train_add_time, opq_dist_time, n_subquantizers, n_centroids]]


# Example usage
header = [
    'method', 'dataset',
    # 'mean_rel_error', 'std_rel_error',
    f'recall@{k}',
    'qr_size', 'db_size', 'train_ratio', 'sample_method',
    'train_add_time', 'dist_time', 'subspaces', 'centroids', 'bits'
]

rows_to_append = [[
    'OPQ', dataset_name,
    # rel_error_mean, rel_error_std,
    recallk,
    nq, nb, train_size_ratio, sampling_method,
    opq_train_add_time, opq_dist_time,
    n_subquantizers, 2**nbits, nbits
]]

# Specify the CSV file name
file_name = '/data/cpanourg/2-hdvc/results/rel_error.csv'

# Call the function to append or create the file
append_or_create_csv(file_name, header, rows_to_append)
