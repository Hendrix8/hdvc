export LD_LIBRARY_PATH=/home/cpanourg/projects/2-hdvc/local/openblas/lib:$LD_LIBRARY_PATH
make FAISS_INCLUDE=/home/cpanourg/projects/2-hdvc/lib/faiss \
     FAISS_LIB_PATH=/home/cpanourg/projects/2-hdvc/lib/faiss/build/faiss \
     OPENBLAS_PREFIX=/home/cpanourg/projects/2-hdvc/local/openblas