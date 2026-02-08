# Rebuild FAISS with GPU support (for gpu_cpp_scripts)
cd /home/cpanourg/projects/2-hdvc/lib/faiss
cmake -B build -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF -DCMAKE_PREFIX_PATH=$CONDA_PREFIX .
make -C build -j$(nproc)
cd /home/cpanourg/projects/2-hdvc/gpu_cpp_scripts