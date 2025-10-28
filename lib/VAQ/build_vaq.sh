#!/usr/bin/env bash
cd ~/projects/2-hdvc/lib/VAQ
rm -rf build
mkdir build && cd build

cmake .. \
  -DCMAKE_CXX_STANDARD=14 \
  -DCMAKE_PREFIX_PATH="$HOME/local/armadillo;$HOME/local/glpk;$CONDA_PREFIX" \
  -DGLPK_INCLUDE_DIR=$HOME/local/glpk/include \
  -DGLPK_LIBRARY=$HOME/local/glpk/lib/libglpk.so \
  -DBLAS_LIBRARIES=$CONDA_PREFIX/lib/libopenblas.so \
  -DLAPACK_LIBRARIES=$CONDA_PREFIX/lib/libopenblas.so \
  -DLAPACKE_LIBRARIES=$CONDA_PREFIX/lib/liblapacke.so \
  -DARMA_DONT_USE_WRAPPER=ON \
  -DCMAKE_EXE_LINKER_FLAGS="-L$CONDA_PREFIX/lib -L$HOME/local/glpk/lib -L$HOME/local/armadillo/lib"

make -j