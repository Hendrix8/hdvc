# Makefile for opq_eval C++ program (GPU training)

# FAISS paths - requires FAISS built with -DFAISS_ENABLE_GPU=ON
FAISS_INCLUDE = /home/cpanourg/projects/2-hdvc/lib/faiss
FAISS_LIB_PATH = /home/cpanourg/projects/2-hdvc/lib/faiss/build/faiss

# CUDA paths (for GPU-enabled FAISS)
CUDA_HOME ?= /usr/local/cuda
CUDA_INCLUDES = -I$(CUDA_HOME)/include
CUDA_LDFLAGS = -L$(CUDA_HOME)/lib64 -lcudart -lcublas

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -march=native
INCLUDES = -I$(FAISS_INCLUDE) $(CUDA_INCLUDES) -I..
LDFLAGS = -L$(FAISS_LIB_PATH) -lfaiss -fopenmp $(CUDA_LDFLAGS) -lrt

# OpenBLAS (local or conda)
CONDA_PREFIX ?= $(HOME)/.miniconda3
OPENBLAS_PREFIX ?= $(CONDA_PREFIX)
LDFLAGS += -L$(OPENBLAS_PREFIX)/lib -lopenblas -lgfortran

# Source files
SOURCES = opq_eval.cpp ../io_utils.cpp
OBJECTS = opq_eval.o io_utils.o
TARGET = opq_eval

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS)

# Compile source files
opq_eval.o: opq_eval.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

io_utils.o: ../io_utils.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET)

# Check if FAISS (GPU) is found
check:
	@echo "Checking FAISS GPU installation..."
	@test -f $(FAISS_INCLUDE)/faiss/VectorTransform.h || (echo "Error: VectorTransform.h not found" && exit 1)
	@test -f $(FAISS_INCLUDE)/faiss/gpu/GpuIndexFlat.h || (echo "Error: FAISS GPU headers not found. Rebuild FAISS with -DFAISS_ENABLE_GPU=ON" && exit 1)
	@test -f $(FAISS_LIB_PATH)/libfaiss.a || (echo "Error: libfaiss.a not found" && exit 1)
	@echo "✓ FAISS GPU headers found"
	@echo "✓ FAISS library found"

# Help
help:
	@echo "Available targets:"
	@echo "  all     - Build opq_eval (default)"
	@echo "  clean   - Remove build artifacts"
	@echo "  check   - Verify FAISS installation"
	@echo "  help    - Show this help message"
	@echo ""
	@echo "Usage:"
	@echo "  make                    # Build opq_eval"
	@echo "  make check              # Verify FAISS paths"
	@echo "  make clean              # Clean build files"
	@echo ""
	@echo "To override paths:"
	@echo "  make FAISS_INCLUDE=/path/to/faiss/include"
	@echo "  make FAISS_LIB_PATH=/path/to/faiss/lib"
	@echo "  make CUDA_HOME=/path/to/cuda"
	@echo "  make OPENBLAS_PREFIX=/path/to/openblas"
	@echo ""
	@echo "Prerequisite: Rebuild FAISS with GPU support:"
	@echo "  cd lib/faiss && cmake -B build -DFAISS_ENABLE_GPU=ON ..."

.PHONY: all clean check help

