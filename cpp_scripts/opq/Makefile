# Makefile for opq_eval C++ program

# FAISS paths (adjust these if needed)
FAISS_INCLUDE = /home/cpanourg/projects/2-hdvc/lib/faiss
FAISS_LIB_PATH = /home/cpanourg/projects/2-hdvc/lib/faiss/build/faiss

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -march=native
INCLUDES = -I$(FAISS_INCLUDE) -I..
LDFLAGS = -L$(FAISS_LIB_PATH) -lfaiss -fopenmp

# OpenBLAS (local or conda)
CONDA_PREFIX ?= $(HOME)/.miniconda3
OPENBLAS_PREFIX ?= $(CONDA_PREFIX)
LDFLAGS += -L$(OPENBLAS_PREFIX)/lib -lopenblas

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

# Check if FAISS is found
check:
	@echo "Checking FAISS installation..."
	@test -f $(FAISS_INCLUDE)/faiss/VectorTransform.h || (echo "Error: VectorTransform.h not found at $(FAISS_INCLUDE)/faiss/VectorTransform.h" && exit 1)
	@test -f $(FAISS_LIB_PATH)/libfaiss.a || (echo "Error: libfaiss.a not found at $(FAISS_LIB_PATH)/libfaiss.a" && exit 1)
	@echo "✓ FAISS headers found"
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
	@echo "  make OPENBLAS_PREFIX=/path/to/openblas"

.PHONY: all clean check help

