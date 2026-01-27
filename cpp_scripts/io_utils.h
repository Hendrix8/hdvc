#ifndef IO_UTILS_H
#define IO_UTILS_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cstring>

// Read .fvecs file format
// Format: Each vector is [dim (int32), data (dim * float32)]
std::vector<std::vector<float>> read_fvecs(const std::string& filename, int max_vectors = -1);

// Read .bin (fbin) file format  
// Format: [nvecs (int32), dim (int32), data (nvecs * dim * float32)]
std::vector<std::vector<float>> read_fbin(const std::string& filename, int start_idx = 0, int chunk_size = -1);

// Auto-detect file format and read
std::vector<std::vector<float>> load_dataset(const std::string& filename, int dim = -1, int max_vectors = -1);

#endif // IO_UTILS_H

