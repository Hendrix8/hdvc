#include "io_utils.h"
#include <algorithm>
#include <cstdint>

std::vector<std::vector<float>> read_fvecs(const std::string& filename, int max_vectors) {
    std::vector<std::vector<float>> data;
    std::ifstream file(filename, std::ios::binary);
    
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return data;
    }
    
    int count = 0;
    while (file.good() && (max_vectors < 0 || count < max_vectors)) {
        int dim;
        file.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
        
        if (!file.good() || file.eof()) {
            break;
        }
        
        std::vector<float> vec(dim);
        file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(float));
        
        if (!file.good()) {
            break;
        }
        
        data.push_back(std::move(vec));
        count++;
    }
    
    file.close();
    return data;
}

std::vector<std::vector<float>> read_fbin(const std::string& filename, int start_idx, int chunk_size) {
    std::vector<std::vector<float>> data;
    std::ifstream file(filename, std::ios::binary);
    
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return data;
    }
    
    // Read header: nvecs, dim
    int32_t nvecs, dim;
    file.read(reinterpret_cast<char*>(&nvecs), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
    
    if (!file.good()) {
        std::cerr << "Error: Cannot read header from " << filename << std::endl;
        file.close();
        return data;
    }
    
    // Determine how many vectors to read
    int actual_chunk_size = (chunk_size < 0) ? (nvecs - start_idx) : chunk_size;
    actual_chunk_size = std::min(actual_chunk_size, nvecs - start_idx);
    
    if (start_idx < 0 || start_idx >= nvecs || actual_chunk_size <= 0) {
        file.close();
        return data;
    }
    
    // Seek to start position
    file.seekg(sizeof(int32_t) * 2 + start_idx * dim * sizeof(float), std::ios::beg);
    
    // Read vectors
    for (int i = 0; i < actual_chunk_size; i++) {
        std::vector<float> vec(dim);
        file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(float));
        
        if (!file.good()) {
            break;
        }
        
        data.push_back(std::move(vec));
    }
    
    file.close();
    return data;
}

std::vector<std::vector<float>> read_bvecs(const std::string& filename, int max_vectors) {
    std::vector<std::vector<float>> data;
    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return data;
    }

    int count = 0;
    while (file.good() && (max_vectors < 0 || count < max_vectors)) {
        int32_t dim;
        file.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));

        if (!file.good() || file.eof()) {
            break;
        }

        std::vector<uint8_t> raw(dim);
        file.read(reinterpret_cast<char*>(raw.data()), dim * sizeof(uint8_t));

        if (!file.good()) {
            break;
        }

        std::vector<float> vec(dim);
        for (int i = 0; i < dim; i++) {
            vec[i] = static_cast<float>(raw[i]);
        }
        data.push_back(std::move(vec));
        count++;
    }

    file.close();
    return data;
}

std::vector<std::vector<float>> load_dataset(const std::string& filename, int dim, int max_vectors) {
    (void)dim; // Parameter kept for API compatibility but not used for format detection
    // Try to detect file format by extension
    std::string ext = filename.substr(filename.find_last_of(".") + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    if (ext == "fvecs") {
        return read_fvecs(filename, max_vectors);
    } else if (ext == "bin" || ext == "fbin") {
        return read_fbin(filename, 0, max_vectors);
    } else if (ext == "bvecs") {
        return read_bvecs(filename, max_vectors);
    } else {
        // Try fvecs first, then fbin
        auto data = read_fvecs(filename, max_vectors);
        if (data.empty()) {
            data = read_fbin(filename, 0, max_vectors);
        }
        return data;
    }
}

