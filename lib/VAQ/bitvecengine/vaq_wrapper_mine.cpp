#include "VAQ.hpp"
#include <Eigen/Dense>
#include <cstdint>
#include <vector>
#include <cstring>

extern "C" {

// --- Train and encode a dataset, returning codebook as uint16 array --- //
void vaq_train_and_encode(
    float* data, int n, int d,
    int total_bits, int n_subspaces,
    int min_bits, int max_bits,
    float var_threshold,
    uint16_t* out_codes
) {
    // 1. Wrap input data in Eigen view
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> XTrain(data, n, d);

    // 2. Create VAQ instance
    VAQ vaq;
    std::string method = "VAQ" + std::to_string(total_bits) +
                         "m" + std::to_string(n_subspaces) +
                         "min" + std::to_string(min_bits) +
                         "max" + std::to_string(max_bits) +
                         "var" + std::to_string(var_threshold);
    vaq.parseMethodString(method);

    // 3. Train
    vaq.train(XTrain, false);

    // 4. Encode
    vaq.encode(XTrain);

    // 5. Get resulting codebook (rows × cols)
    const auto& codes = vaq.mCodebook;
    std::memcpy(out_codes, codes.data(), codes.size() * sizeof(uint16_t));
}

}
