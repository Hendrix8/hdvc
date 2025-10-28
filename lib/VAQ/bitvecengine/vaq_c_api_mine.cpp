#include "VAQ.hpp"
#include <vector>
#include <cstdint>
#include <cstring>
extern "C" {

// Quantize an array (float32[N * D]) into codes (uint8)
// The caller owns output memory
void vaq_quantize(float* data, int n, int d, int nbits, int nclusters, uint8_t* out_codes) {
    // Example pseudo-interface; you need to adapt this to your VAQ class
    VAQ vaq;
    vaq.train(data, n, d, nbits, nclusters);
    vaq.encode(data, out_codes);
}
}
