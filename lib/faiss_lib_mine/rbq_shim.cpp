// rbq_shim.cpp — C ABI wrapper around Faiss RaBitQ for ctypes
#include <cstdint>
#include <cstring>
#include <memory>

#include <faiss/impl/RaBitQuantizer.h>
#include <faiss/impl/DistanceComputer.h>  // FlatCodesDistanceComputer
#include <faiss/MetricType.h>

#if defined(_WIN32)
  #define RBQ_API __declspec(dllexport)
#else
  #define RBQ_API __attribute__((visibility("default")))
#endif

extern "C" {

// Return the per-vector code size (bytes) for dimension d
RBQ_API size_t rbq_code_size(size_t d) {
    faiss::RaBitQuantizer q(d, faiss::METRIC_L2);
    return q.code_size; // includes bits + FactorsData
}

// Encode a batch X[n,d] (row-major float32) into RaBitQ codes
// codes must be preallocated with size n * rbq_code_size(d) bytes.
RBQ_API int rbq_encode_batch(
    const float* X, int64_t n, int64_t d,
    const float* centroid,            // nullptr if no centering
    uint8_t* codes,                   // out: n * code_size bytes
    int metric_type                   // 0=L2, 1=IP
) {
    if (!X || !codes || n <= 0 || d <= 0) return 2;
    faiss::MetricType mt =
        (metric_type == 1) ? faiss::METRIC_INNER_PRODUCT : faiss::METRIC_L2;

    faiss::RaBitQuantizer q((size_t)d, mt);
    // q.train() is a no-op here; we just pass a centroid if you have one
    q.compute_codes_core(X, codes, (size_t)n, centroid); // uses provided centroid
    return 0;
}

// Compute L2 (or IP) distances from a single query to a block of codes.
// codes: n vectors, each code_size bytes (same size as rbq_code_size(d))
// out_dist: float32[n]
RBQ_API int rbq_l2_distances_qbits(
    const float* query, int64_t d,
    const uint8_t* codes, int64_t n,
    const float* centroid,           // nullptr if no centering
    int qb,                          // 0 => float query; 1..8 => quantized query bits
    int centered,                    // 0/1: use centered query quantizer mode
    int metric_type,                 // 0=L2, 1=IP
    float* out_dist                  // out: distances (size n)
) {
    if (!query || !codes || !out_dist || n <= 0 || d <= 0) return 2;

    faiss::MetricType mt =
        (metric_type == 1) ? faiss::METRIC_INNER_PRODUCT : faiss::METRIC_L2;
    faiss::RaBitQuantizer q((size_t)d, mt);

    // Build a distance computer (qb==0 -> float query path; else quantized query with popcounts)
    std::unique_ptr<faiss::FlatCodesDistanceComputer> dc(
        q.get_distance_computer((uint8_t)qb, centroid, centered != 0));
    if (!dc) return 3;

    dc->set_query(query);

    size_t code_size = q.code_size;
    for (int64_t i = 0; i < n; ++i) {
        const uint8_t* code_i = codes + i * code_size;
        out_dist[i] = dc->distance_to_code(code_i);
    }
    return 0;
}

} // extern "C"
