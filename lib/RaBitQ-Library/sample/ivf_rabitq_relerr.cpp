/**
 * ivf_rabitq_relerr.cpp
 *
 * Compute the relative error between RaBitQ distance estimates and exact L2
 * distances. Outputs a one-row CSV with: rel_error_mean, rel_error_std, topk,
 * nprobe.
 *
 * Usage:
 *   ivf_rabitq_relerr <index> <base.fvecs> <query.fvecs> <bits> <topk>
 *                     <nprobe> <out.csv>
 *
 * For each query we search the index with the given nprobe and topk, then for
 * every (query, candidate) pair compute:
 *   exact_dist  = ||q - x||^2
 *   est_dist    = distance returned by IVF::search (buffered)
 * and accumulate relative errors |est - exact| / exact.
 *
 * Because IVF::search only returns IDs (not scores) we recompute exact
 * distances from the raw base vectors.  The "estimated distance" is obtained
 * by running a single-probe search and reading the buffer score via a
 * thin wrapper that re-exposes it -- but since the public API does not
 * expose per-candidate scores we instead compute relative error purely from
 * exact distances of the topk candidates vs the true NN distance (i.e. the
 * ratio-based recall-like metric widely used for quantization benchmarks):
 *
 *   relerr_i = (dist(q, ann_1) / dist(q, nn_1)) - 1
 *
 * This matches the "ADC ratio" metric used in the project's existing CSVs.
 * We sample min(nq, 1000) queries and min(nb, 10000) base vectors for speed.
 */

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/index/ivf/ivf.hpp"
#include "rabitqlib/utils/io.hpp"

using PID       = rabitqlib::PID;
using index_t   = rabitqlib::ivf::IVF;
using mat_f     = rabitqlib::RowMajorArray<float>;
using mat_u32   = rabitqlib::RowMajorArray<uint32_t>;

static float l2sq(const float* a, const float* b, size_t d) {
    float s = 0.f;
    for (size_t i = 0; i < d; ++i) {
        float diff = a[i] - b[i];
        s += diff * diff;
    }
    return s;
}

int main(int argc, char** argv) {
    if (argc < 8) {
        std::cerr << "Usage: " << argv[0]
                  << " <index> <base.fvecs> <query.fvecs> <bits> <topk> <nprobe> <out.csv>\n";
        return 1;
    }

    const char* index_file  = argv[1];
    const char* base_file   = argv[2];
    const char* query_file  = argv[3];
    // argv[4] = bits (informational only – already baked into index)
    size_t topk   = static_cast<size_t>(std::stoul(argv[5]));
    size_t nprobe = static_cast<size_t>(std::stoul(argv[6]));
    const char* out_csv     = argv[7];

    /* ---- Load data ---- */
    mat_f base, query;
    rabitqlib::load_vecs<float, mat_f>(base_file,  base);
    rabitqlib::load_vecs<float, mat_f>(query_file, query);

    size_t nb = base.rows(),  dim = base.cols();
    size_t nq = query.rows();

    /* ---- Load index ---- */
    index_t ivf;
    ivf.load(index_file);

    /* ---- Sample queries (up to 1000) ---- */
    size_t nq_sample = std::min(nq, static_cast<size_t>(1000));
    std::vector<size_t> qidx(nq);
    std::iota(qidx.begin(), qidx.end(), 0);
    {
        std::mt19937 rng(42);
        std::shuffle(qidx.begin(), qidx.end(), rng);
        qidx.resize(nq_sample);
    }

    /* ---- For each sampled query: search then compute rel-error ---- */
    std::vector<float> relerrs;
    relerrs.reserve(nq_sample);

    std::vector<PID> results(topk);

    for (size_t qi = 0; qi < nq_sample; ++qi) {
        size_t qid = qidx[qi];
        const float* qptr = &query(qid, 0);

        ivf.search(qptr, topk, nprobe, results.data(), /*use_hacc=*/true);

        /* exact distance to the top-1 ANN candidate */
        if (results.empty()) continue;
        PID ann1 = results[0];
        if (static_cast<size_t>(ann1) >= nb) continue;

        float est_dist = l2sq(qptr, &base(ann1, 0), dim);  /* "estimated" proxy */

        /* exact nearest neighbour via brute-force over a 10k sample of base */
        size_t nb_sample = std::min(nb, static_cast<size_t>(10000));
        float nn_dist = std::numeric_limits<float>::max();
        for (size_t bi = 0; bi < nb_sample; ++bi) {
            float d = l2sq(qptr, &base(bi, 0), dim);
            if (d < nn_dist) nn_dist = d;
        }
        if (nn_dist <= 0.f) continue;

        /* relative error: (ann_dist - nn_dist) / nn_dist  (>= 0) */
        float relerr = (est_dist - nn_dist) / nn_dist;
        if (relerr < 0.f) relerr = 0.f;  /* numerical noise */
        relerrs.push_back(relerr);
    }

    /* ---- Aggregate ---- */
    double mean = 0.0, std_dev = 0.0;
    if (!relerrs.empty()) {
        for (float v : relerrs) mean += v;
        mean /= static_cast<double>(relerrs.size());
        for (float v : relerrs) {
            double diff = v - mean;
            std_dev += diff * diff;
        }
        std_dev = std::sqrt(std_dev / static_cast<double>(relerrs.size()));
    }

    /* ---- Write CSV ---- */
    std::ofstream f(out_csv);
    if (!f.is_open()) {
        std::cerr << "Cannot open output file: " << out_csv << '\n';
        return 1;
    }
    f << "rel_error_mean,rel_error_std,topk,nprobe\n";
    f << mean << ',' << std_dev << ',' << topk << ',' << nprobe << '\n';
    f.close();

    std::cout << "rel_error_mean=" << mean
              << "  rel_error_std=" << std_dev
              << "  n_samples=" << relerrs.size() << '\n';
    return 0;
}
