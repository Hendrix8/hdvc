#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/index/ivf/ivf.hpp"
#include "rabitqlib/utils/io.hpp"

using PID = rabitqlib::PID;
using index_type = rabitqlib::ivf::IVF;
using data_type = rabitqlib::RowMajorArray<float>;

static float l2sq(const float* a, const float* b, size_t d) {
    float s = 0.0f;
    for (size_t i = 0; i < d; ++i) {
        float diff = a[i] - b[i];
        s += diff * diff;
    }
    return s;
}

int main(int argc, char** argv) {
    if (argc < 8) {
        std::cerr << "Usage: " << argv[0]
                  << " <index> <base.fvecs> <query.fvecs> <bits> <topk> <nprobe> <out.csv>"
                  << " [n_runs] [warmup_runs] [use_hacc=true|false]\n";
        return 1;
    }

    const char* index_file = argv[1];
    const char* base_file = argv[2];
    const char* query_file = argv[3];
    const size_t bits = static_cast<size_t>(std::stoul(argv[4]));
    const size_t topk = static_cast<size_t>(std::stoul(argv[5]));
    const size_t nprobe = static_cast<size_t>(std::stoul(argv[6]));
    const char* out_csv = argv[7];
    const size_t n_runs = argc > 8 ? static_cast<size_t>(std::stoul(argv[8])) : 5;
    const size_t warmup_runs = argc > 9 ? static_cast<size_t>(std::stoul(argv[9])) : 2;
    bool use_hacc = true;
    if (argc > 10) {
        std::string flag(argv[10]);
        if (flag == "false") {
            use_hacc = false;
        }
    }

    data_type base;
    data_type query;
    rabitqlib::load_vecs<float, data_type>(base_file, base);
    rabitqlib::load_vecs<float, data_type>(query_file, query);

    const size_t nb = base.rows();
    const size_t nq = query.rows();
    const size_t dim = base.cols();

    index_type ivf;
    ivf.load(index_file);

    const size_t effective_topk = std::min(topk, nb);
    std::vector<PID> ids(effective_topk);
    std::vector<float> approx(effective_topk);
    std::vector<double> relerrs;
    relerrs.reserve(nq * effective_topk);

    for (size_t qi = 0; qi < nq; ++qi) {
        const float* qptr = &query(qi, 0);
        std::fill(ids.begin(), ids.end(), static_cast<PID>(0));
        std::fill(approx.begin(), approx.end(), std::numeric_limits<float>::infinity());
        ivf.search(qptr, effective_topk, nprobe, ids.data(), approx.data(), use_hacc);
        for (size_t j = 0; j < effective_topk; ++j) {
            PID id = ids[j];
            if (!(approx[j] == approx[j]) || approx[j] > 1e30f) {
                continue;
            }
            if (static_cast<size_t>(id) >= nb) {
                continue;
            }
            float exact = l2sq(qptr, &base(id, 0), dim);
            if (!(exact == exact) || exact <= 1e-9f) {
                continue;
            }
            double denom = static_cast<double>(exact);
            double rel = std::abs(static_cast<double>(approx[j]) - static_cast<double>(exact)) / denom;
            if (rel == rel && rel < 1e20) {
                relerrs.push_back(rel);
            }
        }
    }

    double rel_mean = 0.0;
    double rel_std = 0.0;
    if (!relerrs.empty()) {
        rel_mean = std::accumulate(relerrs.begin(), relerrs.end(), 0.0) /
                   static_cast<double>(relerrs.size());
        for (double v : relerrs) {
            double diff = v - rel_mean;
            rel_std += diff * diff;
        }
        rel_std = std::sqrt(rel_std / static_cast<double>(relerrs.size()));
    }

    for (size_t r = 0; r < warmup_runs; ++r) {
        for (size_t qi = 0; qi < nq; ++qi) {
            ivf.search(&query(qi, 0), effective_topk, nprobe, ids.data(), use_hacc);
        }
    }

    std::vector<double> run_seconds;
    run_seconds.reserve(std::max<size_t>(n_runs, 1));
    for (size_t r = 0; r < std::max<size_t>(n_runs, 1); ++r) {
        auto start = std::chrono::steady_clock::now();
        for (size_t qi = 0; qi < nq; ++qi) {
            ivf.search(&query(qi, 0), effective_topk, nprobe, ids.data(), use_hacc);
        }
        auto end = std::chrono::steady_clock::now();
        run_seconds.push_back(std::chrono::duration<double>(end - start).count());
    }

    double adc_time_mean = 0.0;
    double adc_time_std = 0.0;
    if (!run_seconds.empty()) {
        adc_time_mean =
            std::accumulate(run_seconds.begin(), run_seconds.end(), 0.0) /
            static_cast<double>(run_seconds.size());
        for (double v : run_seconds) {
            double diff = v - adc_time_mean;
            adc_time_std += diff * diff;
        }
        adc_time_std = std::sqrt(adc_time_std / static_cast<double>(run_seconds.size()));
    }

    double per_query_us = nq > 0 ? (adc_time_mean * 1e6 / static_cast<double>(nq)) : 0.0;
    double per_pair_ns =
        (nq > 0 && effective_topk > 0)
            ? (adc_time_mean * 1e9 / static_cast<double>(nq * effective_topk))
            : 0.0;

    std::ofstream out(out_csv);
    if (!out.is_open()) {
        std::cerr << "Cannot open output file: " << out_csv << '\n';
        return 1;
    }
    out << "bits,topk,nprobe,nq,nb,rel_error_mean,rel_error_std,"
           "adc_time_s_mean,adc_time_s_std,per_query_us_mean,per_pair_ns_mean,"
           "n_runs,warmup_runs,use_hacc\n";
    out << bits << ',' << effective_topk << ',' << nprobe << ',' << nq << ',' << nb << ','
        << rel_mean << ',' << rel_std << ',' << adc_time_mean << ',' << adc_time_std << ','
        << per_query_us << ',' << per_pair_ns << ',' << n_runs << ',' << warmup_runs << ','
        << (use_hacc ? "true" : "false") << '\n';
    out.close();

    std::cout << "bits=" << bits << " rel_error_mean=" << rel_mean
              << " adc_time_s_mean=" << adc_time_mean << " nprobe=" << nprobe << '\n';
    return 0;
}
