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
using IVFT = rabitqlib::ivf::IVF;
using DT = rabitqlib::RowMajorArray<float>;
static float l2sq(const float* a, const float* b, size_t d) {
    float s = 0;
    for (size_t i = 0; i < d; ++i) { float df = a[i] - b[i]; s += df * df; }
    return s;
}
int main(int argc, char** argv) {
    if (argc < 8) return 1;
    const char* idxf = argv[1];
    const char* basef = argv[2];
    const char* qryf = argv[3];
    size_t bits = std::stoul(argv[4]);
    size_t topk = std::stoul(argv[5]);
    size_t nprobe = std::stoul(argv[6]);
    const char* out_csv = argv[7];
    size_t nruns = argc>8 ? std::stoul(argv[8]) : 5;
    size_t warm = argc>9 ? std::stoul(argv[9]) : 2;
    bool hacc = argc<=10 || std::string(argv[10])!="false";
    DT base, query;
    rabitqlib::load_vecs<float,DT>(basef, base);
    rabitqlib::load_vecs<float,DT>(qryf, query);
    size_t nb = base.rows(), nq = query.rows(), dim = base.cols();
    IVFT ivf;
    ivf.load(idxf);
    size_t eftopk = std::min(topk, nb);
    std::vector<PID> ids(eftopk);
    std::vector<float> approx(eftopk);
    std::vector<double> relerrs;
    // First pass: full search for rel_error
    for (size_t qi = 0; qi < nq; ++qi) {
        const float* qp = &query(qi,0);
        ivf.search(qp, eftopk, nprobe, ids.data(), approx.data(), hacc);
        for (size_t j=0; j<eftopk; ++j) {
            PID id = ids[j];
            if (j>=nb || !(approx[j]==approx[j]) || approx[j]>1e30f || (size_t)id>=nb) continue;
            float ex = l2sq(qp, &base(id,0), dim);
            if (!(ex==ex) || ex<=1e-9f) continue;
            double den = ex;
            double r = std::abs((double)approx[j]-den)/den;
            if (r==r && r<1e20) relerrs.push_back(r);
        }
    }
    double rm=0, rs=0;
    if (!relerrs.empty()) { for (double v:relerrs) rm+=v; rm/=(double)relerrs.size();
        for (double v:relerrs) {double d=v-rm; rs+=d*d;} rs=std::sqrt(rs/(double)relerrs.size());}
    // Warmup with flat_scan
    for (size_t r=0; r<warm; ++r)
        for (size_t qi=0; qi<nq; ++qi)
            ivf.flat_scan(&query(qi,0), eftopk, ids.data(), approx.data(), hacc);
    // Timing: flat_scan (rotate + scan all clusters, no centroid search)
    std::vector<double> secs;
    for (size_t r=0; r<std::max<size_t>(nruns,1); ++r) {
        auto st = std::chrono::steady_clock::now();
        for (size_t qi=0; qi<nq; ++qi)
            ivf.flat_scan(&query(qi,0), eftopk, ids.data(), approx.data(), hacc);
        auto en = std::chrono::steady_clock::now();
        secs.push_back(std::chrono::duration<double>(en-st).count());
    }
    double am=0, as=0;
    if (!secs.empty()) { for (double v:secs) am+=v; am/=(double)secs.size();
        for (double v:secs) {double d=v-am; as+=d*d;} as=std::sqrt(as/(double)secs.size());}
    double pqu = nq>0 ? (am*1e6/(double)nq) : 0;
    double ppn = (nq>0 && eftopk>0) ? (am*1e9/(double)(nq*eftopk)) : 0;
    std::ofstream out(out_csv);
    if (!out.is_open()) { std::cerr<<"Cannot open "<<out_csv<<"\n"; return 1; }
    out<<"bits,topk,nprobe,nq,nb,rel_error_mean,rel_error_std,"
          "adc_time_s_mean,adc_time_s_std,per_query_us_mean,per_pair_ns_mean,"
          "n_runs,warmup_runs,use_hacc\n";
    out<<bits<<","<<eftopk<<","<<nprobe<<","<<nq<<","<<nb<<","
       <<rm<<","<<rs<<","<<am<<","<<as<<","<<pqu<<","<<ppn<<","
       <<nruns<<","<<warm<<","<<(hacc?"true":"false")<<"\n";
    out.close();
    std::cout<<"bits="<<bits<<" rel="<<rm<<" adc_flat="<<am<<" nprobe="<<nprobe<<"\n";
    return 0;
}
