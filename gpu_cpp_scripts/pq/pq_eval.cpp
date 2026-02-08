#include <faiss/IndexPQ.h>
#include <faiss/index_io.h>
#include <faiss/utils/distances.h>
#include <faiss/impl/ProductQuantizer-inl.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include "../io_utils.h"

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <sstream>
#include <filesystem>

namespace fs = std::filesystem;

// Convert vector of vectors to flat array
std::vector<float> flatten_vectors(const std::vector<std::vector<float>>& vecs) {
    if (vecs.empty()) return {};
    size_t dim = vecs[0].size();
    std::vector<float> flat(vecs.size() * dim);
    for (size_t i = 0; i < vecs.size(); i++) {
        std::memcpy(flat.data() + i * dim, vecs[i].data(), dim * sizeof(float));
    }
    return flat;
}

// Compute exact squared Euclidean distances
void compute_exact_distances(
    const float* queries, size_t nq, 
    const float* database, size_t nb, 
    size_t dim, 
    float* distances) {
    
    for (size_t i = 0; i < nq; i++) {
        const float* q = queries + i * dim;
        for (size_t j = 0; j < nb; j++) {
            const float* d = database + j * dim;
            float dist = 0.0f;
            for (size_t k = 0; k < dim; k++) {
                float diff = q[k] - d[k];
                dist += diff * diff;
            }
            // Clip to prevent overflow
            const float max_safe = std::numeric_limits<float>::max() / 10.0f;
            distances[i * nb + j] = std::min(dist, max_safe);
        }
    }
}

// Compute ADC distances using distance tables and codes
void compute_adc_distances(
    const float* dis_tables,  // nq * M * ksub
    const uint8_t* codes,     // nb * code_size
    size_t nq, size_t nb,
    size_t M, size_t ksub,
    size_t code_size,
    size_t nbits,
    float* adc_distances) {
    
    for (size_t i = 0; i < nq; i++) {
        const float* table = dis_tables + i * M * ksub;
        for (size_t j = 0; j < nb; j++) {
            const uint8_t* code = codes + j * code_size;
            float dist = 0.0f;
            
            // Decode code based on nbits
            if (nbits == 8) {
                // Simple case: each subquantizer index is one byte
                for (size_t m = 0; m < M; m++) {
                    uint8_t idx = code[m];
                    dist += table[m * ksub + idx];
                }
            } else {
                // General case: use PQDecoder to decode indices
                faiss::PQDecoderGeneric decoder(code, nbits);
                for (size_t m = 0; m < M; m++) {
                    uint64_t idx = decoder.decode();
                    dist += table[m * ksub + idx];
                }
            }
            
            // Clip to prevent overflow
            const float max_safe = std::numeric_limits<float>::max() / 10.0f;
            adc_distances[i * nb + j] = std::min(dist, max_safe);
        }
    }
}

// Compute relative error
void compute_relative_error(
    const float* adc_dist, const float* exact_dist,
    size_t nq, size_t nb,
    float& mean_rel, float& std_rel) {
    
    std::vector<double> rel_errors;
    const double epsilon = 1e-6;
    
    for (size_t i = 0; i < nq * nb; i++) {
        double exact = static_cast<double>(exact_dist[i]);
        double adc = static_cast<double>(adc_dist[i]);
        double denominator = std::max(exact, epsilon);
        double diff = std::abs(adc - exact);
        double rel_err = diff / denominator;
        
        // Filter valid values
        if (std::isfinite(rel_err) && rel_err >= 0 && rel_err < 1e6) {
            rel_errors.push_back(rel_err);
        }
    }
    
    if (rel_errors.empty()) {
        mean_rel = std::nan("");
        std_rel = std::nan("");
        return;
    }
    
    // Compute mean
    double sum = 0.0;
    for (double err : rel_errors) {
        sum += err;
    }
    mean_rel = static_cast<float>(sum / rel_errors.size());
    
    // Compute std
    double sum_sq_diff = 0.0;
    for (double err : rel_errors) {
        double diff = err - mean_rel;
        sum_sq_diff += diff * diff;
    }
    std_rel = static_cast<float>(std::sqrt(sum_sq_diff / rel_errors.size()));
}

// Save results to CSV
void save_summary_csv(
    const std::string& csv_path,
    const std::string& dataset_name,
    const std::string& experiment_folder,
    size_t nq, size_t nb, size_t nb_sample,
    size_t dim, size_t M, size_t nbits,
    size_t train_size,
    double train_time, double encoding_time,
    double distance_table_time, double cdist_time, double adc_time,
    float mean_rel, float std_rel) {
    
    bool need_header = !fs::exists(csv_path) || fs::file_size(csv_path) == 0;
    
    std::ofstream file(csv_path, std::ios::app);
    if (need_header) {
        file << "method,dataset,experiment_folder,nq,nb,nb_sample,dim,n_subquantizers,nbits,"
             << "bits_per_vector,train_size,train_time_s,encoding_time_s,distance_table_time_s,"
             << "cdist_time_s,adc_time_s,rel_error_mean,rel_error_std\n";
    }
    
    file << "PQ," << dataset_name << "," << experiment_folder << ","
         << nq << "," << nb << "," << nb_sample << "," << dim << ","
         << M << "," << nbits << "," << (M * nbits) << ","
         << train_size << ","
         << std::fixed << std::setprecision(6)
         << train_time << "," << encoding_time << ","
         << distance_table_time << "," << cdist_time << ","
         << adc_time << ",";
    
    if (std::isnan(mean_rel)) {
        file << "nan,";
    } else {
        file << mean_rel << ",";
    }
    
    if (std::isnan(std_rel)) {
        file << "nan\n";
    } else {
        file << std_rel << "\n";
    }
    
    file.close();
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0] << " --dataset_path PATH --query_path PATH --train_path PATH "
                  << "[--dim DIM] [--dataset_name NAME] [--data_root ROOT] "
                  << "[--n_subquantizers M] [--nbits BITS] [--train_size SIZE] "
                  << "[--sample_db SIZE] [--sample_queries SIZE] [--results_dir DIR] "
                  << "[--load_model] [--model_path PATH] [--gpu_device DEVICE]\n";
        return 1;
    }
    
    // Default parameters
    std::string dataset_path, query_path, train_path;
    int dim = -1;
    std::string dataset_name = "custom";
    std::string data_root = "/data/cpanourg/2-hdvc/";
    size_t M = 32;
    size_t nbits = 8;
    size_t train_size = 100000;
    size_t sample_db = 10000;
    size_t sample_queries = 1000;
    std::string results_dir = "results/relerr";
    bool load_model = false;
    std::string model_path;
    int gpu_device = 0;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--dataset_path" && i + 1 < argc) {
            dataset_path = argv[++i];
        } else if (arg == "--query_path" && i + 1 < argc) {
            query_path = argv[++i];
        } else if (arg == "--train_path" && i + 1 < argc) {
            train_path = argv[++i];
        } else if (arg == "--dim" && i + 1 < argc) {
            dim = std::stoi(argv[++i]);
        } else if (arg == "--dataset_name" && i + 1 < argc) {
            dataset_name = argv[++i];
        } else if (arg == "--data_root" && i + 1 < argc) {
            data_root = argv[++i];
        } else if (arg == "--n_subquantizers" && i + 1 < argc) {
            M = std::stoul(argv[++i]);
        } else if (arg == "--nbits" && i + 1 < argc) {
            nbits = std::stoul(argv[++i]);
        } else if (arg == "--train_size" && i + 1 < argc) {
            train_size = std::stoul(argv[++i]);
        } else if (arg == "--sample_db" && i + 1 < argc) {
            sample_db = std::stoul(argv[++i]);
        } else if (arg == "--sample_queries" && i + 1 < argc) {
            sample_queries = std::stoul(argv[++i]);
        } else if (arg == "--results_dir" && i + 1 < argc) {
            results_dir = argv[++i];
        } else if (arg == "--load_model") {
            load_model = true;
        } else if (arg == "--model_path" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--gpu_device" && i + 1 < argc) {
            gpu_device = std::stoi(argv[++i]);
        }
    }
    
    if (dataset_path.empty() || query_path.empty() || train_path.empty()) {
        std::cerr << "Error: dataset_path, query_path, and train_path are required\n";
        return 1;
    }
    
    if (load_model && model_path.empty()) {
        std::cerr << "Error: model_path is required when --load_model is set\n";
        return 1;
    }
    
    // Load datasets
    std::cout << "📂 Loading database from " << dataset_path << std::endl;
    auto db_vecs = load_dataset(dataset_path, dim, 1000000);
    if (db_vecs.empty()) {
        std::cerr << "Error: Failed to load database\n";
        return 1;
    }
    std::cout << "Loaded database: " << db_vecs.size() << " vectors" << std::endl;
    
    std::cout << "📂 Loading training data from " << train_path << std::endl;
    auto train_db_full = load_dataset(train_path, dim, train_size + 10000);
    if (train_db_full.empty()) {
        std::cerr << "Error: Failed to load training data\n";
        return 1;
    }
    std::cout << "Loaded training data: " << train_db_full.size() << " vectors" << std::endl;
    
    std::cout << "📂 Loading queries from " << query_path << std::endl;
    auto qr_vecs = load_dataset(query_path, dim, sample_queries);
    if (qr_vecs.empty()) {
        std::cerr << "Error: Failed to load queries\n";
        return 1;
    }
    std::cout << "Loaded queries: " << qr_vecs.size() << " vectors" << std::endl;
    
    // Get dimensions
    size_t actual_dim = db_vecs[0].size();
    if (dim > 0 && static_cast<size_t>(dim) != actual_dim) {
        std::cerr << "Warning: Dimension mismatch. Expected " << dim 
                  << ", got " << actual_dim << std::endl;
    }
    dim = actual_dim;
    
    // Prepare training set
    size_t actual_train_size = std::min(train_size, train_db_full.size());
    std::vector<std::vector<float>> train_db(
        train_db_full.begin(), 
        train_db_full.begin() + actual_train_size);
    
    // Prepare test set
    size_t test_size = std::min(1000000UL, db_vecs.size());
    std::vector<std::vector<float>> test_db(
        db_vecs.begin(),
        db_vecs.begin() + test_size);
    
    size_t nb = test_db.size();
    size_t nq = qr_vecs.size();
    
    std::cout << "Training on " << train_db.size() << " samples, "
              << "testing on " << nb << " samples, "
              << "queries: " << nq << ", dim=" << dim << std::endl;
    
    // Setup PQ
    faiss::IndexPQ* index_pq = nullptr;
    double train_time = 0.0;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    if (load_model) {
        std::cout << "📥 Loading PQ model from " << model_path << std::endl;
        index_pq = dynamic_cast<faiss::IndexPQ*>(faiss::read_index(model_path.c_str()));
        if (!index_pq) {
            std::cerr << "Error: Failed to load model or model is not IndexPQ\n";
            return 1;
        }
        if (index_pq->d != static_cast<int>(dim) || 
            index_pq->pq.M != M || 
            index_pq->pq.nbits != nbits) {
            std::cerr << "Error: Model parameters don't match\n";
            return 1;
        }
        std::cout << "✅ PQ model loaded: " << M << "x" << nbits 
                  << " => " << (M * nbits) << " bits/vector" << std::endl;
    } else {
        std::cout << "Training PQ with " << M << "x" << nbits 
                  << " => " << (M * nbits) << " bits/vector (GPU)" << std::endl;
        
        index_pq = new faiss::IndexPQ(dim, M, nbits);
        
        // Use GPU for k-means assignment during PQ training
        faiss::gpu::StandardGpuResources gpu_res;
        faiss::gpu::GpuIndexFlatConfig flat_config;
        flat_config.device = gpu_device;
        faiss::gpu::GpuIndexFlatL2 gpu_assign_index(&gpu_res, index_pq->pq.dsub, flat_config);
        index_pq->pq.assign_index = &gpu_assign_index;
        
        auto train_data = flatten_vectors(train_db);
        index_pq->train(train_db.size(), train_data.data());
        
        index_pq->pq.assign_index = nullptr;
        
        auto end = std::chrono::high_resolution_clock::now();
        train_time = std::chrono::duration<double>(end - start).count();
        std::cout << "✅ PQ trained in " << train_time << "s" << std::endl;
        
        // Save model
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
        std::string timestamp = ss.str();
        
        std::stringstream folder_ss;
        folder_ss << "subq" << M << "_nbits" << nbits 
                  << "_train" << actual_train_size << "_" << timestamp;
        std::string folder_name = folder_ss.str();
        
        fs::path out_dir = fs::path(data_root) / results_dir / "pq" / dataset_name / folder_name;
        fs::create_directories(out_dir);
        
        std::string model_file = (out_dir / "pq_model.index").string();
        faiss::write_index(index_pq, model_file.c_str());
        std::cout << "✅ PQ model saved to " << model_file << std::endl;
    }
    
    // Encode database
    std::cout << "Encoding database..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    
    auto test_data = flatten_vectors(test_db);
    size_t code_size = index_pq->pq.code_size;
    std::vector<uint8_t> codes(nb * code_size);
    index_pq->pq.compute_codes(test_data.data(), codes.data(), nb);
    
    auto end = std::chrono::high_resolution_clock::now();
    double encoding_time = std::chrono::duration<double>(end - start).count();
    std::cout << "✅ Encoding done in " << encoding_time << "s | "
              << "Codes shape: " << nb << " x " << code_size << std::endl;
    
    // Compute distance tables
    std::cout << "Computing distance tables..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    
    size_t ksub = 1 << nbits;
    auto query_data = flatten_vectors(qr_vecs);
    std::vector<float> dis_tables(nq * M * ksub);
    index_pq->pq.compute_distance_tables(nq, query_data.data(), dis_tables.data());
    
    end = std::chrono::high_resolution_clock::now();
    double distance_table_time = std::chrono::duration<double>(end - start).count();
    std::cout << "✅ Distance tables computed in " << distance_table_time << "s" << std::endl;
    
    // Sample subset for ADC vs exact
    size_t n_sample_q = std::min(sample_queries, nq);
    size_t n_sample_db = std::min(sample_db, nb);
    
    std::vector<float> qr_sample(n_sample_q * dim);
    std::vector<float> db_sample(n_sample_db * dim);
    
    for (size_t i = 0; i < n_sample_q; i++) {
        std::memcpy(qr_sample.data() + i * dim, qr_vecs[i].data(), dim * sizeof(float));
    }
    for (size_t i = 0; i < n_sample_db; i++) {
        std::memcpy(db_sample.data() + i * dim, test_db[i].data(), dim * sizeof(float));
    }
    
    // Compute ADC distances
    std::cout << "Computing ADC distances..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    
    std::vector<float> adc_sample(n_sample_q * n_sample_db);
    std::vector<uint8_t> codes_sample(n_sample_db * code_size);
    for (size_t i = 0; i < n_sample_db; i++) {
        std::memcpy(codes_sample.data() + i * code_size, 
                   codes.data() + i * code_size, code_size);
    }
    
    std::vector<float> dis_tables_sample(n_sample_q * M * ksub);
    for (size_t i = 0; i < n_sample_q; i++) {
        std::memcpy(dis_tables_sample.data() + i * M * ksub,
                   dis_tables.data() + i * M * ksub,
                   M * ksub * sizeof(float));
    }
    
    compute_adc_distances(
        dis_tables_sample.data(), codes_sample.data(),
        n_sample_q, n_sample_db, M, ksub, code_size, nbits,
        adc_sample.data());
    
    end = std::chrono::high_resolution_clock::now();
    double adc_time = std::chrono::duration<double>(end - start).count();
    std::cout << "✅ ADC distances computed in " << adc_time << "s" << std::endl;
    
    // Compute exact distances
    std::cout << "Computing exact distances..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    
    std::vector<float> exact_sample(n_sample_q * n_sample_db);
    compute_exact_distances(
        qr_sample.data(), n_sample_q,
        db_sample.data(), n_sample_db,
        dim, exact_sample.data());
    
    end = std::chrono::high_resolution_clock::now();
    double cdist_time = std::chrono::duration<double>(end - start).count();
    std::cout << "✅ Exact distances computed in " << cdist_time << "s" << std::endl;
    
    // Compute relative error
    float mean_rel, std_rel;
    compute_relative_error(
        adc_sample.data(), exact_sample.data(),
        n_sample_q, n_sample_db,
        mean_rel, std_rel);
    
    std::cout << "Mean rel. error: " << mean_rel 
              << ", std: " << std_rel << std::endl;
    
    // Save results
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
    std::string timestamp = ss.str();
    
    std::stringstream folder_ss;
    folder_ss << "subq" << M << "_nbits" << nbits 
              << "_train" << actual_train_size << "_" << timestamp;
    std::string folder_name = folder_ss.str();
    
    fs::path out_dir = fs::path(data_root) / results_dir / "pq" / dataset_name / folder_name;
    if (!load_model) {
        // Directory already created when saving model
    } else {
        fs::create_directories(out_dir);
    }
    
    // Save binary relative error
    std::stringstream bin_ss;
    bin_ss << "rel_error_subq" << M << "_nbits" << nbits 
           << "_db" << (n_sample_db / 1000) << "k_qr" << (n_sample_q / 1000) << "k.bin";
    std::string bin_file = (out_dir / bin_ss.str()).string();
    std::ofstream bin_out(bin_file, std::ios::binary);
    bin_out.write(reinterpret_cast<const char*>(adc_sample.data()), 
                  n_sample_q * n_sample_db * sizeof(float));
    bin_out.close();
    
    // Save summary CSV
    std::string csv_path = (fs::path(data_root) / results_dir / 
                           (dataset_name + "_PQ_adc_vs_exact_eval.csv")).string();
    save_summary_csv(
        csv_path, dataset_name, out_dir.string(),
        nq, nb, n_sample_db, dim, M, nbits, actual_train_size,
        train_time, encoding_time, distance_table_time, cdist_time, adc_time,
        mean_rel, std_rel);
    
    // Save run-specific CSV
    std::string run_csv = (out_dir / "summary.csv").string();
    save_summary_csv(
        run_csv, dataset_name, out_dir.string(),
        nq, nb, n_sample_db, dim, M, nbits, actual_train_size,
        train_time, encoding_time, distance_table_time, cdist_time, adc_time,
        mean_rel, std_rel);
    
    std::cout << "✅ Results saved to " << out_dir << std::endl;
    
    delete index_pq;
    return 0;
}

