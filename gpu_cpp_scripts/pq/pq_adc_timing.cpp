#include <faiss/IndexPQ.h>
#include <faiss/index_io.h>
#include <faiss/utils/distances.h>
#include <faiss/impl/ProductQuantizer-inl.h>
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
#include <sstream>
#include <ctime>
#include <map>

// For CPU time measurement (excludes system interference)
#ifdef __linux__
#include <time.h>
#endif

namespace fs = std::filesystem;

// Get CPU time (process time, excludes interference from other processes)
double get_cpu_time() {
#ifdef __linux__
    timespec ts;
    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts) == 0) {
        return double(ts.tv_sec) + 1e-9 * double(ts.tv_nsec);
    }
#endif
    // Fallback to high_resolution_clock if CLOCK_PROCESS_CPUTIME_ID not available
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double>(duration).count();
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

// Parse CSV line
std::vector<std::string> parse_csv_line(const std::string& line) {
    std::vector<std::string> result;
    std::stringstream ss(line);
    std::string item;
    
    bool in_quotes = false;
    std::string current;
    
    for (char c : line) {
        if (c == '"') {
            in_quotes = !in_quotes;
        } else if (c == ',' && !in_quotes) {
            result.push_back(current);
            current.clear();
        } else {
            current += c;
        }
    }
    result.push_back(current);
    
    return result;
}

// Read CSV and return rows
std::vector<std::vector<std::string>> read_csv(const std::string& csv_path) {
    std::vector<std::vector<std::string>> rows;
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open CSV file: " << csv_path << std::endl;
        return rows;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            rows.push_back(parse_csv_line(line));
        }
    }
    return rows;
}

// Write CSV
void write_csv(const std::string& csv_path, const std::vector<std::vector<std::string>>& rows) {
    std::ofstream file(csv_path);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot write CSV file: " << csv_path << std::endl;
        return;
    }
    
    for (const auto& row : rows) {
        for (size_t i = 0; i < row.size(); i++) {
            file << row[i];
            if (i < row.size() - 1) file << ",";
        }
        file << "\n";
    }
}

// Find column index by name
int find_column_index(const std::vector<std::string>& header, const std::string& col_name) {
    for (size_t i = 0; i < header.size(); i++) {
        if (header[i] == col_name) {
            return i;
        }
    }
    return -1;
}

// Run ADC timing experiment for one configuration
bool run_adc_timing(
    const std::string& model_path,
    const std::string& query_path,
    const std::string& database_path,
    size_t n_sample_q,
    size_t n_sample_db,
    double& adc_time_pp,
    double& distance_table_time_pp) {
    
    // Load model
    faiss::IndexPQ* index_pq = dynamic_cast<faiss::IndexPQ*>(faiss::read_index(model_path.c_str()));
    if (!index_pq) {
        std::cerr << "Error: Failed to load PQ model from " << model_path << std::endl;
        return false;
    }
    
    size_t dim = index_pq->d;
    size_t M = index_pq->pq.M;
    size_t nbits = index_pq->pq.nbits;
    size_t ksub = 1 << nbits;
    size_t code_size = index_pq->pq.code_size;
    
    // Load queries (read more than needed to ensure we have enough)
    std::vector<std::vector<float>> queries_vec = read_fvecs(query_path, n_sample_q + 100);
    if (queries_vec.empty()) {
        std::cerr << "Error: Failed to load queries from " << query_path << std::endl;
        delete index_pq;
        return false;
    }
    size_t nq = queries_vec.size();
    size_t actual_n_sample_q = std::min(n_sample_q, nq);
    
    // Load database (read more than needed to ensure we have enough)
    std::vector<std::vector<float>> database_vec = read_fvecs(database_path, n_sample_db + 100);
    if (database_vec.empty()) {
        std::cerr << "Error: Failed to load database from " << database_path << std::endl;
        delete index_pq;
        return false;
    }
    size_t nb = database_vec.size();
    size_t actual_n_sample_db = std::min(n_sample_db, nb);
    
    // Convert to flat arrays
    std::vector<float> query_data(actual_n_sample_q * dim);
    for (size_t i = 0; i < actual_n_sample_q; i++) {
        std::memcpy(query_data.data() + i * dim, queries_vec[i].data(), dim * sizeof(float));
    }
    
    std::vector<float> database_data(actual_n_sample_db * dim);
    for (size_t i = 0; i < actual_n_sample_db; i++) {
        std::memcpy(database_data.data() + i * dim, database_vec[i].data(), dim * sizeof(float));
    }
    
    // Encode database
    std::vector<uint8_t> codes(actual_n_sample_db * code_size);
    index_pq->pq.compute_codes(database_data.data(), codes.data(), actual_n_sample_db);
    
    // Compute distance tables (for first actual_n_sample_q queries)
    std::vector<float> dis_tables(actual_n_sample_q * M * ksub);
    
    double start_cpu = get_cpu_time();
    index_pq->pq.compute_distance_tables(actual_n_sample_q, query_data.data(), dis_tables.data());
    double end_cpu = get_cpu_time();
    double distance_table_time = end_cpu - start_cpu;
    
    // Compute ADC distances
    std::vector<float> adc_distances(actual_n_sample_q * actual_n_sample_db);
    
    start_cpu = get_cpu_time();
    compute_adc_distances(
        dis_tables.data(), codes.data(),
        actual_n_sample_q, actual_n_sample_db, M, ksub, code_size, nbits,
        adc_distances.data());
    end_cpu = get_cpu_time();
    double adc_time = end_cpu - start_cpu;
    
    // Normalize to time per pair
    size_t total_pairs = actual_n_sample_q * actual_n_sample_db;
    adc_time_pp = adc_time / total_pairs;
    distance_table_time_pp = distance_table_time / total_pairs;
    
    delete index_pq;
    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <csv_path> <data_root> <n_sample_q> <n_sample_db>" << std::endl;
        std::cerr << "Example: " << argv[0] << " results.csv /data/cpanourg/2-hdvc 1000 10000" << std::endl;
        return 1;
    }
    
    std::string csv_path = argv[1];
    std::string data_root = argv[2];
    size_t n_sample_q = std::stoull(argv[3]);
    size_t n_sample_db = std::stoull(argv[4]);
    
    std::cout << "Reading CSV: " << csv_path << std::endl;
    std::vector<std::vector<std::string>> rows = read_csv(csv_path);
    if (rows.empty()) {
        std::cerr << "Error: CSV file is empty or cannot be read" << std::endl;
        return 1;
    }
    
    // Parse header
    std::vector<std::string> header = rows[0];
    
    // Find required columns
    int idx_method = find_column_index(header, "method");
    int idx_dataset = find_column_index(header, "dataset");
    int idx_experiment_folder = find_column_index(header, "experiment_folder");
    int idx_n_subquantizers = find_column_index(header, "n_subquantizers");
    int idx_nbits = find_column_index(header, "nbits");
    
    if (idx_method < 0 || idx_dataset < 0 || idx_experiment_folder < 0 || 
        idx_n_subquantizers < 0 || idx_nbits < 0) {
        std::cerr << "Error: Required columns not found in CSV" << std::endl;
        std::cerr << "Required: method, dataset, experiment_folder, n_subquantizers, nbits" << std::endl;
        return 1;
    }
    
    // Check if adc_pp and distance_table_time_pp columns exist, add if not
    int idx_adc_pp = find_column_index(header, "adc_pp");
    int idx_distance_table_time_pp = find_column_index(header, "distance_table_time_pp");
    
    bool need_add_columns = (idx_adc_pp < 0 || idx_distance_table_time_pp < 0);
    
    if (need_add_columns) {
        if (idx_adc_pp < 0) {
            header.push_back("adc_pp");
            idx_adc_pp = header.size() - 1;
        }
        if (idx_distance_table_time_pp < 0) {
            header.push_back("distance_table_time_pp");
            idx_distance_table_time_pp = header.size() - 1;
        }
        rows[0] = header;
    }
    
    // Process each row
    size_t processed = 0;
    size_t skipped = 0;
    
    for (size_t i = 1; i < rows.size(); i++) {
        if (rows[i].size() < header.size()) {
            rows[i].resize(header.size(), "");
        }
        
        std::string method = rows[i][idx_method];
        std::string dataset = rows[i][idx_dataset];
        std::string experiment_folder = rows[i][idx_experiment_folder];
        std::string n_subquantizers_str = rows[i][idx_n_subquantizers];
        std::string nbits_str = rows[i][idx_nbits];
        
        // Skip if already processed
        if (!rows[i][idx_adc_pp].empty() && rows[i][idx_adc_pp] != "N/A") {
            std::cout << "Skipping row " << i << " (already has adc_pp)" << std::endl;
            skipped++;
            continue;
        }
        
        // Construct paths
        fs::path results_dir = fs::path(data_root) / "results" / "relerr_cpp" / "pq" / dataset;
        fs::path model_path_file = results_dir / experiment_folder / "pq_model.index";
        std::string model_path = model_path_file.string();
        
        // Determine query and database paths based on dataset
        fs::path query_path, database_path;
        if (dataset == "deep") {
            query_path = fs::path(data_root) / "data" / "deep1b" / "dataset" / "fvecs" / "query_10k.fvecs";
            database_path = fs::path(data_root) / "data" / "deep1b" / "dataset" / "fvecs" / "test_1m.fvecs";
        } else if (dataset == "gist") {
            query_path = fs::path(data_root) / "data" / "gist" / "gist_query.fvecs";
            database_path = fs::path(data_root) / "data" / "gist" / "gist_base.fvecs";
        } else if (dataset == "bigann") {
            query_path = fs::path(data_root) / "data" / "bigann" / "sift1m" / "sift_query.fvecs";
            database_path = fs::path(data_root) / "data" / "bigann" / "sift1m" / "sift_base.fvecs";
        } else {
            std::cerr << "Warning: Unknown dataset '" << dataset << "', skipping row " << i << std::endl;
            skipped++;
            continue;
        }
        
        if (!fs::exists(model_path_file)) {
            std::cerr << "Warning: Model not found: " << model_path << ", skipping row " << i << std::endl;
            skipped++;
            continue;
        }
        
        std::cout << "\n[" << i << "/" << (rows.size() - 1) << "] Processing: " 
                  << dataset << " M=" << n_subquantizers_str << " nbits=" << nbits_str << std::endl;
        
        // Run timing experiment
        double adc_time_pp, distance_table_time_pp;
        if (run_adc_timing(model_path, query_path.string(), database_path.string(), 
                          n_sample_q, n_sample_db,
                          adc_time_pp, distance_table_time_pp)) {
            // Update row
            std::stringstream ss_adc, ss_dt;
            ss_adc << std::fixed << std::setprecision(12) << adc_time_pp;
            ss_dt << std::fixed << std::setprecision(12) << distance_table_time_pp;
            
            rows[i][idx_adc_pp] = ss_adc.str();
            rows[i][idx_distance_table_time_pp] = ss_dt.str();
            
            std::cout << "  ✓ adc_pp: " << adc_time_pp << " s/pair" << std::endl;
            std::cout << "  ✓ distance_table_time_pp: " << distance_table_time_pp << " s/pair" << std::endl;
            
            processed++;
            
            // Save CSV after each successful update
            write_csv(csv_path, rows);
            std::cout << "  ✓ CSV updated" << std::endl;
        } else {
            std::cerr << "  ✗ Failed to run timing experiment" << std::endl;
            rows[i][idx_adc_pp] = "N/A";
            rows[i][idx_distance_table_time_pp] = "N/A";
            skipped++;
        }
    }
    
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Processed: " << processed << std::endl;
    std::cout << "Skipped: " << skipped << std::endl;
    std::cout << "CSV saved to: " << csv_path << std::endl;
    
    return 0;
}
