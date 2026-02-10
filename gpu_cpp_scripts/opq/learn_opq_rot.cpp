// learn_opq_rot.cpp
// Standalone tool to learn and cache OPQ rotations per (dataset, M).
// It trains only the OPQMatrix, saves it to --opq_model_path, and records
// the training time to a small sidecar metadata file (\".meta\").

#include <faiss/VectorTransform.h>
#include <faiss/index_io.h>
#include <faiss/impl/ProductQuantizer.h>

#include "../io_utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

std::vector<float> flatten_vectors(const std::vector<std::vector<float>>& vecs) {
    if (vecs.empty()) {
        return {};
    }
    size_t dim = vecs[0].size();
    std::vector<float> flat(vecs.size() * dim);
    for (size_t i = 0; i < vecs.size(); i++) {
        std::memcpy(flat.data() + i * dim, vecs[i].data(), dim * sizeof(float));
    }
    return flat;
}

// Write a small metadata file next to the OPQ transform with:
//   opq_train_time_s, rot_train_size, opq_max_train_points
void write_opq_meta(
        const std::string& opq_model_path,
        double opq_train_time,
        size_t rot_train_size,
        size_t opq_max_train_points) {
    try {
        fs::path meta_path = fs::path(opq_model_path).concat(".meta");
        std::ofstream meta(meta_path);
        if (!meta) {
            std::cerr << "Warning: Failed to open OPQ meta file for writing: "
                      << meta_path << std::endl;
            return;
        }
        meta << std::fixed << std::setprecision(6)
             << opq_train_time << " "
             << rot_train_size << " "
             << opq_max_train_points << "\n";
        meta.close();
        std::cout << "💾 OPQ meta written to " << meta_path
                  << " (train_time_s=" << opq_train_time
                  << ", rot_train_size=" << rot_train_size
                  << ", max_train_pts=" << opq_max_train_points << ")"
                  << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Warning: Exception while writing OPQ meta: " << e.what() << std::endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " --train_path PATH "
                  << "[--dim DIM] [--dataset_name NAME] "
                  << "[--n_subquantizers M] [--rot_train_size SIZE] "
                  << "[--opq_max_train_points N] [--results_dir DIR] "
                  << "[--opq_model_path PATH]\n";
        return 1;
    }

    std::string train_path;
    int dim = -1;
    std::string dataset_name = "custom";
    size_t M = 32;
    // Number of vectors to load and offer to OPQ training (before its own max_train_points cap).
    size_t rot_train_size = 100000;
    size_t opq_max_train_points = 256 * 256;
    std::string results_dir = "results/relerr_cpp";
    std::string opq_model_path;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--train_path" && i + 1 < argc) {
            train_path = argv[++i];
        } else if (arg == "--dim" && i + 1 < argc) {
            dim = std::stoi(argv[++i]);
        } else if (arg == "--dataset_name" && i + 1 < argc) {
            dataset_name = argv[++i];
        } else if (arg == "--n_subquantizers" && i + 1 < argc) {
            M = std::stoul(argv[++i]);
        } else if (arg == "--rot_train_size" && i + 1 < argc) {
            rot_train_size = std::stoul(argv[++i]);
        } else if (arg == "--opq_max_train_points" && i + 1 < argc) {
            opq_max_train_points = std::stoul(argv[++i]);
        } else if (arg == "--results_dir" && i + 1 < argc) {
            results_dir = argv[++i];
        } else if (arg == "--opq_model_path" && i + 1 < argc) {
            opq_model_path = argv[++i];
        }
    }

    if (train_path.empty()) {
        std::cerr << "Error: --train_path is required\n";
        return 1;
    }

    if (M == 0) {
        std::cerr << "Error: n_subquantizers (M) must be > 0\n";
        return 1;
    }

    if (opq_model_path.empty()) {
        // Default layout matches the shell scripts:
        //   results_dir/opq_transforms/<dataset>/opq_M{M}_{opq_max_train_points}.vt
        fs::path root(results_dir);
        fs::path model_root = root / "opq_transforms" / dataset_name;
        fs::create_directories(model_root);
        std::ostringstream oss;
        oss << "opq_M" << M << "_" << opq_max_train_points << ".vt";
        opq_model_path = (model_root / oss.str()).string();
    } else {
        fs::path opq_path(opq_model_path);
        if (!opq_path.parent_path().empty()) {
            fs::create_directories(opq_path.parent_path());
        }
    }

    std::cout << "==========================================" << std::endl;
    std::cout << "Learning OPQ rotation" << std::endl;
    std::cout << "  dataset_name   : " << dataset_name << std::endl;
    std::cout << "  train_path     : " << train_path << std::endl;
    std::cout << "  M (subq)       : " << M << std::endl;
    std::cout << "  rot_train_size : " << rot_train_size << std::endl;
    std::cout << "  max_train_pts  : " << opq_max_train_points << std::endl;
    std::cout << "  model_path     : " << opq_model_path << std::endl;
    std::cout << "==========================================" << std::endl;

    if (fs::exists(opq_model_path)) {
        std::cout << "🛈 OPQ model already exists at " << opq_model_path
                  << " – skipping training." << std::endl;
        return 0;
    }

    std::cout << "📂 Loading training data from " << train_path << std::endl;
    auto train_db_full = load_dataset(train_path, dim, rot_train_size + 10000);
    if (train_db_full.empty()) {
        std::cerr << "Error: Failed to load training data\n";
        return 1;
    }
    std::cout << "Loaded training data: " << train_db_full.size() << " vectors" << std::endl;

    size_t actual_dim = train_db_full[0].size();
    if (dim > 0 && static_cast<size_t>(dim) != actual_dim) {
        std::cerr << "Warning: Dimension mismatch. Expected " << dim
                  << ", got " << actual_dim << std::endl;
    }
    dim = static_cast<int>(actual_dim);

    if (dim % static_cast<int>(M) != 0) {
        std::cerr << "Error: dim must be divisible by n_subquantizers (M)\n";
        return 1;
    }

    size_t actual_train_size = std::min(rot_train_size, train_db_full.size());
    std::vector<std::vector<float>> train_db(
            train_db_full.begin(),
            train_db_full.begin() + actual_train_size);

    std::cout << "Training OPQ on " << train_db.size()
              << " samples, dim=" << dim << std::endl;

    faiss::OPQMatrix opq(dim, static_cast<int>(M));
    opq.max_train_points = opq_max_train_points;

    auto train_flat = flatten_vectors(train_db);

    auto opq_start = std::chrono::high_resolution_clock::now();
    opq.train(train_db.size(), train_flat.data());
    auto opq_end = std::chrono::high_resolution_clock::now();
    double opq_train_time =
            std::chrono::duration<double>(opq_end - opq_start).count();

    std::cout << "✅ OPQ trained in " << opq_train_time << "s" << std::endl;

    try {
        fs::path opq_path(opq_model_path);
        if (!opq_path.parent_path().empty()) {
            fs::create_directories(opq_path.parent_path());
        }
        faiss::write_VectorTransform(&opq, opq_model_path.c_str());
        std::cout << "💾 OPQ transform saved to " << opq_model_path << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: Failed to save OPQ transform: " << e.what() << std::endl;
        return 1;
    }

    // Persist metadata so train_opq can record OPQ training configuration.
    write_opq_meta(opq_model_path, opq_train_time, train_db.size(), opq_max_train_points);

    std::cout << "✅ Finished learning OPQ rotation for dataset=" << dataset_name
              << ", M=" << M << std::endl;
    return 0;
}

