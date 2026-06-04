set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_executable(symqg_indexing symqg_indexing.cpp)
add_executable(symqg_querying symqg_querying.cpp)

add_executable(ivf_rabitq_indexing ivf_rabitq_indexing.cpp)
add_executable(ivf_rabitq_querying ivf_rabitq_querying.cpp)
add_executable(ivf_rabitq_relerr   ivf_rabitq_relerr.cpp)
add_executable(ivf_rabitq_eval     ivf_rabitq_eval.cpp)
add_executable(ivf_rabitq_flat_adc ivf_rabitq_flat_adc.cpp)

add_executable(hnsw_rabitq_indexing hnsw_rabitq_indexing.cpp)
add_executable(hnsw_rabitq_querying hnsw_rabitq_querying.cpp)
