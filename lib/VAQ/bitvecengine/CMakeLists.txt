# ============================================================
# bitvecengine/CMakeLists.txt
# ============================================================

# Main VAQ library
add_library(
  VAQ SHARED
  utils/Heap.cpp
  VAQ.cpp
)

# FP-Growth utility library
add_library(
  FPGrowth SHARED
  utils/FPGrowth/fptree.cpp
)

# ------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------

# --- OpenMP ---
find_package(OpenMP)
if(OPENMP_FOUND)
  target_compile_options(VAQ PRIVATE ${OpenMP_CXX_FLAGS})
  target_link_libraries(VAQ PRIVATE ${OpenMP_CXX_LIBRARIES})
  target_compile_options(FPGrowth PRIVATE ${OpenMP_CXX_FLAGS})
  target_link_libraries(FPGrowth PRIVATE ${OpenMP_CXX_LIBRARIES})
  message(STATUS "✅ OpenMP linked to VAQ and FPGrowth")
else()
  message(STATUS "⚠️ OpenMP not found — continuing without it")
endif()

# --- Armadillo ---
find_package(Armadillo REQUIRED)
target_include_directories(VAQ PRIVATE ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(VAQ PRIVATE ${ARMADILLO_LIBRARIES})
message(STATUS "✅ Armadillo include: ${ARMADILLO_INCLUDE_DIRS}")
message(STATUS "✅ Armadillo lib: ${ARMADILLO_LIBRARIES}")

# --- GLPK (from root CMake) ---
if (DEFINED GLPK_INCLUDE_DIR AND DEFINED GLPK_LIBRARY)
  target_include_directories(VAQ PRIVATE ${GLPK_INCLUDE_DIR})
  target_link_libraries(VAQ PRIVATE ${GLPK_LIBRARY})
  message(STATUS "✅ Linked GLPK: ${GLPK_LIBRARY}")
else()
  message(WARNING "⚠️ GLPK variables not found; please define GLPK_INCLUDE_DIR and GLPK_LIBRARY in parent CMake.")
endif()

# --- Conda BLAS/LAPACKE ---
if (DEFINED BLAS_LIBRARIES)
  target_link_libraries(VAQ PRIVATE ${BLAS_LIBRARIES})
  message(STATUS "✅ Linked BLAS: ${BLAS_LIBRARIES}")
endif()

if (DEFINED LAPACKE_LIBRARIES)
  target_link_libraries(VAQ PRIVATE ${LAPACKE_LIBRARIES})
  message(STATUS "✅ Linked LAPACKE: ${LAPACKE_LIBRARIES}")
endif()

# ------------------------------------------------------------
# General settings
# ------------------------------------------------------------
set_target_properties(VAQ PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(FPGrowth PROPERTIES POSITION_INDEPENDENT_CODE ON)
