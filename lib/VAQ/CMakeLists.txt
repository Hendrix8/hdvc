cmake_minimum_required(VERSION 3.13)
project(BitVecEngine LANGUAGES CXX)

# --------------------------------------------------------
# Compiler settings
# --------------------------------------------------------
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -Wno-psabi -Wno-sign-compare")

# Optimization levels
if (NOT OPTIMIZATION_LEVEL)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -mfma")
    add_definitions(-DVAQ_OPTIMIZE)
    add_definitions(-march=haswell -mavx -mavx2)
elseif (OPTIMIZATION_LEVEL STREQUAL "full")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
elseif (OPTIMIZATION_LEVEL STREQUAL "aggressive")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
elseif (OPTIMIZATION_LEVEL STREQUAL "generic")
    # no additional optimization flags
endif()

# --------------------------------------------------------
# Dependencies
# --------------------------------------------------------

# --- Eigen (header-only) ---
include_directories(external/eigen)

# --- Armadillo ---
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})
link_directories(${ARMADILLO_LIBRARY_DIRS})
message(STATUS "✅ Found Armadillo: ${ARMADILLO_INCLUDE_DIRS}")

# --- GLPK ---
find_path(GLPK_INCLUDE_DIR glpk.h
  HINTS
    $ENV{GLPK_INCLUDE_DIR}
    ${GLPK_INCLUDE_DIR}
    $ENV{HOME}/local/glpk/include
)
find_library(GLPK_LIBRARY NAMES glpk
  HINTS
    $ENV{GLPK_LIBRARY}
    ${GLPK_LIBRARY}
    $ENV{HOME}/local/glpk/lib
)
if (GLPK_INCLUDE_DIR AND GLPK_LIBRARY)
    message(STATUS "✅ Found GLPK:")
    message(STATUS "   Include: ${GLPK_INCLUDE_DIR}")
    message(STATUS "   Library: ${GLPK_LIBRARY}")
    include_directories(${GLPK_INCLUDE_DIR})
else()
    message(FATAL_ERROR "❌ GLPK not found — please set GLPK_INCLUDE_DIR and GLPK_LIBRARY")
endif()

# --------------------------------------------------------
# BLAS / LAPACKE via Conda OpenBLAS
# --------------------------------------------------------
# Try to use Conda BLAS/LAPACKE if available
if (DEFINED ENV{CONDA_PREFIX})
    set(BLAS_LIBRARIES "$ENV{CONDA_PREFIX}/lib/libopenblas.so" CACHE FILEPATH "Path to OpenBLAS library")
    set(LAPACKE_LIBRARIES "$ENV{CONDA_PREFIX}/lib/liblapacke.so" CACHE FILEPATH "Path to LAPACKE library")
    message(STATUS "✅ Using OpenBLAS from Conda: ${BLAS_LIBRARIES}")
    message(STATUS "✅ Using LAPACKE from Conda: ${LAPACKE_LIBRARIES}")
else()
    message(WARNING "⚠️  CONDA_PREFIX not set — assuming system BLAS/LAPACKE")
endif()

# Avoid Armadillo's internal wrapper so it uses the above BLAS/LAPACK
add_definitions(-DARMA_DONT_USE_WRAPPER)

# --------------------------------------------------------
# Optional OpenMP
# --------------------------------------------------------
find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "✅ OpenMP enabled")
else()
    message(STATUS "⚠️  OpenMP not found — continuing without it")
endif()

# --------------------------------------------------------
# Subdirectories
# --------------------------------------------------------
add_subdirectory(bitvecengine)
add_subdirectory(examples)

if (BUILD_TESTING)
    add_subdirectory(test)
    add_custom_target(test COMMAND Test --durations yes)
endif()

# --------------------------------------------------------
# Eigen build flags
# --------------------------------------------------------
add_definitions(-DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE)
add_definitions(-DEIGEN_DONT_PARALLELIZE)

# --------------------------------------------------------
# Export vars for subprojects
# --------------------------------------------------------
set(GLPK_INCLUDE_DIR ${GLPK_INCLUDE_DIR} CACHE PATH "GLPK include directory")
set(GLPK_LIBRARY ${GLPK_LIBRARY} CACHE FILEPATH "GLPK library path")
set(BLAS_LIBRARIES ${BLAS_LIBRARIES} CACHE FILEPATH "BLAS library path")
set(LAPACKE_LIBRARIES ${LAPACKE_LIBRARIES} CACHE FILEPATH "LAPACKE library path")
