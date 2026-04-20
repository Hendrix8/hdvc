cmake_minimum_required(VERSION 3.13)
project(BitVecEngine LANGUAGES CXX)

# --------------------------------------------------------
# Compiler settings
# --------------------------------------------------------
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -Wno-psabi -Wno-sign-compare")

# # Prevent Eigen deprecation warnings from becoming errors
# string(REPLACE "-Werror" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
# add_compile_options(-Wno-error=deprecated-copy -Wno-deprecated-copy -Wno-error)

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
# BLAS / LAPACKE (OpenBLAS preferred; conda path may not exist if env moved)
# --------------------------------------------------------
# Allow override: cmake -DBLAS_LIBRARIES=... -DLAPACKE_LIBRARIES=...
if(BLAS_LIBRARIES AND NOT EXISTS "${BLAS_LIBRARIES}")
    message(WARNING "Clearing stale BLAS_LIBRARIES (file missing): ${BLAS_LIBRARIES}")
    unset(BLAS_LIBRARIES CACHE)
endif()
if(LAPACKE_LIBRARIES AND NOT EXISTS "${LAPACKE_LIBRARIES}")
    message(WARNING "Clearing stale LAPACKE_LIBRARIES (file missing): ${LAPACKE_LIBRARIES}")
    unset(LAPACKE_LIBRARIES CACHE)
endif()

if(NOT BLAS_LIBRARIES)
    set(_blas_candidates "")
    if(DEFINED ENV{CONDA_PREFIX})
        list(APPEND _blas_candidates "$ENV{CONDA_PREFIX}/lib/libopenblas.so")
        list(APPEND _blas_candidates "$ENV{CONDA_PREFIX}/lib/libopenblasp-r0.3.so")
    endif()
    list(APPEND _blas_candidates "$ENV{HOME}/.miniconda3/lib/libopenblas.so")
    list(APPEND _blas_candidates "$ENV{HOME}/anaconda3/lib/libopenblas.so")
    list(APPEND _blas_candidates "$ENV{HOME}/mambaforge/lib/libopenblas.so")
    foreach(_cand IN LISTS _blas_candidates)
        if(EXISTS "${_cand}")
            set(BLAS_LIBRARIES "${_cand}")
            break()
        endif()
    endforeach()
    if(NOT BLAS_LIBRARIES)
        find_library(BLAS_LIBRARIES NAMES openblas openblasp-r0.3
            PATHS
                "$ENV{CONDA_PREFIX}/lib"
                /usr/lib/x86_64-linux-gnu
                /usr/lib64
        )
    endif()
endif()

if(NOT LAPACKE_LIBRARIES)
    set(_lap_candidates "")
    if(DEFINED ENV{CONDA_PREFIX})
        list(APPEND _lap_candidates "$ENV{CONDA_PREFIX}/lib/liblapacke.so")
    endif()
    list(APPEND _lap_candidates "$ENV{HOME}/.miniconda3/lib/liblapacke.so")
    foreach(_cand IN LISTS _lap_candidates)
        if(EXISTS "${_cand}")
            set(LAPACKE_LIBRARIES "${_cand}")
            break()
        endif()
    endforeach()
    if(NOT LAPACKE_LIBRARIES)
        find_library(LAPACKE_LIBRARIES NAMES lapacke
            PATHS
                "$ENV{CONDA_PREFIX}/lib"
                /usr/lib/x86_64-linux-gnu
                /usr/lib64
        )
    endif()
endif()

if(BLAS_LIBRARIES)
    set(BLAS_LIBRARIES "${BLAS_LIBRARIES}" CACHE FILEPATH "Path to OpenBLAS library" FORCE)
    message(STATUS "✅ OpenBLAS: ${BLAS_LIBRARIES}")
else()
    message(FATAL_ERROR "OpenBLAS not found. Install openblas (conda: conda install openblas) or pass -DBLAS_LIBRARIES=/path/to/libopenblas.so")
endif()

if(LAPACKE_LIBRARIES)
    set(LAPACKE_LIBRARIES "${LAPACKE_LIBRARIES}" CACHE FILEPATH "Path to LAPACKE library" FORCE)
    message(STATUS "✅ LAPACKE: ${LAPACKE_LIBRARIES}")
else()
    message(FATAL_ERROR "LAPACKE not found. Install lapacke (conda: conda install lapack) or pass -DLAPACKE_LIBRARIES=/path/to/liblapacke.so")
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
