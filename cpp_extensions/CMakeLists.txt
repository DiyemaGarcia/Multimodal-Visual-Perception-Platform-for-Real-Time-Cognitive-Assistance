cmake_minimum_required(VERSION 3.18)
project(vae_morphosyntax_extensions LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# -----------------------------------------------------------------------
# Python headers — for torch/extension.h via Python.h
# -----------------------------------------------------------------------

set(PYTHON_INCLUDE_DIR
    "/home/rene/miniconda3/envs/image_mining/include/python3.10"
)

# -----------------------------------------------------------------------
# PyTorch paths
# -----------------------------------------------------------------------

set(TORCH_INCLUDE_DIR
    "/home/rene/miniconda3/envs/image_mining/lib/python3.10/site-packages/torch/include"
)
set(TORCH_API_INCLUDE_DIR
    "/home/rene/miniconda3/envs/image_mining/lib/python3.10/site-packages/torch/include/torch/csrc/api/include"
)
set(TORCH_LIB_DIR
    "/home/rene/miniconda3/envs/image_mining/lib/python3.10/site-packages/torch/lib"
)

set(TORCH_INCLUDE_DIRS
    ${PYTHON_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIR}
    ${TORCH_API_INCLUDE_DIR}
)

# -----------------------------------------------------------------------
# .so files present in TORCH_LIB_DIR
# -----------------------------------------------------------------------

set(TORCH_LIBRARIES
    "${TORCH_LIB_DIR}/libtorch.so"
    "${TORCH_LIB_DIR}/libtorch_cpu.so"
    "${TORCH_LIB_DIR}/libtorch_python.so"
    "${TORCH_LIB_DIR}/libtorch_global_deps.so"
    "${TORCH_LIB_DIR}/libc10.so"
    "${TORCH_LIB_DIR}/libshm.so"
)

set(TORCH_CUDA_LIBRARIES
    "${TORCH_LIB_DIR}/libtorch_cuda.so"
    "${TORCH_LIB_DIR}/libtorch_cuda_linalg.so"
    "${TORCH_LIB_DIR}/libc10_cuda.so"
    "${TORCH_LIB_DIR}/libcaffe2_nvrtc.so"
)

# -----------------------------------------------------------------------
# CUDA — CMake find the system CUDA installation
# -----------------------------------------------------------------------

find_package(CUDA REQUIRED)

# -----------------------------------------------------------------------
# Compiler flags
# Must match the ABI used to compile PyTorch in conda environment
# -----------------------------------------------------------------------

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")

# -----------------------------------------------------------------------
# RPATH — so compiled .so files find libtorch at runtime
# without needing to export LD_LIBRARY_PATH manually
# -----------------------------------------------------------------------

set(CMAKE_INSTALL_RPATH "${TORCH_LIB_DIR}")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_SKIP_BUILD_RPATH FALSE)

# -----------------------------------------------------------------------
# C++ extension: fast_edit_distance
# -----------------------------------------------------------------------

add_library(fast_edit_distance SHARED
    fast_edit_distance.cpp
)
target_include_directories(fast_edit_distance PRIVATE
    ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(fast_edit_distance PRIVATE
    ${TORCH_LIBRARIES}
)
set_target_properties(fast_edit_distance PROPERTIES
    CXX_STANDARD 17
    PREFIX ""
    SUFFIX ".so"
)

# -----------------------------------------------------------------------
# C++ extension: morpheme_segmenter
# -----------------------------------------------------------------------

add_library(morpheme_segmenter SHARED
    morpheme_segmenter.cpp
)
target_include_directories(morpheme_segmenter PRIVATE
    ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(morpheme_segmenter PRIVATE
    ${TORCH_LIBRARIES}
)
set_target_properties(morpheme_segmenter PROPERTIES
    CXX_STANDARD 17
    PREFIX ""
    SUFFIX ".so"
)

# -----------------------------------------------------------------------
# CUDA extension: attention_kernel
# -----------------------------------------------------------------------

add_library(attention_kernel SHARED
    cuda_kernels/attention_kernel.cu
)
target_include_directories(attention_kernel PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(attention_kernel PRIVATE
    ${TORCH_LIBRARIES}
    ${TORCH_CUDA_LIBRARIES}
    ${CUDA_LIBRARIES}
)
set_target_properties(attention_kernel PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    PREFIX ""
    SUFFIX ".so"
)

# -----------------------------------------------------------------------
# CUDA extension: elbo_kernel
# -----------------------------------------------------------------------

add_library(elbo_kernel SHARED
    cuda_kernels/elbo_kernel.cu
)
target_include_directories(elbo_kernel PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(elbo_kernel PRIVATE
    ${TORCH_LIBRARIES}
    ${TORCH_CUDA_LIBRARIES}
    ${CUDA_LIBRARIES}
)
set_target_properties(elbo_kernel PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    PREFIX ""
    SUFFIX ".so"
)