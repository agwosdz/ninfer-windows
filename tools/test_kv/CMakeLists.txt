# E8 compressed-KV test suite:
#   ninfer_kv_e8_verify          - ported parallel-implementation codec oracle + 1M needle
#                                  retrieval microbenchmark (self-contained kernels)
#   ninfer_kv_e8_lattice_oracle  - exact oracle for the production codec: compiles
#                                  src/ops/kernel/e8_lattice.cuh and checks the nearest-
#                                  lattice projection and doubled codes against an
#                                  independent CPU reference (algebraic FP64 decoder
#                                  cross-validated by exact enumeration)
#
# Builds two ways:
#   1. From the NInfer top level with BUILD_TESTING=ON (wired into ctest):
#        cmake -S . -B build -GNinja -DBUILD_TESTING=ON
#        cmake --build build -j && ctest --test-dir build
#   2. Standalone with its own configure:
#        cmake -S tools/test_kv -B build -GNinja && cmake --build build -j
#        ctest --test-dir build

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.20)
  project(ninfer_kv_e8_verify LANGUAGES CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120a)
  endif()
  enable_testing()
endif()

add_executable(ninfer_kv_e8_verify
  test_e8_codec.cu
  verify_1m_retrieval.cu)
set_target_properties(ninfer_kv_e8_verify PROPERTIES
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON)

# 1M tokens x 256 dims FP32 keys is ~1 GB on-device; the tool fails cleanly with a
# smaller-token hint when the allocation does not fit (see the OOM guard).
add_test(NAME ninfer_kv_e8_verify COMMAND ninfer_kv_e8_verify 1000000)

add_executable(ninfer_kv_e8_lattice_oracle
  test_e8_lattice_oracle.cu)
set_target_properties(ninfer_kv_e8_lattice_oracle PROPERTIES
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON)
target_include_directories(ninfer_kv_e8_lattice_oracle PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
add_test(NAME ninfer_kv_e8_lattice_oracle COMMAND ninfer_kv_e8_lattice_oracle)

# On-device round-trip check of the rk* rotate primitive on the current arch.
add_executable(ninfer_kv_rotate_roundtrip
  test_rotate_roundtrip.cu)
set_target_properties(ninfer_kv_rotate_roundtrip PROPERTIES
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON)
target_include_directories(ninfer_kv_rotate_roundtrip PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
add_test(NAME ninfer_kv_rotate_roundtrip COMMAND ninfer_kv_rotate_roundtrip)

# shfl_xor_sync primitive semantics probe (diagnostic; not a ctest).
add_executable(ninfer_kv_shfl_probe test_shfl_probe.cu)
set_target_properties(ninfer_kv_shfl_probe PROPERTIES
  CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

# hadamard64 per-stage dump (diagnostic; not a ctest).
add_executable(ninfer_kv_hadamard_dump test_hadamard_dump.cu)
set_target_properties(ninfer_kv_hadamard_dump PROPERTIES
  CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

# packed-V codec round trip (diagnostic; not a ctest).
add_executable(ninfer_kv_v_codec_roundtrip test_v_codec_roundtrip.cu)
set_target_properties(ninfer_kv_v_codec_roundtrip PROPERTIES
  CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
target_include_directories(ninfer_kv_v_codec_roundtrip PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
