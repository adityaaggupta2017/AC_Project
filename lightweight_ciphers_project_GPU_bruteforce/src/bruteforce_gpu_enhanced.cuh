#pragma once

#include <cstdint>
#include <cstdio>
#include <functional>
#include <vector>
#include <thread>
#include <chrono>
#include <mutex>
#include <algorithm>
#include <numeric>
#include <random>

#include "util.hpp"
#include "ciphers_enhanced.cuh"

#ifdef __CUDACC__

// ============================================================
// Helpers
// ============================================================

static inline uint64_t bf_space_size_gpu(int unknown_bits) {
  if (unknown_bits < 0) return 0ULL;
  if (unknown_bits >= 63) return 0ULL;   // avoid undefined shift / unrealistic demo sizes
  return 1ULL << (uint64_t)unknown_bits;
}

// GPU timing helper: measures KERNEL time only (memset/reset is excluded)
static inline double time_kernel_seconds_stream(const std::function<void()>& reset,
                                                const std::function<void()>& launch,
                                                int warmup_iters,
                                                int timed_iters,
                                                cudaStream_t stream) {
  cuda_check(cudaStreamSynchronize(stream), "stream sync before timing");

  cudaEvent_t start, stop;
  cuda_check(cudaEventCreate(&start), "cudaEventCreate start");
  cuda_check(cudaEventCreate(&stop), "cudaEventCreate stop");

  // Warmup
  for (int i = 0; i < warmup_iters; i++) {
    reset();
    launch();
  }
  cuda_check(cudaStreamSynchronize(stream), "stream sync after warmup");

  float total_ms = 0.0f;
  if (timed_iters <= 0) timed_iters = 1;

  for (int i = 0; i < timed_iters; i++) {
    // Queue reset first (excluded)
    reset();

    // Start timing AFTER reset has been enqueued on the same stream
    cuda_check(cudaEventRecord(start, stream), "event record start");

    // Queue kernel work
    launch();

    cuda_check(cudaEventRecord(stop, stream), "event record stop");
    cuda_check(cudaEventSynchronize(stop), "event sync stop");

    float ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&ms, start, stop), "event elapsed time");
    total_ms += ms;
  }

  cuda_check(cudaEventDestroy(start), "event destroy start");
  cuda_check(cudaEventDestroy(stop), "event destroy stop");

  const double avg_ms = (double)total_ms / (double)timed_iters;
  return avg_ms * 1e-3;
}

enum class GpuVariant {
  BASELINE = 0,
  OPTIMIZED = 1,
  OPTIMIZED_ILP = 2,
  OPTIMIZED_SHARED = 3,
  BITSLICED = 4,
};

inline const char* gpu_variant_name(GpuVariant v) {
  switch (v) {
    case GpuVariant::BASELINE: return "gpu0_baseline";
    case GpuVariant::OPTIMIZED: return "gpu1_optimized";
    case GpuVariant::OPTIMIZED_ILP: return "gpu2_optimized+ilp";
    case GpuVariant::OPTIMIZED_SHARED: return "gpu3_optimized+shared";
    case GpuVariant::BITSLICED: return "gpu4_bitsliced";
    default: return "gpu_unknown";
  }
}

struct GpuBFResult {
  bool found = false;
  uint64_t found_key = 0;
  double seconds = 0.0;
  uint64_t keys_tested = 0;
};

// ============================================================
// SIMON 32/64 GPU Kernels
// ============================================================

__device__ inline void try_key_simon_opt(uint32_t pt, uint32_t ct, uint64_t k,
                                         uint64_t* found_key, int* found_flag) {
  uint16_t rk[SIMON32_64_ROUNDS];
  Simon32_64_Enhanced::expand_key_vectorized(k, rk);
  uint32_t out = Simon32_64_Enhanced::encrypt_optimized(pt, rk);

  if (out == ct) {
    if (atomicCAS(found_flag, 0, 1) == 0) {
      *found_key = k;
    }
  }
}

__global__ void bf_kernel_simon_opt(uint32_t pt, uint32_t ct, uint64_t base_key, uint64_t N,
                                    int use_ilp, uint64_t* found_key, int* found_flag) {
  uint64_t tid = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
  uint64_t stride = (uint64_t)gridDim.x * (uint64_t)blockDim.x;

  if (use_ilp) {
    // ILP: Process 4 keys per iteration
    for (uint64_t i = tid; i < N; i += stride * 4ULL) {
      uint64_t k0 = base_key | i;
      uint64_t k1 = base_key | (i + stride);
      uint64_t k2 = base_key | (i + 2ULL * stride);
      uint64_t k3 = base_key | (i + 3ULL * stride);

      if (i < N)                 try_key_simon_opt(pt, ct, k0, found_key, found_flag);
      if (i + stride < N)        try_key_simon_opt(pt, ct, k1, found_key, found_flag);
      if (i + 2ULL * stride < N) try_key_simon_opt(pt, ct, k2, found_key, found_flag);
      if (i + 3ULL * stride < N) try_key_simon_opt(pt, ct, k3, found_key, found_flag);
    }
  } else {
    for (uint64_t i = tid; i < N; i += stride) {
      uint64_t k = base_key | i;
      try_key_simon_opt(pt, ct, k, found_key, found_flag);
    }
  }
}

// ============================================================
// PRESENT-80 GPU Kernels (64-bit block)
// BASELINE: scalar S-box + pLayer_opt + pre-expanded round keys
// OPTIMIZED: byte SP-box tables + on-the-fly key schedule
// OPTIMIZED_SHARED: same SP-box round, but the table is staged in shared memory
// ============================================================

__device__ __forceinline__ void key64_to_present80_zero_hi16_dev(uint64_t key64, uint8_t key[10]) {
  #pragma unroll
  for (int j = 0; j < 8; j++) key[j] = (uint8_t)((key64 >> (j * 8)) & 0xFFu);
  key[8] = 0;
  key[9] = 0;
}

__device__ __forceinline__ void try_key_present_baseline(uint64_t pt, uint64_t ct, uint64_t key64,
                                                         uint64_t* found_key, int* found_flag) {
  uint8_t key[10];
  key64_to_present80_zero_hi16_dev(key64, key);

  uint64_t rk[PRESENT_ROUNDS + 1];
  Present80::expand_key(key, rk);
  const uint64_t out = Present80::encrypt(pt, rk);

  if (out == ct) {
    if (atomicCAS(found_flag, 0, 1) == 0) *found_key = key64;
  }
}

__device__ __forceinline__ void try_key_present_opt(uint64_t pt, uint64_t ct, uint64_t key64,
                                                    uint64_t* found_key, int* found_flag) {
  uint8_t key[10];
  key64_to_present80_zero_hi16_dev(key64, key);

  const uint64_t out = Present80::encrypt_spbox_otf_dev(pt, key);

  if (out == ct) {
    if (atomicCAS(found_flag, 0, 1) == 0) *found_key = key64;
  }
}

__device__ __forceinline__ void try_key_present_shared(uint64_t pt, uint64_t ct, uint64_t key64,
                                                       const uint64_t* spbox_flat,
                                                       uint64_t* found_key, int* found_flag) {
  uint8_t key[10];
  key64_to_present80_zero_hi16_dev(key64, key);

  const uint64_t out = Present80::encrypt_spbox_otf_shared(pt, key, spbox_flat);

  if (out == ct) {
    if (atomicCAS(found_flag, 0, 1) == 0) *found_key = key64;
  }
}

template<bool ILP4>
__global__ void bf_kernel_present_baseline(uint64_t pt, uint64_t ct, uint64_t base_key, uint64_t N,
                                           uint64_t* found_key, int* found_flag) {
  const uint64_t tid = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
  const uint64_t stride = (uint64_t)gridDim.x * (uint64_t)blockDim.x;

  if constexpr (ILP4) {
    for (uint64_t i = tid; i < N; i += stride * 4ULL) {
      if (i < N)                 try_key_present_baseline(pt, ct, base_key | i, found_key, found_flag);
      if (i + stride < N)        try_key_present_baseline(pt, ct, base_key | (i + stride), found_key, found_flag);
      if (i + 2ULL * stride < N) try_key_present_baseline(pt, ct, base_key | (i + 2ULL * stride), found_key, found_flag);
      if (i + 3ULL * stride < N) try_key_present_baseline(pt, ct, base_key | (i + 3ULL * stride), found_key, found_flag);
    }
  } else {
    for (uint64_t i = tid; i < N; i += stride) {
      try_key_present_baseline(pt, ct, base_key | i, found_key, found_flag);
    }
  }
}

template<bool ILP4>
__global__ void bf_kernel_present_opt(uint64_t pt, uint64_t ct, uint64_t base_key, uint64_t N,
                                      uint64_t* found_key, int* found_flag) {
  const uint64_t tid = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
  const uint64_t stride = (uint64_t)gridDim.x * (uint64_t)blockDim.x;

  if constexpr (ILP4) {
    for (uint64_t i = tid; i < N; i += stride * 4ULL) {
      if (i < N)                 try_key_present_opt(pt, ct, base_key | i, found_key, found_flag);
      if (i + stride < N)        try_key_present_opt(pt, ct, base_key | (i + stride), found_key, found_flag);
      if (i + 2ULL * stride < N) try_key_present_opt(pt, ct, base_key | (i + 2ULL * stride), found_key, found_flag);
      if (i + 3ULL * stride < N) try_key_present_opt(pt, ct, base_key | (i + 3ULL * stride), found_key, found_flag);
    }
  } else {
    for (uint64_t i = tid; i < N; i += stride) {
      try_key_present_opt(pt, ct, base_key | i, found_key, found_flag);
    }
  }
}

template<bool ILP4>
__global__ void bf_kernel_present_shared(uint64_t pt, uint64_t ct, uint64_t base_key, uint64_t N,
                                         uint64_t* found_key, int* found_flag) {
  __shared__ uint64_t spbox_shared[8 * 256];
  for (int idx = threadIdx.x; idx < 8 * 256; idx += blockDim.x) {
    spbox_shared[idx] = PRESENT_SPBOX8_DEV[idx >> 8][idx & 0xFF];
  }
  __syncthreads();

  const uint64_t tid = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
  const uint64_t stride = (uint64_t)gridDim.x * (uint64_t)blockDim.x;

  if constexpr (ILP4) {
    for (uint64_t i = tid; i < N; i += stride * 4ULL) {
      if (i < N)                 try_key_present_shared(pt, ct, base_key | i, spbox_shared, found_key, found_flag);
      if (i + stride < N)        try_key_present_shared(pt, ct, base_key | (i + stride), spbox_shared, found_key, found_flag);
      if (i + 2ULL * stride < N) try_key_present_shared(pt, ct, base_key | (i + 2ULL * stride), spbox_shared, found_key, found_flag);
      if (i + 3ULL * stride < N) try_key_present_shared(pt, ct, base_key | (i + 3ULL * stride), spbox_shared, found_key, found_flag);
    }
  } else {
    for (uint64_t i = tid; i < N; i += stride) {
      try_key_present_shared(pt, ct, base_key | i, spbox_shared, found_key, found_flag);
    }
  }
}

// ============================================================
// SPECK64/128 GPU Kernels (ARX, 64-bit block)
// (We brute-force a 64-bit portion of the 128-bit key: low64 varies, high64=0.)
// ============================================================

__device__ inline void try_key_speck(uint64_t pt, uint64_t ct, uint64_t key64,
                                     uint64_t* found_key, int* found_flag) {
  uint8_t key128[16];
  #pragma unroll
  for (int j = 0; j < 8; j++) key128[j] = (uint8_t)((key64 >> (j * 8)) & 0xFF);
  #pragma unroll
  for (int j = 8; j < 16; j++) key128[j] = 0;

  uint32_t rk[SPECK64_128_ROUNDS];
  Speck64_128::expand_key(key128, rk);
  uint64_t out = Speck64_128::encrypt(pt, rk);

  if (out == ct) {
    if (atomicCAS(found_flag, 0, 1) == 0) {
      *found_key = key64;
    }
  }
}

__global__ void bf_kernel_speck(uint64_t pt, uint64_t ct, uint64_t base_key, uint64_t N,
                                int use_ilp, uint64_t* found_key, int* found_flag) {
  uint64_t tid = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
  uint64_t stride = (uint64_t)gridDim.x * (uint64_t)blockDim.x;

  if (use_ilp) {
    for (uint64_t i = tid; i < N; i += stride * 4ULL) {
      if (i < N)                 try_key_speck(pt, ct, (base_key | i), found_key, found_flag);
      if (i + stride < N)        try_key_speck(pt, ct, (base_key | (i + stride)), found_key, found_flag);
      if (i + 2ULL * stride < N) try_key_speck(pt, ct, (base_key | (i + 2ULL * stride)), found_key, found_flag);
      if (i + 3ULL * stride < N) try_key_speck(pt, ct, (base_key | (i + 3ULL * stride)), found_key, found_flag);
    }
  } else {
    for (uint64_t i = tid; i < N; i += stride) {
      try_key_speck(pt, ct, (base_key | i), found_key, found_flag);
    }
  }
}

// ============================================================
// Grain v1 GPU Kernels
// ============================================================

__device__ inline void try_key_grain(const uint8_t* pt, const uint8_t* ct, int length,
                                     const uint8_t iv[8], uint64_t key64,
                                     uint64_t* found_key, int* found_flag) {
  uint8_t key[10];
  #pragma unroll
  for (int j = 0; j < 8; j++) key[j] = (uint8_t)((key64 >> (j * 8)) & 0xFF);
  key[8] = 0; key[9] = 0;

  uint8_t out[32];
  GrainV1::process(pt, out, length, key, iv);

  bool match = true;
  for (int j = 0; j < length; j++) {
    if (out[j] != ct[j]) { match = false; break; }
  }

  if (match) {
    if (atomicCAS(found_flag, 0, 1) == 0) {
      *found_key = key64;
    }
  }
}

__global__ void bf_kernel_grain(const uint8_t* pt, const uint8_t* ct, int length,
                                const uint8_t* iv, uint64_t base_key, uint64_t N,
                                uint64_t* found_key, int* found_flag) {
  uint64_t tid = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
  uint64_t stride = (uint64_t)gridDim.x * (uint64_t)blockDim.x;

  uint8_t iv_local[8];
  #pragma unroll
  for (int j = 0; j < 8; j++) iv_local[j] = iv[j];

  for (uint64_t i = tid; i < N; i += stride) {
    uint64_t k = base_key | i;
    try_key_grain(pt, ct, length, iv_local, k, found_key, found_flag);
  }
}

// ============================================================
// Trivium GPU Kernels
// ============================================================

__device__ inline void try_key_trivium(const uint8_t* pt, const uint8_t* ct, int length,
                                       const uint8_t iv[10], uint64_t key64,
                                       uint64_t* found_key, int* found_flag) {
  uint8_t key[10];
  #pragma unroll
  for (int j = 0; j < 8; j++) key[j] = (uint8_t)((key64 >> (j * 8)) & 0xFF);
  key[8] = 0; key[9] = 0;

  uint8_t out[32];
  Trivium::process(pt, out, length, key, iv);

  bool match = true;
  for (int j = 0; j < length; j++) {
    if (out[j] != ct[j]) { match = false; break; }
  }

  if (match) {
    if (atomicCAS(found_flag, 0, 1) == 0) {
      *found_key = key64;
    }
  }
}

__global__ void bf_kernel_trivium(const uint8_t* pt, const uint8_t* ct, int length,
                                  const uint8_t* iv, uint64_t base_key, uint64_t N,
                                  uint64_t* found_key, int* found_flag) {
  uint64_t tid = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
  uint64_t stride = (uint64_t)gridDim.x * (uint64_t)blockDim.x;

  uint8_t iv_local[10];
  #pragma unroll
  for (int j = 0; j < 10; j++) iv_local[j] = iv[j];

  for (uint64_t i = tid; i < N; i += stride) {
    uint64_t k = base_key | i;
    try_key_trivium(pt, ct, length, iv_local, k, found_key, found_flag);
  }
}

// ============================================================
// ChaCha20 GPU Kernels (ARX stream cipher)
// (We brute-force a 64-bit portion of the 256-bit key: low64 varies, rest=0.)
// ============================================================

__device__ inline void try_key_chacha20(const uint8_t* pt, const uint8_t* ct, int length,
                                        const uint8_t nonce[12], uint64_t key64,
                                        uint64_t* found_key, int* found_flag) {
  uint8_t key256[32];
  #pragma unroll
  for (int j = 0; j < 8; j++) key256[j] = (uint8_t)((key64 >> (j * 8)) & 0xFF);
  #pragma unroll
  for (int j = 8; j < 32; j++) key256[j] = 0;

  uint8_t out[64];
  ChaCha20::process(pt, out, length, key256, 1 /*counter*/, nonce);

  bool match = true;
  for (int j = 0; j < length; j++) {
    if (out[j] != ct[j]) { match = false; break; }
  }

  if (match) {
    if (atomicCAS(found_flag, 0, 1) == 0) {
      *found_key = key64;
    }
  }
}

__global__ void bf_kernel_chacha20(const uint8_t* pt, const uint8_t* ct, int length,
                                   const uint8_t* nonce12, uint64_t base_key, uint64_t N,
                                   uint64_t* found_key, int* found_flag) {
  uint64_t tid = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
  uint64_t stride = (uint64_t)gridDim.x * (uint64_t)blockDim.x;

  uint8_t nonce_local[12];
  #pragma unroll
  for (int j = 0; j < 12; j++) nonce_local[j] = nonce12[j];

  for (uint64_t i = tid; i < N; i += stride) {
    uint64_t k = base_key | i;
    try_key_chacha20(pt, ct, length, nonce_local, k, found_key, found_flag);
  }
}

// ============================================================
// Salsa20 GPU Kernels (ARX stream cipher)
// (We brute-force a 64-bit portion of the 256-bit key: low64 varies, rest=0.)
// ============================================================

__device__ inline void try_key_salsa20(const uint8_t* pt, const uint8_t* ct, int length,
                                       const uint8_t nonce[8], uint64_t key64,
                                       uint64_t* found_key, int* found_flag) {
  uint8_t key256[32];
  #pragma unroll
  for (int j = 0; j < 8; j++) key256[j] = (uint8_t)((key64 >> (j * 8)) & 0xFF);
  #pragma unroll
  for (int j = 8; j < 32; j++) key256[j] = 0;

  uint8_t out[64];
  Salsa20::process(pt, out, length, key256, nonce);

  bool match = true;
  for (int j = 0; j < length; j++) {
    if (out[j] != ct[j]) { match = false; break; }
  }

  if (match) {
    if (atomicCAS(found_flag, 0, 1) == 0) {
      *found_key = key64;
    }
  }
}

__global__ void bf_kernel_salsa20(const uint8_t* pt, const uint8_t* ct, int length,
                                  const uint8_t* nonce8, uint64_t base_key, uint64_t N,
                                  uint64_t* found_key, int* found_flag) {
  uint64_t tid = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
  uint64_t stride = (uint64_t)gridDim.x * (uint64_t)blockDim.x;

  uint8_t nonce_local[8];
  #pragma unroll
  for (int j = 0; j < 8; j++) nonce_local[j] = nonce8[j];

  for (uint64_t i = tid; i < N; i += stride) {
    uint64_t k = base_key | i;
    try_key_salsa20(pt, ct, length, nonce_local, k, found_key, found_flag);
  }
}

__device__ __forceinline__ bool salsa_match_prefix_words4(const uint32_t out4[4],
                                                          const uint8_t* target, int len) {
  int n = (len > 16) ? 16 : len;
  #pragma unroll
  for (int i = 0; i < 16; i++) {
    if (i >= n) break;
    uint8_t b = (uint8_t)((out4[i >> 2] >> (8 * (i & 3))) & 0xFF);
    if (b != target[i]) return false;
  }
  return true;
}

// gpu1_optimized: early-exit by matching keystream prefix (block_words4, counter=0)
// gpu2_optimized+ilp: ILP2 (2 keys per loop) + early-exit
template<bool ILP2, bool STOP_ON_FOUND>
__global__ void bf_kernel_salsa20_match(const uint8_t* target, int data_len, const uint8_t* nonce8,
                                        uint64_t known_high, int unknown_bits, uint64_t N,
                                        volatile int* d_found, uint64_t* d_found_key) {
  uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t stride = (uint64_t)gridDim.x * blockDim.x;

  auto test_one = [=] __device__ (uint64_t low) {
    if (STOP_ON_FOUND && *d_found) return;
    uint64_t key64 = (known_high << unknown_bits) | low;

    uint8_t key256[32] = {0};
    #pragma unroll
    for (int i = 0; i < 8; i++) key256[i] = (uint8_t)((key64 >> (8*i)) & 0xFF);

    uint32_t out4[4];
    Salsa20::block_words4(key256, 0ULL, nonce8, out4);

    if (salsa_match_prefix_words4(out4, target, data_len)) {
      if (atomicCAS((int*)d_found, 0, 1) == 0) *d_found_key = key64;
    }
  };

  if constexpr (ILP2) {
    for (uint64_t k = tid; k < N && (!STOP_ON_FOUND || !(*d_found)); k += stride * 2ULL) {
      test_one(k);
      uint64_t k2 = k + stride;
      if (k2 < N) test_one(k2);
    }
  } else {
    for (uint64_t k = tid; k < N && (!STOP_ON_FOUND || !(*d_found)); k += stride) test_one(k);
  }
}

// ============================================================
// Unified GPU Brute Force Interface
// ============================================================

enum class CipherType {
  SIMON32_64,
  PRESENT80,
  SPECK64_128,
  GRAIN_V1,
  TRIVIUM,
  CHACHA20,
  TINYJAMBU_128,
  SNOW_V,
  ZUC_128,
  AES_128,
  SALSA20,
  GRAIN128_AEADV2
};

// ============================================================
// Stream/ARX cipher optimized GPU kernels (match keystream prefix)
//   OPTIMIZED: early-reject by comparing keystream bytes (pt^ct)
//   OPTIMIZED_ILP: 2 keys per loop (ILP2) + early-reject
// ============================================================

template<bool ILP2, bool STOP_ON_FOUND>
__global__ void bf_kernel_grain_match(const uint8_t* target, int data_len, const uint8_t* iv8,
                                      uint64_t known_high, int unknown_bits, uint64_t N,
                                      volatile int* d_found, uint64_t* d_found_key) {
  uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t stride = (uint64_t)gridDim.x * blockDim.x;

  auto test_one = [=] __device__ (uint64_t low) {
    if (STOP_ON_FOUND && *d_found) return;
    uint64_t key64 = (known_high << unknown_bits) | low;

    uint8_t key80[10] = {0};
    // low 64 bits in little-endian bytes, top 16 bits = 0
    #pragma unroll
    for (int i = 0; i < 8; i++) key80[i] = (uint8_t)((key64 >> (8*i)) & 0xFF);

    if (GrainV1::match_keystream(key80, iv8, target, data_len)) {
      if (atomicCAS((int*)d_found, 0, 1) == 0) *d_found_key = key64;
    }
  };

  if constexpr (ILP2) {
    for (uint64_t k = tid; k < N && (!STOP_ON_FOUND || !(*d_found)); k += stride * 2ULL) {
      test_one(k);
      uint64_t k2 = k + stride;
      if (k2 < N) test_one(k2);
    }
  } else {
    for (uint64_t k = tid; k < N && (!STOP_ON_FOUND || !(*d_found)); k += stride) test_one(k);
  }
}

template<bool ILP2, bool STOP_ON_FOUND>
__global__ void bf_kernel_trivium_match(const uint8_t* target, int data_len, const uint8_t* iv10,
                                        uint64_t known_high, int unknown_bits, uint64_t N,
                                        volatile int* d_found, uint64_t* d_found_key) {
  uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t stride = (uint64_t)gridDim.x * blockDim.x;

  auto test_one = [=] __device__ (uint64_t low) {
    if (STOP_ON_FOUND && *d_found) return;
    uint64_t key64 = (known_high << unknown_bits) | low;

    uint8_t key80[10] = {0};
    #pragma unroll
    for (int i = 0; i < 8; i++) key80[i] = (uint8_t)((key64 >> (8*i)) & 0xFF);

    if (Trivium::match_keystream(key80, iv10, target, data_len)) {
      if (atomicCAS((int*)d_found, 0, 1) == 0) *d_found_key = key64;
    }
  };

  if constexpr (ILP2) {
    for (uint64_t k = tid; k < N && (!STOP_ON_FOUND || !(*d_found)); k += stride * 2ULL) {
      test_one(k);
      uint64_t k2 = k + stride;
      if (k2 < N) test_one(k2);
    }
  } else {
    for (uint64_t k = tid; k < N && (!STOP_ON_FOUND || !(*d_found)); k += stride) test_one(k);
  }
}

__device__ __forceinline__ bool chacha_match_prefix_words4(const uint32_t out4[4],
                                                          const uint8_t* target, int len) {
  // Compare up to 16 bytes (len <= 16 in our benchmark)
  int n = (len > 16) ? 16 : len;
  #pragma unroll
  for (int i = 0; i < 16; i++) {
    if (i >= n) break;
    uint8_t b = (uint8_t)((out4[i >> 2] >> (8 * (i & 3))) & 0xFF);
    if (b != target[i]) return false;
  }
  return true;
}

template<bool ILP2, bool STOP_ON_FOUND>
__global__ void bf_kernel_chacha20_match(const uint8_t* target, int data_len, const uint8_t* nonce12,
                                         uint64_t known_high, int unknown_bits, uint64_t N,
                                         volatile int* d_found, uint64_t* d_found_key) {
  uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t stride = (uint64_t)gridDim.x * blockDim.x;

  auto test_one = [=] __device__ (uint64_t low) {
    if (STOP_ON_FOUND && *d_found) return;
    uint64_t key64 = (known_high << unknown_bits) | low;

    uint8_t key256[32] = {0};
    #pragma unroll
    for (int i = 0; i < 8; i++) key256[i] = (uint8_t)((key64 >> (8*i)) & 0xFF);

    uint32_t out4[4];
    ChaCha20::block_words4(key256, 1u, nonce12, out4);

    if (chacha_match_prefix_words4(out4, target, data_len)) {
      if (atomicCAS((int*)d_found, 0, 1) == 0) *d_found_key = key64;
    }
  };

  if constexpr (ILP2) {
    for (uint64_t k = tid; k < N && (!STOP_ON_FOUND || !(*d_found)); k += stride * 2ULL) {
      test_one(k);
      uint64_t k2 = k + stride;
      if (k2 < N) test_one(k2);
    }
  } else {
    for (uint64_t k = tid; k < N && (!STOP_ON_FOUND || !(*d_found)); k += stride) test_one(k);
  }
}

// ============================================================
// TinyJAMBU-128 GPU Kernels (AEAD)
// ============================================================

__device__ inline void try_key_tinyjambu(const uint8_t* pt, const uint8_t* ct, int length,
                                         const uint8_t* expected_tag, const uint8_t nonce[12],
                                         const uint8_t* ad, int ad_len, uint64_t key64,
                                         uint64_t* found_key, int* found_flag) {
  uint8_t key[16] = {0};
  #pragma unroll
  for (int j = 0; j < 8; j++) key[j] = (uint8_t)((key64 >> (j * 8)) & 0xFF);

  uint8_t out_ct[64];
  uint8_t out_tag[8];
  TinyJAMBU128::encrypt(pt, out_ct, length, out_tag, key, nonce, ad, ad_len);

  bool match = true;
  for (int j = 0; j < length; j++) {
    if (out_ct[j] != ct[j]) { match = false; break; }
  }
  if (match) {
    for (int j = 0; j < 8; j++) {
      if (out_tag[j] != expected_tag[j]) { match = false; break; }
    }
  }

  if (match && atomicCAS(found_flag, 0, 1) == 0) *found_key = key64;
}

template<bool ILP2, bool STOP_ON_FOUND>
__global__ void bf_kernel_tinyjambu(const uint8_t* pt, const uint8_t* ct, int length,
                                    const uint8_t* expected_tag, const uint8_t* nonce,
                                    const uint8_t* ad, int ad_len, uint64_t base_key, uint64_t N,
                                    uint64_t* found_key, int* found_flag) {
  uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t stride = (uint64_t)gridDim.x * blockDim.x;

  uint8_t nonce_local[12];
  #pragma unroll
  for (int j = 0; j < 12; j++) nonce_local[j] = nonce[j];

  auto test_one = [=] __device__ (uint64_t low) {
    if (STOP_ON_FOUND && *found_flag) return;
    try_key_tinyjambu(pt, ct, length, expected_tag, nonce_local, ad, ad_len, base_key | low, found_key, found_flag);
  };

  if constexpr (ILP2) {
    for (uint64_t k = tid; k < N && (!STOP_ON_FOUND || !(*found_flag)); k += stride * 2ULL) {
      test_one(k);
      if (k + stride < N) test_one(k + stride);
    }
  } else {
    for (uint64_t i = tid; i < N && (!STOP_ON_FOUND || !(*found_flag)); i += stride) test_one(i);
  }
}

__global__ void bf_kernel_tinyjambu_bitsliced(const uint8_t* pt, const uint8_t* ct, int length,
                                              const uint8_t* expected_tag, const uint8_t* nonce,
                                              const uint8_t* ad, int ad_len, uint64_t base_key, uint64_t N,
                                              uint64_t* found_key, int* found_flag) {
  uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t stride = (uint64_t)gridDim.x * blockDim.x;

  uint64_t global_key_idx = tid * 32ULL;
  uint64_t step = stride * 32ULL;

  uint8_t nonce_local[12];
  #pragma unroll
  for (int j = 0; j < 12; j++) nonce_local[j] = nonce[j];

  for (uint64_t i = global_key_idx; i < N && !(*found_flag); i += step) {
    uint32_t key_bitsliced[128] = {0};

    #pragma unroll
    for (int k = 0; k < 32; k++) {
      uint64_t current_key = (i + k < N) ? (base_key | (i + k)) : base_key;
      #pragma unroll
      for (int b = 0; b < 64; b++) {
        uint32_t bit_val = (current_key >> b) & 1;
        key_bitsliced[b] |= (bit_val << k);
      }
    }

    uint32_t match_mask = TinyJAMBU128_Bitsliced::match_keys(
      pt, ct, length, expected_tag, key_bitsliced, nonce_local, ad, ad_len
    );

    if (match_mask != 0) {
      for (int k = 0; k < 32; k++) {
        if ((match_mask >> k) & 1) {
          uint64_t actual_key = base_key | (i + k);
          if (i + k < N && atomicCAS(found_flag, 0, 1) == 0) {
            *found_key = actual_key;
          }
        }
      }
    }
  }
}

// ============================================================
// ZUC-128 GPU Kernels (stream cipher, 128-bit key/IV)
// ============================================================

__device__ inline void try_key_zuc(const uint8_t* pt, const uint8_t* ct, int length,
                                   const uint8_t iv[16], uint64_t key64,
                                   uint64_t* found_key, int* found_flag) {
  uint8_t key[16] = {0};
  #pragma unroll
  for (int j = 0; j < 8; j++) key[j] = (uint8_t)((key64 >> (j * 8)) & 0xFF);

  uint8_t out[64];
  ZUC::process(pt, out, length, key, iv);

  bool match = true;
  for (int j = 0; j < length; j++) {
    if (out[j] != ct[j]) { match = false; break; }
  }
  if (match && atomicCAS(found_flag, 0, 1) == 0) *found_key = key64;
}

__global__ void bf_kernel_zuc(const uint8_t* pt, const uint8_t* ct, int length,
                              const uint8_t* iv, uint64_t base_key, uint64_t N,
                              uint64_t* found_key, int* found_flag) {
  uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t stride = (uint64_t)gridDim.x * blockDim.x;

  uint8_t iv_local[16];
  #pragma unroll
  for (int j = 0; j < 16; j++) iv_local[j] = iv[j];

  for (uint64_t i = tid; i < N; i += stride) {
    try_key_zuc(pt, ct, length, iv_local, base_key | i, found_key, found_flag);
  }
}

template<bool ILP2, bool STOP_ON_FOUND>
__global__ void bf_kernel_zuc_match(const uint8_t* target, int data_len, const uint8_t* iv16,
                                    uint64_t known_high, int unknown_bits, uint64_t N,
                                    volatile int* d_found, uint64_t* d_found_key) {
  uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t stride = (uint64_t)gridDim.x * blockDim.x;

  auto test_one = [=] __device__ (uint64_t low) {
    if (STOP_ON_FOUND && *d_found) return;
    uint64_t key64 = (known_high << unknown_bits) | low;
    uint8_t key[16] = {0};
    #pragma unroll
    for (int i = 0; i < 8; i++) key[i] = (uint8_t)((key64 >> (8 * i)) & 0xFF);

    if (ZUC::match_keystream(key, iv16, target, data_len)) {
      if (atomicCAS((int*)d_found, 0, 1) == 0) *d_found_key = key64;
    }
  };

  if constexpr (ILP2) {
    for (uint64_t k = tid; k < N && (!STOP_ON_FOUND || !(*d_found)); k += stride * 2ULL) {
      test_one(k);
      if (k + stride < N) test_one(k + stride);
    }
  } else {
    for (uint64_t k = tid; k < N && (!STOP_ON_FOUND || !(*d_found)); k += stride) test_one(k);
  }
}

// ============================================================
// SNOW-V GPU Kernels (stream cipher, 256-bit key / 128-bit IV)
// ============================================================

__device__ inline void try_key_snow_v(const uint8_t* pt, const uint8_t* ct, int length,
                                      const uint8_t iv[16], uint64_t key64,
                                      uint64_t* found_key, int* found_flag) {
  uint8_t key[32] = {0};
  #pragma unroll
  for (int j = 0; j < 8; j++) key[j] = (uint8_t)((key64 >> (j * 8)) & 0xFF);

  uint8_t out[64];
  SNOW_V::process<false>(pt, out, length, key, iv);

  bool match = true;
  for (int j = 0; j < length; j++) {
    if (out[j] != ct[j]) { match = false; break; }
  }
  if (match && atomicCAS(found_flag, 0, 1) == 0) *found_key = key64;
}

__global__ void bf_kernel_snow_v(const uint8_t* pt, const uint8_t* ct, int length,
                                 const uint8_t* iv, uint64_t base_key, uint64_t N,
                                 uint64_t* found_key, int* found_flag) {
  uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t stride = (uint64_t)gridDim.x * blockDim.x;

  uint8_t iv_local[16];
  #pragma unroll
  for (int j = 0; j < 16; j++) iv_local[j] = iv[j];

  for (uint64_t i = tid; i < N; i += stride) {
    try_key_snow_v(pt, ct, length, iv_local, base_key | i, found_key, found_flag);
  }
}

template<bool ILP2, bool STOP_ON_FOUND>
__global__ void bf_kernel_snow_v_match(const uint8_t* target, int data_len, const uint8_t* iv16,
                                       uint64_t known_high, int unknown_bits, uint64_t N,
                                       volatile int* d_found, uint64_t* d_found_key) {
  uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t stride = (uint64_t)gridDim.x * blockDim.x;

  auto test_one = [=] __device__ (uint64_t low) {
    if (STOP_ON_FOUND && *d_found) return;
    uint64_t key64 = (known_high << unknown_bits) | low;
    uint8_t key[32] = {0};
    #pragma unroll
    for (int i = 0; i < 8; i++) key[i] = (uint8_t)((key64 >> (8 * i)) & 0xFF);

    if (SNOW_V::match_keystream<true>(key, iv16, target, data_len)) {
      if (atomicCAS((int*)d_found, 0, 1) == 0) *d_found_key = key64;
    }
  };

  if constexpr (ILP2) {
    for (uint64_t k = tid; k < N && (!STOP_ON_FOUND || !(*d_found)); k += stride * 2ULL) {
      test_one(k);
      if (k + stride < N) test_one(k + stride);
    }
  } else {
    for (uint64_t k = tid; k < N && (!STOP_ON_FOUND || !(*d_found)); k += stride) test_one(k);
  }
}

// ============================================================
// AES-128 GPU Kernels (block cipher, 128-bit block/key)
// ============================================================

__device__ __forceinline__ void key64_to_aes_key_dev(uint64_t key64, uint8_t key[16]) {
  #pragma unroll
  for (int j = 0; j < 8; j++) key[j] = (uint8_t)((key64 >> (j * 8)) & 0xFFu);
  #pragma unroll
  for (int j = 8; j < 16; j++) key[j] = 0u;
}

__device__ __forceinline__ uint32_t aes_rot_right(uint32_t val, uint32_t selector) {
    return __byte_perm(val, val, selector);
}

__device__ __forceinline__ void aes128_encrypt_tezcan(
    const uint8_t* pt, const uint32_t* rk, uint8_t* ct,
    const uint32_t T0_shr[256][32], const uint8_t S_shr[64][32][4])
{
    int lane = threadIdx.x % 32;

    uint32_t w0 = ((const uint32_t*)pt)[0] ^ rk[0];
    uint32_t w1 = ((const uint32_t*)pt)[1] ^ rk[1];
    uint32_t w2 = ((const uint32_t*)pt)[2] ^ rk[2];
    uint32_t w3 = ((const uint32_t*)pt)[3] ^ rk[3];

    #pragma unroll
    for (int round = 1; round < 10; round++) {
        uint32_t t0 = T0_shr[ w0 & 0xff ][lane] ^
                      aes_rot_right(T0_shr[ (w1 >> 8) & 0xff ][lane], 0x6543) ^
                      aes_rot_right(T0_shr[ (w2 >> 16) & 0xff ][lane], 0x5432) ^
                      aes_rot_right(T0_shr[ w3 >> 24 ][lane], 0x4321) ^ rk[4*round + 0];

        uint32_t t1 = T0_shr[ w1 & 0xff ][lane] ^
                      aes_rot_right(T0_shr[ (w2 >> 8) & 0xff ][lane], 0x6543) ^
                      aes_rot_right(T0_shr[ (w3 >> 16) & 0xff ][lane], 0x5432) ^
                      aes_rot_right(T0_shr[ w0 >> 24 ][lane], 0x4321) ^ rk[4*round + 1];

        uint32_t t2 = T0_shr[ w2 & 0xff ][lane] ^
                      aes_rot_right(T0_shr[ (w3 >> 8) & 0xff ][lane], 0x6543) ^
                      aes_rot_right(T0_shr[ (w0 >> 16) & 0xff ][lane], 0x5432) ^
                      aes_rot_right(T0_shr[ w1 >> 24 ][lane], 0x4321) ^ rk[4*round + 2];

        uint32_t t3 = T0_shr[ w3 & 0xff ][lane] ^
                      aes_rot_right(T0_shr[ (w0 >> 8) & 0xff ][lane], 0x6543) ^
                      aes_rot_right(T0_shr[ (w1 >> 16) & 0xff ][lane], 0x5432) ^
                      aes_rot_right(T0_shr[ w2 >> 24 ][lane], 0x4321) ^ rk[4*round + 3];
        w0 = t0; w1 = t1; w2 = t2; w3 = t3;
    }

    // Last round
    uint32_t f0 = (uint32_t)S_shr[(w0 & 0xff)/4][lane][(w0 & 0xff)%4] |
                 ((uint32_t)S_shr[((w1 >> 8) & 0xff)/4][lane][((w1 >> 8) & 0xff)%4] << 8) |
                 ((uint32_t)S_shr[((w2 >> 16) & 0xff)/4][lane][((w2 >> 16) & 0xff)%4] << 16) |
                 ((uint32_t)S_shr[(w3 >> 24)/4][lane][(w3 >> 24)%4] << 24);
    f0 ^= rk[40];

    uint32_t f1 = (uint32_t)S_shr[(w1 & 0xff)/4][lane][(w1 & 0xff)%4] |
                 ((uint32_t)S_shr[((w2 >> 8) & 0xff)/4][lane][((w2 >> 8) & 0xff)%4] << 8) |
                 ((uint32_t)S_shr[((w3 >> 16) & 0xff)/4][lane][((w3 >> 16) & 0xff)%4] << 16) |
                 ((uint32_t)S_shr[(w0 >> 24)/4][lane][(w0 >> 24)%4] << 24);
    f1 ^= rk[41];

    uint32_t f2 = (uint32_t)S_shr[(w2 & 0xff)/4][lane][(w2 & 0xff)%4] |
                 ((uint32_t)S_shr[((w3 >> 8) & 0xff)/4][lane][((w3 >> 8) & 0xff)%4] << 8) |
                 ((uint32_t)S_shr[((w0 >> 16) & 0xff)/4][lane][((w0 >> 16) & 0xff)%4] << 16) |
                 ((uint32_t)S_shr[(w1 >> 24)/4][lane][(w1 >> 24)%4] << 24);
    f2 ^= rk[42];

    uint32_t f3 = (uint32_t)S_shr[(w3 & 0xff)/4][lane][(w3 & 0xff)%4] |
                 ((uint32_t)S_shr[((w0 >> 8) & 0xff)/4][lane][((w0 >> 8) & 0xff)%4] << 8) |
                 ((uint32_t)S_shr[((w1 >> 16) & 0xff)/4][lane][((w1 >> 16) & 0xff)%4] << 16) |
                 ((uint32_t)S_shr[(w2 >> 24)/4][lane][(w2 >> 24)%4] << 24);
    f3 ^= rk[43];

    ((uint32_t*)ct)[0] = f0;
    ((uint32_t*)ct)[1] = f1;
    ((uint32_t*)ct)[2] = f2;
    ((uint32_t*)ct)[3] = f3;
}
// UseTTable=false → mixcolumns_naive (baseline, no GF bit-trick)
// UseTTable=true  → mixcolumns_fast  (optimized, gmul2/gmul3 bit-shifts)
template<bool UseTTable>
__device__ __forceinline__ void try_key_aes_opt(const uint8_t* pt, const uint8_t* ct, uint64_t key64,
                                                uint64_t* found_key, int* found_flag) {
  uint8_t key[16];
  key64_to_aes_key_dev(key64, key);

  uint8_t ct_computed[16];
  AES128::encrypt<UseTTable>(pt, key, ct_computed);

  bool match = true;
  #pragma unroll
  for (int i = 0; i < 16; i++) {
    if (ct_computed[i] != ct[i]) match = false;
  }

  if (match) {
    atomicExch((unsigned long long*)found_key, (unsigned long long)key64);
    atomicExch((int*)found_flag, 1);
  }
}

template<bool ILP4, bool UseTTable>
__global__ void bf_kernel_aes_opt(const uint8_t* pt, const uint8_t* ct, uint64_t base_key, uint64_t N,
                                  uint64_t* found_key, int* found_flag) {
  const uint64_t tid = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
  const uint64_t stride = (uint64_t)gridDim.x * (uint64_t)blockDim.x;

  if constexpr (ILP4) {
    for (uint64_t k = tid; k < N; k += stride * 4ULL) {
      try_key_aes_opt<UseTTable>(pt, ct, base_key | k, found_key, found_flag);
      if (k + stride < N) try_key_aes_opt<UseTTable>(pt, ct, base_key | (k + stride), found_key, found_flag);
      if (k + stride * 2ULL < N) try_key_aes_opt<UseTTable>(pt, ct, base_key | (k + stride * 2ULL), found_key, found_flag);
      if (k + stride * 3ULL < N) try_key_aes_opt<UseTTable>(pt, ct, base_key | (k + stride * 3ULL), found_key, found_flag);
    }
  } else {
    for (uint64_t k = tid; k < N; k += stride) {
      try_key_aes_opt<UseTTable>(pt, ct, base_key | k, found_key, found_flag);
    }
  }
}

// ============================================================
// Grain-128AEADv2 GPU Kernels
// ============================================================

__device__ inline void try_key_grain128aeadv2(const uint8_t* pt, const uint8_t* ct, int length,
                                             const uint8_t* expected_tag, const uint8_t nonce[12],
                                             const uint8_t* ad, int ad_len, uint64_t key64,
                                             uint64_t* found_key, int* found_flag) {
  uint8_t key[16] = {0};
  #pragma unroll
  for (int j = 0; j < 8; j++) key[j] = (uint8_t)((key64 >> (j * 8)) & 0xFF);

  uint8_t out_ct[64];
  uint8_t out_tag[8];
  Grain128AEADv2::process(pt, out_ct, length, ad, ad_len, out_tag, key, nonce);

  bool match = true;
  for (int j = 0; j < length; j++) {
    if (out_ct[j] != ct[j]) { match = false; break; }
  }
  if (match) {
    for (int j = 0; j < 8; j++) {
      if (out_tag[j] != expected_tag[j]) { match = false; break; }
    }
  }
  if (match && atomicCAS(found_flag, 0, 1) == 0) *found_key = key64;
}

template<bool ILP2, bool STOP_ON_FOUND>
__global__ void bf_kernel_grain128aeadv2(const uint8_t* pt, const uint8_t* ct, int length,
                                         const uint8_t* expected_tag, const uint8_t* nonce,
                                         const uint8_t* ad, int ad_len, uint64_t base_key, uint64_t N,
                                         uint64_t* found_key, int* found_flag) {
  uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t stride = (uint64_t)gridDim.x * blockDim.x;

  uint8_t nonce_local[12];
  #pragma unroll
  for (int j = 0; j < 12; j++) nonce_local[j] = nonce[j];

  auto test_one = [=] __device__ (uint64_t low) {
    if (STOP_ON_FOUND && *found_flag) return;
    try_key_grain128aeadv2(pt, ct, length, expected_tag, nonce_local, ad, ad_len, base_key | low, found_key, found_flag);
  };

  if constexpr (ILP2) {
    for (uint64_t k = tid; k < N && (!STOP_ON_FOUND || !(*found_flag)); k += stride * 2ULL) {
      test_one(k);
      if (k + stride < N) test_one(k + stride);
    }
  } else {
    for (uint64_t i = tid; i < N && (!STOP_ON_FOUND || !(*found_flag)); i += stride) test_one(i);
  }
}

__global__ void bf_kernel_grain128aeadv2_bitsliced(const uint8_t* pt, const uint8_t* ct, int length,
                                                   const uint8_t* nonce, const uint8_t* ad, int ad_len,
                                                   uint64_t base_key, uint64_t N,
                                                   uint64_t* found_key, int* found_flag) {
  uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t stride = (uint64_t)gridDim.x * blockDim.x;

  uint64_t global_key_idx = tid * 32ULL;
  uint64_t step = stride * 32ULL;

  uint8_t nonce_local[12];
  #pragma unroll
  for (int j = 0; j < 12; j++) nonce_local[j] = nonce[j];

  uint32_t nonce_bitsliced[96] = {0};
  #pragma unroll
  for (int i = 0; i < 96; i++) {
    uint32_t bit_val = (nonce_local[i / 8] >> (i % 8)) & 1;
    nonce_bitsliced[i] = bit_val ? 0xFFFFFFFF : 0x00000000;
  }

  for (uint64_t i = global_key_idx; i < N && !(*found_flag); i += step) {
    uint32_t key_bitsliced[128] = {0};
    #pragma unroll
    for (int k = 0; k < 32; k++) {
      uint64_t current_key = (i + k < N) ? (base_key | (i + k)) : base_key;
      #pragma unroll
      for (int b = 0; b < 64; b++) {
        uint32_t bit_val = (current_key >> b) & 1;
        key_bitsliced[b] |= (bit_val << k);
      }
    }
    uint32_t match_mask = Grain128AEADv2_Bitsliced::match_keys(
      key_bitsliced, nonce_bitsliced, ad_len, pt, ct, length
    );
    if (match_mask != 0) {
      for (int k = 0; k < 32; k++) {
        if ((match_mask >> k) & 1) {
          if (i + k < N && atomicCAS(found_flag, 0, 1) == 0)
            *found_key = base_key | (i + k);
        }
      }
    }
  }
}

inline GpuBFResult brute_force_gpu_enhanced(CipherType cipher,
                                           const void* pt,
                                           const void* ct,
                                           const void* iv_or_nonce,
                                           int data_len,
                                           uint64_t known_high,
                                           int unknown_bits,
                                           GpuVariant variant,
                                           int blocks,
                                           int threads,
                                           int repeats,
                                           cudaStream_t stream = 0) {
  GpuBFResult res;

  const uint64_t N = bf_space_size_gpu(unknown_bits);
  res.keys_tested = N;
  if (N == 0) return res;

  const uint64_t base_key = (known_high << (uint64_t)unknown_bits);

  uint64_t* d_found_key = nullptr;
  int* d_found_flag = nullptr;
  cuda_check(cudaMalloc(&d_found_key, sizeof(uint64_t)), "cudaMalloc found_key");
  cuda_check(cudaMalloc(&d_found_flag, sizeof(int)), "cudaMalloc found_flag");

  // Stream ciphers: allocate device memory for iv/nonce and (optionally) data buffers.
  uint8_t* d_pt = nullptr;
  uint8_t* d_ct = nullptr;
  uint8_t* d_target = nullptr;  // pt^ct (keystream) for OPTIMIZED variants
  uint8_t* d_iv = nullptr;

  const bool is_stream = (cipher == CipherType::GRAIN_V1 ||
                          cipher == CipherType::TRIVIUM ||
                          cipher == CipherType::CHACHA20 ||
                          cipher == CipherType::ZUC_128 ||
                          cipher == CipherType::SNOW_V ||
                          cipher == CipherType::SALSA20);

  const bool use_target = is_stream && (variant != GpuVariant::BASELINE);

  if (is_stream) {
    // IV / nonce
    int iv_len = 0;
    if (cipher == CipherType::GRAIN_V1) iv_len = 8;
    else if (cipher == CipherType::TRIVIUM) iv_len = 10;
    else if (cipher == CipherType::CHACHA20) iv_len = 12;
    else if (cipher == CipherType::ZUC_128) iv_len = 16;
    else if (cipher == CipherType::SNOW_V) iv_len = 16;
    else if (cipher == CipherType::SALSA20) iv_len = 8;

    cuda_check(cudaMalloc(&d_iv, (size_t)iv_len), "cudaMalloc iv/nonce");
    cuda_check(cudaMemcpy(d_iv, iv_or_nonce, (size_t)iv_len, cudaMemcpyHostToDevice), "memcpy iv/nonce");

    if (use_target) {
      // Precompute expected keystream prefix: ks = pt ^ ct
      const uint8_t* ptb = static_cast<const uint8_t*>(pt);
      const uint8_t* ctb = static_cast<const uint8_t*>(ct);
      std::vector<uint8_t> host_target((size_t)data_len);
      for (int i = 0; i < data_len; i++) host_target[(size_t)i] = (uint8_t)(ptb[i] ^ ctb[i]);

      cuda_check(cudaMalloc(&d_target, (size_t)data_len), "cudaMalloc target");
      cuda_check(cudaMemcpy(d_target, host_target.data(), (size_t)data_len, cudaMemcpyHostToDevice), "memcpy target");
    } else {
      cuda_check(cudaMalloc(&d_pt, (size_t)data_len), "cudaMalloc pt");
      cuda_check(cudaMalloc(&d_ct, (size_t)data_len), "cudaMalloc ct");
      cuda_check(cudaMemcpy(d_pt, pt, (size_t)data_len, cudaMemcpyHostToDevice), "memcpy pt");
      cuda_check(cudaMemcpy(d_ct, ct, (size_t)data_len, cudaMemcpyHostToDevice), "memcpy ct");
    }
  }

  // Block cipher (AES-128): allocate 16-byte device buffers for plaintext and ciphertext
  if (cipher == CipherType::AES_128) {
    cuda_check(cudaMalloc(&d_pt, 16), "cudaMalloc pt (AES)");
    cuda_check(cudaMalloc(&d_ct, 16), "cudaMalloc ct (AES)");
    cuda_check(cudaMemcpy(d_pt, pt, 16, cudaMemcpyHostToDevice), "memcpy pt (AES)");
    cuda_check(cudaMemcpy(d_ct, ct, 16, cudaMemcpyHostToDevice), "memcpy ct (AES)");
  }

  const int use_ilp = (variant == GpuVariant::OPTIMIZED_ILP) ? 1 : 0;

  auto reset = [&]() {
    cuda_check(cudaMemsetAsync(d_found_flag, 0, sizeof(int), stream), "memset flag");
    cuda_check(cudaMemsetAsync(d_found_key, 0, sizeof(uint64_t), stream), "memset key");
  };

  auto launch_timed = [&]() {
    switch (cipher) {
      case CipherType::SIMON32_64:
        bf_kernel_simon_opt<<<blocks, threads, 0, stream>>>(
          *(const uint32_t*)pt, *(const uint32_t*)ct, base_key, N, use_ilp, d_found_key, d_found_flag
        );
        break;

      case CipherType::PRESENT80:
        if (variant == GpuVariant::BASELINE) {
          if (use_ilp) {
            bf_kernel_present_baseline<true><<<blocks, threads, 0, stream>>>(
              *(const uint64_t*)pt, *(const uint64_t*)ct, base_key, N, d_found_key, d_found_flag
            );
          } else {
            bf_kernel_present_baseline<false><<<blocks, threads, 0, stream>>>(
              *(const uint64_t*)pt, *(const uint64_t*)ct, base_key, N, d_found_key, d_found_flag
            );
          }
        } else if (variant == GpuVariant::OPTIMIZED_SHARED) {
          bf_kernel_present_shared<false><<<blocks, threads, 0, stream>>>(
            *(const uint64_t*)pt, *(const uint64_t*)ct, base_key, N, d_found_key, d_found_flag
          );
        } else if (use_ilp) {
          bf_kernel_present_opt<true><<<blocks, threads, 0, stream>>>(
            *(const uint64_t*)pt, *(const uint64_t*)ct, base_key, N, d_found_key, d_found_flag
          );
        } else {
          bf_kernel_present_opt<false><<<blocks, threads, 0, stream>>>(
            *(const uint64_t*)pt, *(const uint64_t*)ct, base_key, N, d_found_key, d_found_flag
          );
        }
        break;

      case CipherType::SPECK64_128:
        bf_kernel_speck<<<blocks, threads, 0, stream>>>(
          *(const uint64_t*)pt, *(const uint64_t*)ct, base_key, N, use_ilp, d_found_key, d_found_flag
        );
        break;

      case CipherType::GRAIN_V1:
        if (variant == GpuVariant::BASELINE) {
          bf_kernel_grain<<<blocks, threads, 0, stream>>>(
            d_pt, d_ct, data_len, d_iv, base_key, N, d_found_key, d_found_flag
          );
        } else if (variant == GpuVariant::OPTIMIZED) {
          bf_kernel_grain_match<false, false><<<blocks, threads, 0, stream>>>(
            d_target, data_len, d_iv, known_high, unknown_bits, N, d_found_flag, d_found_key
          );
        } else {
          bf_kernel_grain_match<true, false><<<blocks, threads, 0, stream>>>(
            d_target, data_len, d_iv, known_high, unknown_bits, N, d_found_flag, d_found_key
          );
        }
        break;

      case CipherType::TRIVIUM:
        if (variant == GpuVariant::BASELINE) {
          bf_kernel_trivium<<<blocks, threads, 0, stream>>>(
            d_pt, d_ct, data_len, d_iv, base_key, N, d_found_key, d_found_flag
          );
        } else if (variant == GpuVariant::OPTIMIZED) {
          bf_kernel_trivium_match<false, false><<<blocks, threads, 0, stream>>>(
            d_target, data_len, d_iv, known_high, unknown_bits, N, d_found_flag, d_found_key
          );
        } else {
          bf_kernel_trivium_match<true, false><<<blocks, threads, 0, stream>>>(
            d_target, data_len, d_iv, known_high, unknown_bits, N, d_found_flag, d_found_key
          );
        }
        break;

      case CipherType::CHACHA20:
        if (variant == GpuVariant::BASELINE) {
          bf_kernel_chacha20<<<blocks, threads, 0, stream>>>(
            d_pt, d_ct, data_len, d_iv, base_key, N, d_found_key, d_found_flag
          );
        } else if (variant == GpuVariant::OPTIMIZED) {
          bf_kernel_chacha20_match<false, false><<<blocks, threads, 0, stream>>>(
            d_target, data_len, d_iv, known_high, unknown_bits, N, d_found_flag, d_found_key
          );
        } else {
          bf_kernel_chacha20_match<true, false><<<blocks, threads, 0, stream>>>(
            d_target, data_len, d_iv, known_high, unknown_bits, N, d_found_flag, d_found_key
          );
        }
        break;

      case CipherType::ZUC_128:
        if (variant == GpuVariant::BASELINE) {
          bf_kernel_zuc<<<blocks, threads, 0, stream>>>(
            d_pt, d_ct, data_len, d_iv, base_key, N, d_found_key, d_found_flag
          );
        } else if (variant == GpuVariant::OPTIMIZED) {
          bf_kernel_zuc_match<false, false><<<blocks, threads, 0, stream>>>(
            d_target, data_len, d_iv, known_high, unknown_bits, N, d_found_flag, d_found_key
          );
        } else {
          bf_kernel_zuc_match<true, false><<<blocks, threads, 0, stream>>>(
            d_target, data_len, d_iv, known_high, unknown_bits, N, d_found_flag, d_found_key
          );
        }
        break;

      case CipherType::SNOW_V:
        if (variant == GpuVariant::BASELINE) {
          bf_kernel_snow_v<<<blocks, threads, 0, stream>>>(
            d_pt, d_ct, data_len, d_iv, base_key, N, d_found_key, d_found_flag
          );
        } else if (variant == GpuVariant::OPTIMIZED) {
          bf_kernel_snow_v_match<false, false><<<blocks, threads, 0, stream>>>(
            d_target, data_len, d_iv, known_high, unknown_bits, N, d_found_flag, d_found_key
          );
        } else {
          bf_kernel_snow_v_match<true, false><<<blocks, threads, 0, stream>>>(
            d_target, data_len, d_iv, known_high, unknown_bits, N, d_found_flag, d_found_key
          );
        }
        break;

      case CipherType::AES_128:
        if (variant == GpuVariant::BASELINE) {
          // gpu0_baseline: mixcolumns_naive (no GF bit-trick)
          bf_kernel_aes_opt<false, false><<<blocks, threads, 0, stream>>>(
            d_pt, d_ct, base_key, N, d_found_key, d_found_flag
          );
        } else if (use_ilp) {
          // gpu2_optimized+ilp: mixcolumns_fast + ILP4
          bf_kernel_aes_opt<true, true><<<blocks, threads, 0, stream>>>(
            d_pt, d_ct, base_key, N, d_found_key, d_found_flag
          );
        } else {
          // gpu1_optimized: mixcolumns_fast, no ILP
          bf_kernel_aes_opt<false, true><<<blocks, threads, 0, stream>>>(
            d_pt, d_ct, base_key, N, d_found_key, d_found_flag
          );
        }
        break;

      case CipherType::SALSA20:
        if (variant == GpuVariant::BASELINE) {
          bf_kernel_salsa20<<<blocks, threads, 0, stream>>>(
            d_pt, d_ct, data_len, d_iv, base_key, N, d_found_key, d_found_flag
          );
        } else if (variant == GpuVariant::OPTIMIZED) {
          bf_kernel_salsa20_match<false, false><<<blocks, threads, 0, stream>>>(
            d_target, data_len, d_iv, known_high, unknown_bits, N, d_found_flag, d_found_key
          );
        } else {
          bf_kernel_salsa20_match<true, false><<<blocks, threads, 0, stream>>>(
            d_target, data_len, d_iv, known_high, unknown_bits, N, d_found_flag, d_found_key
          );
        }
        break;

      case CipherType::TINYJAMBU_128:
        // AEAD cipher: use brute_force_gpu_enhanced_aead() instead
        break;
    }

    cuda_check(cudaGetLastError(), "kernel launch");
  };

  auto launch_verify = [&]() {
    switch (cipher) {
      case CipherType::SIMON32_64:
        bf_kernel_simon_opt<<<blocks, threads, 0, stream>>>(
          *(const uint32_t*)pt, *(const uint32_t*)ct, base_key, N, use_ilp, d_found_key, d_found_flag
        );
        break;

      case CipherType::PRESENT80:
        if (variant == GpuVariant::BASELINE) {
          if (use_ilp) {
            bf_kernel_present_baseline<true><<<blocks, threads, 0, stream>>>(
              *(const uint64_t*)pt, *(const uint64_t*)ct, base_key, N, d_found_key, d_found_flag
            );
          } else {
            bf_kernel_present_baseline<false><<<blocks, threads, 0, stream>>>(
              *(const uint64_t*)pt, *(const uint64_t*)ct, base_key, N, d_found_key, d_found_flag
            );
          }
        } else if (variant == GpuVariant::OPTIMIZED_SHARED) {
          bf_kernel_present_shared<false><<<blocks, threads, 0, stream>>>(
            *(const uint64_t*)pt, *(const uint64_t*)ct, base_key, N, d_found_key, d_found_flag
          );
        } else if (use_ilp) {
          bf_kernel_present_opt<true><<<blocks, threads, 0, stream>>>(
            *(const uint64_t*)pt, *(const uint64_t*)ct, base_key, N, d_found_key, d_found_flag
          );
        } else {
          bf_kernel_present_opt<false><<<blocks, threads, 0, stream>>>(
            *(const uint64_t*)pt, *(const uint64_t*)ct, base_key, N, d_found_key, d_found_flag
          );
        }
        break;

      case CipherType::SPECK64_128:
        bf_kernel_speck<<<blocks, threads, 0, stream>>>(
          *(const uint64_t*)pt, *(const uint64_t*)ct, base_key, N, use_ilp, d_found_key, d_found_flag
        );
        break;

      case CipherType::GRAIN_V1:
        if (variant == GpuVariant::BASELINE) {
          bf_kernel_grain<<<blocks, threads, 0, stream>>>(
            d_pt, d_ct, data_len, d_iv, base_key, N, d_found_key, d_found_flag
          );
        } else if (variant == GpuVariant::OPTIMIZED) {
          bf_kernel_grain_match<false, true><<<blocks, threads, 0, stream>>>(
            d_target, data_len, d_iv, known_high, unknown_bits, N, d_found_flag, d_found_key
          );
        } else {
          bf_kernel_grain_match<true, true><<<blocks, threads, 0, stream>>>(
            d_target, data_len, d_iv, known_high, unknown_bits, N, d_found_flag, d_found_key
          );
        }
        break;

      case CipherType::TRIVIUM:
        if (variant == GpuVariant::BASELINE) {
          bf_kernel_trivium<<<blocks, threads, 0, stream>>>(
            d_pt, d_ct, data_len, d_iv, base_key, N, d_found_key, d_found_flag
          );
        } else if (variant == GpuVariant::OPTIMIZED) {
          bf_kernel_trivium_match<false, true><<<blocks, threads, 0, stream>>>(
            d_target, data_len, d_iv, known_high, unknown_bits, N, d_found_flag, d_found_key
          );
        } else {
          bf_kernel_trivium_match<true, true><<<blocks, threads, 0, stream>>>(
            d_target, data_len, d_iv, known_high, unknown_bits, N, d_found_flag, d_found_key
          );
        }
        break;

      case CipherType::CHACHA20:
        if (variant == GpuVariant::BASELINE) {
          bf_kernel_chacha20<<<blocks, threads, 0, stream>>>(
            d_pt, d_ct, data_len, d_iv, base_key, N, d_found_key, d_found_flag
          );
        } else if (variant == GpuVariant::OPTIMIZED) {
          bf_kernel_chacha20_match<false, true><<<blocks, threads, 0, stream>>>(
            d_target, data_len, d_iv, known_high, unknown_bits, N, d_found_flag, d_found_key
          );
        } else {
          bf_kernel_chacha20_match<true, true><<<blocks, threads, 0, stream>>>(
            d_target, data_len, d_iv, known_high, unknown_bits, N, d_found_flag, d_found_key
          );
        }
        break;

      case CipherType::ZUC_128:
        if (variant == GpuVariant::BASELINE) {
          bf_kernel_zuc<<<blocks, threads, 0, stream>>>(
            d_pt, d_ct, data_len, d_iv, base_key, N, d_found_key, d_found_flag
          );
        } else if (variant == GpuVariant::OPTIMIZED) {
          bf_kernel_zuc_match<false, true><<<blocks, threads, 0, stream>>>(
            d_target, data_len, d_iv, known_high, unknown_bits, N, d_found_flag, d_found_key
          );
        } else {
          bf_kernel_zuc_match<true, true><<<blocks, threads, 0, stream>>>(
            d_target, data_len, d_iv, known_high, unknown_bits, N, d_found_flag, d_found_key
          );
        }
        break;

      case CipherType::SNOW_V:
        if (variant == GpuVariant::BASELINE) {
          bf_kernel_snow_v<<<blocks, threads, 0, stream>>>(
            d_pt, d_ct, data_len, d_iv, base_key, N, d_found_key, d_found_flag
          );
        } else if (variant == GpuVariant::OPTIMIZED) {
          bf_kernel_snow_v_match<false, true><<<blocks, threads, 0, stream>>>(
            d_target, data_len, d_iv, known_high, unknown_bits, N, d_found_flag, d_found_key
          );
        } else {
          bf_kernel_snow_v_match<true, true><<<blocks, threads, 0, stream>>>(
            d_target, data_len, d_iv, known_high, unknown_bits, N, d_found_flag, d_found_key
          );
        }
        break;

      case CipherType::AES_128:
        if (variant == GpuVariant::BASELINE) {
          // gpu0_baseline: mixcolumns_naive (no GF bit-trick)
          bf_kernel_aes_opt<false, false><<<blocks, threads, 0, stream>>>(
            d_pt, d_ct, base_key, N, d_found_key, d_found_flag
          );
        } else if (use_ilp) {
          // gpu2_optimized+ilp: mixcolumns_fast + ILP4
          bf_kernel_aes_opt<true, true><<<blocks, threads, 0, stream>>>(
            d_pt, d_ct, base_key, N, d_found_key, d_found_flag
          );
        } else {
          // gpu1_optimized: mixcolumns_fast, no ILP
          bf_kernel_aes_opt<false, true><<<blocks, threads, 0, stream>>>(
            d_pt, d_ct, base_key, N, d_found_key, d_found_flag
          );
        }
        break;

      case CipherType::SALSA20:
        if (variant == GpuVariant::BASELINE) {
          bf_kernel_salsa20<<<blocks, threads, 0, stream>>>(
            d_pt, d_ct, data_len, d_iv, base_key, N, d_found_key, d_found_flag
          );
        } else if (variant == GpuVariant::OPTIMIZED) {
          bf_kernel_salsa20_match<false, true><<<blocks, threads, 0, stream>>>(
            d_target, data_len, d_iv, known_high, unknown_bits, N, d_found_flag, d_found_key
          );
        } else {
          bf_kernel_salsa20_match<true, true><<<blocks, threads, 0, stream>>>(
            d_target, data_len, d_iv, known_high, unknown_bits, N, d_found_flag, d_found_key
          );
        }
        break;

      case CipherType::TINYJAMBU_128:
        // AEAD cipher: use brute_force_gpu_enhanced_aead() instead
        break;
    }

    cuda_check(cudaGetLastError(), "kernel launch");
  };

  const int warmup_iters = 2;
  const int timed_iters = (repeats > 0) ? repeats : 1;

  res.seconds = time_kernel_seconds_stream(reset, launch_timed, warmup_iters, timed_iters, stream);

  // Run one final time to check found key
  reset();
  launch_verify();
  cuda_check(cudaStreamSynchronize(stream), "stream sync after final run");

  int h_flag = 0;
  uint64_t h_key = 0;
  cuda_check(cudaMemcpy(&h_flag, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost), "memcpy flag D2H");
  cuda_check(cudaMemcpy(&h_key, d_found_key, sizeof(uint64_t), cudaMemcpyDeviceToHost), "memcpy key D2H");

  res.found = (h_flag != 0);
  res.found_key = h_key;

  cudaFree(d_found_key);
  cudaFree(d_found_flag);
  if (d_pt) cudaFree(d_pt);
  if (d_ct) cudaFree(d_ct);
  if (d_target) cudaFree(d_target);
  if (d_iv) cudaFree(d_iv);

  return res;
}

// ============================================================
// AEAD brute-force interface (TinyJAMBU-128)
// ============================================================

inline GpuBFResult brute_force_gpu_enhanced_aead(CipherType cipher,
                                                 const void* pt, const void* ct,
                                                 const void* nonce, int pt_len,
                                                 const void* ad, int ad_len,
                                                 const void* expected_tag,
                                                 uint64_t known_high, int unknown_bits,
                                                 GpuVariant variant,
                                                 int blocks, int threads, int repeats,
                                                 cudaStream_t stream = 0) {
  GpuBFResult res;
  const uint64_t N = bf_space_size_gpu(unknown_bits);
  res.keys_tested = N;
  if (N == 0) return res;

  const uint64_t base_key = (known_high << (uint64_t)unknown_bits);

  uint64_t* d_found_key = nullptr;
  int*      d_found_flag = nullptr;
  cuda_check(cudaMalloc(&d_found_key, sizeof(uint64_t)), "cudaMalloc found_key");
  cuda_check(cudaMalloc(&d_found_flag, sizeof(int)),     "cudaMalloc found_flag");

  uint8_t *d_pt = nullptr, *d_ct = nullptr, *d_nonce = nullptr, *d_ad = nullptr, *d_tag = nullptr;

  cuda_check(cudaMalloc(&d_nonce, 12), "cudaMalloc nonce");
  cuda_check(cudaMemcpy(d_nonce, nonce, 12, cudaMemcpyHostToDevice), "memcpy nonce");
  cuda_check(cudaMalloc(&d_tag, 8), "cudaMalloc tag");
  cuda_check(cudaMemcpy(d_tag, expected_tag, 8, cudaMemcpyHostToDevice), "memcpy tag");

  if (pt_len > 0) {
    cuda_check(cudaMalloc(&d_pt, (size_t)pt_len), "cudaMalloc pt");
    cuda_check(cudaMalloc(&d_ct, (size_t)pt_len), "cudaMalloc ct");
    cuda_check(cudaMemcpy(d_pt, pt, (size_t)pt_len, cudaMemcpyHostToDevice), "memcpy pt");
    cuda_check(cudaMemcpy(d_ct, ct, (size_t)pt_len, cudaMemcpyHostToDevice), "memcpy ct");
  }

  if (ad_len > 0) {
    cuda_check(cudaMalloc(&d_ad, (size_t)ad_len), "cudaMalloc ad");
    cuda_check(cudaMemcpy(d_ad, ad, (size_t)ad_len, cudaMemcpyHostToDevice), "memcpy ad");
  }

  const bool use_ilp = (variant == GpuVariant::OPTIMIZED_ILP);

  auto reset = [&]() {
    cuda_check(cudaMemsetAsync(d_found_flag, 0, sizeof(int),      stream), "memset flag");
    cuda_check(cudaMemsetAsync(d_found_key,  0, sizeof(uint64_t), stream), "memset key");
  };

  // launch_timed: STOP_ON_FOUND=false for fair throughput measurement
  auto launch_timed = [&]() {
    if (cipher == CipherType::TINYJAMBU_128) {
      if (variant == GpuVariant::BITSLICED) {
        bf_kernel_tinyjambu_bitsliced<<<blocks, threads, 0, stream>>>(
          d_pt, d_ct, pt_len, d_tag, d_nonce, d_ad, ad_len, base_key, N, d_found_key, d_found_flag
        );
      } else if (use_ilp) {
        bf_kernel_tinyjambu<true, false><<<blocks, threads, 0, stream>>>(
          d_pt, d_ct, pt_len, d_tag, d_nonce, d_ad, ad_len, base_key, N, d_found_key, d_found_flag
        );
      } else {
        bf_kernel_tinyjambu<false, false><<<blocks, threads, 0, stream>>>(
          d_pt, d_ct, pt_len, d_tag, d_nonce, d_ad, ad_len, base_key, N, d_found_key, d_found_flag
        );
      }
    } else if (cipher == CipherType::GRAIN128_AEADV2) {
      if (variant == GpuVariant::BITSLICED) {
        bf_kernel_grain128aeadv2_bitsliced<<<blocks, threads, 0, stream>>>(
          d_pt, d_ct, pt_len, d_nonce, d_ad, ad_len, base_key, N, d_found_key, d_found_flag
        );
      } else if (use_ilp) {
        bf_kernel_grain128aeadv2<true, false><<<blocks, threads, 0, stream>>>(
          d_pt, d_ct, pt_len, d_tag, d_nonce, d_ad, ad_len, base_key, N, d_found_key, d_found_flag
        );
      } else {
        bf_kernel_grain128aeadv2<false, false><<<blocks, threads, 0, stream>>>(
          d_pt, d_ct, pt_len, d_tag, d_nonce, d_ad, ad_len, base_key, N, d_found_key, d_found_flag
        );
      }
    }
    cuda_check(cudaGetLastError(), "AEAD kernel launch timed");
  };

  // launch_verify: STOP_ON_FOUND=true to get the actual key on final run
  auto launch_verify = [&]() {
    if (cipher == CipherType::TINYJAMBU_128) {
      if (variant == GpuVariant::BITSLICED) {
        bf_kernel_tinyjambu_bitsliced<<<blocks, threads, 0, stream>>>(
          d_pt, d_ct, pt_len, d_tag, d_nonce, d_ad, ad_len, base_key, N, d_found_key, d_found_flag
        );
      } else if (use_ilp) {
        bf_kernel_tinyjambu<true, true><<<blocks, threads, 0, stream>>>(
          d_pt, d_ct, pt_len, d_tag, d_nonce, d_ad, ad_len, base_key, N, d_found_key, d_found_flag
        );
      } else {
        bf_kernel_tinyjambu<false, true><<<blocks, threads, 0, stream>>>(
          d_pt, d_ct, pt_len, d_tag, d_nonce, d_ad, ad_len, base_key, N, d_found_key, d_found_flag
        );
      }
    } else if (cipher == CipherType::GRAIN128_AEADV2) {
      if (variant == GpuVariant::BITSLICED) {
        bf_kernel_grain128aeadv2_bitsliced<<<blocks, threads, 0, stream>>>(
          d_pt, d_ct, pt_len, d_nonce, d_ad, ad_len, base_key, N, d_found_key, d_found_flag
        );
      } else if (use_ilp) {
        bf_kernel_grain128aeadv2<true, true><<<blocks, threads, 0, stream>>>(
          d_pt, d_ct, pt_len, d_tag, d_nonce, d_ad, ad_len, base_key, N, d_found_key, d_found_flag
        );
      } else {
        bf_kernel_grain128aeadv2<false, true><<<blocks, threads, 0, stream>>>(
          d_pt, d_ct, pt_len, d_tag, d_nonce, d_ad, ad_len, base_key, N, d_found_key, d_found_flag
        );
      }
    }
    cuda_check(cudaGetLastError(), "AEAD kernel launch verify");
  };

  const int warmup_iters = 2;
  const int timed_iters  = (repeats > 0) ? repeats : 1;

  res.seconds = time_kernel_seconds_stream(reset, launch_timed, warmup_iters, timed_iters, stream);

  reset();
  launch_verify();
  cuda_check(cudaStreamSynchronize(stream), "stream sync after final run");

  int h_flag = 0;
  uint64_t h_key = 0;
  cuda_check(cudaMemcpy(&h_flag, d_found_flag, sizeof(int),      cudaMemcpyDeviceToHost), "memcpy flag D2H");
  cuda_check(cudaMemcpy(&h_key,  d_found_key,  sizeof(uint64_t), cudaMemcpyDeviceToHost), "memcpy key D2H");

  res.found     = (h_flag != 0);
  res.found_key = h_key;

  cudaFree(d_found_key);
  cudaFree(d_found_flag);
  cudaFree(d_nonce);
  cudaFree(d_tag);
  if (d_pt)  cudaFree(d_pt);
  if (d_ct)  cudaFree(d_ct);
  if (d_ad)  cudaFree(d_ad);

  return res;
}

// ============================================================
// Bonus: PDEP (Bit Deposit) — maps sweep index to key bits
// Deposits bits of 'v' into positions marked by 'mask'
// Example: pdep64(0b11, 0b1010) -> 0b1010
// For LOW_BITS strategy (mask=(1<<b)-1) this is identical to 'v'
// ============================================================

__host__ __device__ inline uint64_t pdep64(uint64_t v, uint64_t mask) {
  uint64_t result = 0;
  uint64_t bit = 1ULL;
  uint64_t m = mask;
  while (m) {
    uint64_t lsb = m & (uint64_t)(-(int64_t)m);
    if (v & bit) result |= lsb;
    bit <<= 1;
    m &= m - 1ULL;
  }
  return result;
}

// ============================================================
// Bonus: Key Bit Selection Strategy
// ============================================================

enum class KeyStrategy {
  LOW_BITS    = 0,  // default: bits [0..b-1] are unknown
  HIGH_BITS   = 1,  // bits [64-b..63] are unknown
  INTERLEAVED = 2,  // every other bit: 0,2,4,... (b bits)
  RANDOM_BITS = 3,  // b randomly chosen bit positions
};

inline const char* key_strategy_name(KeyStrategy s) {
  switch (s) {
    case KeyStrategy::LOW_BITS:    return "strategy_low_bits";
    case KeyStrategy::HIGH_BITS:   return "strategy_high_bits";
    case KeyStrategy::INTERLEAVED: return "strategy_interleaved";
    case KeyStrategy::RANDOM_BITS: return "strategy_random_bits";
    default: return "strategy_unknown";
  }
}

// A KeyMask encodes which bits are unknown (mask) and what
// values the known bits hold (fixed_bits).
// Full key candidate = fixed_bits | pdep64(sweep_index, mask)
struct KeyMask {
  uint64_t mask;       // exactly b bits set = positions of unknown bits
  uint64_t fixed_bits; // value at all non-mask positions (known bits)
};

// Build a KeyMask for a given strategy.
// full_key64: the true key (sets fixed_bits correctly)
// b: number of unknown bits (sweep space = 2^b)
inline KeyMask make_key_mask(KeyStrategy s, int b, uint64_t full_key64, unsigned seed = 42) {
  if (b <= 0)  return {0ULL, full_key64};
  if (b >= 64) return {~0ULL, 0ULL};

  uint64_t mask = 0;
  switch (s) {
    case KeyStrategy::LOW_BITS:
      mask = (1ULL << b) - 1ULL;
      break;
    case KeyStrategy::HIGH_BITS:
      mask = ((1ULL << b) - 1ULL) << (64 - b);
      break;
    case KeyStrategy::INTERLEAVED: {
      int count = 0;
      for (int pos = 0; pos < 64 && count < b; pos += 2, ++count) mask |= (1ULL << pos);
      for (int pos = 1; pos < 64 && count < b; pos += 2, ++count) mask |= (1ULL << pos);
      break;
    }
    case KeyStrategy::RANDOM_BITS: {
      int perm[64]; std::iota(perm, perm + 64, 0);
      std::mt19937 rng(seed);
      for (int i = 63; i > 0; --i) {
        int j = (int)std::uniform_int_distribution<int>(0, i)(rng);
        std::swap(perm[i], perm[j]);
      }
      for (int i = 0; i < b; ++i) mask |= (1ULL << perm[i]);
      break;
    }
  }
  return {mask, full_key64 & ~mask};
}

// ============================================================
// Bonus: Strategy-Aware GPU Kernel for SIMON 32/64
// candidate key = km.fixed_bits | pdep64(sweep_idx, km.mask)
// Generalises the default LOW_BITS sweep to any b-bit subset
// ============================================================

template<bool ILP2, bool STOP_ON_FOUND>
__global__ void bf_kernel_simon_keymask(uint32_t pt, uint32_t ct,
                                         uint64_t kmask, uint64_t kfixed,
                                         uint64_t N,
                                         uint64_t* found_key, int* found_flag) {
  const uint64_t tid    = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
  const uint64_t stride = (uint64_t)gridDim.x  * (uint64_t)blockDim.x;

  if constexpr (ILP2) {
    for (uint64_t i = tid; i < N; i += stride * 2ULL) {
      if constexpr (STOP_ON_FOUND) { if (*found_flag) return; }
      uint64_t k0 = kfixed | pdep64(i,          kmask);
      uint64_t k1 = kfixed | pdep64(i + stride, kmask);
      uint16_t rk0[SIMON32_64_ROUNDS], rk1[SIMON32_64_ROUNDS];
      Simon32_64_Enhanced::expand_key_vectorized(k0, rk0);
      Simon32_64_Enhanced::expand_key_vectorized(k1, rk1);
      if (Simon32_64_Enhanced::encrypt_optimized(pt, rk0) == ct)
        if (atomicCAS(found_flag, 0, 1) == 0) *found_key = k0;
      if ((i + stride) < N)
        if (Simon32_64_Enhanced::encrypt_optimized(pt, rk1) == ct)
          if (atomicCAS(found_flag, 0, 1) == 0) *found_key = k1;
    }
  } else {
    for (uint64_t i = tid; i < N; i += stride) {
      if constexpr (STOP_ON_FOUND) { if (*found_flag) return; }
      uint64_t k = kfixed | pdep64(i, kmask);
      uint16_t rk[SIMON32_64_ROUNDS];
      Simon32_64_Enhanced::expand_key_vectorized(k, rk);
      if (Simon32_64_Enhanced::encrypt_optimized(pt, rk) == ct)
        if (atomicCAS(found_flag, 0, 1) == 0) *found_key = k;
    }
  }
}

// Launch the strategy-aware SIMON keymask brute force
inline GpuBFResult brute_force_gpu_keymask_simon(
    uint32_t pt, uint32_t ct,
    KeyMask km, int unknown_bits,
    int blocks, int threads, int repeats)
{
  GpuBFResult res;
  const uint64_t N = (unknown_bits >= 63) ? 0ULL : (1ULL << (uint64_t)unknown_bits);
  res.keys_tested = N;
  if (N == 0) return res;

  cudaStream_t stream;
  cuda_check(cudaStreamCreate(&stream), "stream create keymask");

  uint64_t *d_found_key  = nullptr;
  int      *d_found_flag = nullptr;
  cuda_check(cudaMalloc(&d_found_key,  sizeof(uint64_t)), "malloc key keymask");
  cuda_check(cudaMalloc(&d_found_flag, sizeof(int)),      "malloc flag keymask");

  auto reset = [&]() {
    cuda_check(cudaMemsetAsync(d_found_key,  0, sizeof(uint64_t), stream), "memset key keymask");
    cuda_check(cudaMemsetAsync(d_found_flag, 0, sizeof(int),      stream), "memset flag keymask");
  };
  auto launch_timed = [&]() {
    bf_kernel_simon_keymask<true, false><<<blocks, threads, 0, stream>>>(
      pt, ct, km.mask, km.fixed_bits, N, d_found_key, d_found_flag);
    cuda_check(cudaGetLastError(), "kernel keymask timed");
  };

  const int iters = (repeats > 0) ? repeats : 1;
  res.seconds = time_kernel_seconds_stream(reset, launch_timed, 2, iters, stream);

  // Final verify run with early-exit to capture the found key
  reset();
  bf_kernel_simon_keymask<true, true><<<blocks, threads, 0, stream>>>(
    pt, ct, km.mask, km.fixed_bits, N, d_found_key, d_found_flag);
  cuda_check(cudaStreamSynchronize(stream), "sync keymask verify");

  int h_flag = 0; uint64_t h_key = 0;
  cuda_check(cudaMemcpy(&h_flag, d_found_flag, sizeof(int),      cudaMemcpyDeviceToHost), "D2H flag keymask");
  cuda_check(cudaMemcpy(&h_key,  d_found_key,  sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H key keymask");
  res.found     = (h_flag != 0);
  res.found_key = h_key;

  cudaFree(d_found_key); cudaFree(d_found_flag);
  cudaStreamDestroy(stream);
  return res;
}

// ============================================================
// Bonus: Multi-GPU Brute Force
// Partitions [base_key, base_key+2^b) across all available GPUs.
// Each GPU runs on its own CUDA stream in a parallel std::thread.
// Wall-clock time measures the true parallel execution benefit.
// ============================================================

struct MultiGpuBFResult {
  bool found = false;
  uint64_t found_key = 0;
  double wall_seconds = 0.0;        // actual elapsed wall time (parallel)
  std::vector<double> per_gpu_kps;  // keys/s measured on each GPU
  uint64_t total_keys = 0;
  int num_gpus = 0;
};

// Multi-GPU brute force for SIMON 32/64 (ILP2 kernel on each device)
// known_high: fixed upper bits of key (same convention as brute_force_gpu_enhanced)
// unknown_bits: number of low bits to sweep (space = 2^unknown_bits)
inline MultiGpuBFResult brute_force_multi_gpu_simon(
    uint32_t pt, uint32_t ct,
    uint64_t known_high, int unknown_bits,
    int blocks, int threads, int repeats)
{
  MultiGpuBFResult res;
  int ngpu = 0;
  cuda_check(cudaGetDeviceCount(&ngpu), "cudaGetDeviceCount");
  if (ngpu < 1) { res.num_gpus = 0; return res; }
  res.num_gpus = ngpu;

  const uint64_t N = (unknown_bits >= 63) ? 0ULL : (1ULL << (uint64_t)unknown_bits);
  res.total_keys = N;
  if (N == 0) return res;

  // Align base_key the same way brute_force_gpu_enhanced does
  const uint64_t base_key = known_high << (uint64_t)unknown_bits;

  // Partition key space evenly: GPU g handles indices [g*chunk, (g+1)*chunk)
  const uint64_t chunk = (N + (uint64_t)ngpu - 1) / (uint64_t)ngpu;
  std::vector<GpuBFResult>   gpu_results(ngpu);
  res.per_gpu_kps.resize(ngpu, 0.0);
  std::mutex result_mutex;

  auto wall_start = std::chrono::high_resolution_clock::now();

  {
    std::vector<std::thread> workers;
    workers.reserve(ngpu);
    for (int g = 0; g < ngpu; ++g) {
      workers.emplace_back([&, g]() {
        cudaSetDevice(g);
        const uint64_t lo      = (uint64_t)g * chunk;
        const uint64_t hi      = std::min(lo + chunk, N);
        const uint64_t local_N = (hi > lo) ? (hi - lo) : 0ULL;
        if (local_N == 0) return;
        const uint64_t local_base = base_key + lo;

        cudaStream_t stream; cudaStreamCreate(&stream);
        uint64_t *d_key = nullptr; int *d_flag = nullptr;
        cudaMalloc(&d_key,  sizeof(uint64_t));
        cudaMalloc(&d_flag, sizeof(int));

        auto reset = [&]() {
          cudaMemsetAsync(d_key,  0, sizeof(uint64_t), stream);
          cudaMemsetAsync(d_flag, 0, sizeof(int),      stream);
        };
        auto launch = [&]() {
          bf_kernel_simon_opt<<<blocks, threads, 0, stream>>>(
            pt, ct, local_base, local_N, /*use_ilp=*/1, d_key, d_flag);
        };

        const int iters = (repeats > 0) ? repeats : 1;
        gpu_results[g].seconds    = time_kernel_seconds_stream(reset, launch, 2, iters, stream);
        gpu_results[g].keys_tested = local_N;
        res.per_gpu_kps[g] = (gpu_results[g].seconds > 0.0) ?
                              (double)local_N / gpu_results[g].seconds : 0.0;

        // Final key-find pass
        reset();
        bf_kernel_simon_opt<<<blocks, threads, 0, stream>>>(
          pt, ct, local_base, local_N, 1, d_key, d_flag);
        cudaStreamSynchronize(stream);

        int h_flag = 0; uint64_t h_key = 0;
        cudaMemcpy(&h_flag, d_flag, sizeof(int),      cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_key,  d_key,  sizeof(uint64_t), cudaMemcpyDeviceToHost);
        if (h_flag) {
          std::lock_guard<std::mutex> lk(result_mutex);
          if (!res.found) { res.found = true; res.found_key = h_key; }
        }

        cudaFree(d_key); cudaFree(d_flag);
        cudaStreamDestroy(stream);
      });
    }
    for (auto& t : workers) t.join();
  }

  auto wall_end = std::chrono::high_resolution_clock::now();
  res.wall_seconds = std::chrono::duration<double>(wall_end - wall_start).count();
  cudaSetDevice(0); // restore default
  return res;
}

#endif // __CUDACC__
