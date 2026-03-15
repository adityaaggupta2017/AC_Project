#pragma once

#include <cstdint>

// ============================================================
// Rotate helpers for 16-bit operations
// ============================================================

__host__ __device__ static inline uint16_t rotl16_u16(uint16_t x, unsigned r) {
  r &= 15u;
  if (r == 0) return x;
  return (uint16_t)((uint16_t)(x << r) | (uint16_t)(x >> (16u - r)));
}

__host__ __device__ static inline uint16_t rotr16_u16(uint16_t x, unsigned r) {
  r &= 15u;
  if (r == 0) return x;
  return (uint16_t)((uint16_t)(x >> r) | (uint16_t)(x << (16u - r)));
}

__host__ __device__ static inline uint16_t rotl16_u32mask(uint32_t x, unsigned r) {
  r &= 15u;
  if (r == 0) return (uint16_t)(x & 0xFFFFu);
  uint32_t y = ((x << r) | (x >> (16u - r))) & 0xFFFFu;
  return (uint16_t)y;
}

__host__ __device__ static inline uint16_t rotr16_u32mask(uint32_t x, unsigned r) {
  r &= 15u;
  if (r == 0) return (uint16_t)(x & 0xFFFFu);
  uint32_t y = ((x >> r) | (x << (16u - r))) & 0xFFFFu;
  return (uint16_t)y;
}

// ============================================================
// SIMON 32/64 - Enhanced GPU Optimizations
// ============================================================

static constexpr int SIMON32_64_ROUNDS = 32;

struct Simon32_64_Enhanced {
  uint16_t rk[SIMON32_64_ROUNDS];

  __host__ __device__ static inline uint8_t z0_bit(int i) {
    constexpr uint64_t z0 = 0x3E8958737D12B0E6ULL;
    int j = i % 62;
    int shift = 61 - j;
    return (uint8_t)((z0 >> shift) & 1ULL);
  }

  // Optimized Simon F function using 32-bit operations
  __host__ __device__ static inline uint16_t simon_f_opt(uint32_t x) {
    uint32_t s1 = ((x << 1) | (x >> 15)) & 0xFFFFu;
    uint32_t s8 = ((x << 8) | (x >> 8)) & 0xFFFFu;
    uint32_t s2 = ((x << 2) | (x >> 14)) & 0xFFFFu;
    return (uint16_t)((s1 & s8) ^ s2);
  }

  // Vectorized key expansion for better ILP
  __host__ __device__ static inline void expand_key_vectorized(uint64_t K, uint16_t out_rk[SIMON32_64_ROUNDS]) {
    // Load all 4 key words at once
    uint32_t k0 = (uint32_t)(K & 0xFFFFu);
    uint32_t k1 = (uint32_t)((K >> 16) & 0xFFFFu);
    uint32_t k2 = (uint32_t)((K >> 32) & 0xFFFFu);
    uint32_t k3 = (uint32_t)((K >> 48) & 0xFFFFu);

    out_rk[0] = (uint16_t)k0;
    out_rk[1] = (uint16_t)k1;
    out_rk[2] = (uint16_t)k2;
    out_rk[3] = (uint16_t)k3;

    const uint32_t c = 0xFFFCu;

    // Unroll first few iterations explicitly for better scheduling
    #pragma unroll 4
    for (int i = 4; i < 8; i++) {
      uint32_t tmp = rotr16_u32mask(out_rk[i-1], 3);
      tmp ^= out_rk[i-3];
      tmp ^= rotr16_u32mask(tmp, 1);
      uint32_t zi = (uint32_t)z0_bit(i - 4);
      out_rk[i] = (uint16_t)(c ^ zi ^ out_rk[i-4] ^ tmp);
    }

    #pragma unroll 8
    for (int i = 8; i < SIMON32_64_ROUNDS; i++) {
      uint32_t tmp = rotr16_u32mask(out_rk[i-1], 3);
      tmp ^= out_rk[i-3];
      tmp ^= rotr16_u32mask(tmp, 1);
      uint32_t zi = (uint32_t)z0_bit(i - 4);
      out_rk[i] = (uint16_t)(c ^ zi ^ out_rk[i-4] ^ tmp);
    }
  }

  // Optimized encryption with 32-bit operations and ILP
  __host__ __device__ static inline uint32_t encrypt_optimized(uint32_t pt, const uint16_t rk[SIMON32_64_ROUNDS]) {
    uint32_t x = pt & 0xFFFFu;
    uint32_t y = (pt >> 16) & 0xFFFFu;

    // Process 4 rounds at a time for better ILP
    #pragma unroll 8
    for (int i = 0; i < SIMON32_64_ROUNDS; i += 4) {
      // Round i
      uint32_t tmp0 = x;
      x = y ^ simon_f_opt(x) ^ rk[i];
      y = tmp0;

      // Round i+1
      uint32_t tmp1 = x;
      x = y ^ simon_f_opt(x) ^ rk[i+1];
      y = tmp1;

      // Round i+2
      uint32_t tmp2 = x;
      x = y ^ simon_f_opt(x) ^ rk[i+2];
      y = tmp2;

      // Round i+3
      uint32_t tmp3 = x;
      x = y ^ simon_f_opt(x) ^ rk[i+3];
      y = tmp3;
    }

    return ((y & 0xFFFFu) << 16) | (x & 0xFFFFu);
  }
};

// ============================================================
// PRESENT - Ultra-Lightweight Block Cipher
// Block size: 64 bits, Key size: 80 bits
// ============================================================


static constexpr int PRESENT_ROUNDS = 31;

#include "present_spbox_tables.inc"

struct Present80 {
  uint64_t round_keys[PRESENT_ROUNDS + 1];

  __host__ __device__ static inline uint8_t sbox(uint8_t x) {
    constexpr uint8_t S[16] = {
      0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD,
      0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2
    };
    return S[x & 0xF];
  }

  __host__ __device__ static inline uint64_t pLayer(uint64_t state) {
    uint64_t result = 0;
    #pragma unroll
    for (int i = 0; i < 64; i++) {
      int j = (i * 16) % 63;
      if (i == 63) j = 63;
      if (state & (1ULL << i)) result |= (1ULL << j);
    }
    return result;
  }

  __host__ __device__ static inline uint64_t pLayer_opt(uint64_t state) {
    uint64_t result = 0;
    #pragma unroll
    for (int nibble = 0; nibble < 16; nibble++) {
      uint8_t bits = (uint8_t)((state >> (nibble * 4)) & 0xFULL);
      #pragma unroll
      for (int bit = 0; bit < 4; bit++) {
        if (bits & (1u << bit)) {
          int src_pos = nibble * 4 + bit;
          int dst_pos = (src_pos * 16) % 63;
          if (src_pos == 63) dst_pos = 63;
          result |= (1ULL << dst_pos);
        }
      }
    }
    return result;
  }

  __host__ __device__ static inline uint64_t sBoxLayer(uint64_t state) {
    uint64_t result = 0;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
      const uint8_t nibble = (uint8_t)((state >> (i * 4)) & 0xFULL);
      result |= ((uint64_t)sbox(nibble)) << (i * 4);
    }
    return result;
  }

  __host__ __device__ static inline void load_key_words(const uint8_t key[10], uint64_t& w0, uint16_t& w1) {
    w0 = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
      w0 |= ((uint64_t)key[i]) << (8 * i);
    }
    w1 = (uint16_t)((uint16_t)key[8] | ((uint16_t)key[9] << 8));
  }

  __host__ __device__ static inline uint64_t extract_round_key(uint64_t w0, uint16_t w1) {
    return (((uint64_t)w1) << 48) | (w0 >> 16);
  }

  __host__ __device__ static inline void update_key_words(uint64_t& w0, uint16_t& w1, int round_counter) {
    const uint64_t new_w0 = (w0 >> 19) | (((uint64_t)w1) << 45) | (w0 << 61);
    uint16_t new_w1 = (uint16_t)(w0 >> 3);

    uint8_t top = (uint8_t)((new_w1 >> 12) & 0xF);
    top = sbox(top);
    new_w1 = (uint16_t)((new_w1 & 0x0FFFu) | ((uint16_t)top << 12));

    w0 = new_w0 ^ (((uint64_t)(round_counter & 0x1F)) << 15);
    w1 = new_w1;
  }

  __host__ __device__ static inline void expand_key(const uint8_t key[10], uint64_t out_rk[PRESENT_ROUNDS + 1]) {
    uint64_t w0;
    uint16_t w1;
    load_key_words(key, w0, w1);

    #pragma unroll
    for (int round = 0; round <= PRESENT_ROUNDS; round++) {
      out_rk[round] = extract_round_key(w0, w1);
      if (round != PRESENT_ROUNDS) update_key_words(w0, w1, round + 1);
    }
  }

  __host__ static inline uint64_t spbox_round_host(uint64_t state) {
    return PRESENT_SPBOX8_HOST[0][(unsigned)((state      ) & 0xFFu)] |
           PRESENT_SPBOX8_HOST[1][(unsigned)((state >>  8) & 0xFFu)] |
           PRESENT_SPBOX8_HOST[2][(unsigned)((state >> 16) & 0xFFu)] |
           PRESENT_SPBOX8_HOST[3][(unsigned)((state >> 24) & 0xFFu)] |
           PRESENT_SPBOX8_HOST[4][(unsigned)((state >> 32) & 0xFFu)] |
           PRESENT_SPBOX8_HOST[5][(unsigned)((state >> 40) & 0xFFu)] |
           PRESENT_SPBOX8_HOST[6][(unsigned)((state >> 48) & 0xFFu)] |
           PRESENT_SPBOX8_HOST[7][(unsigned)((state >> 56) & 0xFFu)];
  }

#ifdef __CUDACC__
  __device__ __forceinline__ static uint64_t spbox_round_dev(uint64_t state) {
    return PRESENT_SPBOX8_DEV[0][(unsigned)((state      ) & 0xFFu)] |
           PRESENT_SPBOX8_DEV[1][(unsigned)((state >>  8) & 0xFFu)] |
           PRESENT_SPBOX8_DEV[2][(unsigned)((state >> 16) & 0xFFu)] |
           PRESENT_SPBOX8_DEV[3][(unsigned)((state >> 24) & 0xFFu)] |
           PRESENT_SPBOX8_DEV[4][(unsigned)((state >> 32) & 0xFFu)] |
           PRESENT_SPBOX8_DEV[5][(unsigned)((state >> 40) & 0xFFu)] |
           PRESENT_SPBOX8_DEV[6][(unsigned)((state >> 48) & 0xFFu)] |
           PRESENT_SPBOX8_DEV[7][(unsigned)((state >> 56) & 0xFFu)];
  }

  __device__ __forceinline__ static uint64_t spbox_round_shared(uint64_t state, const uint64_t* spbox_flat) {
    return spbox_flat[0 * 256 + (unsigned)((state      ) & 0xFFu)] |
           spbox_flat[1 * 256 + (unsigned)((state >>  8) & 0xFFu)] |
           spbox_flat[2 * 256 + (unsigned)((state >> 16) & 0xFFu)] |
           spbox_flat[3 * 256 + (unsigned)((state >> 24) & 0xFFu)] |
           spbox_flat[4 * 256 + (unsigned)((state >> 32) & 0xFFu)] |
           spbox_flat[5 * 256 + (unsigned)((state >> 40) & 0xFFu)] |
           spbox_flat[6 * 256 + (unsigned)((state >> 48) & 0xFFu)] |
           spbox_flat[7 * 256 + (unsigned)((state >> 56) & 0xFFu)];
  }
#endif

  __host__ __device__ static inline uint64_t encrypt(uint64_t plaintext, const uint64_t rk[PRESENT_ROUNDS + 1]) {
    uint64_t state = plaintext;
    #pragma unroll
    for (int round = 0; round < PRESENT_ROUNDS; round++) {
      state ^= rk[round];
      state = sBoxLayer(state);
      state = pLayer_opt(state);
    }
    state ^= rk[PRESENT_ROUNDS];
    return state;
  }

  __host__ static inline uint64_t encrypt_spbox_otf_host(uint64_t plaintext, const uint8_t key[10]) {
    uint64_t w0;
    uint16_t w1;
    load_key_words(key, w0, w1);
    uint64_t state = plaintext;

    #pragma unroll
    for (int round = 0; round < PRESENT_ROUNDS; round++) {
      state ^= extract_round_key(w0, w1);
      state = spbox_round_host(state);
      update_key_words(w0, w1, round + 1);
    }
    state ^= extract_round_key(w0, w1);
    return state;
  }

#ifdef __CUDACC__
  __device__ __forceinline__ static uint64_t encrypt_spbox_otf_dev(uint64_t plaintext, const uint8_t key[10]) {
    uint64_t w0;
    uint16_t w1;
    load_key_words(key, w0, w1);
    uint64_t state = plaintext;

    #pragma unroll
    for (int round = 0; round < PRESENT_ROUNDS; round++) {
      state ^= extract_round_key(w0, w1);
      state = spbox_round_dev(state);
      update_key_words(w0, w1, round + 1);
    }
    state ^= extract_round_key(w0, w1);
    return state;
  }

  __device__ __forceinline__ static uint64_t encrypt_spbox_otf_shared(uint64_t plaintext, const uint8_t key[10],
                                                                       const uint64_t* spbox_flat) {
    uint64_t w0;
    uint16_t w1;
    load_key_words(key, w0, w1);
    uint64_t state = plaintext;

    #pragma unroll
    for (int round = 0; round < PRESENT_ROUNDS; round++) {
      state ^= extract_round_key(w0, w1);
      state = spbox_round_shared(state, spbox_flat);
      update_key_words(w0, w1, round + 1);
    }
    state ^= extract_round_key(w0, w1);
    return state;
  }
#endif
};

// ============================================================
// GRAIN v1 - Lightweight Stream Cipher
// Key size: 80 bits, IV size: 64 bits
// ============================================================

static constexpr int GRAIN_V1_INIT_ROUNDS = 160;

struct GrainV1 {
  uint8_t lfsr[80];   // Linear Feedback Shift Register
  uint8_t nfsr[80];   // Nonlinear Feedback Shift Register

  // LFSR feedback function
  __host__ __device__ static inline uint8_t lfsr_fb(const uint8_t lfsr[80]) {
    return lfsr[62] ^ lfsr[51] ^ lfsr[38] ^ lfsr[23] ^ lfsr[13] ^ lfsr[0];
  }

  // NFSR feedback function (nonlinear)
  __host__ __device__ static inline uint8_t nfsr_fb(const uint8_t lfsr[80], const uint8_t nfsr[80]) {
    uint8_t out = lfsr[0] ^ nfsr[62] ^ nfsr[60] ^ nfsr[52] ^ nfsr[45] ^ nfsr[37] ^
                  nfsr[33] ^ nfsr[28] ^ nfsr[21] ^ nfsr[14] ^ nfsr[9] ^ nfsr[0];
    out ^= (nfsr[63] & nfsr[60]);
    out ^= (nfsr[37] & nfsr[33]);
    out ^= (nfsr[15] & nfsr[9]);
    out ^= (nfsr[60] & nfsr[52] & nfsr[45]);
    out ^= (nfsr[33] & nfsr[28] & nfsr[21]);
    out ^= (nfsr[63] & nfsr[45] & nfsr[28] & nfsr[9]);
    out ^= (nfsr[60] & nfsr[52] & nfsr[37] & nfsr[33]);
    out ^= (nfsr[63] & nfsr[60] & nfsr[21] & nfsr[15]);
    out ^= (nfsr[63] & nfsr[60] & nfsr[52] & nfsr[45] & nfsr[37]);
    out ^= (nfsr[33] & nfsr[28] & nfsr[21] & nfsr[15] & nfsr[9]);
    out ^= (nfsr[52] & nfsr[45] & nfsr[37] & nfsr[33] & nfsr[28] & nfsr[21]);
    return out;
  }

  // Boolean function h
  __host__ __device__ static inline uint8_t h_func(uint8_t x0, uint8_t x1, uint8_t x2, uint8_t x3, uint8_t x4) {
    return x0 ^ x4 ^ (x0 & x3) ^ (x2 & x3) ^ (x3 & x4) ^
           (x0 & x1 & x2) ^ (x0 & x2 & x3) ^ (x0 & x2 & x4) ^
           (x1 & x2 & x4) ^ (x2 & x3 & x4);
  }

  // Output function
  __host__ __device__ static inline uint8_t output_bit(const uint8_t lfsr[80], const uint8_t nfsr[80]) {
    uint8_t h_out = h_func(lfsr[3], lfsr[25], lfsr[46], lfsr[64], nfsr[63]);
    uint8_t out = nfsr[1] ^ nfsr[2] ^ nfsr[4] ^ nfsr[10] ^ nfsr[31] ^ nfsr[43] ^ nfsr[56];
    return out ^ h_out;
  }

  // Initialize Grain with key and IV
  __host__ __device__ static inline void init(const uint8_t key[10], const uint8_t iv[8],
                                               uint8_t lfsr_out[80], uint8_t nfsr_out[80]) {
    // Load key into NFSR
    for (int i = 0; i < 80; i++) {
      int byte_idx = i / 8;
      int bit_idx = i % 8;
      nfsr_out[i] = (key[byte_idx] >> bit_idx) & 1;
    }

    // Load IV into LFSR (first 64 bits)
    for (int i = 0; i < 64; i++) {
      int byte_idx = i / 8;
      int bit_idx = i % 8;
      lfsr_out[i] = (iv[byte_idx] >> bit_idx) & 1;
    }

    // Fill remaining LFSR bits with 1s
    for (int i = 64; i < 80; i++) {
      lfsr_out[i] = 1;
    }

    // Initialize: clock 160 times without producing output
    for (int i = 0; i < GRAIN_V1_INIT_ROUNDS; i++) {
      uint8_t out_bit = output_bit(lfsr_out, nfsr_out);
      uint8_t lfsr_new = lfsr_fb(lfsr_out) ^ out_bit;
      uint8_t nfsr_new = nfsr_fb(lfsr_out, nfsr_out) ^ out_bit;

      // Shift registers
      for (int j = 0; j < 79; j++) {
        lfsr_out[j] = lfsr_out[j + 1];
        nfsr_out[j] = nfsr_out[j + 1];
      }
      lfsr_out[79] = lfsr_new;
      nfsr_out[79] = nfsr_new;
    }
  }

  // Generate keystream
  __host__ __device__ static inline uint8_t generate_keystream_bit(uint8_t lfsr[80], uint8_t nfsr[80]) {
    uint8_t out_bit = output_bit(lfsr, nfsr);
    uint8_t lfsr_new = lfsr_fb(lfsr);
    uint8_t nfsr_new = nfsr_fb(lfsr, nfsr);

    // Shift registers
    for (int j = 0; j < 79; j++) {
      lfsr[j] = lfsr[j + 1];
      nfsr[j] = nfsr[j + 1];
    }
    lfsr[79] = lfsr_new;
    nfsr[79] = nfsr_new;

    return out_bit;
  }

  // Encrypt/decrypt data (stream cipher)
  __host__ __device__ static inline void process(const uint8_t* input, uint8_t* output,
                                                  int length, const uint8_t key[10],
                                                  const uint8_t iv[8]) {
    uint8_t lfsr[80], nfsr[80];
    init(key, iv, lfsr, nfsr);

    for (int i = 0; i < length; i++) {
      uint8_t keystream_byte = 0;
      for (int bit = 0; bit < 8; bit++) {
        keystream_byte |= (generate_keystream_bit(lfsr, nfsr) << bit);
      }
      output[i] = input[i] ^ keystream_byte;
    }
  }

  // Optimized matcher: generate keystream bytes and early-exit on mismatch.
  // target[i] should be the expected keystream byte (pt[i] ^ ct[i]).
  static __host__ __device__ bool match_keystream(const uint8_t key[10], const uint8_t iv[8],
                                        const uint8_t* target, int length) {
    uint8_t lfsr[80] = {0};
    uint8_t nfsr[80] = {0};

    // Key in NFSR (80 bits, little-endian byte order)
    for (int i = 0; i < 80; i++) nfsr[i] = (key[i/8] >> (i%8)) & 1;

    // IV in LFSR (64 bits) + fixed padding
    for (int i = 0; i < 64; i++) lfsr[i] = (iv[i/8] >> (i%8)) & 1;
    for (int i = 64; i < 80; i++) lfsr[i] = 1;

    // Initialization: 160 clocks, output fed back
    for (int i = 0; i < 160; i++) {
      uint8_t out = generate_keystream_bit(lfsr, nfsr);
      nfsr[79] ^= out;
      lfsr[79] ^= out;
    }

    // Generate/compare keystream
    for (int i = 0; i < length; i++) {
      uint8_t ks = 0;
      #pragma unroll
      for (int bit = 0; bit < 8; bit++) {
        ks |= (generate_keystream_bit(lfsr, nfsr) << bit);
      }
      if (ks != target[i]) return false;
    }
    return true;
  }
};

// ============================================================
// TRIVIUM - Hardware-Oriented Stream Cipher
// Key size: 80 bits, IV size: 80 bits
// Internal state: 288 bits
// ============================================================

static constexpr int TRIVIUM_INIT_ROUNDS = 1152;

struct Trivium {
  uint8_t state[288];  // 288-bit internal state

  // Initialize Trivium with 80-bit key and 80-bit IV
  __host__ __device__ static inline void init(const uint8_t key[10], const uint8_t iv[10],
                                               uint8_t state_out[288]) {
    // Clear state
    for (int i = 0; i < 288; i++) {
      state_out[i] = 0;
    }

    // Load key into positions 0-79
    for (int i = 0; i < 80; i++) {
      int byte_idx = i / 8;
      int bit_idx = i % 8;
      state_out[i] = (key[byte_idx] >> bit_idx) & 1;
    }

    // Load IV into positions 93-172
    for (int i = 0; i < 80; i++) {
      int byte_idx = i / 8;
      int bit_idx = i % 8;
      state_out[93 + i] = (iv[byte_idx] >> bit_idx) & 1;
    }

    // Set last 3 bits to 1
    state_out[285] = 1;
    state_out[286] = 1;
    state_out[287] = 1;

    // Clock 1152 times (4 * 288) for initialization
    for (int i = 0; i < TRIVIUM_INIT_ROUNDS; i++) {
      uint8_t t1 = state_out[65] ^ state_out[92];
      uint8_t t2 = state_out[161] ^ state_out[176];
      uint8_t t3 = state_out[242] ^ state_out[287];

      t1 ^= (state_out[90] & state_out[91]) ^ state_out[170];
      t2 ^= (state_out[174] & state_out[175]) ^ state_out[263];
      t3 ^= (state_out[285] & state_out[286]) ^ state_out[68];

      // Update state
      for (int j = 92; j > 0; j--) {
        state_out[j] = state_out[j - 1];
      }
      state_out[0] = t3;

      for (int j = 176; j > 93; j--) {
        state_out[j] = state_out[j - 1];
      }
      state_out[93] = t1;

      for (int j = 287; j > 177; j--) {
        state_out[j] = state_out[j - 1];
      }
      state_out[177] = t2;
    }
  }

  // Generate one keystream bit
  __host__ __device__ static inline uint8_t generate_keystream_bit(uint8_t state[288]) {
    uint8_t t1 = state[65] ^ state[92];
    uint8_t t2 = state[161] ^ state[176];
    uint8_t t3 = state[242] ^ state[287];

    uint8_t z = t1 ^ t2 ^ t3;

    t1 ^= (state[90] & state[91]) ^ state[170];
    t2 ^= (state[174] & state[175]) ^ state[263];
    t3 ^= (state[285] & state[286]) ^ state[68];

    // Update state
    for (int j = 92; j > 0; j--) {
      state[j] = state[j - 1];
    }
    state[0] = t3;

    for (int j = 176; j > 93; j--) {
      state[j] = state[j - 1];
    }
    state[93] = t1;

    for (int j = 287; j > 177; j--) {
      state[j] = state[j - 1];
    }
    state[177] = t2;

    return z;
  }

  // Encrypt/decrypt data (stream cipher)
  __host__ __device__ static inline void process(const uint8_t* input, uint8_t* output,
                                                  int length, const uint8_t key[10],
                                                  const uint8_t iv[10]) {
    uint8_t state[288];
    init(key, iv, state);

    for (int i = 0; i < length; i++) {
      uint8_t keystream_byte = 0;
      for (int bit = 0; bit < 8; bit++) {
        keystream_byte |= (generate_keystream_bit(state) << bit);
      }
      output[i] = input[i] ^ keystream_byte;
    }
  }

  // Optimized matcher: generate keystream bytes and early-exit on mismatch.
  // target[i] should be the expected keystream byte (pt[i] ^ ct[i]).
  static __host__ __device__ bool match_keystream(const uint8_t key[10], const uint8_t iv[10],
                                        const uint8_t* target, int length) {
    uint8_t state[288] = {0};

    // Load key into state[0..79] (LSB-first per byte)
    for (int i = 0; i < 80; i++) state[i] = (key[i/8] >> (i%8)) & 1;

    // Load IV into state[93..172]
    for (int i = 0; i < 80; i++) state[93+i] = (iv[i/8] >> (i%8)) & 1;

    // state[285..287] = 1
    state[285] = state[286] = state[287] = 1;

    // Initialization: 4*288 cycles
    for (int i = 0; i < 4*288; i++) generate_keystream_bit(state);

    for (int i = 0; i < length; i++) {
      uint8_t ks = 0;
      #pragma unroll
      for (int bit = 0; bit < 8; bit++) {
        ks |= (generate_keystream_bit(state) << bit);
      }
      if (ks != target[i]) return false;
    }
    return true;
  }
};

// ============================================================
// SPECK64/128 (ARX block cipher)
// Block size: 64 bits (2x32)
// Key size: 128 bits (4x32)
// Rounds: 27
// ============================================================

#define SPECK64_128_ROUNDS 27

struct Speck64_128 {
  __host__ __device__ static inline uint32_t rotl32(uint32_t x, int r) {
    return (x << r) | (x >> (32 - r));
  }
  __host__ __device__ static inline uint32_t rotr32(uint32_t x, int r) {
    return (x >> r) | (x << (32 - r));
  }

  __host__ __device__ static inline void er32(uint32_t &x, uint32_t &y, uint32_t k) {
    x = rotr32(x, 8);
    x = x + y;
    x ^= k;
    y = rotl32(y, 3);
    y ^= x;
  }

  // Key schedule (little-endian word order), per NSA implementation guide
  __host__ __device__ static inline void expand_key(const uint8_t key[16], uint32_t rk[SPECK64_128_ROUNDS]) {
    auto load32_le = [](const uint8_t* p) -> uint32_t {
      return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    };

    uint32_t K[4];
    K[0] = load32_le(key + 0);
    K[1] = load32_le(key + 4);
    K[2] = load32_le(key + 8);
    K[3] = load32_le(key + 12);

    uint32_t A = K[0];
    uint32_t B = K[1];
    uint32_t C = K[2];
    uint32_t D = K[3];

    uint32_t i = 0;
    while (i < SPECK64_128_ROUNDS) {
      rk[i] = A; er32(B, A, i); i++;
      if (i >= SPECK64_128_ROUNDS) break;
      rk[i] = A; er32(C, A, i); i++;
      if (i >= SPECK64_128_ROUNDS) break;
      rk[i] = A; er32(D, A, i); i++;
    }
  }

  // Encrypt a 64-bit block (Pt[1]|Pt[0])
  __host__ __device__ static inline uint64_t encrypt(uint64_t pt, const uint32_t rk[SPECK64_128_ROUNDS]) {
    uint32_t y = (uint32_t)(pt & 0xFFFFFFFFu);        // Pt[0] (right)
    uint32_t x = (uint32_t)((pt >> 32) & 0xFFFFFFFFu); // Pt[1] (left)

    #pragma unroll
    for (int i = 0; i < SPECK64_128_ROUNDS; i++) {
      er32(x, y, rk[i]);
    }

    return ((uint64_t)x << 32) | (uint64_t)y;
  }
};

// ============================================================
// ChaCha20 (ARX stream cipher) - RFC 8439
// Key: 256 bits, Nonce: 96 bits, Counter: 32 bits
// ============================================================

struct ChaCha20 {
  __host__ __device__ static inline uint32_t rotl32(uint32_t x, int r) {
    return (x << r) | (x >> (32 - r));
  }

  __host__ __device__ static inline uint32_t load32_le(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
  }

  __host__ __device__ static inline void store32_le(uint8_t* p, uint32_t x) {
    p[0] = (uint8_t)(x & 0xFF);
    p[1] = (uint8_t)((x >> 8) & 0xFF);
    p[2] = (uint8_t)((x >> 16) & 0xFF);
    p[3] = (uint8_t)((x >> 24) & 0xFF);
  }

  __host__ __device__ static inline void quarter_round(uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d) {
    a += b; d ^= a; d = rotl32(d, 16);
    c += d; b ^= c; b = rotl32(b, 12);
    a += b; d ^= a; d = rotl32(d, 8);
    c += d; b ^= c; b = rotl32(b, 7);
  }

  // Produce 64-byte keystream block
  __host__ __device__ static inline void block(const uint8_t key[32], uint32_t counter, const uint8_t nonce[12], uint8_t out[64]) {
    // Constants "expand 32-byte k"
    uint32_t state[16];
    state[0]  = 0x61707865u;
    state[1]  = 0x3320646eu;
    state[2]  = 0x79622d32u;
    state[3]  = 0x6b206574u;

    // Key (8 words)
    #pragma unroll
    for (int i = 0; i < 8; i++) {
      state[4 + i] = load32_le(key + 4 * i);
    }

    state[12] = counter;
    state[13] = load32_le(nonce + 0);
    state[14] = load32_le(nonce + 4);
    state[15] = load32_le(nonce + 8);

    uint32_t working[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) working[i] = state[i];

    // 20 rounds = 10 double-rounds
    #pragma unroll
    for (int i = 0; i < 10; i++) {
      // Column rounds
      quarter_round(working[0], working[4], working[8],  working[12]);
      quarter_round(working[1], working[5], working[9],  working[13]);
      quarter_round(working[2], working[6], working[10], working[14]);
      quarter_round(working[3], working[7], working[11], working[15]);

      // Diagonal rounds
      quarter_round(working[0], working[5], working[10], working[15]);
      quarter_round(working[1], working[6], working[11], working[12]);
      quarter_round(working[2], working[7], working[8],  working[13]);
      quarter_round(working[3], working[4], working[9],  working[14]);
    }

    // Add original state
    #pragma unroll
    for (int i = 0; i < 16; i++) working[i] += state[i];

    // Serialize little-endian
    #pragma unroll
    for (int i = 0; i < 16; i++) {
      store32_le(out + 4 * i, working[i]);
    }
  }
    // Optimized helper: produce only the first 16 bytes (4 words) of the ChaCha20 block.
    // This avoids materializing the full 64-byte keystream when we only need a short prefix.
    //
    // IMPORTANT: This is called from GPU brute-force kernels, so it must be device-callable.
    // Mark it __host__ __device__ (and keep it header-only/inline) to avoid
    // "calling a __host__ function from a __device__ function" build errors.
    __host__ __device__ static inline void block_words4(const uint8_t key[32], uint32_t counter,
                                                        const uint8_t nonce[12], uint32_t out4[4]) {
      // Setup state
      uint32_t x[16];
      x[0]  = 0x61707865; x[1]  = 0x3320646e; x[2]  = 0x79622d32; x[3]  = 0x6b206574;

      // key (little-endian words)
      for (int i = 0; i < 8; i++) {
        x[4+i] = (uint32_t)key[4*i] | ((uint32_t)key[4*i+1] << 8) |
                 ((uint32_t)key[4*i+2] << 16) | ((uint32_t)key[4*i+3] << 24);
      }

      x[12] = counter;

      // nonce (3 little-endian words)
      x[13] = (uint32_t)nonce[0] | ((uint32_t)nonce[1] << 8) | ((uint32_t)nonce[2] << 16) | ((uint32_t)nonce[3] << 24);
      x[14] = (uint32_t)nonce[4] | ((uint32_t)nonce[5] << 8) | ((uint32_t)nonce[6] << 16) | ((uint32_t)nonce[7] << 24);
      x[15] = (uint32_t)nonce[8] | ((uint32_t)nonce[9] << 8) | ((uint32_t)nonce[10] << 16) | ((uint32_t)nonce[11] << 24);

      uint32_t w[16];
      #pragma unroll
      for (int i = 0; i < 16; i++) w[i] = x[i];

      // 20 rounds (10 double-rounds)
      #pragma unroll
      for (int i = 0; i < 10; i++) {
        // Column rounds
        quarter_round(w[0], w[4], w[8],  w[12]);
        quarter_round(w[1], w[5], w[9],  w[13]);
        quarter_round(w[2], w[6], w[10], w[14]);
        quarter_round(w[3], w[7], w[11], w[15]);
        // Diagonal rounds
        quarter_round(w[0], w[5], w[10], w[15]);
        quarter_round(w[1], w[6], w[11], w[12]);
        quarter_round(w[2], w[7], w[8],  w[13]);
        quarter_round(w[3], w[4], w[9],  w[14]);
      }

      // Add original state and output first 4 words
      out4[0] = w[0] + x[0];
      out4[1] = w[1] + x[1];
      out4[2] = w[2] + x[2];
      out4[3] = w[3] + x[3];
    }


  // XOR keystream with input
  __host__ __device__ static inline void process(const uint8_t* input, uint8_t* output, int length,
                                                 const uint8_t key[32], uint32_t counter,
                                                 const uint8_t nonce[12]) {
    uint32_t block_count = 0;
    int offset = 0;
    while (offset < length) {
      uint8_t ks[64];
      block(key, counter + block_count, nonce, ks);
      int n = (length - offset > 64) ? 64 : (length - offset);
      for (int i = 0; i < n; i++) {
        output[offset + i] = input[offset + i] ^ ks[i];
      }
      offset += n;
      block_count++;
    }
  }
};

// ============================================================
// TinyJAMBU-128 (v2)
// Key: 128 bits, Nonce: 96 bits, Tag: 64 bits, State: 128 bits
// Ref: https://csrc.nist.gov/CSRC/media/Projects/lightweight-cryptography/documents/finalist-round/updated-spec-doc/tinyjambu-spec-final.pdf
// ============================================================
static constexpr int TINYJAMBU_P1024_STEPS = 1024;
static constexpr int TINYJAMBU_P640_STEPS  = 640;
static constexpr int TINYJAMBU_P384_STEPS  = 384;

struct TinyJAMBU128 {
  __host__ __device__ static inline uint32_t load32_le(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
  }
  __host__ __device__ static inline void store32_le(uint8_t* p, uint32_t x) {
    p[0] = (uint8_t)(x & 0xFF); p[1] = (uint8_t)((x >> 8) & 0xFF);
    p[2] = (uint8_t)((x >> 16) & 0xFF); p[3] = (uint8_t)((x >> 24) & 0xFF);
  }

  // One permutation step (advances state by 1 bit)
  __host__ __device__ static inline void step(uint32_t state[4], uint32_t keybit) {
    uint32_t s0 = state[0], s1 = state[1], s2 = state[2], s3 = state[3];
    uint32_t t1 = (s1 >> 15) & 1;  // s47
    uint32_t t2 = (s2 >> 6)  & 1;  // s70
    uint32_t t3 = (s2 >> 21) & 1;  // s85
    uint32_t t4 = (s2 >> 27) & 1;  // s91
    uint32_t feedback = (s0 & 1) ^ t1 ^ (~(t2 & t3) & 1) ^ t4 ^ keybit;
    state[0] = (s0 >> 1) | (s1 << 31);
    state[1] = (s1 >> 1) | (s2 << 31);
    state[2] = (s2 >> 1) | (s3 << 31);
    state[3] = (s3 >> 1) | (feedback << 31);
  }

  // Apply permutation Pn (n steps); key bits cycle with period 128
  __host__ __device__ static inline void permute(uint32_t state[4], const uint32_t key[4], int steps) {
    for (int i = 0; i < steps; i++) {
      uint32_t keybit = (key[(i >> 5) & 3] >> (i & 31)) & 1;
      step(state, keybit);
    }
  }

  // XOR framebits into s{36..38}
  __host__ __device__ static inline void framebits(uint32_t state[4], uint32_t v) {
    state[1] ^= (v & 7) << 4;
  }

  __host__ __device__ static inline void init(uint32_t state[4], const uint32_t key[4], const uint8_t nonce[12]) {
    state[0] = state[1] = state[2] = state[3] = 0;
    permute(state, key, TINYJAMBU_P1024_STEPS);
    #pragma unroll
    for (int i = 0; i < 3; i++) {
      framebits(state, 1);
      permute(state, key, TINYJAMBU_P640_STEPS);
      state[3] ^= load32_le(nonce + 4 * i);
    }
  }

  __host__ __device__ static inline void absorb_ad(uint32_t state[4], const uint8_t* ad, int adlen, const uint32_t key[4]) {
    int blocks = adlen / 4, rem_ad = adlen % 4;
    for (int i = 0; i < blocks; i++) {
      framebits(state, 3); permute(state, key, TINYJAMBU_P640_STEPS);
      state[3] ^= load32_le(ad + 4 * i);
    }
    if (rem_ad > 0) {
      framebits(state, 3); permute(state, key, TINYJAMBU_P640_STEPS);
      uint32_t partial = 0;
      for (int i = 0; i < rem_ad; i++) partial |= ((uint32_t)ad[blocks * 4 + i]) << (8 * i);
      state[3] ^= partial;
      state[1] ^= rem_ad;
    }
  }

  __host__ __device__ static inline void encrypt_blocks(uint32_t state[4], const uint8_t* pt, uint8_t* ct, int mlen, const uint32_t key[4]) {
    int blocks = mlen / 4, rem = mlen % 4;
    for (int i = 0; i < blocks; i++) {
      framebits(state, 5); permute(state, key, TINYJAMBU_P1024_STEPS);
      uint32_t m = load32_le(pt + 4 * i);
      store32_le(ct + 4 * i, state[2] ^ m);
      state[3] ^= m;
    }
    if (rem > 0) {
      framebits(state, 5); permute(state, key, TINYJAMBU_P1024_STEPS);
      uint32_t partial = 0;
      for (int i = 0; i < rem; i++) partial |= ((uint32_t)pt[blocks * 4 + i]) << (8 * i);
      uint32_t c = state[2] ^ partial;
      for (int i = 0; i < rem; i++) ct[blocks * 4 + i] = (uint8_t)((c >> (8 * i)) & 0xFF);
      state[3] ^= partial;
      state[1] ^= rem;
    }
  }

  __host__ __device__ static inline void finalize(uint32_t state[4], uint8_t tag[8], const uint32_t key[4]) {
    framebits(state, 7); permute(state, key, TINYJAMBU_P1024_STEPS);
    store32_le(tag, state[2]);
    framebits(state, 7); permute(state, key, TINYJAMBU_P640_STEPS); // v2 uses P640 for second tag word
    store32_le(tag + 4, state[2]);
  }

  __host__ __device__ static inline void encrypt(const uint8_t* pt, uint8_t* ct, int len, uint8_t tag[8],
                                                 const uint8_t key_bytes[16], const uint8_t nonce[12],
                                                 const uint8_t* ad, int ad_len) {
    uint32_t key[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) key[i] = load32_le(key_bytes + 4 * i);
    uint32_t state[4];
    init(state, key, nonce);
    absorb_ad(state, ad, ad_len, key);
    encrypt_blocks(state, pt, ct, len, key);
    finalize(state, tag, key);
  }
};

// ============================================================
// TinyJAMBU-128 Bitsliced (32 parallel instances per thread)
// ============================================================
struct TinyJAMBU128_Bitsliced {
  __host__ __device__ static inline uint32_t broadcast_bit(uint32_t word, int bit_idx) {
    return ((word >> bit_idx) & 1) ? 0xFFFFFFFF : 0x00000000;
  }

  // Bitsliced permutation: state[128] where each uint32 holds 32 parallel bits
  __host__ __device__ static void permute(uint32_t state[128], const uint32_t key[128], int steps) {
    int rounds = steps / 128;
    for (int r = 0; r < rounds; r++) {
      #pragma unroll 128
      for (int i = 0; i < 128; i++) {
        uint32_t t1 = state[(i + 47) % 128];
        uint32_t t2 = state[(i + 70) % 128];
        uint32_t t3 = state[(i + 85) % 128];
        uint32_t t4 = state[(i + 91) % 128];
        state[i] ^= t1 ^ ~(t2 & t3) ^ t4 ^ key[i];
      }
    }
  }

  __host__ __device__ static inline void framebits(uint32_t state[128], uint32_t v) {
    state[36] ^= broadcast_bit(v, 0);
    state[37] ^= broadcast_bit(v, 1);
    state[38] ^= broadcast_bit(v, 2);
  }

  __host__ __device__ static inline void init(uint32_t state[128], const uint32_t key[128], const uint8_t nonce[12]) {
    #pragma unroll
    for (int i = 0; i < 128; i++) state[i] = 0;
    permute(state, key, TINYJAMBU_P1024_STEPS);
    #pragma unroll
    for (int i = 0; i < 3; i++) {
      framebits(state, 1);
      permute(state, key, TINYJAMBU_P640_STEPS);
      uint32_t n_word = (uint32_t)nonce[4*i] | ((uint32_t)nonce[4*i+1]<<8) |
                        ((uint32_t)nonce[4*i+2]<<16) | ((uint32_t)nonce[4*i+3]<<24);
      #pragma unroll
      for (int bit = 0; bit < 32; bit++) state[96 + bit] ^= broadcast_bit(n_word, bit);
    }
  }

  __host__ __device__ static inline void absorb_ad(uint32_t state[128], const uint8_t* ad, int adlen, const uint32_t key[128]) {
    int blocks = adlen / 4, rem_ad = adlen % 4;
    for (int i = 0; i < blocks; i++) {
      framebits(state, 3); permute(state, key, TINYJAMBU_P640_STEPS);
      uint32_t ad_word = (uint32_t)ad[4*i] | ((uint32_t)ad[4*i+1]<<8) |
                         ((uint32_t)ad[4*i+2]<<16) | ((uint32_t)ad[4*i+3]<<24);
      #pragma unroll
      for (int bit = 0; bit < 32; bit++) state[96 + bit] ^= broadcast_bit(ad_word, bit);
    }
    if (rem_ad > 0) {
      framebits(state, 3); permute(state, key, TINYJAMBU_P640_STEPS);
      uint32_t partial = 0;
      for (int i = 0; i < rem_ad; i++) partial |= ((uint32_t)ad[blocks * 4 + i]) << (8 * i);
      #pragma unroll
      for (int bit = 0; bit < 32; bit++) state[96 + bit] ^= broadcast_bit(partial, bit);
      #pragma unroll
      for (int bit = 0; bit < 32; bit++) state[32 + bit] ^= broadcast_bit(rem_ad, bit);
    }
  }

  // Encrypt and check ciphertext bits; updates match_mask (clears bits for wrong keys)
  __host__ __device__ static inline void check_blocks(uint32_t state[128], uint32_t& match_mask,
                                                      const uint8_t* pt, const uint8_t* target_ct,
                                                      int mlen, const uint32_t key[128]) {
    int blocks = mlen / 4, rem = mlen % 4;
    for (int i = 0; i < blocks; i++) {
      framebits(state, 5); permute(state, key, TINYJAMBU_P1024_STEPS);
      uint32_t pt_word = (uint32_t)pt[4*i]|(uint32_t)pt[4*i+1]<<8|(uint32_t)pt[4*i+2]<<16|(uint32_t)pt[4*i+3]<<24;
      uint32_t ct_word = (uint32_t)target_ct[4*i]|(uint32_t)target_ct[4*i+1]<<8|(uint32_t)target_ct[4*i+2]<<16|(uint32_t)target_ct[4*i+3]<<24;
      #pragma unroll
      for (int bit = 0; bit < 32; bit++) {
        uint32_t p_bit = broadcast_bit(pt_word, bit);
        uint32_t expected_c_bit = broadcast_bit(ct_word, bit);
        uint32_t generated_c_bit = state[64 + bit] ^ p_bit;
        match_mask &= ~(generated_c_bit ^ expected_c_bit);
        state[96 + bit] ^= p_bit;
      }
    }
    if (rem > 0) {
      framebits(state, 5); permute(state, key, TINYJAMBU_P1024_STEPS);
      uint32_t partial_pt = 0, partial_ct = 0;
      for (int i = 0; i < rem; i++) {
        partial_pt |= ((uint32_t)pt[blocks * 4 + i]) << (8 * i);
        partial_ct |= ((uint32_t)target_ct[blocks * 4 + i]) << (8 * i);
      }
      for (int bit = 0; bit < rem * 8; bit++) {
        uint32_t p_bit = broadcast_bit(partial_pt, bit);
        uint32_t expected_c_bit = broadcast_bit(partial_ct, bit);
        uint32_t generated_c_bit = state[64 + bit] ^ p_bit;
        match_mask &= ~(generated_c_bit ^ expected_c_bit);
        state[96 + bit] ^= p_bit;
      }
      #pragma unroll
      for (int bit = 0; bit < 32; bit++) state[32 + bit] ^= broadcast_bit(rem, bit);
    }
  }

  __host__ __device__ static inline void finalize(uint32_t state[128], uint32_t& match_mask,
                                                  const uint8_t expected_tag[8], const uint32_t key[128]) {
    framebits(state, 7); permute(state, key, TINYJAMBU_P1024_STEPS);
    uint32_t tag0 = (uint32_t)expected_tag[0]|(uint32_t)expected_tag[1]<<8|(uint32_t)expected_tag[2]<<16|(uint32_t)expected_tag[3]<<24;
    #pragma unroll
    for (int bit = 0; bit < 32; bit++) match_mask &= ~(state[64 + bit] ^ broadcast_bit(tag0, bit));

    framebits(state, 7); permute(state, key, TINYJAMBU_P640_STEPS);
    uint32_t tag1 = (uint32_t)expected_tag[4]|(uint32_t)expected_tag[5]<<8|(uint32_t)expected_tag[6]<<16|(uint32_t)expected_tag[7]<<24;
    #pragma unroll
    for (int bit = 0; bit < 32; bit++) match_mask &= ~(state[64 + bit] ^ broadcast_bit(tag1, bit));
  }

  // Match 32 keys simultaneously; returns bitmask of matching keys
  __host__ __device__ static inline uint32_t match_keys(const uint8_t* pt, const uint8_t* ct, int len,
                                                        const uint8_t expected_tag[8],
                                                        const uint32_t key_bitsliced[128],
                                                        const uint8_t nonce[12],
                                                        const uint8_t* ad, int ad_len) {
    uint32_t state[128];
    uint32_t match_mask = 0xFFFFFFFF;
    init(state, key_bitsliced, nonce);
    absorb_ad(state, ad, ad_len, key_bitsliced);
    check_blocks(state, match_mask, pt, ct, len, key_bitsliced);
    if (match_mask == 0) return 0;
    finalize(state, match_mask, expected_tag, key_bitsliced);
    return match_mask;
  }
};

// ============================================================
// ZUC-128 Stream Cipher
// Key: 128 bits, IV: 128 bits
// Output: 32-bit keystream words (big-endian serialized)
// Ref: https://www.gsma.com/security/wp-content/uploads/2019/05/eea3eia3zucv16.pdf
// ============================================================
struct ZUC {
  struct State {
    uint32_t S[16]; // LFSR registers
    uint32_t R1;    // FSM register 1
    uint32_t R2;    // FSM register 2
  };

  __host__ __device__ static inline uint8_t sbox0(uint8_t x) {
    static const uint8_t S0[256] = {
      0x3e,0x72,0x5b,0x47,0xca,0xe0,0x00,0x33,0x04,0xd1,0x54,0x98,0x09,0xb9,0x6d,0xcb,
      0x7b,0x1b,0xf9,0x32,0xaf,0x9d,0x6a,0xa5,0xb8,0x2d,0xfc,0x1d,0x08,0x53,0x03,0x90,
      0x4d,0x4e,0x84,0x99,0xe4,0xce,0xd9,0x91,0xdd,0xb6,0x85,0x48,0x8b,0x29,0x6e,0xac,
      0xcd,0xc1,0xf8,0x1e,0x73,0x43,0x69,0xc6,0xb5,0xbd,0xfd,0x39,0x63,0x20,0xd4,0x38,
      0x76,0x7d,0xb2,0xa7,0xcf,0xed,0x57,0xc5,0xf3,0x2c,0xbb,0x14,0x21,0x06,0x55,0x9b,
      0xe3,0xef,0x5e,0x31,0x4f,0x7f,0x5a,0xa4,0x0d,0x82,0x51,0x49,0x5f,0xba,0x58,0x1c,
      0x4a,0x16,0xd5,0x17,0xa8,0x92,0x24,0x1f,0x8c,0xff,0xd8,0xae,0x2e,0x01,0xd3,0xad,
      0x3b,0x4b,0xda,0x46,0xeb,0xc9,0xde,0x9a,0x8f,0x87,0xd7,0x3a,0x80,0x6f,0x2f,0xc8,
      0xb1,0xb4,0x37,0xf7,0x0a,0x22,0x13,0x28,0x7c,0xcc,0x3c,0x89,0xc7,0xc3,0x96,0x56,
      0x07,0xbf,0x7e,0xf0,0x0b,0x2b,0x97,0x52,0x35,0x41,0x79,0x61,0xa6,0x4c,0x10,0xfe,
      0xbc,0x26,0x95,0x88,0x8a,0xb0,0xa3,0xfb,0xc0,0x18,0x94,0xf2,0xe1,0xe5,0xe9,0x5d,
      0xd0,0xdc,0x11,0x66,0x64,0x5c,0xec,0x59,0x42,0x75,0x12,0xf5,0x74,0x9c,0xaa,0x23,
      0x0e,0x86,0xab,0xbe,0x2a,0x02,0xe7,0x67,0xe6,0x44,0xa2,0x6c,0xc2,0x93,0x9f,0xf1,
      0xf6,0xfa,0x36,0xd2,0x50,0x68,0x9e,0x62,0x71,0x15,0x3d,0xd6,0x40,0xc4,0xe2,0x0f,
      0x8e,0x83,0x77,0x6b,0x25,0x05,0x3f,0x0c,0x30,0xea,0x70,0xb7,0xa1,0xe8,0xa9,0x65,
      0x8d,0x27,0x1a,0xdb,0x81,0xb3,0xa0,0xf4,0x45,0x7a,0x19,0xdf,0xee,0x78,0x34,0x60
    };
    return S0[x];
  }

  __host__ __device__ static inline uint8_t sbox1(uint8_t x) {
    static const uint8_t S1[256] = {
      0x55,0xc2,0x63,0x71,0x3b,0xc8,0x47,0x86,0x9f,0x3c,0xda,0x5b,0x29,0xaa,0xfd,0x77,
      0x8c,0xc5,0x94,0x0c,0xa6,0x1a,0x13,0x00,0xe3,0xa8,0x16,0x72,0x40,0xf9,0xf8,0x42,
      0x44,0x26,0x68,0x96,0x81,0xd9,0x45,0x3e,0x10,0x76,0xc6,0xa7,0x8b,0x39,0x43,0xe1,
      0x3a,0xb5,0x56,0x2a,0xc0,0x6d,0xb3,0x05,0x22,0x66,0xbf,0xdc,0x0b,0xfa,0x62,0x48,
      0xdd,0x20,0x11,0x06,0x36,0xc9,0xc1,0xcf,0xf6,0x27,0x52,0xbb,0x69,0xf5,0xd4,0x87,
      0x7f,0x84,0x4c,0xd2,0x9c,0x57,0xa4,0xbc,0x4f,0x9a,0xdf,0xfe,0xd6,0x8d,0x7a,0xeb,
      0x2b,0x53,0xd8,0x5c,0xa1,0x14,0x17,0xfb,0x23,0xd5,0x7d,0x30,0x67,0x73,0x08,0x09,
      0xee,0xb7,0x70,0x3f,0x61,0xb2,0x19,0x8e,0x4e,0xe5,0x4b,0x93,0x8f,0x5d,0xdb,0xa9,
      0xad,0xf1,0xae,0x2e,0xcb,0x0d,0xfc,0xf4,0x2d,0x46,0x6e,0x1d,0x97,0xe8,0xd1,0xe9,
      0x4d,0x37,0xa5,0x75,0x5e,0x83,0x9e,0xab,0x82,0x9d,0xb9,0x1c,0xe0,0xcd,0x49,0x89,
      0x01,0xb6,0xbd,0x58,0x24,0xa2,0x5f,0x38,0x78,0x99,0x15,0x90,0x50,0xb8,0x95,0xe4,
      0xd0,0x91,0xc7,0xce,0xed,0x0f,0xb4,0x6f,0xa0,0xcc,0xf0,0x02,0x4a,0x79,0xc3,0xde,
      0xa3,0xef,0xea,0x51,0xe6,0x6b,0x18,0xec,0x1b,0x2c,0x80,0xf7,0x74,0xe7,0xff,0x21,
      0x5a,0x6a,0x54,0x1e,0x41,0x31,0x92,0x35,0xc4,0x33,0x07,0x0a,0xba,0x7e,0x0e,0x34,
      0x88,0xb1,0x98,0x7c,0xf3,0x3d,0x60,0x6c,0x7b,0xca,0xd3,0x1f,0x32,0x65,0x04,0x28,
      0x64,0xbe,0x85,0x9b,0x2f,0x59,0x8a,0xd7,0xb0,0x25,0xac,0xaf,0x12,0x03,0xe2,0xf2
    };
    return S1[x];
  }

  __host__ __device__ static inline uint32_t ek_d(int x) {
    static const uint32_t D[16] = {
      0x44D7,0x26BC,0x626B,0x135E,0x5789,0x35E2,0x7135,0x09AF,
      0x4D78,0x2F13,0x6BC4,0x1AF1,0x5E26,0x3C4D,0x789A,0x47AC
    };
    return D[x];
  }

  __host__ __device__ static inline uint32_t load32_be(const uint8_t* p) {
    return ((uint32_t)p[0]<<24)|((uint32_t)p[1]<<16)|((uint32_t)p[2]<<8)|(uint32_t)p[3];
  }
  __host__ __device__ static inline void store32_be(uint8_t* p, uint32_t x) {
    p[0]=(uint8_t)(x>>24); p[1]=(uint8_t)(x>>16); p[2]=(uint8_t)(x>>8); p[3]=(uint8_t)(x&0xFF);
  }
  __host__ __device__ static inline uint32_t rotl32(uint32_t a, int k) { return (a<<k)|(a>>(32-k)); }

  // Modular addition in GF(2^31 - 1)
  __host__ __device__ static inline uint32_t AddM(uint32_t a, uint32_t b) {
    uint32_t c = a + b; return (c & 0x7FFFFFFF) + (c >> 31);
  }
  __host__ __device__ static inline uint32_t MulByPow2(uint32_t x, int k) {
    return ((x << k) | (x >> (31 - k))) & 0x7FFFFFFF;
  }

  __host__ __device__ static inline uint32_t L1(uint32_t X) {
    return X ^ rotl32(X,2) ^ rotl32(X,10) ^ rotl32(X,18) ^ rotl32(X,24);
  }
  __host__ __device__ static inline uint32_t L2(uint32_t X) {
    return X ^ rotl32(X,8) ^ rotl32(X,14) ^ rotl32(X,22) ^ rotl32(X,30);
  }

  __host__ __device__ static inline void update_LFSR(State& s, uint32_t u) {
    uint32_t f = s.S[0];
    f = AddM(f, MulByPow2(s.S[0], 8));
    f = AddM(f, MulByPow2(s.S[4], 20));
    f = AddM(f, MulByPow2(s.S[10], 21));
    f = AddM(f, MulByPow2(s.S[13], 17));
    f = AddM(f, MulByPow2(s.S[15], 15));
    f = AddM(f, u);
    #pragma unroll
    for (int i = 0; i < 15; i++) s.S[i] = s.S[i + 1];
    s.S[15] = f;
  }

  // FSM update; returns output word W
  __host__ __device__ static inline uint32_t F(State& s, uint32_t X0, uint32_t X1, uint32_t X2) {
    uint32_t W  = (X0 ^ s.R1) + s.R2;
    uint32_t W1 = s.R1 + X1;
    uint32_t W2 = s.R2 ^ X2;
    uint32_t u  = L1((W1 << 16) | (W2 >> 16));
    uint32_t v  = L2((W2 << 16) | (W1 >> 16));
    s.R1 = ((uint32_t)sbox0(u>>24)<<24)|((uint32_t)sbox1((u>>16)&0xFF)<<16)|
           ((uint32_t)sbox0((u>>8)&0xFF)<<8)|((uint32_t)sbox1(u&0xFF));
    s.R2 = ((uint32_t)sbox0(v>>24)<<24)|((uint32_t)sbox1((v>>16)&0xFF)<<16)|
           ((uint32_t)sbox0((v>>8)&0xFF)<<8)|((uint32_t)sbox1(v&0xFF));
    return W;
  }

  __host__ __device__ static inline void init(State& s, const uint8_t key[16], const uint8_t iv[16]) {
    #pragma unroll
    for (int i = 0; i < 16; i++)
      s.S[i] = ((uint32_t)key[i] << 23) | (ek_d(i) << 8) | (uint32_t)iv[i];
    s.R1 = 0; s.R2 = 0;
    #pragma unroll 32
    for (int i = 0; i < 32; i++) {
      uint32_t X0 = ((s.S[15]&0x7FFF8000)<<1)|(s.S[14]&0xFFFF);
      uint32_t X1 = ((s.S[11]&0xFFFF)<<16)|(s.S[9]>>15);
      uint32_t X2 = ((s.S[7]&0xFFFF)<<16)|(s.S[5]>>15);
      uint32_t W  = F(s, X0, X1, X2);
      update_LFSR(s, W >> 1);
    }
  }

  __host__ __device__ static inline void process(const uint8_t* pt, uint8_t* ct, int len,
                                                 const uint8_t key[16], const uint8_t iv[16]) {
    State s; init(s, key, iv);
    // Dummy round
    uint32_t X0=((s.S[15]&0x7FFF8000)<<1)|(s.S[14]&0xFFFF);
    uint32_t X1=((s.S[11]&0xFFFF)<<16)|(s.S[9]>>15);
    uint32_t X2=((s.S[7]&0xFFFF)<<16)|(s.S[5]>>15);
    F(s,X0,X1,X2); update_LFSR(s,0);

    int blocks = len / 4, rem = len % 4;
    for (int i = 0; i < blocks; i++) {
      X0=((s.S[15]&0x7FFF8000)<<1)|(s.S[14]&0xFFFF);
      X1=((s.S[11]&0xFFFF)<<16)|(s.S[9]>>15);
      X2=((s.S[7]&0xFFFF)<<16)|(s.S[5]>>15);
      uint32_t X3=((s.S[2]&0xFFFF)<<16)|(s.S[0]>>15);
      uint32_t Z = F(s,X0,X1,X2) ^ X3; update_LFSR(s,0);
      store32_be(ct + 4*i, load32_be(pt + 4*i) ^ Z);
    }
    if (rem > 0) {
      X0=((s.S[15]&0x7FFF8000)<<1)|(s.S[14]&0xFFFF);
      X1=((s.S[11]&0xFFFF)<<16)|(s.S[9]>>15);
      X2=((s.S[7]&0xFFFF)<<16)|(s.S[5]>>15);
      uint32_t X3=((s.S[2]&0xFFFF)<<16)|(s.S[0]>>15);
      uint32_t Z = F(s,X0,X1,X2) ^ X3;
      uint8_t z_bytes[4]; store32_be(z_bytes, Z);
      for (int i = 0; i < rem; i++) ct[blocks*4+i] = pt[blocks*4+i] ^ z_bytes[i];
    }
  }

  // Early-exit keystream matching for GPU brute-force
  __host__ __device__ static inline bool match_keystream(const uint8_t key[16], const uint8_t iv[16],
                                                         const uint8_t* target, int len) {
    State s; init(s, key, iv);
    uint32_t X0=((s.S[15]&0x7FFF8000)<<1)|(s.S[14]&0xFFFF);
    uint32_t X1=((s.S[11]&0xFFFF)<<16)|(s.S[9]>>15);
    uint32_t X2=((s.S[7]&0xFFFF)<<16)|(s.S[5]>>15);
    F(s,X0,X1,X2); update_LFSR(s,0);
    int blocks = len / 4;
    for (int i = 0; i < blocks; i++) {
      X0=((s.S[15]&0x7FFF8000)<<1)|(s.S[14]&0xFFFF);
      X1=((s.S[11]&0xFFFF)<<16)|(s.S[9]>>15);
      X2=((s.S[7]&0xFFFF)<<16)|(s.S[5]>>15);
      uint32_t X3=((s.S[2]&0xFFFF)<<16)|(s.S[0]>>15);
      uint32_t Z = F(s,X0,X1,X2) ^ X3; update_LFSR(s,0);
      if (Z != load32_be(target + 4*i)) return false;
    }
    return true;
  }
};

// ============================================================
// SNOW-V Stream Cipher (with T-Table AES optimization)
// Key: 256 bits, IV: 128 bits
// Ref: https://eprint.iacr.org/2018/1143.pdf
// ============================================================
struct SNOW_V {
  struct State {
    uint16_t A[16], B[16];
    uint32_t R1[4], R2[4], R3[4];
  };

  __host__ __device__ static inline uint8_t sbox(uint8_t x) {
    static const uint8_t S[256] = {
      0x63,0x7C,0x77,0x7B,0xF2,0x6B,0x6F,0xC5,0x30,0x01,0x67,0x2B,0xFE,0xD7,0xAB,0x76,
      0xCA,0x82,0xC9,0x7D,0xFA,0x59,0x47,0xF0,0xAD,0xD4,0xA2,0xAF,0x9C,0xA4,0x72,0xC0,
      0xB7,0xFD,0x93,0x26,0x36,0x3F,0xF7,0xCC,0x34,0xA5,0xE5,0xF1,0x71,0xD8,0x31,0x15,
      0x04,0xC7,0x23,0xC3,0x18,0x96,0x05,0x9A,0x07,0x12,0x80,0xE2,0xEB,0x27,0xB2,0x75,
      0x09,0x83,0x2C,0x1A,0x1B,0x6E,0x5A,0xA0,0x52,0x3B,0xD6,0xB3,0x29,0xE3,0x2F,0x84,
      0x53,0xD1,0x00,0xED,0x20,0xFC,0xB1,0x5B,0x6A,0xCB,0xBE,0x39,0x4A,0x4C,0x58,0xCF,
      0xD0,0xEF,0xAA,0xFB,0x43,0x4D,0x33,0x85,0x45,0xF9,0x02,0x7F,0x50,0x3C,0x9F,0xA8,
      0x51,0xA3,0x40,0x8F,0x92,0x9D,0x38,0xF5,0xBC,0xB6,0xDA,0x21,0x10,0xFF,0xF3,0xD2,
      0xCD,0x0C,0x13,0xEC,0x5F,0x97,0x44,0x17,0xC4,0xA7,0x7E,0x3D,0x64,0x5D,0x19,0x73,
      0x60,0x81,0x4F,0xDC,0x22,0x2A,0x90,0x88,0x46,0xEE,0xB8,0x14,0xDE,0x5E,0x0B,0xDB,
      0xE0,0x32,0x3A,0x0A,0x49,0x06,0x24,0x5C,0xC2,0xD3,0xAC,0x62,0x91,0x95,0xE4,0x79,
      0xE7,0xC8,0x37,0x6D,0x8D,0xD5,0x4E,0xA9,0x6C,0x56,0xF4,0xEA,0x65,0x7A,0xAE,0x08,
      0xBA,0x78,0x25,0x2E,0x1C,0xA6,0xB4,0xC6,0xE8,0xDD,0x74,0x1F,0x4B,0xBD,0x8B,0x8A,
      0x70,0x3E,0xB5,0x66,0x48,0x03,0xF6,0x0E,0x61,0x35,0x57,0xB9,0x86,0xC1,0x1D,0x9E,
      0xE1,0xF8,0x98,0x11,0x69,0xD9,0x8E,0x94,0x9B,0x1E,0x87,0xE9,0xCE,0x55,0x28,0xDF,
      0x8C,0xA1,0x89,0x0D,0xBF,0xE6,0x42,0x68,0x41,0x99,0x2D,0x0F,0xB0,0x54,0xBB,0x16
    };
    return S[x];
  }

  // T-table: combines AES SubBytes + MixColumns (standard T0 table, little-endian)
  __host__ __device__ static inline uint32_t t0_lookup(uint8_t x) {
    static const uint32_t T0[256] = {
      0xa56363c6,0x847c7cf8,0x997777ee,0x8d7b7bf6,0x0df2f2ff,0xbd6b6bd6,0xb16f6fde,0x54c5c591,
      0x50303060,0x03010102,0xa96767ce,0x7d2b2b56,0x19fefee7,0x62d7d7b5,0xe6abab4d,0x9a7676ec,
      0x45caca8f,0x9d82821f,0x40c9c989,0x877d7dfa,0x15fafaef,0xeb5959b2,0xc947478e,0x0bf0f0fb,
      0xecadad41,0x67d4d4b3,0xfda2a25f,0xeaafaf45,0xbf9c9c23,0xf7a4a453,0x967272e4,0x5bc0c09b,
      0xc2b7b775,0x1cfdfde1,0xae93933d,0x6a26264c,0x5a36366c,0x413f3f7e,0x02f7f7f5,0x4fcccc83,
      0x5c343468,0xf4a5a551,0x34e5e5d1,0x08f1f1f9,0x937171e2,0x73d8d8ab,0x53313162,0x3f15152a,
      0x0c040408,0x52c7c795,0x65232346,0x5ec3c39d,0x28181830,0xa1969637,0x0f05050a,0xb59a9a2f,
      0x0907070e,0x36121224,0x9b80801b,0x3de2e2df,0x26ebebcd,0x6927274e,0xcdb2b27f,0x9f7575ea,
      0x1b090912,0x9e83831d,0x742c2c58,0x2e1a1a34,0x2d1b1b36,0xb26e6edc,0xee5a5ab4,0xfba0a05b,
      0xf65252a4,0x4d3b3b76,0x61d6d6b7,0xceb3b37d,0x7b292952,0x3ee3e3dd,0x712f2f5e,0x97848413,
      0xf55353a6,0x68d1d1b9,0x00000000,0x2cededc1,0x60202040,0x1ffcfce3,0xc8b1b179,0xed5b5bb6,
      0xbe6a6ad4,0x46cbcb8d,0xd9bebe67,0x4b393972,0xde4a4a94,0xd44c4c98,0xe85858b0,0x4acfcf85,
      0x6bd0d0bb,0x2aefefc5,0xe5aaaa4f,0x16fbfbed,0xc5434386,0xd74d4d9a,0x55333366,0x94858511,
      0xcf45458a,0x10f9f9e9,0x06020204,0x817f7ffe,0xf05050a0,0x443c3c78,0xba9f9f25,0xe3a8a84b,
      0xf35151a2,0xfea3a35d,0xc0404080,0x8a8f8f05,0xad92923f,0xbc9d9d21,0x48383870,0x04f5f5f1,
      0xdfbcbc63,0xc1b6b677,0x75dadaaf,0x63212142,0x30101020,0x1affffe5,0x0ef3f3fd,0x6dd2d2bf,
      0x4ccdcd81,0x140c0c18,0x35131326,0x2fececc3,0xe15f5fbe,0xa2979735,0xcc444488,0x3917172e,
      0x57c4c493,0xf2a7a755,0x827e7efc,0x473d3d7a,0xac6464c8,0xe75d5dba,0x2b191932,0x957373e6,
      0xa06060c0,0x98818119,0xd14f4f9e,0x7fdcdca3,0x66222244,0x7e2a2a54,0xab90903b,0x8388880b,
      0xca46468c,0x29eeeec7,0xd3b8b86b,0x3c141428,0x79dedea7,0xe25e5ebc,0x1d0b0b16,0x76dbdbad,
      0x3be0e0db,0x56323264,0x4e3a3a74,0x1e0a0a14,0xdb494992,0x0a06060c,0x6c242448,0xe45c5cb8,
      0x5dc2c29f,0x6ed3d3bd,0xefacac43,0xa66262c4,0xa8919139,0xa4959531,0x37e4e4d3,0x8b7979f2,
      0x32e7e7d5,0x43c8c88b,0x5937376e,0xb76d6dda,0x8c8d8d01,0x64d5d5b1,0xd24e4e9c,0xe0a9a949,
      0xb46c6cd8,0xfa5656ac,0x07f4f4f3,0x25eaeacf,0xaf6565ca,0x8e7a7af4,0xe9aeae47,0x18080810,
      0xd5baba6f,0x887878f0,0x6f25254a,0x722e2e5c,0x241c1c38,0xf1a6a657,0xc7b4b473,0x51c6c697,
      0x23e8e8cb,0x7cdddda1,0x9c7474e8,0x211f1f3e,0xdd4b4b96,0xdcbdbd61,0x868b8b0d,0x858a8a0f,
      0x907070e0,0x423e3e7c,0xc4b5b571,0xaa6666cc,0xd8484890,0x05030306,0x01f6f6f7,0x120e0e1c,
      0xa36161c2,0x5f35356a,0xf95757ae,0xd0b9b969,0x91868617,0x58c1c199,0x271d1d3a,0xb99e9e27,
      0x38e1e1d9,0x13f8f8eb,0xb398982b,0x33111122,0xbb6969d2,0x70d9d9a9,0x898e8e07,0xa7949433,
      0xb69b9b2d,0x221e1e3c,0x92878715,0x20e9e9c9,0x49cece87,0xff5555aa,0x78282850,0x7adfdfa5,
      0x8f8c8c03,0xf8a1a159,0x80898909,0x170d0d1a,0xdabfbf65,0x31e6e6d7,0xc6424284,0xb86868d0,
      0xc3414182,0xb0999929,0x772d2d5a,0x110f0f1e,0xcbb0b07b,0xfc5454a8,0xd6bbbb6d,0x3a16162c,
    };
    return T0[x];
  }

  __host__ __device__ static inline uint8_t sigma_val(uint8_t x) {
    static const uint8_t Sig[16] = {0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15};
    return Sig[x & 0xF];
  }

  __host__ __device__ static inline uint16_t load16_le(const uint8_t* p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
  }
  __host__ __device__ static inline uint32_t load32_le_snow(const uint8_t* p) {
    return (uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);
  }
  __host__ __device__ static inline uint32_t rotl32(uint32_t x, int r) { return (x<<r)|(x>>(32-r)); }
  __host__ __device__ static inline uint16_t mul_x(uint16_t v, uint16_t c) {
    return (v & 0x8000) ? ((v << 1) ^ c) : (v << 1);
  }
  __host__ __device__ static inline uint16_t mul_x_inv(uint16_t v, uint16_t d) {
    return (v & 0x0001) ? ((v >> 1) ^ d) : (v >> 1);
  }

  // AES round using T-tables + rotations (SubBytes + ShiftRows + MixColumns, zero round key)
  __host__ __device__ static void aes_enc_round(uint32_t result[4], const uint32_t state[4]) {
    uint32_t t00=t0_lookup((state[0]>> 0)&0xFF), t01=t0_lookup((state[0]>> 8)&0xFF);
    uint32_t t02=t0_lookup((state[0]>>16)&0xFF), t03=t0_lookup((state[0]>>24)&0xFF);
    uint32_t t10=t0_lookup((state[1]>> 0)&0xFF), t11=t0_lookup((state[1]>> 8)&0xFF);
    uint32_t t12=t0_lookup((state[1]>>16)&0xFF), t13=t0_lookup((state[1]>>24)&0xFF);
    uint32_t t20=t0_lookup((state[2]>> 0)&0xFF), t21=t0_lookup((state[2]>> 8)&0xFF);
    uint32_t t22=t0_lookup((state[2]>>16)&0xFF), t23=t0_lookup((state[2]>>24)&0xFF);
    uint32_t t30=t0_lookup((state[3]>> 0)&0xFF), t31=t0_lookup((state[3]>> 8)&0xFF);
    uint32_t t32=t0_lookup((state[3]>>16)&0xFF), t33=t0_lookup((state[3]>>24)&0xFF);
    result[0] = t00 ^ rotl32(t11, 8) ^ rotl32(t22,16) ^ rotl32(t33,24);
    result[1] = t10 ^ rotl32(t21, 8) ^ rotl32(t32,16) ^ rotl32(t03,24);
    result[2] = t20 ^ rotl32(t31, 8) ^ rotl32(t02,16) ^ rotl32(t13,24);
    result[3] = t30 ^ rotl32(t01, 8) ^ rotl32(t12,16) ^ rotl32(t23,24);
  }

  // Sigma permutation (matrix transpose of the 4x4 byte state)
  __host__ __device__ static inline void permute_sigma(uint32_t state[4]) {
    uint32_t s0=state[0], s1=state[1], s2=state[2], s3=state[3];
    state[0]=(s0&0x000000FF)|((s1&0x000000FF)<<8)|((s2&0x000000FF)<<16)|((s3&0x000000FF)<<24);
    state[1]=((s0&0x0000FF00)>>8)|(s1&0x0000FF00)|((s2&0x0000FF00)<<8)|((s3&0x0000FF00)<<16);
    state[2]=((s0&0x00FF0000)>>16)|((s1&0x00FF0000)>>8)|(s2&0x00FF0000)|((s3&0x00FF0000)<<8);
    state[3]=(s0>>24)|((s1&0xFF000000)>>16)|((s2&0xFF000000)>>8)|(s3&0xFF000000);
  }

  __host__ __device__ static void fsm_update(State& s) {
    uint32_t R1temp[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) R1temp[i] = s.R1[i];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      uint32_t T2 = ((uint32_t)s.A[2*i+1] << 16) | s.A[2*i];
      s.R1[i] = (T2 ^ s.R3[i]) + s.R2[i];
    }
    permute_sigma(s.R1);
    aes_enc_round(s.R3, s.R2);
    aes_enc_round(s.R2, R1temp);
  }

  __host__ __device__ static inline void lfsr_update(State& s) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
      uint16_t u = mul_x(s.A[0], 0x990f) ^ s.A[1] ^ mul_x_inv(s.A[8], 0xcc87) ^ s.B[0];
      uint16_t v = mul_x(s.B[0], 0xc963) ^ s.B[3] ^ mul_x_inv(s.B[8], 0xe4b1) ^ s.A[0];
      #pragma unroll
      for (int j = 0; j < 15; j++) { s.A[j]=s.A[j+1]; s.B[j]=s.B[j+1]; }
      s.A[15] = u; s.B[15] = v;
    }
  }

  // Generate 16 bytes of keystream
  __host__ __device__ static inline void keystream(State& s, uint8_t z[16]) {
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      uint32_t T1 = ((uint32_t)s.B[2*i+9] << 16) | s.B[2*i+8];
      uint32_t v = (T1 + s.R1[i]) ^ s.R2[i];
      z[i*4+0]=(uint8_t)(v&0xff); z[i*4+1]=(uint8_t)((v>>8)&0xff);
      z[i*4+2]=(uint8_t)((v>>16)&0xff); z[i*4+3]=(uint8_t)((v>>24)&0xff);
    }
    fsm_update(s);
    lfsr_update(s);
  }

  __host__ __device__ static inline void init(State& s, const uint8_t key[32], const uint8_t iv[16]) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
      s.A[i]   = load16_le(iv + 2*i);
      s.A[i+8] = load16_le(key + 2*i);
      s.B[i]   = 0x0000;
      s.B[i+8] = load16_le(key + 16 + 2*i);
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) s.R1[i] = s.R2[i] = s.R3[i] = 0;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
      uint8_t z[16]; keystream(s, z);
      #pragma unroll
      for (int j = 0; j < 8; j++) s.A[j+8] ^= load16_le(z + 2*j);
      if (i == 14) { for (int j = 0; j < 4; j++) s.R1[j] ^= load32_le_snow(key + 4*j); }
      if (i == 15) { for (int j = 0; j < 4; j++) s.R1[j] ^= load32_le_snow(key + 16 + 4*j); }
    }
  }

  __host__ __device__ static inline void process(const uint8_t* pt, uint8_t* ct, int len,
                                                 const uint8_t key[32], const uint8_t iv[16]) {
    State s; init(s, key, iv);
    int blocks = len / 16, rem = len % 16;
    for (int i = 0; i < blocks; i++) {
      uint8_t z[16]; keystream(s, z);
      #pragma unroll
      for (int j = 0; j < 16; j++) ct[16*i+j] = pt[16*i+j] ^ z[j];
    }
    if (rem > 0) {
      uint8_t z[16]; keystream(s, z);
      for (int j = 0; j < rem; j++) ct[16*blocks+j] = pt[16*blocks+j] ^ z[j];
    }
  }

  // Early-exit keystream matching for GPU brute-force
  __host__ __device__ static bool match_keystream(const uint8_t key[32], const uint8_t iv[16],
                                                  const uint8_t* target, int len) {
    State s; init(s, key, iv);
    int blocks = len / 16, rem = len % 16;
    for (int i = 0; i < blocks; i++) {
      uint8_t z[16]; keystream(s, z);
      #pragma unroll
      for (int j = 0; j < 16; j++) if (z[j] != target[16*i+j]) return false;
    }
    if (rem > 0) {
      uint8_t z[16]; keystream(s, z);
      for (int j = 0; j < rem; j++) if (z[j] != target[16*blocks+j]) return false;
    }
    return true;
  }
};

