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

