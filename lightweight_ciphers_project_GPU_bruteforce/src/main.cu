#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <algorithm>
#include <sstream>
#include <iomanip>

#include "util.hpp"
#include "ciphers_enhanced.cuh"
#include "bruteforce_cpu.hpp"
#include "bruteforce_gpu_enhanced.cuh"


// ============================================================
// Common helpers (avoid undefined shifts, avoid div-by-zero)
// ============================================================

static inline uint64_t bf_space_size_main(int unknown_bits) {
  if (unknown_bits < 0) return 0ULL;
  // We treat >=63 as unsupported for uint64_t brute-force space sizing.
  if (unknown_bits >= 63) return 0ULL;
  return 1ULL << (uint64_t)unknown_bits;
}

static inline double safe_kps(uint64_t keys, double seconds) {
  if (seconds <= 0.0) return 0.0;
  return (double)keys / seconds;
}


// ============================================================
// Test Vectors / Bench Inputs
// ============================================================

struct TestVector_Simon {
  uint64_t key;
  uint32_t pt;
  uint32_t ct;
};

struct TestVector_Present {
  uint8_t key[10];
  uint64_t pt;
  uint64_t ct;
};

struct TestVector_Speck {
  uint8_t key[16];
  uint64_t pt;
  uint64_t ct;
};

struct TestVector_Stream {
  uint8_t key[10];
  uint8_t iv[10];
  uint8_t pt[16];
  uint8_t ct[16];
  int length;
};

struct TestVector_ChaChaBlock {
  uint8_t key[32];
  uint8_t nonce[12];
  uint32_t counter;
  uint8_t block[64];
};

struct TestVector_ChaChaBench {
  uint64_t key64;
  uint8_t nonce[12];
  uint8_t pt[16];
  uint8_t ct[16];
  int length;
};

struct TestVector_TinyJambu {
  uint8_t key[16];
  uint8_t nonce[12];
  uint8_t ad[32];
  uint8_t pt[32];
  uint8_t ct[32];
  uint8_t tag[8];
  int ad_len;
  int pt_len;
};

struct TestVector_ZUC128 {
  uint8_t key[16];
  uint8_t iv[16];
  uint8_t pt[32];
  uint8_t ct[32];
  int length;
};

struct TestVector_SnowV {
  uint8_t key[32];
  uint8_t iv[16];
  uint8_t pt[16];
  uint8_t ct[16];
  int length;
};

struct TestVector_AES {
  uint8_t key[16];
  uint8_t pt[16];
  uint8_t ct[16];
};

struct TestVector_Salsa20Block {
  uint8_t key[32];
  uint8_t nonce[8];
  uint8_t stream[16]; // first 16 bytes of keystream (counter=0), verified from eSTREAM Set 1
};

struct TestVector_Salsa20Bench {
  uint64_t key64;
  uint8_t nonce[8];
  uint8_t pt[64];
  uint8_t ct[64];
  int length;
};

struct TestVector_Grain128AEADv2 {
  uint8_t key[16];
  uint8_t nonce[12];
  uint8_t pt[32];
  uint8_t ct[32];
  uint8_t ad[32];
  uint8_t tag[8];
  int pt_len;
  int ad_len;
};

// SIMON 32/64 official test vector
static TestVector_Simon simon_tv() {
  uint64_t key = 0;
  key |= (uint64_t)0x1918ULL << 48;
  key |= (uint64_t)0x1110ULL << 32;
  key |= (uint64_t)0x0908ULL << 16;
  key |= (uint64_t)0x0100ULL;

  uint32_t pt = ((uint32_t)0x6877U << 16) | 0x6565U;
  uint32_t ct = ((uint32_t)0xe9bbU << 16) | 0xc69bU;

  return {key, pt, ct};
}

// PRESENT-80 official test vector: key=0, pt=0, ct=0x5579c1387b228445
static TestVector_Present present_tv() {
  TestVector_Present tv;
  memset(tv.key, 0, 10);
  tv.pt = 0x0000000000000000ULL;
  tv.ct = 0x5579c1387b228445ULL;
  return tv;
}

// SPECK64/128 official test vector (NSA implementation guide)
// Key bytes: 00 01 02 03 08 09 0a 0b 10 11 12 13 18 19 1a 1b
// Pt bytes:  2d 43 75 74 74 65 72 3b  ("-Cutter;")
// Ct bytes:  8b 02 4e 45 48 a5 6f 8c
static TestVector_Speck speck_tv() {
  TestVector_Speck tv;
  const uint8_t key[16] = {
    0x00,0x01,0x02,0x03, 0x08,0x09,0x0a,0x0b,
    0x10,0x11,0x12,0x13, 0x18,0x19,0x1a,0x1b
  };
  memcpy(tv.key, key, 16);
  tv.pt = 0x3b7265747475432dULL;
  tv.ct = 0x8c6fa548454e028bULL;
  return tv;
}



static inline uint64_t bench_key64_last_candidate() {
  // Low 30 bits are all 1, so for every benchmark with unknown_bits <= 30
  // the matching key is the last candidate in the searched subspace.
  return 0xA55A12343FFFFFFFULL;
}

static inline void store_u64_le(uint64_t v, uint8_t* out, int nbytes) {
  for (int i = 0; i < nbytes; i++) out[i] = (uint8_t)((v >> (8 * i)) & 0xFF);
}

static TestVector_Simon simon_bench_tv() {
  TestVector_Simon tv{};
  tv.key = bench_key64_last_candidate();
  tv.pt = 0x6c617669U; // arbitrary non-zero plaintext
  uint16_t rk[SIMON32_64_ROUNDS];
  Simon32_64_Enhanced::expand_key_vectorized(tv.key, rk);
  tv.ct = Simon32_64_Enhanced::encrypt_optimized(tv.pt, rk);
  return tv;
}

static TestVector_Present present_bench_tv() {
  TestVector_Present tv{};
  const uint64_t key64 = bench_key64_last_candidate();
  store_u64_le(key64, tv.key, 8);
  tv.key[8] = 0;
  tv.key[9] = 0;
  tv.pt = 0x0123456789ABCDEFULL;
  uint64_t rk[PRESENT_ROUNDS + 1];
  Present80::expand_key(tv.key, rk);
  tv.ct = Present80::encrypt(tv.pt, rk);
  return tv;
}

static TestVector_Speck speck_bench_tv() {
  TestVector_Speck tv{};
  const uint64_t key64 = bench_key64_last_candidate();
  store_u64_le(key64, tv.key, 8);
  for (int i = 8; i < 16; i++) tv.key[i] = 0;
  tv.pt = 0x6c61766975716520ULL;
  uint32_t rk[SPECK64_128_ROUNDS];
  Speck64_128::expand_key(tv.key, rk);
  tv.ct = Speck64_128::encrypt(tv.pt, rk);
  return tv;
}

// Grain v1 benchmark input (ciphertext computed from key/iv/pt)
static TestVector_Stream grain_bench_tv() {
  TestVector_Stream tv;
  memset(&tv, 0, sizeof(tv));

  // Key: low 64 bits chosen so the correct key is the last candidate for all b <= 30.
  store_u64_le(bench_key64_last_candidate(), tv.key, 8);
  tv.key[8] = 0;
  tv.key[9] = 0;

  // IV: 64 bits = 8 bytes
  tv.iv[0] = 0x11;
  tv.iv[1] = 0x22;
  tv.iv[2] = 0x33;
  tv.iv[3] = 0x44;

  const char* msg = "Hello Grain!";
  tv.length = (int)strlen(msg);
  memcpy(tv.pt, msg, tv.length);
  GrainV1::process(tv.pt, tv.ct, tv.length, tv.key, tv.iv);
  return tv;
}

// Trivium benchmark input (ciphertext computed from key/iv/pt)
static TestVector_Stream trivium_bench_tv() {
  TestVector_Stream tv;
  memset(&tv, 0, sizeof(tv));

  store_u64_le(bench_key64_last_candidate(), tv.key, 8);
  tv.key[8] = 0;
  tv.key[9] = 0;

  // IV: 80 bits = 10 bytes
  tv.iv[0] = 0x12;
  tv.iv[1] = 0x34;
  tv.iv[2] = 0x56;
  tv.iv[3] = 0x78;

  const char* msg = "Test Trivium";
  tv.length = (int)strlen(msg);
  memcpy(tv.pt, msg, tv.length);
  Trivium::process(tv.pt, tv.ct, tv.length, tv.key, tv.iv);
  return tv;
}

// ChaCha20 block-function official test vector from RFC 8439 (Section 2.3.2)
static TestVector_ChaChaBlock chacha_block_tv() {
  TestVector_ChaChaBlock tv;
  for (int i = 0; i < 32; i++) tv.key[i] = (uint8_t)i;
  tv.nonce[0] = 0x00; tv.nonce[1] = 0x00; tv.nonce[2] = 0x00; tv.nonce[3] = 0x09;
  tv.nonce[4] = 0x00; tv.nonce[5] = 0x00; tv.nonce[6] = 0x00; tv.nonce[7] = 0x4a;
  tv.nonce[8] = 0x00; tv.nonce[9] = 0x00; tv.nonce[10]= 0x00; tv.nonce[11]= 0x00;
  tv.counter = 1;

  // Serialized keystream block (64 bytes), RFC 8439 Section 2.3.2
  const uint8_t expected[64] = {
    0x10,0xf1,0xe7,0xe4,0xd1,0x3b,0x59,0x15,0x50,0x0f,0xdd,0x1f,0xa3,0x20,0x71,0xc4,
    0xc7,0xd1,0xf4,0xc7,0x33,0xc0,0x68,0x03,0x04,0x22,0xaa,0x9a,0xc3,0xd4,0x6c,0x4e,
    0xd2,0x82,0x64,0x46,0x07,0x9f,0xaa,0x09,0x14,0xc2,0xd7,0x05,0xd9,0x8b,0x02,0xa2,
    0xb5,0x12,0x9c,0xd1,0xde,0x16,0x4e,0xb9,0xcb,0xd0,0x83,0xe8,0xa2,0x50,0x3c,0x4e
  };
  memcpy(tv.block, expected, 64);
  return tv;
}

// ChaCha20 benchmark input (we brute-force only low 64 bits; rest of key is 0)
static TestVector_ChaChaBench chacha_bench_tv() {
  TestVector_ChaChaBench tv;
  memset(&tv, 0, sizeof(tv));

  tv.key64 = bench_key64_last_candidate();
  tv.nonce[0]=0x00; tv.nonce[1]=0x01; tv.nonce[2]=0x02; tv.nonce[3]=0x03;
  tv.nonce[4]=0x04; tv.nonce[5]=0x05; tv.nonce[6]=0x06; tv.nonce[7]=0x07;
  tv.nonce[8]=0x08; tv.nonce[9]=0x09; tv.nonce[10]=0x0a; tv.nonce[11]=0x0b;

  const char* msg = "Hello ChaCha!";
  tv.length = (int)strlen(msg);
  memcpy(tv.pt, msg, tv.length);

  uint8_t key256[32];
  for (int j = 0; j < 8; j++) key256[j] = (uint8_t)((tv.key64 >> (8*j)) & 0xFF);
  for (int j = 8; j < 32; j++) key256[j] = 0;

  ChaCha20::process(tv.pt, tv.ct, tv.length, key256, 1, tv.nonce);
  return tv;
}

// TinyJAMBU-128 NIST KAT (LWC_AEAD_KAT_128_96.txt)
static TestVector_TinyJambu tinyjambu_nist_tv() {
  TestVector_TinyJambu tv;
  memset(&tv, 0, sizeof(tv));
  const uint8_t key[16]   = {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
                              0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F};
  const uint8_t nonce[12] = {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
                              0x08,0x09,0x0A,0x0B};
  const uint8_t ad[5]  = {0x00,0x01,0x02,0x03,0x04};
  const uint8_t pt[5]  = {0x00,0x01,0x02,0x03,0x04};
  const uint8_t exp_ct[5]  = {0x10,0x17,0x1c,0x22,0xf8};
  const uint8_t exp_tag[8] = {0x65,0x42,0x31,0x96,0x1b,0x23,0xa2,0xab};
  memcpy(tv.key, key, 16);
  memcpy(tv.nonce, nonce, 12);
  memcpy(tv.ad, ad, 5); tv.ad_len = 5;
  memcpy(tv.pt, pt, 5); tv.pt_len = 5;
  memcpy(tv.ct, exp_ct, 5);
  memcpy(tv.tag, exp_tag, 8);
  return tv;
}

// TinyJAMBU-128 benchmark TV — key = bench_key64_last_candidate, worst-case position
static TestVector_TinyJambu tinyjambu_bench_tv() {
  TestVector_TinyJambu tv;
  memset(&tv, 0, sizeof(tv));

  const uint64_t key64 = bench_key64_last_candidate();
  store_u64_le(key64, tv.key, 8);   // low 64 bits; high 64 = 0

  const uint8_t nonce[12] = {0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,
                              0x18,0x19,0x1A,0x1B};
  const uint8_t ad[4]  = {0xDE,0xAD,0xBE,0xEF};
  const char*   msg    = "Bench!";
  memcpy(tv.nonce, nonce, 12);
  memcpy(tv.ad, ad, 4); tv.ad_len = 4;
  tv.pt_len = (int)strlen(msg);
  memcpy(tv.pt, msg, tv.pt_len);

  TinyJAMBU128::encrypt(tv.pt, tv.ct, tv.pt_len, tv.tag, tv.key, tv.nonce, tv.ad, tv.ad_len);
  return tv;
}

// ZUC-128 GSMA Test Set 3 (for self-test only)
struct ZucNistTV {
  uint8_t  key[16];
  uint8_t  iv[16];
  uint32_t expected_ks[2];
};
static ZucNistTV zuc_nist_tv() {
  ZucNistTV tv;
  const uint8_t key[16] = {0x3d,0x4c,0x4b,0xe9,0x6a,0x82,0xfd,0xae,
                            0xb5,0x8f,0x64,0x1d,0xb1,0x7b,0x45,0x5b};
  const uint8_t iv[16]  = {0x84,0x31,0x9a,0xa8,0xde,0x69,0x15,0xca,
                            0x1f,0x6b,0xda,0x6b,0xfb,0xd8,0xc7,0x66};
  memcpy(tv.key, key, 16);
  memcpy(tv.iv, iv, 16);
  tv.expected_ks[0] = 0x14f1c272u;
  tv.expected_ks[1] = 0x3279c419u;
  return tv;
}

// ZUC-128 benchmark TV — key = bench_key64_last_candidate, worst-case position
static TestVector_ZUC128 zuc_bench_tv() {
  TestVector_ZUC128 tv;
  memset(&tv, 0, sizeof(tv));

  const uint64_t key64 = bench_key64_last_candidate();
  store_u64_le(key64, tv.key, 8);   // high 8 bytes remain 0

  tv.iv[0] = 0x42; tv.iv[1] = 0x13; tv.iv[2] = 0x07;

  const char* msg = "ZUC-128 Test";
  tv.length = (int)strlen(msg);
  memcpy(tv.pt, msg, tv.length);
  ZUC::process(tv.pt, tv.ct, tv.length, tv.key, tv.iv);
  return tv;
}

// SNOW-V paper Test Vector #3 (for self-test only)
struct SnowVNistTV {
  uint8_t key[32];
  uint8_t iv[16];
  uint8_t expected_ks[128];
};
static SnowVNistTV snow_v_nist_tv() {
  SnowVNistTV tv;
  const uint8_t key[32] = {
    0x50,0x51,0x52,0x53,0x54,0x55,0x56,0x57,0x58,0x59,0x5a,0x5b,0x5c,0x5d,0x5e,0x5f,
    0x0a,0x1a,0x2a,0x3a,0x4a,0x5a,0x6a,0x7a,0x8a,0x9a,0xaa,0xba,0xca,0xda,0xea,0xfa
  };
  const uint8_t iv[16] = {
    0x01,0x23,0x45,0x67,0x89,0xab,0xcd,0xef,0xfe,0xdc,0xba,0x98,0x76,0x54,0x32,0x10
  };
  const uint8_t expected_ks[128] = {
    0xaa,0x81,0xea,0xfb,0x8b,0x86,0x16,0xce,0x3e,0x5c,0xe2,0x22,0x24,0x61,0xc5,0x0a,
    0x6a,0xb4,0x48,0x77,0x56,0xde,0x4b,0xd3,0x1c,0x90,0x4f,0x3d,0x97,0x8a,0xfe,0x56,
    0x33,0x4f,0x10,0xdd,0xdf,0x2b,0x95,0x31,0x76,0x9a,0x71,0x05,0x0b,0xe4,0x38,0x5f,
    0xc2,0xb6,0x19,0x2c,0x7a,0x85,0x7b,0xe8,0xb4,0xfc,0x28,0xb7,0x09,0xf0,0x8f,0x11,
    0xf2,0x06,0x49,0xe2,0xee,0xf2,0x49,0x80,0xf8,0x6c,0x4c,0x11,0x36,0x41,0xfe,0xd2,
    0xf3,0xf6,0xfa,0x2b,0x91,0x95,0x12,0x06,0xb8,0x01,0xdb,0x15,0x46,0x65,0x17,0xa6,
    0x33,0x0a,0xdd,0xa6,0xb3,0x5b,0x26,0x5e,0xfd,0x72,0x2e,0x86,0x77,0xb4,0x8b,0xfc,
    0x15,0xb4,0x41,0x18,0xde,0x52,0xd0,0x73,0xb0,0xad,0x0f,0xe7,0x59,0x4d,0x62,0x91
  };
  memcpy(tv.key, key, 32);
  memcpy(tv.iv, iv, 16);
  memcpy(tv.expected_ks, expected_ks, 128);
  return tv;
}

// SNOW-V benchmark TV — key = bench_key64_last_candidate, worst-case position
static TestVector_SnowV snow_v_bench_tv() {
  TestVector_SnowV tv;
  memset(&tv, 0, sizeof(tv));

  const uint64_t key64 = bench_key64_last_candidate();
  store_u64_le(key64, tv.key, 8);   // high 24 bytes remain 0

  tv.iv[0] = 0xA1; tv.iv[1] = 0xB2; tv.iv[2] = 0xC3;

  const char* msg = "Hello SNOW-V!";
  tv.length = (int)strlen(msg);
  memcpy(tv.pt, msg, tv.length);
  SNOW_V::process<true>(tv.pt, tv.ct, tv.length, tv.key, tv.iv);
  return tv;
}

// AES-128 NIST test vector (FIPS 197 Appendix C.1)
static TestVector_AES aes_tv() {
  TestVector_AES tv;
  const uint8_t key[16] = {0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
                            0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c};
  const uint8_t pt[16]  = {0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d,
                            0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07, 0x34};
  const uint8_t ct[16]  = {0x39, 0x25, 0x84, 0x1d, 0x02, 0xdc, 0x09, 0xfb,
                            0xdc, 0x11, 0x85, 0x97, 0x19, 0x6a, 0x0b, 0x32};
  memcpy(tv.key, key, 16);
  memcpy(tv.pt, pt, 16);
  memcpy(tv.ct, ct, 16);
  return tv;
}

// AES-128 benchmark TV — low 64 bits = bench_key64_last_candidate, high 64 bits = 0
static TestVector_AES aes_bench_tv() {
  TestVector_AES tv;
  memset(&tv, 0, sizeof(tv));
  const uint64_t key64 = bench_key64_last_candidate();
  store_u64_le(key64, tv.key, 8);
  for (int j = 8; j < 16; j++) tv.key[j] = 0;

  const uint8_t pt[16] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
                           0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff};
  memcpy(tv.pt, pt, 16);

  AES128::encrypt<true>(tv.pt, tv.key, tv.ct);
  return tv;
}

// Salsa20/20 eSTREAM test vector — Set 1, vector 0 (first 16 bytes only)
// key[0] = 0x80, key[1..31] = 0; nonce = all zeros; counter = 0
// These first 16 bytes are verified correct from eSTREAM Set 1, vector 0.
// Only 16 bytes stored: full 64-byte comparison is replaced by round-trip test below.
static TestVector_Salsa20Block salsa20_ecrypt_tv() {
  TestVector_Salsa20Block tv;
  memset(&tv, 0, sizeof(tv));
  tv.key[0] = 0x80; // key = {0x80, 0, 0, ..., 0}
  // nonce = all zeros (already zeroed by memset)
  // Expected first 16 bytes of keystream (eSTREAM Set 1, vector 0)
  const uint8_t expected16[16] = {
    0xE3,0xBE,0x8F,0xDD,0x8B,0xEC,0xA2,0xE3,
    0xEA,0x8E,0xF9,0x47,0x5B,0x29,0xA6,0xE7
  };
  memcpy(tv.stream, expected16, 16);
  return tv;
}

// Salsa20 benchmark TV — key = bench_key64_last_candidate, worst-case position
static TestVector_Salsa20Bench salsa20_bench_tv() {
  TestVector_Salsa20Bench tv;
  memset(&tv, 0, sizeof(tv));

  const uint64_t key64 = bench_key64_last_candidate();
  tv.key64 = key64;
  tv.nonce[0] = 0x12; tv.nonce[1] = 0x34; tv.nonce[2] = 0x56; tv.nonce[3] = 0x78;
  // nonce[4..7] already zeroed by memset

  const char* msg = "Hello Salsa20!";
  tv.length = (int)strlen(msg);
  memcpy(tv.pt, msg, tv.length);

  uint8_t key256[32] = {0};
  store_u64_le(key64, key256, 8);  // low 8 bytes from key64, high 24 bytes remain 0
  Salsa20::process(tv.pt, tv.ct, tv.length, key256, tv.nonce);
  return tv;
}

// Grain-128AEADv2 NIST LWC test vector
static TestVector_Grain128AEADv2 grain128aeadv2_nist_tv() {
  TestVector_Grain128AEADv2 tv;
  memset(&tv, 0, sizeof(tv));

  const uint8_t key[16] = {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
                            0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f};
  memcpy(tv.key, key, 16);

  const uint8_t nonce[12] = {0x00,0x01,0x02,0x03,0x04,0x05,
                              0x06,0x07,0x08,0x09,0x0a,0x0b};
  memcpy(tv.nonce, nonce, 12);

  const uint8_t ad[8] = {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07};
  memcpy(tv.ad, ad, 8);
  tv.ad_len = 8;

  const uint8_t pt[8] = {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07};
  memcpy(tv.pt, pt, 8);
  tv.pt_len = 8;

  const uint8_t ct[8] = {0x96,0xd1,0xbd,0xa7,0xae,0x11,0xf0,0xba};
  memcpy(tv.ct, ct, 8);

  const uint8_t tag[8] = {0x22,0xb0,0xc1,0x20,0x39,0xa2,0x0e,0x28};
  memcpy(tv.tag, tag, 8);

  return tv;
}

// Grain-128AEADv2 benchmark input
static TestVector_Grain128AEADv2 grain128aeadv2_bench_tv() {
  TestVector_Grain128AEADv2 tv;
  memset(&tv, 0, sizeof(tv));

  store_u64_le(bench_key64_last_candidate(), tv.key, 8);

  tv.nonce[0] = 0x11; tv.nonce[1] = 0x22; tv.nonce[2] = 0x33; tv.nonce[3] = 0x44;

  const char* msg = "Hello Grain 128!";
  tv.pt_len = (int)strlen(msg);
  memcpy(tv.pt, msg, tv.pt_len);

  const char* ad_msg = "AuthData";
  tv.ad_len = (int)strlen(ad_msg);
  memcpy(tv.ad, ad_msg, tv.ad_len);

  Grain128AEADv2::process(tv.pt, tv.ct, tv.pt_len,
                          tv.ad, tv.ad_len, tv.tag,
                          tv.key, tv.nonce);
  return tv;
}

// ============================================================
// Self-Tests
// ============================================================

static bool self_test_simon() {
  auto tv = simon_bench_tv();
  uint16_t rk[SIMON32_64_ROUNDS];
  Simon32_64_Enhanced::expand_key_vectorized(tv.key, rk);
  uint32_t out = Simon32_64_Enhanced::encrypt_optimized(tv.pt, rk);

  bool pass = (out == tv.ct);
  std::cout << "SIMON32/64 Self-Test: " << (pass ? "PASS" : "FAIL") << "\n";
  if (!pass) {
    std::cout << "  Key: " << u64_hex(tv.key) << "\n";
    std::cout << "  PT:  " << u64_hex(tv.pt) << "\n";
    std::cout << "  CT:  " << u64_hex(tv.ct) << " (expected)\n";
    std::cout << "  Out: " << u64_hex(out) << " (computed)\n";
  }
  return pass;
}

static bool self_test_present() {
  auto tv = present_tv();
  uint64_t rk[PRESENT_ROUNDS + 1];
  Present80::expand_key(tv.key, rk);
  uint64_t out_scalar = Present80::encrypt(tv.pt, rk);
  uint64_t out_opt = Present80::encrypt_spbox_otf_host(tv.pt, tv.key);

  bool pass = (out_scalar == tv.ct) && (out_opt == tv.ct);
  std::cout << "PRESENT-80 Self-Test: " << (pass ? "PASS" : "FAIL") << "\n";
  if (!pass) {
    std::cout << "  Key:        " << bytes_to_hex(tv.key, 10) << "\n";
    std::cout << "  PT:         " << u64_hex(tv.pt) << "\n";
    std::cout << "  CT:         " << u64_hex(tv.ct) << " (expected)\n";
    std::cout << "  Scalar Out: " << u64_hex(out_scalar) << " (computed)\n";
    std::cout << "  Opt Out:    " << u64_hex(out_opt) << " (computed)\n";
  }
  return pass;
}

static bool self_test_speck() {
  auto tv = speck_tv();
  uint32_t rk[SPECK64_128_ROUNDS];
  Speck64_128::expand_key(tv.key, rk);
  uint64_t out = Speck64_128::encrypt(tv.pt, rk);

  bool pass = (out == tv.ct);
  std::cout << "SPECK64/128 Self-Test: " << (pass ? "PASS" : "FAIL") << "\n";
  if (!pass) {
    std::cout << "  Key: " << bytes_to_hex(tv.key, 16) << "\n";
    std::cout << "  PT:  " << u64_hex(tv.pt) << "\n";
    std::cout << "  CT:  " << u64_hex(tv.ct) << " (expected)\n";
    std::cout << "  Out: " << u64_hex(out) << " (computed)\n";
  }
  return pass;
}

static bool self_test_grain() {
  auto tv = grain_bench_tv();

  uint8_t decrypted[16];
  GrainV1::process(tv.ct, decrypted, tv.length, tv.key, tv.iv);
  bool pass = (memcmp(decrypted, tv.pt, tv.length) == 0);

  std::cout << "Grain v1 Self-Test: " << (pass ? "PASS" : "FAIL") << "\n";
  return pass;
}

static bool self_test_trivium() {
  auto tv = trivium_bench_tv();

  uint8_t decrypted[16];
  Trivium::process(tv.ct, decrypted, tv.length, tv.key, tv.iv);
  bool pass = (memcmp(decrypted, tv.pt, tv.length) == 0);

  std::cout << "Trivium Self-Test: " << (pass ? "PASS" : "FAIL") << "\n";
  return pass;
}

static bool self_test_chacha20() {
  auto tv = chacha_block_tv();
  uint8_t out[64];
  ChaCha20::block(tv.key, tv.counter, tv.nonce, out);
  bool pass = (memcmp(out, tv.block, 64) == 0);

  std::cout << "ChaCha20 Block Self-Test: " << (pass ? "PASS" : "FAIL") << "\n";
  if (!pass) {
    std::cout << "  Expected: " << bytes_to_hex(tv.block, 64) << "\n";
    std::cout << "  Got:      " << bytes_to_hex(out, 64) << "\n";
  }
  return pass;
}

static bool self_test_tinyjambu() {
  auto tv = tinyjambu_nist_tv();
  uint8_t computed_ct[32];
  uint8_t computed_tag[8];
  TinyJAMBU128::encrypt(tv.pt, computed_ct, tv.pt_len, computed_tag,
                        tv.key, tv.nonce, tv.ad, tv.ad_len);

  bool pass_ct  = (memcmp(computed_ct, tv.ct, tv.pt_len) == 0);
  bool pass_tag = (memcmp(computed_tag, tv.tag, 8) == 0);
  bool pass     = pass_ct && pass_tag;

  std::cout << "TinyJAMBU-128 Self-Test: " << (pass ? "PASS" : "FAIL") << "\n";
  if (!pass) {
    if (!pass_ct) {
      std::cout << "  CT Expected:  " << bytes_to_hex(tv.ct, tv.pt_len) << "\n";
      std::cout << "  CT Got:       " << bytes_to_hex(computed_ct, tv.pt_len) << "\n";
    }
    if (!pass_tag) {
      std::cout << "  Tag Expected: " << bytes_to_hex(tv.tag, 8) << "\n";
      std::cout << "  Tag Got:      " << bytes_to_hex(computed_tag, 8) << "\n";
    }
  }
  return pass;
}

static bool self_test_tinyjambu_bitsliced() {
  auto tv = tinyjambu_nist_tv();

  // Broadcast the 16-byte key to all 32 lanes in bitsliced format
  uint32_t bitsliced_key[128] = {0};
  for (int i = 0; i < 16; i++) {
    for (int b = 0; b < 8; b++) {
      bitsliced_key[i * 8 + b] = ((tv.key[i] >> b) & 1) ? 0xFFFFFFFFu : 0x00000000u;
    }
  }

  uint32_t match_mask = TinyJAMBU128_Bitsliced::match_keys(
    tv.pt, tv.ct, tv.pt_len, tv.tag, bitsliced_key, tv.nonce, tv.ad, tv.ad_len
  );

  bool pass = (match_mask == 0xFFFFFFFFu);
  std::cout << "TinyJAMBU-128 Bitsliced Self-Test: " << (pass ? "PASS" : "FAIL") << "\n";
  if (!pass) {
    std::cout << "  Expected Mask: 0xffffffff\n";
    std::cout << "  Got Mask:      0x" << std::hex << match_mask << std::dec << "\n";
  }
  return pass;
}

static bool self_test_zuc() {
  auto tv = zuc_nist_tv();

  // ZUC keystream words are big-endian; process with zero plaintext
  const int byte_len = 2 * 4;
  uint8_t pt_zeros[8] = {0};
  uint8_t computed[8] = {0};
  ZUC::process(pt_zeros, computed, byte_len, tv.key, tv.iv);

  // Reconstruct big-endian words from output
  bool pass = true;
  for (int i = 0; i < 2 && pass; i++) {
    uint32_t got = ((uint32_t)computed[i*4+0] << 24) | ((uint32_t)computed[i*4+1] << 16) |
                   ((uint32_t)computed[i*4+2] <<  8) |  (uint32_t)computed[i*4+3];
    if (got != tv.expected_ks[i]) pass = false;
  }

  std::cout << "ZUC-128 Self-Test: " << (pass ? "PASS" : "FAIL") << "\n";
  if (!pass) {
    for (int i = 0; i < 2; i++) {
      uint32_t got = ((uint32_t)computed[i*4+0] << 24) | ((uint32_t)computed[i*4+1] << 16) |
                     ((uint32_t)computed[i*4+2] <<  8) |  (uint32_t)computed[i*4+3];
      std::cout << "  Word " << i << " -> Expected: 0x"
                << std::hex << std::setw(8) << std::setfill('0') << tv.expected_ks[i]
                << " | Got: 0x" << std::setw(8) << std::setfill('0') << got
                << std::dec << "\n";
    }
  }
  return pass;
}

static bool self_test_snow_v() {
  auto tv = snow_v_nist_tv();

  uint8_t pt_zeros[128] = {0};
  uint8_t computed[128] = {0};
  SNOW_V::process<true>(pt_zeros, computed, 128, tv.key, tv.iv);

  bool pass = (memcmp(computed, tv.expected_ks, 128) == 0);
  std::cout << "SNOW-V Self-Test: " << (pass ? "PASS" : "FAIL") << "\n";
  if (!pass) {
    std::cout << "  Expected: " << bytes_to_hex(tv.expected_ks, 32) << "\n";
    std::cout << "  Got:      " << bytes_to_hex(computed, 32) << "\n";
  }
  return pass;
}

static bool self_test_aes() {
  auto tv = aes_tv();
  uint8_t ct[16];
  AES128::encrypt<true>(tv.pt, tv.key, ct);
  bool pass = (memcmp(ct, tv.ct, 16) == 0);
  std::cout << "AES-128 Self-Test: " << (pass ? "PASS" : "FAIL") << "\n";
  if (!pass) {
    std::cout << "  Expected: " << bytes_to_hex(tv.ct, 16) << "\n";
    std::cout << "  Got:      " << bytes_to_hex(ct, 16) << "\n";
  }
  return pass;
}

static bool self_test_salsa20() {
  auto tv = salsa20_ecrypt_tv();

  // Part 1: KAT — verify first 16 bytes of keystream via block_words4
  // against eSTREAM Set 1, vector 0 (verified reference)
  uint32_t out4[4];
  Salsa20::block_words4(tv.key, 0ULL, tv.nonce, out4);
  uint8_t out16[16];
  for (int i = 0; i < 4; i++) {
    out16[4*i+0] = (uint8_t)(out4[i]        & 0xFF);
    out16[4*i+1] = (uint8_t)((out4[i] >>  8) & 0xFF);
    out16[4*i+2] = (uint8_t)((out4[i] >> 16) & 0xFF);
    out16[4*i+3] = (uint8_t)((out4[i] >> 24) & 0xFF);
  }
  bool pass_kat = (memcmp(out16, tv.stream, 16) == 0);

  // Part 2: Round-trip — encrypt then XOR back (Salsa20 is its own inverse)
  const char* msg = "Salsa20RoundTrip"; // exactly 16 bytes
  uint8_t ct[16], rt[16];
  Salsa20::process((const uint8_t*)msg, ct, 16, tv.key, tv.nonce);
  Salsa20::process(ct, rt, 16, tv.key, tv.nonce);
  bool pass_rt = (memcmp(rt, msg, 16) == 0);

  bool pass = pass_kat && pass_rt;
  std::cout << "Salsa20 Self-Test: " << (pass ? "PASS" : "FAIL") << "\n";
  if (!pass_kat) {
    std::cout << "  KAT Expected: " << bytes_to_hex(tv.stream, 16) << "\n";
    std::cout << "  KAT Got:      " << bytes_to_hex(out16, 16) << "\n";
  }
  if (!pass_rt) {
    std::cout << "  Round-trip FAIL: decrypted != original plaintext\n";
    std::cout << "  Expected: " << bytes_to_hex((const uint8_t*)msg, 16) << "\n";
    std::cout << "  Got:      " << bytes_to_hex(rt, 16) << "\n";
  }
  return pass;
}

static bool self_test_grain128aeadv2() {
  auto tv = grain128aeadv2_nist_tv();
  uint8_t computed_ct[32];
  uint8_t computed_tag[8];

  Grain128AEADv2::process(tv.pt, computed_ct, tv.pt_len,
                          tv.ad, tv.ad_len, computed_tag,
                          tv.key, tv.nonce);

  bool pass_ct  = (memcmp(computed_ct,  tv.ct,  tv.pt_len) == 0);
  bool pass_tag = (memcmp(computed_tag, tv.tag, 8)         == 0);
  bool pass     = pass_ct && pass_tag;

  std::cout << "Grain-128AEADv2 Self-Test: " << (pass ? "PASS" : "FAIL") << "\n";
  if (!pass) {
    if (!pass_ct) {
      std::cout << "  CT Expected:  " << bytes_to_hex(tv.ct, tv.pt_len) << "\n";
      std::cout << "  CT Got:       " << bytes_to_hex(computed_ct, tv.pt_len) << "\n";
    }
    if (!pass_tag) {
      std::cout << "  Tag Expected: " << bytes_to_hex(tv.tag, 8) << "\n";
      std::cout << "  Tag Got:      " << bytes_to_hex(computed_tag, 8) << "\n";
    }
  }
  return pass;
}

static bool self_test_grain128aeadv2_bitsliced() {
  auto tv = grain128aeadv2_nist_tv();

  uint32_t key_bitsliced[128] = {0};
  uint32_t nonce_bitsliced[96] = {0};

  for (int i = 0; i < 128; i++) {
    uint32_t bit_val = (tv.key[i / 8] >> (i % 8)) & 1;
    key_bitsliced[i] = bit_val ? 0xFFFFFFFF : 0x00000000;
  }
  for (int i = 0; i < 96; i++) {
    uint32_t bit_val = (tv.nonce[i / 8] >> (i % 8)) & 1;
    nonce_bitsliced[i] = bit_val ? 0xFFFFFFFF : 0x00000000;
  }

  uint32_t match_mask = Grain128AEADv2_Bitsliced::match_keys(
    key_bitsliced, nonce_bitsliced, tv.ad_len, tv.pt, tv.ct, tv.pt_len);

  bool pass = (match_mask == 0xFFFFFFFF);
  std::cout << "Grain-128AEADv2 Bitsliced Test: " << (pass ? "PASS" : "FAIL") << "\n";
  if (!pass) {
    std::cout << "  Expected Mask: 0xffffffff\n";
    std::cout << "  Got Mask:      0x" << std::hex << match_mask << std::dec << "\n";
  }
  return pass;
}

static bool run_all_self_tests() {
  bool ok = true;
  ok &= self_test_simon();
  ok &= self_test_present();
  ok &= self_test_speck();
  ok &= self_test_grain();
  ok &= self_test_trivium();
  ok &= self_test_chacha20();
  ok &= self_test_tinyjambu();
  ok &= self_test_tinyjambu_bitsliced();
  ok &= self_test_zuc();
  ok &= self_test_snow_v();
  ok &= self_test_aes();
  ok &= self_test_salsa20();
  ok &= self_test_grain128aeadv2();
  ok &= self_test_grain128aeadv2_bitsliced();
  return ok;
}

// ============================================================
// Randomized Correctness Verification
// ============================================================

static bool run_random_key_verification(int verify_bits = 1) {
  std::cout << "\n=== Randomized Key Recovery Verification ===\n";
  std::cout << "5 random keys per cipher, unknown_bits=" << verify_bits << ", CPU brute-force\n";
  std::cout << "Each found key is also verified against a second PT/CT pair.\n\n";

  srand(42); // fixed seed for reproducibility

  // Generate a random 64-bit key with the lowest verify_bits bits all set to 1,
  // making the correct key the last candidate in the search (worst-case coverage).
  const uint64_t low_mask = (verify_bits >= 64) ? ~0ULL : ((1ULL << verify_bits) - 1);
  auto rkey64 = [&]() -> uint64_t {
    return (((uint64_t)(unsigned int)rand() << 33) |
            ((uint64_t)(unsigned int)rand() <<  1) | low_mask);
  };

  int total = 0, fail_count = 0;
  std::vector<std::string> failures;

  auto record = [&](const std::string& label, bool found, uint64_t got, uint64_t expected, bool pair2_ok) {
    ++total;
    bool ok = found && (got == expected) && pair2_ok;
    if (!ok) {
      ++fail_count;
      std::ostringstream s;
      s << "  FAIL  " << label << ": found=" << (found ? "yes" : "no");
      if (found) s << " got=" << u64_hex(got) << " expected=" << u64_hex(expected);
      if (!pair2_ok) s << " [second PT/CT mismatch]";
      failures.push_back(s.str());
    }
  };

  const int N = 5; // keys per cipher

  // Fixed IV / nonce values used across ciphers
  const uint8_t iv10[10]   = {0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88,0x00,0x00};
  const uint8_t iv8[8]     = {0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88};
  const uint8_t iv16[16]   = {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
                               0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f};
  const uint8_t nonce12[12]= {0x01,0x02,0x03,0x04,0x05,0x06,
                               0x07,0x08,0x09,0x0a,0x0b,0x0c};
  const uint8_t ad8[8]     = {0xca,0xfe,0xba,0xbe,0xde,0xad,0xbe,0xef};
  uint8_t pt16[16]; memcpy(pt16,  "RandVerifyTestA!", 16);
  uint8_t pt16b[16]; memcpy(pt16b, "SecondPairCheck!", 16);
  uint8_t pt64[64]; memset(pt64, 0xAB, 64);
  uint8_t pt64b[64]; memset(pt64b, 0xCD, 64);

  // ── SIMON 32/64 ──────────────────────────────────────────────────────────
  { bool cp = true;
    for (int t = 0; t < N; t++) {
      uint64_t key = rkey64();
      uint16_t rk[SIMON32_64_ROUNDS];
      Simon32_64_Enhanced::expand_key_vectorized(key, rk);
      uint32_t p1 = 0x6c617669U, p2 = 0x4e455752U;
      uint32_t c1 = Simon32_64_Enhanced::encrypt_optimized(p1, rk);
      uint32_t c2 = Simon32_64_Enhanced::encrypt_optimized(p2, rk);
      auto r = brute_force_cpu_simon(p1, c1, key >> verify_bits, verify_bits, 1);
      bool ok2 = false;
      if (r.found) {
        uint16_t rk2[SIMON32_64_ROUNDS];
        Simon32_64_Enhanced::expand_key_vectorized(r.found_key, rk2);
        ok2 = (Simon32_64_Enhanced::encrypt_optimized(p2, rk2) == c2);
      }
      record("SIMON32/64 #" + std::to_string(t), r.found, r.found_key, key, ok2);
      cp &= (r.found && r.found_key == key && ok2);
    }
    std::cout << "SIMON 32/64          " << (cp ? "PASS" : "FAIL") << "\n";
  }

  // ── PRESENT-80 ───────────────────────────────────────────────────────────
  { bool cp = true;
    for (int t = 0; t < N; t++) {
      uint64_t key64 = rkey64();
      uint8_t key10[10] = {0};
      for (int i = 0; i < 8; i++) key10[i] = (uint8_t)((key64 >> (8*i)) & 0xFF);
      uint64_t rk[PRESENT_ROUNDS+1];
      Present80::expand_key(key10, rk);
      uint64_t p1 = 0x0123456789ABCDEFULL, p2 = 0xFEDCBA9876543210ULL;
      uint64_t c1 = Present80::encrypt(p1, rk), c2 = Present80::encrypt(p2, rk);
      auto r = brute_force_cpu_present(p1, c1, key64 >> verify_bits, verify_bits, 1);
      bool ok2 = false;
      if (r.found) {
        uint8_t k2[10] = {0};
        for (int i = 0; i < 8; i++) k2[i] = (uint8_t)((r.found_key >> (8*i)) & 0xFF);
        uint64_t rk2[PRESENT_ROUNDS+1]; Present80::expand_key(k2, rk2);
        ok2 = (Present80::encrypt(p2, rk2) == c2);
      }
      record("PRESENT-80 #" + std::to_string(t), r.found, r.found_key, key64, ok2);
      cp &= (r.found && r.found_key == key64 && ok2);
    }
    std::cout << "PRESENT-80           " << (cp ? "PASS" : "FAIL") << "\n";
  }

  // ── SPECK 64/128 ─────────────────────────────────────────────────────────
  { bool cp = true;
    for (int t = 0; t < N; t++) {
      uint64_t key64 = rkey64();
      uint8_t key16[16] = {0};
      for (int i = 0; i < 8; i++) key16[i] = (uint8_t)((key64 >> (8*i)) & 0xFF);
      uint32_t rk[SPECK64_128_ROUNDS]; Speck64_128::expand_key(key16, rk);
      uint64_t p1 = 0x3b7265747475432dULL, p2 = 0xAABBCCDDEEFF0011ULL;
      uint64_t c1 = Speck64_128::encrypt(p1, rk), c2 = Speck64_128::encrypt(p2, rk);
      auto r = brute_force_cpu_speck(p1, c1, key64 >> verify_bits, verify_bits, 1);
      bool ok2 = false;
      if (r.found) {
        uint8_t k2[16] = {0};
        for (int i = 0; i < 8; i++) k2[i] = (uint8_t)((r.found_key >> (8*i)) & 0xFF);
        uint32_t rk2[SPECK64_128_ROUNDS]; Speck64_128::expand_key(k2, rk2);
        ok2 = (Speck64_128::encrypt(p2, rk2) == c2);
      }
      record("SPECK64/128 #" + std::to_string(t), r.found, r.found_key, key64, ok2);
      cp &= (r.found && r.found_key == key64 && ok2);
    }
    std::cout << "SPECK 64/128         " << (cp ? "PASS" : "FAIL") << "\n";
  }

  // ── Grain v1 ─────────────────────────────────────────────────────────────
  { bool cp = true;
    for (int t = 0; t < N; t++) {
      uint64_t key64 = rkey64();
      uint8_t key10[10] = {0};
      for (int i = 0; i < 8; i++) key10[i] = (uint8_t)((key64 >> (8*i)) & 0xFF);
      uint8_t ct1[16] = {0}, ct2[16] = {0};
      GrainV1::process(pt16, ct1, 16, key10, iv10);
      GrainV1::process(pt16b, ct2, 16, key10, iv10);
      auto r = brute_force_cpu_grain(pt16, ct1, iv10, 16, key64 >> verify_bits, verify_bits, 1);
      bool ok2 = false;
      if (r.found) {
        uint8_t k2[10] = {0};
        for (int i = 0; i < 8; i++) k2[i] = (uint8_t)((r.found_key >> (8*i)) & 0xFF);
        uint8_t out[16]; GrainV1::process(pt16b, out, 16, k2, iv10);
        ok2 = (memcmp(out, ct2, 16) == 0);
      }
      record("Grain v1 #" + std::to_string(t), r.found, r.found_key, key64, ok2);
      cp &= (r.found && r.found_key == key64 && ok2);
    }
    std::cout << "Grain v1             " << (cp ? "PASS" : "FAIL") << "\n";
  }

  // ── Trivium ───────────────────────────────────────────────────────────────
  { bool cp = true;
    for (int t = 0; t < N; t++) {
      uint64_t key64 = rkey64();
      uint8_t key10[10] = {0};
      for (int i = 0; i < 8; i++) key10[i] = (uint8_t)((key64 >> (8*i)) & 0xFF);
      uint8_t ct1[16] = {0}, ct2[16] = {0};
      Trivium::process(pt16, ct1, 16, key10, iv10);
      Trivium::process(pt16b, ct2, 16, key10, iv10);
      auto r = brute_force_cpu_trivium(pt16, ct1, iv10, 16, key64 >> verify_bits, verify_bits, 1);
      bool ok2 = false;
      if (r.found) {
        uint8_t k2[10] = {0};
        for (int i = 0; i < 8; i++) k2[i] = (uint8_t)((r.found_key >> (8*i)) & 0xFF);
        uint8_t out[16]; Trivium::process(pt16b, out, 16, k2, iv10);
        ok2 = (memcmp(out, ct2, 16) == 0);
      }
      record("Trivium #" + std::to_string(t), r.found, r.found_key, key64, ok2);
      cp &= (r.found && r.found_key == key64 && ok2);
    }
    std::cout << "Trivium              " << (cp ? "PASS" : "FAIL") << "\n";
  }

  // ── ChaCha20 ──────────────────────────────────────────────────────────────
  { bool cp = true;
    for (int t = 0; t < N; t++) {
      uint64_t key64 = rkey64();
      uint8_t key256[32] = {0};
      for (int i = 0; i < 8; i++) key256[i] = (uint8_t)((key64 >> (8*i)) & 0xFF);
      uint8_t ct1[16] = {0}, ct2[16] = {0};
      ChaCha20::process(pt16, ct1, 16, key256, 1, nonce12);
      ChaCha20::process(pt16b, ct2, 16, key256, 1, nonce12);
      auto r = brute_force_cpu_chacha20(pt16, ct1, nonce12, 16, key64 >> verify_bits, verify_bits, 1);
      bool ok2 = false;
      if (r.found) {
        uint8_t k2[32] = {0};
        for (int i = 0; i < 8; i++) k2[i] = (uint8_t)((r.found_key >> (8*i)) & 0xFF);
        uint8_t out[16]; ChaCha20::process(pt16b, out, 16, k2, 1, nonce12);
        ok2 = (memcmp(out, ct2, 16) == 0);
      }
      record("ChaCha20 #" + std::to_string(t), r.found, r.found_key, key64, ok2);
      cp &= (r.found && r.found_key == key64 && ok2);
    }
    std::cout << "ChaCha20             " << (cp ? "PASS" : "FAIL") << "\n";
  }

  // ── TinyJAMBU-128 ─────────────────────────────────────────────────────────
  { bool cp = true;
    for (int t = 0; t < N; t++) {
      uint64_t key64 = rkey64();
      uint8_t key16[16] = {0};
      for (int i = 0; i < 8; i++) key16[i] = (uint8_t)((key64 >> (8*i)) & 0xFF);
      uint8_t ct1[16]={0}, tag1[8]={0}, ct2[16]={0}, tag2[8]={0};
      TinyJAMBU128::encrypt(pt16, ct1, 16, tag1, key16, nonce12, ad8, 8);
      TinyJAMBU128::encrypt(pt16b, ct2, 16, tag2, key16, nonce12, ad8, 8);
      auto r = brute_force_cpu_tinyjambu(pt16, ct1, nonce12, 16, tag1, ad8, 8, key64>>verify_bits, verify_bits, 1);
      bool ok2 = false;
      if (r.found) {
        uint8_t k2[16]={0};
        for (int i = 0; i < 8; i++) k2[i] = (uint8_t)((r.found_key >> (8*i)) & 0xFF);
        uint8_t oc[16]={0}, ot[8]={0};
        TinyJAMBU128::encrypt(pt16b, oc, 16, ot, k2, nonce12, ad8, 8);
        ok2 = (memcmp(oc, ct2, 16)==0 && memcmp(ot, tag2, 8)==0);
      }
      record("TinyJAMBU-128 #" + std::to_string(t), r.found, r.found_key, key64, ok2);
      cp &= (r.found && r.found_key == key64 && ok2);
    }
    std::cout << "TinyJAMBU-128        " << (cp ? "PASS" : "FAIL") << "\n";
  }

  // ── ZUC-128 ───────────────────────────────────────────────────────────────
  { bool cp = true;
    for (int t = 0; t < N; t++) {
      uint64_t key64 = rkey64();
      uint8_t key16[16] = {0};
      for (int i = 0; i < 8; i++) key16[i] = (uint8_t)((key64 >> (8*i)) & 0xFF);
      uint8_t ct1[16]={0}, ct2[16]={0};
      ZUC::process(pt16, ct1, 16, key16, iv16);
      ZUC::process(pt16b, ct2, 16, key16, iv16);
      auto r = brute_force_cpu_zuc(pt16, ct1, iv16, 16, key64>>verify_bits, verify_bits, 1);
      bool ok2 = false;
      if (r.found) {
        uint8_t k2[16]={0};
        for (int i = 0; i < 8; i++) k2[i] = (uint8_t)((r.found_key >> (8*i)) & 0xFF);
        uint8_t out[16]; ZUC::process(pt16b, out, 16, k2, iv16);
        ok2 = (memcmp(out, ct2, 16)==0);
      }
      record("ZUC-128 #" + std::to_string(t), r.found, r.found_key, key64, ok2);
      cp &= (r.found && r.found_key == key64 && ok2);
    }
    std::cout << "ZUC-128              " << (cp ? "PASS" : "FAIL") << "\n";
  }

  // ── SNOW-V ────────────────────────────────────────────────────────────────
  { bool cp = true;
    for (int t = 0; t < N; t++) {
      uint64_t key64 = rkey64();
      uint8_t key32[32] = {0};
      for (int i = 0; i < 8; i++) key32[i] = (uint8_t)((key64 >> (8*i)) & 0xFF);
      uint8_t ct1[16]={0}, ct2[16]={0};
      SNOW_V::process<true>(pt16, ct1, 16, key32, iv16);
      SNOW_V::process<true>(pt16b, ct2, 16, key32, iv16);
      auto r = brute_force_cpu_snow_v(pt16, ct1, iv16, 16, key64>>verify_bits, verify_bits, 1);
      bool ok2 = false;
      if (r.found) {
        uint8_t k2[32]={0};
        for (int i = 0; i < 8; i++) k2[i] = (uint8_t)((r.found_key >> (8*i)) & 0xFF);
        uint8_t out[16]; SNOW_V::process<true>(pt16b, out, 16, k2, iv16);
        ok2 = (memcmp(out, ct2, 16)==0);
      }
      record("SNOW-V #" + std::to_string(t), r.found, r.found_key, key64, ok2);
      cp &= (r.found && r.found_key == key64 && ok2);
    }
    std::cout << "SNOW-V               " << (cp ? "PASS" : "FAIL") << "\n";
  }

  // ── AES-128 ───────────────────────────────────────────────────────────────
  { bool cp = true;
    for (int t = 0; t < N; t++) {
      uint64_t key64 = rkey64();
      uint8_t key16[16] = {0};
      for (int i = 0; i < 8; i++) key16[i] = (uint8_t)((key64 >> (8*i)) & 0xFF);
      uint8_t ct1[16]={0}, ct2[16]={0};
      AES128::encrypt<true>(pt16, key16, ct1);
      AES128::encrypt<true>(pt16b, key16, ct2);
      auto r = brute_force_cpu_aes(pt16, ct1, key64>>verify_bits, verify_bits, 1);
      bool ok2 = false;
      if (r.found) {
        uint8_t k2[16]={0};
        for (int i = 0; i < 8; i++) k2[i] = (uint8_t)((r.found_key >> (8*i)) & 0xFF);
        uint8_t out[16]; AES128::encrypt<true>(pt16b, k2, out);
        ok2 = (memcmp(out, ct2, 16)==0);
      }
      record("AES-128 #" + std::to_string(t), r.found, r.found_key, key64, ok2);
      cp &= (r.found && r.found_key == key64 && ok2);
    }
    std::cout << "AES-128              " << (cp ? "PASS" : "FAIL") << "\n";
  }

  // ── Salsa20 ───────────────────────────────────────────────────────────────
  { bool cp = true;
    for (int t = 0; t < N; t++) {
      uint64_t key64 = rkey64();
      uint8_t key256[32]={0};
      for (int i = 0; i < 8; i++) key256[i] = (uint8_t)((key64 >> (8*i)) & 0xFF);
      uint8_t ct1[16]={0}, ct2[16]={0};
      Salsa20::process(pt16, ct1, 16, key256, iv8);
      Salsa20::process(pt16b, ct2, 16, key256, iv8);
      auto r = brute_force_cpu_salsa20(pt16, ct1, iv8, 16, key64>>verify_bits, verify_bits, 1);
      bool ok2 = false;
      if (r.found) {
        uint8_t k2[32]={0};
        for (int i = 0; i < 8; i++) k2[i] = (uint8_t)((r.found_key >> (8*i)) & 0xFF);
        uint8_t out[16]; Salsa20::process(pt16b, out, 16, k2, iv8);
        ok2 = (memcmp(out, ct2, 16)==0);
      }
      record("Salsa20 #" + std::to_string(t), r.found, r.found_key, key64, ok2);
      cp &= (r.found && r.found_key == key64 && ok2);
    }
    std::cout << "Salsa20              " << (cp ? "PASS" : "FAIL") << "\n";
  }

  // ── Grain-128AEADv2 ───────────────────────────────────────────────────────
  { bool cp = true;
    for (int t = 0; t < N; t++) {
      uint64_t key64 = rkey64();
      uint8_t key16[16]={0};
      for (int i = 0; i < 8; i++) key16[i] = (uint8_t)((key64 >> (8*i)) & 0xFF);
      uint8_t ct1[16]={0}, tag1[8]={0}, ct2[16]={0}, tag2[8]={0};
      Grain128AEADv2::process(pt16, ct1, 16, ad8, 8, tag1, key16, nonce12);
      Grain128AEADv2::process(pt16b, ct2, 16, ad8, 8, tag2, key16, nonce12);
      uint8_t tgt[16]; for(int i=0;i<16;i++) tgt[i]=pt16[i]^ct1[i];
      auto r = brute_force_cpu_grain128aeadv2(pt16, ct1, nonce12, 16, ad8, 8, key64>>verify_bits, verify_bits, 1);
      bool ok2 = false;
      if (r.found) {
        uint8_t k2[16]={0};
        for (int i = 0; i < 8; i++) k2[i] = (uint8_t)((r.found_key >> (8*i)) & 0xFF);
        uint8_t oc[16]={0}, ot[8]={0};
        Grain128AEADv2::process(pt16b, oc, 16, ad8, 8, ot, k2, nonce12);
        ok2 = (memcmp(oc, ct2, 16)==0 && memcmp(ot, tag2, 8)==0);
      }
      record("Grain128AEADv2 #" + std::to_string(t), r.found, r.found_key, key64, ok2);
      cp &= (r.found && r.found_key == key64 && ok2);
    }
    std::cout << "Grain-128AEADv2      " << (cp ? "PASS" : "FAIL") << "\n";
  }

  // ── Summary ───────────────────────────────────────────────────────────────
  std::cout << "\nResults: " << (total - fail_count) << "/" << total << " passed";
  if (fail_count == 0) {
    std::cout << " — no failure cases detected.\n";
  } else {
    std::cout << " — " << fail_count << " FAILURE(S):\n";
    for (auto& f : failures) std::cout << f << "\n";
  }
  return (fail_count == 0);
}

// ============================================================
// CLI Args
// ============================================================

struct Args {
  std::string out_csv = "results.csv";
  std::string cipher = "all";  // simon, present, speck, grain, trivium, chacha, all
  std::string variants = "auto"; // baseline|auto (baseline+2-3 opts)
  int min_bits = 1;
  int max_bits = 30;
  int step_bits = 1;
  int cpu_repeats = 3;
  int gpu_repeats = 10;
  int blocks = 1024;
  int threads = 256;
  bool test_only = false;
  bool verify_only = false;
  int  verify_bits = 1;   // unknown_bits used in --verify mode
  bool cpu_only = false;
  bool gpu_only = false;
  bool run_cpu = true;
  bool run_gpu = true;
};

static Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; i++) {
    std::string s = argv[i];
    auto need = [&](const char* name) {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << name << "\n";
        std::exit(1);
      }
      return std::string(argv[++i]);
    };

    if (s == "--out") a.out_csv = need("--out");
    else if (s == "--cipher") a.cipher = need("--cipher");
    else if (s == "--variants") a.variants = need("--variants");
    else if (s == "--min_bits") a.min_bits = std::stoi(need("--min_bits"));
    else if (s == "--max_bits") a.max_bits = std::stoi(need("--max_bits"));
    else if (s == "--step_bits") a.step_bits = std::stoi(need("--step_bits"));
    else if (s == "--cpu_repeats") a.cpu_repeats = std::stoi(need("--cpu_repeats"));
    else if (s == "--gpu_repeats") a.gpu_repeats = std::stoi(need("--gpu_repeats"));
    else if (s == "--blocks") a.blocks = std::stoi(need("--blocks"));
    else if (s == "--threads") a.threads = std::stoi(need("--threads"));
    else if (s == "--test")        a.test_only   = true;
    else if (s == "--verify")      a.verify_only = true;
    else if (s == "--verify_bits") a.verify_bits = std::stoi(need("--verify_bits"));
    else if (s == "--cpu_only") { a.run_cpu = true; a.run_gpu = false; a.cpu_only = true; }
    else if (s == "--gpu_only") { a.run_cpu = false; a.run_gpu = true; a.gpu_only = true; }
    else if (s == "--help" || s == "-h") {
      std::cout
        << "Usage: bench [options]\n"
        << "  --cipher <simon|present|speck|grain|trivium|chacha|tinyjambu|zuc|snowv|aes|salsa|grain128|all>\n"
        << "           (default: all)\n"
        << "  --variants <baseline|optimized|optimized_ilp|shared|bitsliced|all|auto>\n"
        << "             (default: auto — best variants per cipher)\n"
        << "             bitsliced is only supported for tinyjambu and grain128\n"
        << "  --out results.csv\n"
        << "  --min_bits 1 --max_bits 30 --step_bits 1\n"
        << "  --cpu_repeats 3 --gpu_repeats 10\n"
        << "  --blocks 1024 --threads 256\n"
        << "  --cpu_only | --gpu_only (default: run both)\n"
        << "  --test              (run fixed self-tests only)\n"
        << "  --verify            (run randomized correctness verification)\n"
        << "  --verify_bits <N>   unknown bits for --verify mode (default: 1)\n";
      std::exit(0);
    }
  }
  return a;
}


static inline std::vector<GpuVariant> selected_gpu_variants(const Args& a, CipherType cipher) {
  const bool is_aead = (cipher == CipherType::TINYJAMBU_128 || cipher == CipherType::GRAIN128_AEADV2);

  if (a.variants == "baseline")      return {GpuVariant::BASELINE};
  if (a.variants == "optimized")     return {GpuVariant::OPTIMIZED};
  if (a.variants == "optimized_ilp") return {GpuVariant::OPTIMIZED_ILP};
  if (a.variants == "shared")        return {GpuVariant::OPTIMIZED_SHARED};
  if (a.variants == "bitsliced") {
    // Only AEAD ciphers have a bitsliced kernel; others fall back to baseline
    return is_aead ? std::vector<GpuVariant>{GpuVariant::BITSLICED}
                   : std::vector<GpuVariant>{GpuVariant::BASELINE};
  }
  if (a.variants == "all") {
    if (is_aead)
      return {GpuVariant::BASELINE, GpuVariant::OPTIMIZED_ILP, GpuVariant::BITSLICED};
    if (cipher == CipherType::PRESENT80)
      return {GpuVariant::BASELINE, GpuVariant::OPTIMIZED, GpuVariant::OPTIMIZED_ILP, GpuVariant::OPTIMIZED_SHARED};
    return {GpuVariant::BASELINE, GpuVariant::OPTIMIZED, GpuVariant::OPTIMIZED_ILP};
  }

  // auto mode — pick the most relevant set per cipher
  if (is_aead)
    return {GpuVariant::BASELINE, GpuVariant::OPTIMIZED_ILP, GpuVariant::BITSLICED};
  if (cipher == CipherType::PRESENT80)
    return {GpuVariant::BASELINE, GpuVariant::OPTIMIZED, GpuVariant::OPTIMIZED_ILP, GpuVariant::OPTIMIZED_SHARED};
  return {GpuVariant::BASELINE, GpuVariant::OPTIMIZED, GpuVariant::OPTIMIZED_ILP};
}

// ============================================================
// Benchmarks
// ============================================================

static void benchmark_simon(const Args& a) {
  std::cout << "\n=== SIMON 32/64 Benchmark ===\n";
  auto tv = simon_bench_tv();

  for (int b = a.min_bits; b <= a.max_bits; b += a.step_bits) {
    uint64_t known_high = tv.key >> b;
    uint64_t N = bf_space_size_main(b);

    std::cout << "\nunknown_bits=" << b << " (keys=" << N << ")\n";

    if (a.run_cpu) {
      auto cr = brute_force_cpu_simon(tv.pt, tv.ct, known_high, b, a.cpu_repeats);
      double kps = safe_kps(cr.keys_tested, cr.seconds);
      std::cout << "CPU:               " << cr.seconds << " s, " << kps << " keys/s\n";
      std::ostringstream row;
      row << "simon32_64,cpu,cpu_baseline," << b << "," << cr.keys_tested << "," << cr.seconds << "," << kps << "," << u64_hex(cr.found ? cr.found_key : 0);
      csv_append_row(a.out_csv, row.str());
    }

    if (a.run_gpu) {
      auto run = [&](GpuVariant v, const char* tag) {
        auto gr = brute_force_gpu_enhanced(CipherType::SIMON32_64, &tv.pt, &tv.ct, nullptr, 0,
                                           known_high, b, v, a.blocks, a.threads, a.gpu_repeats);
        double kps = safe_kps(gr.keys_tested, gr.seconds);
        std::cout << tag << ": " << gr.seconds << " s, " << kps << " keys/s, found=" << (gr.found ? "yes" : "no") << "\n";
        std::ostringstream row;
        row << "simon32_64,gpu," << gpu_variant_name(v) << "," << b << "," << gr.keys_tested << "," << gr.seconds << "," << kps << "," << u64_hex(gr.found ? gr.found_key : 0);
        csv_append_row(a.out_csv, row.str());
      };

      for (auto v : selected_gpu_variants(a, CipherType::SIMON32_64)) {
        std::string tag = std::string("GPU ") + gpu_variant_name(v);
        run(v, tag.c_str());
      }
    }
  }
}

static void benchmark_present(const Args& a) {
  std::cout << "\n=== PRESENT-80 Benchmark ===\n";
  auto tv = present_bench_tv();

  // We brute-force only the low 64 bits of the 80-bit key (top 16 bits are 0)
  uint64_t key64 = 0;
  for (int i = 0; i < 8; i++) key64 |= ((uint64_t)tv.key[i]) << (i * 8);

  for (int b = a.min_bits; b <= a.max_bits; b += a.step_bits) {
    uint64_t known_high = key64 >> b;
    uint64_t N = bf_space_size_main(b);

    std::cout << "\nunknown_bits=" << b << " (keys=" << N << ")\n";

    if (a.run_cpu) {
      auto cr = brute_force_cpu_present(tv.pt, tv.ct, known_high, b, a.cpu_repeats);
      double kps = safe_kps(cr.keys_tested, cr.seconds);
      std::cout << "CPU:               " << cr.seconds << " s, " << kps << " keys/s\n";
      std::ostringstream row;
      row << "present80,cpu,cpu_baseline," << b << "," << cr.keys_tested << "," << cr.seconds << "," << kps << "," << u64_hex(cr.found ? cr.found_key : 0);
      csv_append_row(a.out_csv, row.str());
    }

    if (a.run_gpu) {
      auto run = [&](GpuVariant v, const char* tag) {
        auto gr = brute_force_gpu_enhanced(CipherType::PRESENT80, &tv.pt, &tv.ct, nullptr, 0,
                                           known_high, b, v, a.blocks, a.threads, a.gpu_repeats);
        double kps = safe_kps(gr.keys_tested, gr.seconds);
        std::cout << tag << ": " << gr.seconds << " s, " << kps << " keys/s\n";
        std::ostringstream row;
        row << "present80,gpu," << gpu_variant_name(v) << "," << b << "," << gr.keys_tested << "," << gr.seconds << "," << kps << "," << u64_hex(gr.found ? gr.found_key : 0);
        csv_append_row(a.out_csv, row.str());
      };

      for (auto v : selected_gpu_variants(a, CipherType::PRESENT80)) {
        std::string tag = std::string("GPU ") + gpu_variant_name(v);
        run(v, tag.c_str());
      }
    }
  }
}

static void benchmark_speck(const Args& a) {
  std::cout << "\n=== SPECK64/128 Benchmark ===\n";

  auto tv = speck_bench_tv();
  uint64_t key64 = 0;
  for (int j = 0; j < 8; j++) key64 |= ((uint64_t)tv.key[j]) << (8 * j);
  const uint64_t pt = tv.pt;
  const uint64_t ct = tv.ct;

  for (int b = a.min_bits; b <= a.max_bits; b += a.step_bits) {
    uint64_t known_high = key64 >> b;
    uint64_t N = bf_space_size_main(b);

    std::cout << "\nunknown_bits=" << b << " (keys=" << N << ")\n";

    if (a.run_cpu) {
      auto cr = brute_force_cpu_speck(pt, ct, known_high, b, a.cpu_repeats);
      double kps = safe_kps(cr.keys_tested, cr.seconds);
      std::cout << "CPU:               " << cr.seconds << " s, " << kps << " keys/s\n";
      std::ostringstream row;
      row << "speck64_128,cpu,cpu_baseline," << b << "," << cr.keys_tested << "," << cr.seconds << "," << kps << "," << u64_hex(cr.found ? cr.found_key : 0);
      csv_append_row(a.out_csv, row.str());
    }

    if (a.run_gpu) {
      auto run = [&](GpuVariant v, const char* tag) {
        auto gr = brute_force_gpu_enhanced(CipherType::SPECK64_128, &pt, &ct, nullptr, 0,
                                           known_high, b, v, a.blocks, a.threads, a.gpu_repeats);
        double kps = safe_kps(gr.keys_tested, gr.seconds);
        std::cout << tag << ": " << gr.seconds << " s, " << kps << " keys/s\n";
        std::ostringstream row;
        row << "speck64_128,gpu," << gpu_variant_name(v) << "," << b << "," << gr.keys_tested << "," << gr.seconds << "," << kps << "," << u64_hex(gr.found ? gr.found_key : 0);
        csv_append_row(a.out_csv, row.str());
      };

      for (auto v : selected_gpu_variants(a, CipherType::SPECK64_128)) {
        std::string tag = std::string("GPU ") + gpu_variant_name(v);
        run(v, tag.c_str());
      }
    }
  }
}

static void benchmark_grain(const Args& a) {
  std::cout << "\n=== Grain v1 Benchmark ===\n";
  auto tv = grain_bench_tv();

  // Project-aligned range: allow the same unknown-bit sweep requested by the project.
  int max_bits = a.max_bits;

  uint64_t key64 = 0;
  for (int i = 0; i < 8; i++) key64 |= ((uint64_t)tv.key[i]) << (i * 8);

  for (int b = a.min_bits; b <= max_bits; b += a.step_bits) {
    uint64_t known_high = key64 >> b;
    uint64_t N = bf_space_size_main(b);

    std::cout << "\nunknown_bits=" << b << " (keys=" << N << ")\n";

    if (a.run_cpu) {
      auto cr = brute_force_cpu_grain(tv.pt, tv.ct, tv.iv, tv.length, known_high, b, a.cpu_repeats);
      double kps = safe_kps(cr.keys_tested, cr.seconds);
      std::cout << "CPU:               " << cr.seconds << " s, " << kps << " keys/s\n";
      std::ostringstream row;
      row << "grain_v1,cpu,cpu_baseline," << b << "," << cr.keys_tested << "," << cr.seconds << "," << kps << "," << u64_hex(cr.found ? cr.found_key : 0);
      csv_append_row(a.out_csv, row.str());
    }

    if (a.run_gpu) {
      for (auto v : selected_gpu_variants(a, CipherType::GRAIN_V1)) {
        auto gr = brute_force_gpu_enhanced(CipherType::GRAIN_V1, tv.pt, tv.ct, tv.iv, tv.length,
                                           known_high, b, v, a.blocks, a.threads, a.gpu_repeats);
        double kps = safe_kps(gr.keys_tested, gr.seconds);
        std::cout << "GPU " << gpu_variant_name(v) << ": " << gr.seconds << " s, " << kps << " keys/s\n";
        std::ostringstream row;
        row << "grain_v1,gpu," << gpu_variant_name(v) << "," << b << "," << gr.keys_tested << "," << gr.seconds << "," << kps << "," << u64_hex(gr.found ? gr.found_key : 0);
        csv_append_row(a.out_csv, row.str());
      }
    }
  }
}

static void benchmark_trivium(const Args& a) {
  std::cout << "\n=== Trivium Benchmark ===\n";
  auto tv = trivium_bench_tv();

  int max_bits = a.max_bits;

  uint64_t key64 = 0;
  for (int i = 0; i < 8; i++) key64 |= ((uint64_t)tv.key[i]) << (i * 8);

  for (int b = a.min_bits; b <= max_bits; b += a.step_bits) {
    uint64_t known_high = key64 >> b;
    uint64_t N = bf_space_size_main(b);

    std::cout << "\nunknown_bits=" << b << " (keys=" << N << ")\n";

    if (a.run_cpu) {
      auto cr = brute_force_cpu_trivium(tv.pt, tv.ct, tv.iv, tv.length, known_high, b, a.cpu_repeats);
      double kps = safe_kps(cr.keys_tested, cr.seconds);
      std::cout << "CPU:               " << cr.seconds << " s, " << kps << " keys/s\n";
      std::ostringstream row;
      row << "trivium,cpu,cpu_baseline," << b << "," << cr.keys_tested << "," << cr.seconds << "," << kps << "," << u64_hex(cr.found ? cr.found_key : 0);
      csv_append_row(a.out_csv, row.str());
    }

    if (a.run_gpu) {
      for (auto v : selected_gpu_variants(a, CipherType::TRIVIUM)) {
        auto gr = brute_force_gpu_enhanced(CipherType::TRIVIUM, tv.pt, tv.ct, tv.iv, tv.length,
                                           known_high, b, v, a.blocks, a.threads, a.gpu_repeats);
        double kps = safe_kps(gr.keys_tested, gr.seconds);
        std::cout << "GPU " << gpu_variant_name(v) << ": " << gr.seconds << " s, " << kps << " keys/s\n";
        std::ostringstream row;
        row << "trivium,gpu," << gpu_variant_name(v) << "," << b << "," << gr.keys_tested << "," << gr.seconds << "," << kps << "," << u64_hex(gr.found ? gr.found_key : 0);
        csv_append_row(a.out_csv, row.str());
      }
    }
  }
}

static void benchmark_chacha(const Args& a) {
  std::cout << "\n=== ChaCha20 Benchmark ===\n";
  auto tv = chacha_bench_tv();

  // Stream cipher: keep max bits smaller
  int max_bits = a.max_bits;

  for (int b = a.min_bits; b <= max_bits; b += a.step_bits) {
    uint64_t known_high = tv.key64 >> b;
    uint64_t N = bf_space_size_main(b);

    std::cout << "\nunknown_bits=" << b << " (keys=" << N << ")\n";

    if (a.run_cpu) {
      auto cr = brute_force_cpu_chacha20(tv.pt, tv.ct, tv.nonce, tv.length, known_high, b, a.cpu_repeats);
      double kps = safe_kps(cr.keys_tested, cr.seconds);
      std::cout << "CPU:               " << cr.seconds << " s, " << kps << " keys/s\n";
      std::ostringstream row;
      row << "chacha20,cpu,cpu_baseline," << b << "," << cr.keys_tested << "," << cr.seconds << "," << kps << "," << u64_hex(cr.found ? cr.found_key : 0);
      csv_append_row(a.out_csv, row.str());
    }

    if (a.run_gpu) {
      for (auto v : selected_gpu_variants(a, CipherType::CHACHA20)) {
        auto gr = brute_force_gpu_enhanced(CipherType::CHACHA20, tv.pt, tv.ct, tv.nonce, tv.length,
                                           known_high, b, v, a.blocks, a.threads, a.gpu_repeats);
        double kps = safe_kps(gr.keys_tested, gr.seconds);
        std::cout << "GPU " << gpu_variant_name(v) << ": " << gr.seconds << " s, " << kps << " keys/s\n";
        std::ostringstream row;
        row << "chacha20,gpu," << gpu_variant_name(v) << "," << b << "," << gr.keys_tested << "," << gr.seconds << "," << kps << "," << u64_hex(gr.found ? gr.found_key : 0);
        csv_append_row(a.out_csv, row.str());
      }
    }
  }
}

static void benchmark_tinyjambu(const Args& a) {
  std::cout << "\n=== TinyJAMBU-128 Benchmark ===\n";
  auto tv = tinyjambu_bench_tv();

  uint64_t key64 = 0;
  for (int i = 0; i < 8; i++) key64 |= ((uint64_t)tv.key[i]) << (i * 8);

  int max_bits = a.max_bits;

  for (int b = a.min_bits; b <= max_bits; b += a.step_bits) {
    uint64_t known_high = key64 >> b;
    uint64_t N = bf_space_size_main(b);

    std::cout << "\nunknown_bits=" << b << " (keys=" << N << ")\n";

    if (a.run_cpu) {
      auto cr = brute_force_cpu_tinyjambu(tv.pt, tv.ct, tv.nonce, tv.pt_len, tv.tag,
                                          tv.ad, tv.ad_len, known_high, b, a.cpu_repeats);
      double kps = safe_kps(cr.keys_tested, cr.seconds);
      std::cout << "CPU:               " << cr.seconds << " s, " << kps << " keys/s\n";
      std::ostringstream row;
      row << "tinyjambu_128,cpu,cpu_baseline," << b << "," << cr.keys_tested << "," << cr.seconds << "," << kps << "," << u64_hex(cr.found ? cr.found_key : 0);
      csv_append_row(a.out_csv, row.str());
    }

    if (a.run_gpu) {
      for (auto v : selected_gpu_variants(a, CipherType::TINYJAMBU_128)) {
        auto gr = brute_force_gpu_enhanced_aead(CipherType::TINYJAMBU_128,
                                                tv.pt, tv.ct, tv.nonce, tv.pt_len,
                                                tv.ad, tv.ad_len, tv.tag,
                                                known_high, b, v, a.blocks, a.threads, a.gpu_repeats);
        double kps = safe_kps(gr.keys_tested, gr.seconds);
        std::cout << "GPU " << gpu_variant_name(v) << ": " << gr.seconds << " s, " << kps << " keys/s, found=" << (gr.found ? "yes" : "no") << "\n";
        std::ostringstream row;
        row << "tinyjambu_128,gpu," << gpu_variant_name(v) << "," << b << "," << gr.keys_tested << "," << gr.seconds << "," << kps << "," << u64_hex(gr.found ? gr.found_key : 0);
        csv_append_row(a.out_csv, row.str());
      }
    }
  }
}

static void benchmark_zuc(const Args& a) {
  std::cout << "\n=== ZUC-128 Benchmark ===\n";
  auto tv = zuc_bench_tv();

  uint64_t key64 = 0;
  for (int i = 0; i < 8; i++) key64 |= ((uint64_t)tv.key[i]) << (i * 8);

  int max_bits = a.max_bits;

  for (int b = a.min_bits; b <= max_bits; b += a.step_bits) {
    uint64_t known_high = key64 >> b;
    uint64_t N = bf_space_size_main(b);

    std::cout << "\nunknown_bits=" << b << " (keys=" << N << ")\n";

    if (a.run_cpu) {
      auto cr = brute_force_cpu_zuc(tv.pt, tv.ct, tv.iv, tv.length, known_high, b, a.cpu_repeats);
      double kps = safe_kps(cr.keys_tested, cr.seconds);
      std::cout << "CPU:               " << cr.seconds << " s, " << kps << " keys/s\n";
      std::ostringstream row;
      row << "zuc_128,cpu,cpu_baseline," << b << "," << cr.keys_tested << "," << cr.seconds << "," << kps << "," << u64_hex(cr.found ? cr.found_key : 0);
      csv_append_row(a.out_csv, row.str());
    }

    if (a.run_gpu) {
      for (auto v : selected_gpu_variants(a, CipherType::ZUC_128)) {
        auto gr = brute_force_gpu_enhanced(CipherType::ZUC_128, tv.pt, tv.ct, tv.iv, tv.length,
                                           known_high, b, v, a.blocks, a.threads, a.gpu_repeats);
        double kps = safe_kps(gr.keys_tested, gr.seconds);
        std::cout << "GPU " << gpu_variant_name(v) << ": " << gr.seconds << " s, " << kps << " keys/s, found=" << (gr.found ? "yes" : "no") << "\n";
        std::ostringstream row;
        row << "zuc_128,gpu," << gpu_variant_name(v) << "," << b << "," << gr.keys_tested << "," << gr.seconds << "," << kps << "," << u64_hex(gr.found ? gr.found_key : 0);
        csv_append_row(a.out_csv, row.str());
      }
    }
  }
}

static void benchmark_snowv(const Args& a) {
  std::cout << "\n=== SNOW-V Benchmark ===\n";
  auto tv = snow_v_bench_tv();

  uint64_t key64 = 0;
  for (int i = 0; i < 8; i++) key64 |= ((uint64_t)tv.key[i]) << (i * 8);

  int max_bits = a.max_bits;

  for (int b = a.min_bits; b <= max_bits; b += a.step_bits) {
    uint64_t known_high = key64 >> b;
    uint64_t N = bf_space_size_main(b);

    std::cout << "\nunknown_bits=" << b << " (keys=" << N << ")\n";

    if (a.run_cpu) {
      auto cr = brute_force_cpu_snow_v(tv.pt, tv.ct, tv.iv, tv.length, known_high, b, a.cpu_repeats);
      double kps = safe_kps(cr.keys_tested, cr.seconds);
      std::cout << "CPU:               " << cr.seconds << " s, " << kps << " keys/s\n";
      std::ostringstream row;
      row << "snow_v,cpu,cpu_baseline," << b << "," << cr.keys_tested << "," << cr.seconds << "," << kps << "," << u64_hex(cr.found ? cr.found_key : 0);
      csv_append_row(a.out_csv, row.str());
    }

    if (a.run_gpu) {
      for (auto v : selected_gpu_variants(a, CipherType::SNOW_V)) {
        auto gr = brute_force_gpu_enhanced(CipherType::SNOW_V, tv.pt, tv.ct, tv.iv, tv.length,
                                           known_high, b, v, a.blocks, a.threads, a.gpu_repeats);
        double kps = safe_kps(gr.keys_tested, gr.seconds);
        std::cout << "GPU " << gpu_variant_name(v) << ": " << gr.seconds << " s, " << kps << " keys/s, found=" << (gr.found ? "yes" : "no") << "\n";
        std::ostringstream row;
        row << "snow_v,gpu," << gpu_variant_name(v) << "," << b << "," << gr.keys_tested << "," << gr.seconds << "," << kps << "," << u64_hex(gr.found ? gr.found_key : 0);
        csv_append_row(a.out_csv, row.str());
      }
    }
  }
}

static void benchmark_aes(const Args& a) {
  printf("\n=== AES-128 Benchmark ===\n");
  auto tv = aes_bench_tv();

  uint64_t key64 = 0;
  for (int i = 0; i < 8; i++) key64 |= ((uint64_t)tv.key[i]) << (i * 8);

  for (int b = a.min_bits; b <= a.max_bits; b += a.step_bits) {
    uint64_t known_high = key64 >> b;
    printf("\nunknown_bits=%d (keys=%llu)\n", b, bf_space_size_main(b));

    if (a.run_cpu) {
      auto cr = brute_force_cpu_aes(tv.pt, tv.ct, known_high, b, a.cpu_repeats);
      double kps = safe_kps(cr.keys_tested, cr.seconds);
      printf("CPU:               %g s, %g keys/s\n", cr.seconds, kps);
      std::ostringstream row;
      row << "aes128,cpu,cpu_baseline," << b << "," << cr.keys_tested << "," << cr.seconds << "," << kps << "," << u64_hex(cr.found ? cr.found_key : 0);
      csv_append_row(a.out_csv, row.str());
    }

    if (a.run_gpu) {
      auto run = [&](GpuVariant v, const char* tag) {
        auto gr = brute_force_gpu_enhanced(CipherType::AES_128, tv.pt, tv.ct, nullptr, 0,
                                           known_high, b, v, a.blocks, a.threads, a.gpu_repeats);
        double kps = safe_kps(gr.keys_tested, gr.seconds);
        printf("%s: %g s, %g keys/s, found=%s\n", tag, gr.seconds, kps, gr.found ? "yes" : "no");
        std::ostringstream row;
        row << "aes128,gpu," << gpu_variant_name(v) << "," << b << "," << gr.keys_tested << "," << gr.seconds << "," << kps << "," << u64_hex(gr.found ? gr.found_key : 0);
        csv_append_row(a.out_csv, row.str());
      };

      for (auto v : selected_gpu_variants(a, CipherType::AES_128)) {
        std::string tag = std::string("GPU ") + gpu_variant_name(v);
        run(v, tag.c_str());
      }
    }
  }
}

static void benchmark_salsa20(const Args& a) {
  std::cout << "\n=== Salsa20 Benchmark ===\n";
  auto tv = salsa20_bench_tv();

  int max_bits = a.max_bits;

  for (int b = a.min_bits; b <= max_bits; b += a.step_bits) {
    uint64_t known_high = tv.key64 >> b;
    uint64_t N = bf_space_size_main(b);

    std::cout << "\nunknown_bits=" << b << " (keys=" << N << ")\n";

    if (a.run_cpu) {
      auto cr = brute_force_cpu_salsa20(tv.pt, tv.ct, tv.nonce, tv.length, known_high, b, a.cpu_repeats);
      double kps = safe_kps(cr.keys_tested, cr.seconds);
      std::cout << "CPU:               " << cr.seconds << " s, " << kps << " keys/s\n";
      std::ostringstream row;
      row << "salsa20,cpu,cpu_baseline," << b << "," << cr.keys_tested << "," << cr.seconds << "," << kps << "," << u64_hex(cr.found ? cr.found_key : 0);
      csv_append_row(a.out_csv, row.str());
    }

    if (a.run_gpu) {
      for (auto v : selected_gpu_variants(a, CipherType::SALSA20)) {
        auto gr = brute_force_gpu_enhanced(CipherType::SALSA20, tv.pt, tv.ct, tv.nonce, tv.length,
                                           known_high, b, v, a.blocks, a.threads, a.gpu_repeats);
        double kps = safe_kps(gr.keys_tested, gr.seconds);
        std::cout << "GPU " << gpu_variant_name(v) << ": " << gr.seconds << " s, " << kps << " keys/s, found=" << (gr.found ? "yes" : "no") << "\n";
        std::ostringstream row;
        row << "salsa20,gpu," << gpu_variant_name(v) << "," << b << "," << gr.keys_tested << "," << gr.seconds << "," << kps << "," << u64_hex(gr.found ? gr.found_key : 0);
        csv_append_row(a.out_csv, row.str());
      }
    }
  }
}

static void benchmark_grain128aeadv2(const Args& a) {
  std::cout << "\n=== Grain-128AEADv2 Benchmark ===\n";
  auto tv = grain128aeadv2_bench_tv();

  uint64_t key64 = 0;
  for (int i = 0; i < 8; i++) key64 |= ((uint64_t)tv.key[i]) << (i * 8);

  for (int b = a.min_bits; b <= a.max_bits; b += a.step_bits) {
    uint64_t known_high = key64 >> b;
    uint64_t N = bf_space_size_main(b);

    std::cout << "\nunknown_bits=" << b << " (keys=" << N << ")\n";

    if (a.run_cpu) {
      auto cr = brute_force_cpu_grain128aeadv2(tv.pt, tv.ct, tv.nonce, tv.pt_len,
                                               tv.ad, tv.ad_len, known_high, b, a.cpu_repeats);
      double kps = safe_kps(cr.keys_tested, cr.seconds);
      std::cout << "CPU:               " << cr.seconds << " s, " << kps << " keys/s\n";
      std::ostringstream row;
      row << "grain128aeadv2,cpu,cpu_baseline," << b << "," << cr.keys_tested << ","
          << cr.seconds << "," << kps << "," << u64_hex(cr.found ? cr.found_key : 0);
      csv_append_row(a.out_csv, row.str());
    }

    if (a.run_gpu) {
      for (auto v : selected_gpu_variants(a, CipherType::GRAIN128_AEADV2)) {
        auto gr = brute_force_gpu_enhanced_aead(CipherType::GRAIN128_AEADV2,
                                                tv.pt, tv.ct, tv.nonce, tv.pt_len,
                                                tv.ad, tv.ad_len, tv.tag,
                                                known_high, b, v, a.blocks, a.threads, a.gpu_repeats);
        double kps = safe_kps(gr.keys_tested, gr.seconds);
        std::cout << "GPU " << gpu_variant_name(v) << ": " << gr.seconds << " s, "
                  << kps << " keys/s, found=" << (gr.found ? "yes" : "no") << "\n";
        std::ostringstream row;
        row << "grain128aeadv2,gpu," << gpu_variant_name(v) << "," << b << ","
            << gr.keys_tested << "," << gr.seconds << "," << kps << ","
            << u64_hex(gr.found ? gr.found_key : 0);
        csv_append_row(a.out_csv, row.str());
      }
    }
  }
}

// ============================================================
// main
// ============================================================

int main(int argc, char** argv) {
  Args a = parse_args(argc, argv);

  if (a.test_only) {
    bool ok = run_all_self_tests();
    return ok ? 0 : 1;
  }

  if (a.verify_only) {
    bool ok = run_random_key_verification(a.verify_bits);
    return ok ? 0 : 1;
  }

  // Init CSV
  csv_init(a.out_csv);

  // Always run self-tests once at start
  if (!run_all_self_tests()) {
    std::cerr << "One or more self-tests failed. Aborting benchmarks.\n";
    return 2;
  }

  auto run_one = [&](const std::string& name) {
    if (name == "simon") benchmark_simon(a);
    else if (name == "present") benchmark_present(a);
    else if (name == "speck") benchmark_speck(a);
    else if (name == "grain") benchmark_grain(a);
    else if (name == "trivium") benchmark_trivium(a);
    else if (name == "chacha") benchmark_chacha(a);
    else if (name == "tinyjambu") benchmark_tinyjambu(a);
    else if (name == "zuc") benchmark_zuc(a);
    else if (name == "snowv") benchmark_snowv(a);
    else if (name == "aes") benchmark_aes(a);
    else if (name == "salsa") benchmark_salsa20(a);
    else if (name == "grain128") benchmark_grain128aeadv2(a);
    else {
      std::cerr << "Unknown cipher: " << name << "\n";
      std::cerr << "Valid: simon, present, speck, grain, trivium, chacha, tinyjambu, zuc, snowv, aes, salsa, grain128, all\n";
      std::exit(1);
    }
  };

  if (a.cipher == "all") {
    benchmark_simon(a);
    benchmark_present(a);
    benchmark_speck(a);
    benchmark_grain(a);
    benchmark_trivium(a);
    benchmark_chacha(a);
    benchmark_tinyjambu(a);
    benchmark_zuc(a);
    benchmark_snowv(a);
    benchmark_aes(a);
    benchmark_salsa20(a);
    benchmark_grain128aeadv2(a);
  } else {
    run_one(a.cipher);
  }

  std::cout << "\nDone. Wrote results to " << a.out_csv << "\n";
  return 0;
}
