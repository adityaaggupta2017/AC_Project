#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <algorithm>
#include <sstream>

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

static bool run_all_self_tests() {
  bool ok = true;
  ok &= self_test_simon();
  ok &= self_test_present();
  ok &= self_test_speck();
  ok &= self_test_grain();
  ok &= self_test_trivium();
  ok &= self_test_chacha20();
  return ok;
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
    else if (s == "--test") a.test_only = true;
    else if (s == "--cpu_only") { a.run_cpu = true; a.run_gpu = false; a.cpu_only = true; }
    else if (s == "--gpu_only") { a.run_cpu = false; a.run_gpu = true; a.gpu_only = true; }
    else if (s == "--help" || s == "-h") {
      std::cout
        << "Usage: bench [options]\n"
        << "  --cipher <simon|present|speck|grain|trivium|chacha|all> (default: all)\n"
        << "  --variants <baseline|optimized|optimized_ilp|shared|all|auto> (default: auto)\n"
        << "  --out results.csv\n"
        << "  --min_bits 1 --max_bits 30 --step_bits 1\n"
        << "  --cpu_repeats 3 --gpu_repeats 10\n"
        << "  --blocks 1024 --threads 256\n"
        << "  --cpu_only | --gpu_only (default: run both)\n"
        << "  --test (run self-tests only)\n";
      std::exit(0);
    }
  }
  return a;
}


static inline std::vector<GpuVariant> selected_gpu_variants(const Args& a, bool include_shared = false) {
  if (a.variants == "baseline") return {GpuVariant::BASELINE};
  if (a.variants == "optimized") return {GpuVariant::OPTIMIZED};
  if (a.variants == "optimized_ilp") return {GpuVariant::OPTIMIZED_ILP};
  if (a.variants == "shared") return {GpuVariant::OPTIMIZED_SHARED};
  if (a.variants == "all") return {GpuVariant::BASELINE, GpuVariant::OPTIMIZED, GpuVariant::OPTIMIZED_ILP, GpuVariant::OPTIMIZED_SHARED};
  if (include_shared) return {GpuVariant::BASELINE, GpuVariant::OPTIMIZED, GpuVariant::OPTIMIZED_ILP, GpuVariant::OPTIMIZED_SHARED};
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

      for (auto v : selected_gpu_variants(a)) {
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

      for (auto v : selected_gpu_variants(a, true)) {
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

      for (auto v : selected_gpu_variants(a)) {
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
      for (auto v : selected_gpu_variants(a)) {
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
      for (auto v : selected_gpu_variants(a)) {
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
      for (auto v : selected_gpu_variants(a)) {
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

// ============================================================
// main
// ============================================================

int main(int argc, char** argv) {
  Args a = parse_args(argc, argv);

  if (a.test_only) {
    bool ok = run_all_self_tests();
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
    else {
      std::cerr << "Unknown cipher: " << name << "\n";
      std::cerr << "Valid: simon, present, speck, grain, trivium, chacha, all\n";
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
  } else {
    run_one(a.cipher);
  }

  std::cout << "\nDone. Wrote results to " << a.out_csv << "\n";
  return 0;
}
