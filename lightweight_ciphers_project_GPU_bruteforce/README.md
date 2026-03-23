# Lightweight Ciphers — GPU Brute-Force Benchmark

Partial-key brute-force throughput benchmarks (CPU and CUDA GPU) for eleven lightweight and
modern stream/block ciphers. The project measures how many candidate keys per second each
implementation can test, sweeping over a configurable range of unknown key bits.

---

## Ciphers

| # | Cipher | Type | Key bits brute-forced | Notes |
|---|--------|------|-----------------------|-------|
| 1 | **SIMON 32/64** | Block | 64-bit key, full sweep | NSA lightweight block cipher |
| 2 | **PRESENT-80** | Block | Low 64 of 80-bit key | ISO/IEC 29192-2; has shared-memory SP-box variant |
| 3 | **SPECK 64/128** | Block (ARX) | Low 64 of 128-bit key | NSA lightweight block cipher |
| 4 | **Grain v1** | Stream | Low 64 of 80-bit key | 64-bit IV; eSTREAM portfolio |
| 5 | **Trivium** | Stream | Low 64 of 80-bit key | 80-bit IV; eSTREAM portfolio |
| 6 | **ChaCha20** | Stream (ARX) | Low 64 of 256-bit key | 96-bit nonce; RFC 8439 |
| 7 | **TinyJAMBU-128** | AEAD | Low 64 of 128-bit key | 96-bit nonce, 64-bit tag; NIST LWC finalist (v2) |
| 8 | **ZUC-128** | Stream | Low 64 of 128-bit key | 128-bit IV; 3GPP/GSMA (EEA3/EIA3) |
| 9 | **SNOW-V** | Stream | Low 64 of 256-bit key | 128-bit IV; AES-based FSM |
| 10 | **AES-128** | Block | Low 64 of 128-bit key | NIST FIPS 197; verified against NIST test vector |
| 11 | **Salsa20** | Stream (ARX) | Low 64 of 256-bit key | 64-bit nonce; eSTREAM portfolio; verified against eSTREAM Set 1 vector |

> **Brute-force model:** only the lowest 64 bits of each key are swept; the remaining bits
> are held fixed at zero. `unknown_bits=N` means the lowest N bits are unknown — the search
> space is 2^N keys.

---

## GPU Variants

| Variant flag | Label in CSV | Description |
|---|---|---|
| `baseline` | `gpu0_baseline` | 1 key/thread, full encrypt call |
| `optimized` | `gpu1_optimized` | Optimized encrypt (fast MixColumns for AES; early-exit for stream ciphers) |
| `optimized_ilp` | `gpu2_optimized+ilp` | 4 keys per loop iteration (ILP4) + optimized encrypt |
| `shared` | `gpu3_optimized+shared` | Shared-memory SP-box table (PRESENT-80 only) |
| `bitsliced` | `gpu4_bitsliced` | 32 keys per thread via bitsliced permutation (TinyJAMBU-128 only) |

Default `auto` mode selects the most relevant variants per cipher:
- **PRESENT-80** — baseline, optimized, optimized_ilp, **shared**
- **TinyJAMBU-128** — baseline, optimized_ilp, **bitsliced**
- **AES-128** — baseline, optimized, optimized_ilp
- **Salsa20** — baseline, optimized, optimized_ilp
- **All others** — baseline, optimized, optimized_ilp

---

## AES-128 GPU Variant Details

AES-128 is a 10-round SPN block cipher with a 128-bit key and 128-bit block. Each round
performs: SubBytes → ShiftRows → MixColumns → AddRoundKey. The three GPU variants differ
in how MixColumns is computed:

### gpu0_baseline — Naive MixColumns (`UseTTable=false`)

MixColumns is implemented using a **general GF(2⁸) multiplication loop**:

```
gmul(a, b):
  result = 0
  for 8 iterations:
    if b & 1: result ^= a
    a = (a << 1) ^ (0x1b if a & 0x80 else 0)  # reduce mod x^8+x^4+x^3+x+1
    b >>= 1
```

Each of the four MixColumns column outputs requires four `gmul` calls (with factors 1, 2, 3).
This loop is unrolled by the compiler but involves data-dependent branching and bit-by-bit
iteration — slower but algorithmically straightforward.

### gpu1_optimized — Fast MixColumns (`UseTTable=true`, no ILP)

MixColumns uses **precomputed bit-shift GF(2⁸) tricks**:

```
gmul2(a) = (a << 1) ^ (0x1b if a & 0x80)   // multiply by 2 in GF(2^8)
gmul3(a) = gmul2(a) ^ a                      // multiply by 3 = 2+1
```

These replace the 8-iteration loop with 2–3 XOR/shift instructions per element.
The MixColumns column update becomes 8 XOR operations total (no loop, no branch on the key
data path). This is **significantly faster** on GPUs because:
- No loop overhead or loop-carried dependency
- No data-dependent branching in the inner loop
- Fewer instructions per AES round

### gpu2_optimized+ilp — Fast MixColumns + ILP4

Same fast MixColumns as `gpu1_optimized`, plus **Instruction Level Parallelism (ILP4)**:
each thread processes **4 candidate keys per loop iteration** instead of 1.

```
for k in range(tid, N, stride*4):
    try_key(base | k)
    try_key(base | (k + stride))
    try_key(base | (k + stride*2))
    try_key(base | (k + stride*3))
```

Because each `try_key` call is independent (different key, no shared state), the GPU's
out-of-order execution units can overlap memory latency (S-box table loads) across
the 4 independent AES encryption pipelines. This hides L1/L2 cache miss latency and
increases arithmetic throughput utilization.

### Summary Table

| Variant | MixColumns | Keys/thread/iter | Throughput |
|---------|-----------|-----------------|------------|
| `gpu0_baseline` | Naive (8-iter GF loop) | 1 | Lowest |
| `gpu1_optimized` | Fast (gmul2/gmul3 bit tricks) | 1 | ~2–4x baseline |
| `gpu2_optimized+ilp` | Fast (gmul2/gmul3 bit tricks) | 4 | Highest |

---

## Salsa20 GPU Variant Details

Salsa20 is a 20-round ARX stream cipher with a 256-bit key, 64-bit nonce, and 64-bit counter. Each block produces 64 bytes of keystream. The three GPU variants differ in how many bytes are computed per key candidate and how many candidates are processed per thread per loop iteration.

### State Layout

Salsa20 uses a 4×4 matrix of 32-bit words (all little-endian):

```
σ0   k0   k1   k2      σ0 = 0x61707865  "expa"
k3   σ1   n0   n1      σ1 = 0x3320646e  "nd 3"
t0   t1   σ2   k4      σ2 = 0x79622d32  "2-by"
k5   k6   k7   σ3      σ3 = 0x6b206574  "te k"
```

where k0..k7 = 256-bit key words, n0/n1 = 64-bit nonce words, t0/t1 = 64-bit counter words.

### Quarter-Round Function

Each column-round and row-round applies this QR(a, b, c, d) to 4 state words:

```
x[b] ^= (x[a] + x[d]) <<< 7
x[c] ^= (x[b] + x[a]) <<< 9
x[d] ^= (x[c] + x[b]) <<< 13
x[a] ^= (x[d] + x[c]) <<< 18
```

Note the different rotation amounts (7, 9, 13, 18) compared to ChaCha20 (16, 12, 8, 7), and the different order of adds.

### gpu0_baseline — Full 64-byte block

Each thread calls `Salsa20::process()` which produces the full 64-byte keystream block, then compares against the ciphertext. No early exit.

### gpu1_optimized — First-16-byte prefix match with early exit

Uses `Salsa20::block_words4()` which computes only the **first 4 output words (16 bytes)** by running the full 20-round core but extracting only `w[0..3] + x[0..3]`. This avoids materializing bytes 16–63. The match check aborts immediately on the first mismatched byte (`STOP_ON_FOUND` propagation in the optimized variant).

The savings: ~4× fewer memory writes per candidate; no loop over full 64 bytes on mismatch.

### gpu2_optimized+ilp — First-16-byte prefix match + ILP2

Same `block_words4` optimization as `gpu1_optimized`, plus **Instruction Level Parallelism (ILP2)**:
each thread processes **2 candidate keys per loop iteration**. Since each `test_one` call is fully independent (separate key, no shared state), the GPU's execution units overlap memory latency across the two ARX pipelines.

### Summary Table

| Variant | Bytes computed | Keys/thread/iter | Throughput |
|---------|----------------|-----------------|------------|
| `gpu0_baseline` | 64 (full block) | 1 | Lowest |
| `gpu1_optimized` | 16 (prefix only) | 1 | ~2–3x baseline |
| `gpu2_optimized+ilp` | 16 (prefix only) | 2 | Highest |

---

## Repository Layout

```
lightweight_ciphers_project_GPU_bruteforce/
├── CMakeLists.txt
├── README.md
├── plot_results.py
├── src/
│   ├── main.cu                    # Benchmark driver, self-tests, CLI
│   ├── ciphers_enhanced.cuh       # All eleven cipher implementations (host + device)
│   ├── bruteforce_gpu_enhanced.cuh # GPU kernels, CipherType/GpuVariant enums, launch helpers
│   ├── bruteforce_cpu.hpp         # CPU brute-force reference implementations
│   ├── present_spbox_tables.inc   # Pre-generated PRESENT-80 SP-box lookup tables
│   └── util.hpp                   # CUDA error checking, CSV helpers, hex formatting
├── results_project_b.csv          # Example full-run results
└── results_present.csv            # Example PRESENT-only results
```

---

## Build

### Prerequisites
- CUDA toolkit ≥ 11.2 with `nvcc`
- CMake ≥ 3.18
- C++17-capable host compiler (gcc / clang)

### Find your GPU architecture

```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
# e.g. "8.6" → use -DCMAKE_CUDA_ARCHITECTURES=86
```

Common values: `75` (Turing T4), `80` (Ampere A100), `86` (Ampere RTX 30xx), `89` (Ada RTX 40xx).

### Configure CUDA paths (if nvcc is not in PATH)

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDACXX=$CUDA_HOME/bin/nvcc
```

### Build

```bash
cd lightweight_ciphers_project_GPU_bruteforce
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 ..
cmake --build . -j$(nproc)
# Binary is: build/bench
```

---

## Self-Tests

Verifies all ten cipher implementations against official test vectors before benchmarking.

```bash
./bench --test
```

Expected output:
```
SIMON32/64 Self-Test:              PASS
PRESENT-80 Self-Test:              PASS
SPECK64/128 Self-Test:             PASS
Grain v1 Self-Test:                PASS
Trivium Self-Test:                 PASS
ChaCha20 Block Self-Test:          PASS
TinyJAMBU-128 Self-Test:           PASS
TinyJAMBU-128 Bitsliced Self-Test: PASS
ZUC-128 Self-Test:                 PASS
SNOW-V Self-Test:                  PASS
AES-128 Self-Test:                 PASS
Salsa20 Self-Test:                 PASS
```

---

## Running Experiments

All commands below assume you are inside the `build/` directory.

### Run all 11 ciphers (recommended full benchmark)

```bash
./bench --cipher all \
        --min_bits 2 --max_bits 30 --step_bits 1 \
        --cpu_repeats 3 --gpu_repeats 10 \
        --blocks 1024 --threads 256 \
        --out ../results_all.csv
```

### GPU only (skip CPU baselines)

```bash
./bench --cipher all --gpu_only \
        --min_bits 2 --max_bits 20 --step_bits 2 \
        --out ../results_gpu_only.csv
```

### CPU only

```bash
./bench --cipher all --cpu_only \
        --min_bits 2 --max_bits 20 --step_bits 2 \
        --out ../results_cpu_only.csv
```

---

### Run individual ciphers

#### 1. SIMON 32/64

```bash
./bench --cipher simon --min_bits 1 --max_bits 30 --step_bits 1 --out ../results_simon.csv
```

#### 2. PRESENT-80 (with all variants including shared-memory)

```bash
./bench --cipher present --variants all \
        --min_bits 1 --max_bits 30 --step_bits 1 \
        --out ../results_present.csv
```

#### 3. SPECK 64/128

```bash
./bench --cipher speck --min_bits 1 --max_bits 30 --step_bits 1 --out ../results_speck.csv
```

#### 4. Grain v1

```bash
./bench --cipher grain --min_bits 1 --max_bits 20 --step_bits 1 --out ../results_grain.csv
```

#### 5. Trivium

```bash
./bench --cipher trivium --min_bits 1 --max_bits 20 --step_bits 1 --out ../results_trivium.csv
```

#### 6. ChaCha20

```bash
./bench --cipher chacha --min_bits 1 --max_bits 20 --step_bits 1 --out ../results_chacha.csv
```

#### 7. TinyJAMBU-128 (AEAD — includes bitsliced variant)

```bash
./bench --cipher tinyjambu --min_bits 1 --max_bits 20 --step_bits 1 --out ../results_tinyjambu.csv
```

To benchmark only the bitsliced kernel:
```bash
./bench --cipher tinyjambu --variants bitsliced \
        --min_bits 1 --max_bits 20 --step_bits 1 \
        --out ../results_tinyjambu_bs.csv
```

#### 8. ZUC-128

```bash
./bench --cipher zuc --min_bits 1 --max_bits 20 --step_bits 1 --out ../results_zuc.csv
```

#### 9. SNOW-V

```bash
./bench --cipher snowv --min_bits 1 --max_bits 20 --step_bits 1 --out ../results_snowv.csv
```

#### 10. AES-128

```bash
./bench --cipher aes --min_bits 1 --max_bits 30 --step_bits 1 --out ../results_aes.csv
```

Run only specific AES GPU variants:
```bash
# Baseline only (naive MixColumns)
./bench --cipher aes --variants baseline --min_bits 1 --max_bits 30 --out ../results_aes_baseline.csv

# Optimized only (fast MixColumns, no ILP)
./bench --cipher aes --variants optimized --min_bits 1 --max_bits 30 --out ../results_aes_opt.csv

# ILP4 only (fast MixColumns + ILP4)
./bench --cipher aes --variants optimized_ilp --min_bits 1 --max_bits 30 --out ../results_aes_ilp.csv
```

#### 11. Salsa20

```bash
./bench --cipher salsa --min_bits 1 --max_bits 30 --step_bits 1 --out ../results_salsa20.csv
```

Run only specific Salsa20 GPU variants:
```bash
# Baseline only (full 64-byte block per candidate)
./bench --cipher salsa --variants baseline --min_bits 1 --max_bits 30 --out ../results_salsa20_baseline.csv

# Optimized only (first 16-byte prefix match, early exit)
./bench --cipher salsa --variants optimized --min_bits 1 --max_bits 30 --out ../results_salsa20_opt.csv

# ILP2 only (2 keys per loop + early exit)
./bench --cipher salsa --variants optimized_ilp --min_bits 1 --max_bits 30 --out ../results_salsa20_ilp.csv
```

---

## Selecting GPU Variants Manually

```bash
# Baseline only
./bench --cipher grain --variants baseline

# Optimized (early-exit / fast MixColumns) only
./bench --cipher trivium --variants optimized

# ILP4 only
./bench --cipher chacha --variants optimized_ilp

# Shared-memory (PRESENT-80 only)
./bench --cipher present --variants shared

# Bitsliced (TinyJAMBU-128 only)
./bench --cipher tinyjambu --variants bitsliced

# All variants
./bench --cipher present --variants all
```

---

## CLI Reference

```
Usage: bench [options]
  --cipher   <simon|present|speck|grain|trivium|chacha|tinyjambu|zuc|snowv|aes|salsa|all>
             (default: all)
  --variants <baseline|optimized|optimized_ilp|shared|bitsliced|all|auto>
             (default: auto — best variants per cipher)
  --out      <path>          Output CSV file (default: results.csv)
  --min_bits <N>             Minimum unknown bits to sweep (default: 1)
  --max_bits <N>             Maximum unknown bits to sweep (default: 30)
  --step_bits <N>            Step size for bit sweep (default: 1)
  --cpu_repeats <N>          CPU timing repetitions (default: 3)
  --gpu_repeats <N>          GPU timing repetitions (default: 10)
  --blocks <N>               CUDA grid blocks (default: 1024)
  --threads <N>              CUDA threads per block (default: 256)
  --cpu_only                 Run CPU benchmarks only
  --gpu_only                 Run GPU benchmarks only
  --test                     Run self-tests only (no benchmarks)
```

---

## CSV Output Format

Each row in the output CSV:

```
cipher, platform, variant, unknown_bits, keys_tested, seconds, keys_per_second, found_key_hex
```

Example rows:
```
simon32_64,cpu,cpu_baseline,10,1024,0.000012,85333333,0x0000000000000000
simon32_64,gpu,gpu0_baseline,10,1024,0.0000031,330322580,0x0000000000000000
simon32_64,gpu,gpu2_optimized+ilp,10,1024,0.0000019,539000000,0x0000000000000000
aes_128,cpu,cpu_baseline,10,1024,0.000015,68266666,0x0000000000000000
aes_128,gpu,gpu0_baseline,10,1024,0.0000058,176551724,0x0000000000000000
aes_128,gpu,gpu1_optimized,10,1024,0.0000021,487619047,0x0000000000000000
aes_128,gpu,gpu2_optimized+ilp,10,1024,0.0000014,731428571,0x0000000000000000
salsa20,gpu,gpu0_baseline,10,1024,0.0000049,208979591,0x0000000000000000
salsa20,gpu,gpu1_optimized,10,1024,0.0000018,568888888,0x0000000000000000
salsa20,gpu,gpu2_optimized+ilp,10,1024,0.0000012,853333333,0x0000000000000000
tinyjambu_128,gpu,gpu4_bitsliced,10,1024,0.0000045,227555555,0xa55a12343fffffff
```

The `found_key_hex` field is non-zero only when the correct key is found within the search space (verification run).

---

## Plotting Results

```bash
cd ..   # back to lightweight_ciphers_project_GPU_bruteforce/

# Plot all ciphers from a CSV
python3 plot_results.py results_all.csv --outdir plots/

# Plot a single cipher
python3 plot_results.py results_all.csv --only_cipher aes_128 --outdir plots/

# Specify CSV with flag
python3 plot_results.py --csv results_all.csv --outdir plots/
```

Plots are saved as PNG files in the `--outdir` directory:
- `<cipher>__time_vs_bits.png` — wall-clock time vs unknown_bits per variant
- `<cipher>__kps_vs_bits.png` — throughput (keys/s) vs unknown_bits per variant

---

## Tuning Tips

| Situation | Suggested change |
|-----------|-----------------|
| GPU is slow / low occupancy | Increase `--blocks` (try 2048 or 4096) |
| Want faster benchmark runs | Reduce `--max_bits` and `--gpu_repeats` |
| Comparing only GPU variants | Use `--gpu_only` |
| PRESENT-80 shared-memory study | `--cipher present --variants all` |
| TinyJAMBU bitsliced study | `--cipher tinyjambu --variants all --max_bits 16` |
| AES MixColumns optimization study | `--cipher aes --variants all` |
| Salsa20 vs ChaCha20 comparison | `--cipher salsa` then `--cipher chacha` |
| Stream cipher comparison | `--cipher grain` then `--cipher trivium` then `--cipher chacha` then `--cipher salsa` |
