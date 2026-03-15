# Lightweight Ciphers — GPU Brute-Force Benchmark

Partial-key brute-force throughput benchmarks (CPU and CUDA GPU) for nine lightweight and
modern stream ciphers. The project measures how many candidate keys per second each
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

> **Brute-force model:** only the lowest 64 bits of each key are swept; the remaining bits
> are held fixed at zero. `unknown_bits=N` means the lowest N bits are unknown — the search
> space is 2^N keys.

---

## GPU Variants

| Variant flag | Label in CSV | Description |
|---|---|---|
| `baseline` | `gpu0_baseline` | 1 key/thread, full process/encrypt call |
| `optimized` | `gpu1_optimized` | Early-exit keystream comparison (stream ciphers) |
| `optimized_ilp` | `gpu2_optimized+ilp` | 2 keys per loop iteration (ILP2) + early-exit |
| `shared` | `gpu3_optimized+shared` | Shared-memory SP-box table (PRESENT-80 only) |
| `bitsliced` | `gpu4_bitsliced` | 32 keys per thread via bitsliced permutation (TinyJAMBU-128 only) |

Default `auto` mode selects the most relevant variants per cipher:
- **PRESENT-80** — baseline, optimized, optimized_ilp, **shared**
- **TinyJAMBU-128** — baseline, optimized_ilp, **bitsliced**
- **All others** — baseline, optimized, optimized_ilp

---

## Repository Layout

```
lightweight_ciphers_project_GPU_bruteforce/
├── CMakeLists.txt
├── README.md
├── plot_results.py
├── src/
│   ├── main.cu                    # Benchmark driver, self-tests, CLI
│   ├── ciphers_enhanced.cuh       # All nine cipher implementations (host + device)
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

Verifies all nine cipher implementations against official test vectors before benchmarking.

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
```

---

## Running Experiments

All commands below assume you are inside the `build/` directory.

### Run all 9 ciphers (recommended full benchmark)

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

---

## Selecting GPU Variants Manually

```bash
# Baseline only
./bench --cipher grain --variants baseline

# Optimized (early-exit) only
./bench --cipher trivium --variants optimized

# ILP2 only
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
  --cipher   <simon|present|speck|grain|trivium|chacha|tinyjambu|zuc|snowv|all>
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
python3 plot_results.py results_all.csv --only_cipher simon32_64 --outdir plots/

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
| Stream cipher comparison | `--cipher grain` then `--cipher trivium` then `--cipher chacha` |
