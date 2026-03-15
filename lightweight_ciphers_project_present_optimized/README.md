# Lightweight Ciphers Project (PRESENT Optimized)

This package contains corrected source code for partial-key brute-force benchmarks on:

- SIMON 32/64
- PRESENT-80
- SPECK 64/128
- Grain v1
- Trivium
- ChaCha20

The main change compared with the earlier corrected package is a stronger PRESENT-80 path:

- baseline PRESENT kept for reference
- optimized PRESENT uses **byte SP-box tables**
- optimized PRESENT also uses **on-the-fly key schedule**
- shared-memory PRESENT variant stages the SP-box table per block

This keeps the code structure close to your original project while improving the PRESENT kernel substantially without rewriting the entire cipher into a full bit-sliced implementation.

## Files

- `src/ciphers_enhanced.cuh` — cipher implementations
- `src/present_spbox_tables.inc` — generated PRESENT SP-box tables
- `src/bruteforce_cpu.hpp` — CPU brute force
- `src/bruteforce_gpu_enhanced.cuh` — GPU kernels and wrappers
- `src/main.cu` — benchmark driver
- `src/util.hpp` — helpers
- `plot_results.py` — plotting utility
- `PRESENT_OPTIMIZATIONS.md` — explanation of the PRESENT changes

## Build

```bash
cd lightweight_ciphers_project_present_optimized
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 ..
cmake --build . -j$(nproc)
```

If your server uses a newer CUDA install outside `/usr/bin/nvcc`, point CMake to it first:

```bash
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDACXX=$CUDA_HOME/bin/nvcc
```

## Run self-tests

```bash
./bench --test
```

## Run all ciphers for 1..30 unknown bits

```bash
./bench --cipher all --min_bits 1 --max_bits 30 --step_bits 1 --out ../results_project_b.csv
```

## Run only PRESENT with all variants

```bash
./bench --cipher present --variants all --min_bits 1 --max_bits 30 --step_bits 1 --out ../results_present.csv
```

## Plot results

```bash
cd ..
python3 plot_results.py --csv results_project_b.csv
```
