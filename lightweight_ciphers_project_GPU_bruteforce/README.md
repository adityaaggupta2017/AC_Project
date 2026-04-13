        # Lightweight Ciphers — GPU Brute-Force Benchmark

        Partial-key brute-force throughput benchmarks (CPU and CUDA GPU) for twelve lightweight and
        modern stream/block/AEAD ciphers. The project sweeps over a configurable range of unknown key
        bits and measures how many candidate keys per second each implementation can test.

        ---

        ## Table of Contents

        1. [Ciphers](#ciphers)
        2. [Brute-Force Model](#brute-force-model)
        3. [GPU Variant Overview](#gpu-variant-overview)
        4. [Per-Cipher GPU Optimizations](#per-cipher-gpu-optimizations)
        5. [CPU Optimizations](#cpu-optimizations)
        6. [Repository Layout](#repository-layout)
        7. [Build](#build)
        8. [Self-Tests](#self-tests)
        9. [Running Experiments](#running-experiments)
        10. [Randomized Correctness Verification](#randomized-correctness-verification)
        11. [CLI Reference](#cli-reference)
        12. [CSV Output Format](#csv-output-format)
        13. [Plotting Results](#plotting-results)
        14. [Tuning Tips](#tuning-tips)

        ---

        ## Ciphers

        | # | Cipher | Type | Key bits brute-forced | Standard / Notes |
        |---|--------|------|-----------------------|-----------------|
        | 1 | **SIMON 32/64** | Block | 64-bit key, full sweep | NSA lightweight; 32-round Feistel |
        | 2 | **PRESENT-80** | Block | Low 64 of 80-bit key | ISO/IEC 29192-2; 31-round SPN |
        | 3 | **SPECK 64/128** | Block (ARX) | Low 64 of 128-bit key | NSA lightweight; 27-round ARX |
        | 4 | **Grain v1** | Stream | Low 64 of 80-bit key | eSTREAM portfolio; 80-bit IV |
        | 5 | **Trivium** | Stream | Low 64 of 80-bit key | eSTREAM portfolio; 80-bit IV |
        | 6 | **ChaCha20** | Stream (ARX) | Low 64 of 256-bit key | RFC 8439; 96-bit nonce |
        | 7 | **TinyJAMBU-128** | AEAD | Low 64 of 128-bit key | NIST LWC finalist v2; 96-bit nonce, 64-bit tag |
        | 8 | **ZUC-128** | Stream | Low 64 of 128-bit key | 3GPP/GSMA EEA3/EIA3; 128-bit IV |
        | 9 | **SNOW-V** | Stream | Low 64 of 256-bit key | AES-based FSM; 128-bit IV |
        | 10 | **AES-128** | Block | Low 64 of 128-bit key | NIST FIPS 197; 10-round SPN |
        | 11 | **Salsa20** | Stream (ARX) | Low 64 of 256-bit key | eSTREAM portfolio; 64-bit nonce |
        | 12 | **Grain-128AEADv2** | AEAD | Low 64 of 128-bit key | NIST LWC finalist; 96-bit nonce, 64-bit tag |

        ---

        ## Brute-Force Model

        Only the **lowest 64 bits** of each key are swept; the remaining key bits are held fixed
        at their correct values (taken from the benchmark test vector). `unknown_bits=N` means
        the lowest N bits are unknown — search space = 2^N keys.

        ```
        key = (known_high << N) | candidate_low      candidate_low ∈ [0, 2^N)
        ```

        The benchmark measures keys-per-second throughput at each bit level and reports the time
        needed to exhaust the full search space. This models the scenario where an attacker has
        partial key information and must sweep only the unknown portion.

        ---

        ## GPU Variant Overview

        | Variant flag | CSV label | Description |
        |---|---|---|
        | `baseline` | `gpu0_baseline` | 1 key/thread; full encrypt per candidate; reference implementation |
        | `optimized` | `gpu1_optimized` | Faster encrypt path per cipher (see details below) |
        | `optimized_ilp` | `gpu2_optimized+ilp` | Same as optimized + Instruction Level Parallelism (2–4 keys per thread per loop) |
        | `shared` | `gpu3_optimized+shared` | Shared-memory SP-box table (PRESENT-80 only) |
        | `bitsliced` | `gpu4_bitsliced` | 32 keys per thread via bitsliced permutation (TinyJAMBU-128, Grain-128AEADv2) |

        Default `auto` mode selects the most relevant variants per cipher:

        | Cipher | Auto variants |
        |--------|--------------|
        | SIMON 32/64 | baseline, optimized, optimized_ilp |
        | PRESENT-80 | baseline, optimized, optimized_ilp, **shared** |
        | SPECK 64/128 | baseline, optimized, optimized_ilp |
        | Grain v1 | baseline, optimized, optimized_ilp |
        | Trivium | baseline, optimized, optimized_ilp |
        | ChaCha20 | baseline, optimized, optimized_ilp |
        | TinyJAMBU-128 | baseline, optimized_ilp, **bitsliced** |
        | ZUC-128 | baseline, optimized, optimized_ilp |
        | SNOW-V | baseline, optimized, optimized_ilp |
        | AES-128 | baseline, optimized, optimized_ilp |
        | Salsa20 | baseline, optimized, optimized_ilp |
        | Grain-128AEADv2 | baseline, optimized_ilp, **bitsliced** |

        ---

        ## Per-Cipher GPU Optimizations

        ### 1. SIMON 32/64

        SIMON 32/64 is a 32-round Feistel block cipher with a 64-bit key and 32-bit block. Each round
        applies: `x[i+1] = x[i-1] XOR (x[i] <<< 1 AND x[i] <<< 8) XOR (x[i] <<< 2) XOR k[i]`.

        #### gpu0_baseline — Precomputed round keys, 1 key/thread

        Calls `Simon32_64_Enhanced::expand_key_vectorized(k, rk)` to expand the 64-bit key into 32
        round keys packed as `uint16_t[32]`, then calls `encrypt_optimized(pt, rk)`.

        **Key optimization even in baseline**: `expand_key_vectorized` packs the key schedule
        into 4-round batches using `uint64_t` shifts, reducing the number of extract operations
        compared to naive byte-by-byte expansion. Each encrypt call is a tight 32-round loop
        with fully unrolled round key loads.

        #### gpu1_optimized — Same core, registered for dispatch

        Dispatches the same `encrypt_optimized` path. SIMON's ARX structure is already efficient
        — the round function is 3 rotations + 1 AND + 2 XORs, all register-only operations.
        No memory table lookups; no improvement from a separate "optimized" path.

        #### gpu2_optimized+ilp — ILP4 (4 keys per loop iteration)

        Each thread evaluates **4 independent candidate keys** per loop iteration:
        ```
        for i in [tid, N, stride×4]:
        try_key(base | i)
        try_key(base | (i + stride))
        try_key(base | (i + 2×stride))
        try_key(base | (i + 3×stride))
        ```
        Because the 4 Feistel pipelines are completely independent (different keys, no shared state),
        the GPU's warp scheduler can issue instructions from all 4 pipelines simultaneously, hiding
        register-file read latency and improving arithmetic utilization.

        | Variant | Keys/thread/iter | Notes |
        |---------|-----------------|-------|
        | `gpu0_baseline` | 1 | Optimized key schedule + encrypt |
        | `gpu1_optimized` | 1 | Same as baseline for SIMON |
        | `gpu2_optimized+ilp` | 4 | ILP4 overlaps 4 Feistel pipelines |

        ---

        ### 2. PRESENT-80

        PRESENT-80 is a 31-round SPN block cipher with an 80-bit key and 64-bit block. Each round
        applies a 4-bit S-box to each nibble (16 S-boxes) then a bit-permutation (pLayer).

        #### gpu0_baseline — Pre-expanded round keys

        Calls `Present80::expand_key(key, rk)` to generate all 32 round keys (each 64-bit), then
        `Present80::encrypt(pt, rk)` which applies the S-box + pLayer per round using the precomputed
        key material. The S-box is applied nibble by nibble (16 operations per round).

        #### gpu1_optimized — Combined SP-box lookup table (on-the-fly key schedule)

        Replaces the separate S-box + pLayer with a **precomputed 8-bit SP-box table** (`PRESENT_SPBOX8`):
        each byte of the 64-bit state maps to a 64-bit value that encodes both the S-box substitution
        and the pLayer bit positions for that byte's 8 bits. One table lookup per byte (8 lookups per
        round) replaces 2 nibble S-box + 64 pLayer bit operations. The key schedule is computed
        **on-the-fly** (`encrypt_spbox_otf_dev`) to avoid register pressure from storing 32 round keys.

        #### gpu2_optimized+ilp — SP-box + ILP4

        Same SP-box on-the-fly path plus **ILP4**: 4 candidate keys per thread per loop.

        #### gpu3_optimized+shared — SP-box table in shared memory + ILP4

        The 8-entry SP-box table (8 × 256 × 8 bytes = 16 KB) is **loaded into shared memory** at
        the start of each thread block with a cooperative load:
        ```
        for idx in [threadIdx.x, 8×256, blockDim.x]:
        spbox_shared[idx] = PRESENT_SPBOX8_DEV[idx >> 8][idx & 0xFF]
        __syncthreads()
        ```
        All 16 table lookups per round then hit L1 shared memory (~4 cycle latency) instead of
        global/L2 memory (~100 cycle latency). ILP4 is included. **Best throughput** for PRESENT-80
        when the table fits in shared memory.

        | Variant | S-box method | Key schedule | Keys/thread/iter |
        |---------|-------------|-------------|-----------------|
        | `gpu0_baseline` | Nibble S-box | Pre-expanded | 1 |
        | `gpu1_optimized` | SP-box (global mem) | On-the-fly | 1 |
        | `gpu2_optimized+ilp` | SP-box (global mem) | On-the-fly | 4 |
        | `gpu3_optimized+shared` | SP-box (shared mem) | On-the-fly | 4 |

        ---

        ### 3. SPECK 64/128

        SPECK 64/128 is a 27-round ARX block cipher with a 128-bit key and 64-bit block. Each round:
        `x = (x >>> 8) + y XOR k; y = (y <<< 3) XOR x`.

        #### gpu0_baseline — Pre-expanded round keys, 1 key/thread

        Calls `Speck64_128::expand_key(key128, rk)` to unroll the 27 round keys into `uint32_t[27]`,
        then `Speck64_128::encrypt(pt, rk)`. The round function is 2 additions + 3 XORs + 2 rotations,
        all register-only — very efficient on GPU.

        #### gpu1_optimized — Same core as baseline

        SPECK's ARX structure has no memory dependencies (no S-box tables, no MixColumns). The
        optimized path is identical to baseline — there is no further algorithmic simplification.

        #### gpu2_optimized+ilp — ILP4

        Each thread evaluates **4 independent candidate keys** per loop, overlapping 4 independent
        27-round ARX pipelines. Since SPECK has no data-dependent memory accesses, ILP is the
        primary lever for throughput improvement.

        | Variant | Keys/thread/iter | Notes |
        |---------|-----------------|-------|
        | `gpu0_baseline` | 1 | Pre-expanded key, tight ARX loop |
        | `gpu1_optimized` | 1 | Same as baseline for SPECK |
        | `gpu2_optimized+ilp` | 4 | ILP4 overlaps 4 ARX pipelines |

        ---

        ### 4. Grain v1

        Grain v1 is an eSTREAM stream cipher with an 80-bit key and 80-bit IV. It uses an LFSR and
        an NFSR with nonlinear output function h(x). Initialization requires 160 clock cycles.

        #### gpu0_baseline — Full process, 1 key/thread

        Calls `GrainV1::process(pt, out, len, key, iv)` which runs full initialization (160 clocks)
        and generates the full keystream, then compares all output bytes against the ciphertext.

        #### gpu1_optimized — Early-exit keystream match

        Calls `GrainV1::match_keystream(key, iv, target, len)` which computes the target keystream
        (`pt XOR ct`) and compares byte-by-byte, **returning false immediately** at the first
        mismatch. For random wrong keys, rejection typically occurs after the very first byte
        (probability ~255/256 of mismatch), saving nearly all of the keystream generation work.

        ```
        target = pt XOR ct        (precomputed once)
        for each key:
        init(key, iv)          // 160 clocks — always runs
        for each keystream byte:
                if generate_byte() != target[i]: return false   // early exit
        return true
        ```

        #### gpu2_optimized+ilp — Early-exit + ILP2

        Same `match_keystream` as optimized plus **ILP2**: 2 keys per thread per loop iteration.

        | Variant | Keystream bytes computed (avg) | Keys/thread/iter |
        |---------|-------------------------------|-----------------|
        | `gpu0_baseline` | All (full message) | 1 |
        | `gpu1_optimized` | ~1 (early exit on mismatch) | 1 |
        | `gpu2_optimized+ilp` | ~1 (early exit on mismatch) | 2 |

        ---

        ### 5. Trivium

        Trivium is an eSTREAM stream cipher based on three interconnected shift registers (93+84+111
        = 288 bits total state). Initialization requires 1152 clock cycles (4× state size).

        #### gpu0_baseline — Full process, 1 key/thread

        Calls `Trivium::process(pt, out, len, key, iv)` — full 1152-clock init + full keystream.
        All output bytes compared after generation.

        #### gpu1_optimized — Early-exit keystream match

        Calls `Trivium::match_keystream(key, iv, target, len)` — same 1152-clock init (unavoidable),
        then byte-by-byte match with **immediate return on first mismatch**.

        #### gpu2_optimized+ilp — Early-exit + ILP2

        `match_keystream` + **ILP2** (2 keys per thread per loop).

        | Variant | Keystream bytes computed (avg) | Keys/thread/iter |
        |---------|-------------------------------|-----------------|
        | `gpu0_baseline` | All (full message) | 1 |
        | `gpu1_optimized` | ~1 (early exit on mismatch) | 1 |
        | `gpu2_optimized+ilp` | ~1 (early exit on mismatch) | 2 |

        > **Note:** Trivium's 1152-clock init dominates cost regardless of variant. Early-exit
        > saves only the keystream comparison work, not the init. ILP2 is the most effective
        > lever for Trivium throughput.

        ---

        ### 6. ChaCha20

        ChaCha20 is a 20-round ARX stream cipher with a 256-bit key, 96-bit nonce, and 32-bit
        counter. One full block produces 64 bytes of keystream from a 4×4 matrix of 32-bit words.

        #### gpu0_baseline — Full 64-byte block, 1 key/thread

        Calls `ChaCha20::process(pt, out, len, key, nonce)` which runs the full 20-round core and
        produces the entire 64-byte keystream block. All bytes compared after generation.

        #### gpu1_optimized — 16-byte prefix match (block_words4)

        Calls `ChaCha20::block_words4(key, counter, nonce, out4)` which runs the same 20-round
        core but **only extracts the first 4 output words (16 bytes)**:
        ```
        out4[0..3] = (state[0..3] + initial_state[0..3]) after 20 rounds
        ```
        The `chacha_match_prefix_words4()` helper then compares up to 16 bytes directly from the
        `uint32_t[4]` words without writing to a byte buffer. **No intermediate memory allocation;
        comparison short-circuits on first mismatch byte.**

        Savings: ~4× fewer store instructions, no byte-buffer materialization, early exit.

        #### gpu2_optimized+ilp — 16-byte prefix + ILP2

        Same `block_words4` path plus **ILP2** (2 keys per thread per loop). The two independent
        ARX pipelines overlap each other's instruction latency.

        | Variant | Bytes computed | Keys/thread/iter |
        |---------|---------------|-----------------|
        | `gpu0_baseline` | 64 (full block) | 1 |
        | `gpu1_optimized` | 16 (first 4 words) | 1 |
        | `gpu2_optimized+ilp` | 16 (first 4 words) | 2 |

        ---

        ### 7. TinyJAMBU-128

        TinyJAMBU-128 is a NIST LWC finalist AEAD cipher based on a 128-bit keystream-based
        permutation. It uses a 128-bit key, 96-bit nonce, and produces a 64-bit tag.

        #### gpu0_baseline — Full AEAD (CT + tag check), 1 key/thread

        Calls `TinyJAMBU128::encrypt(pt, ct_out, len, tag_out, key, nonce, ad, ad_len)` which
        runs the full AEAD: P1024 nonce init + P640 AD processing + P1024 encrypt blocks +
        P640 finalization (tag). Both the ciphertext and tag are compared.

        #### gpu2_optimized+ilp — Full AEAD + ILP2

        Same full AEAD as baseline plus **ILP2** (2 keys per thread per loop).

        #### gpu4_bitsliced — 32-lane bitsliced matching

        Processes **32 candidate keys simultaneously** using `uint32_t` bitsliced words (each bit
        position = one lane):

        ```
        Step 1: Pack 32 sequential keys into key_bitsliced[128]
                key_bitsliced[b] has bit k set iff key[k] has bit b set

        Step 2: TinyJAMBU128_Bitsliced::match_keys(pt, ct, len, tag, key_bs, nonce, ad, ad_len)
                - Runs P1024 init (bitsliced, all 32 lanes in parallel)
                - For each keystream word: compute 32 outputs simultaneously
                - Compare all 32 against expected ct byte
                - If match_mask == 0 (all 32 failed): abort early

        Step 3: If match_mask != 0: report which lane(s) matched
        ```

        **Throughput advantage**: 32 keys per thread per iteration; early exit eliminates tag
        computation for wrong keys, saving the P640 finalization step.

        | Variant | Verification | Keys/thread/iter |
        |---------|-------------|-----------------|
        | `gpu0_baseline` | Full CT + tag | 1 |
        | `gpu2_optimized+ilp` | Full CT + tag | 2 |
        | `gpu4_bitsliced` | Keystream-only (early reject) | 32 |

        ---

        ### 8. ZUC-128

        ZUC-128 is the 3GPP/GSMA stream cipher (EEA3/EIA3) with a 128-bit key and 128-bit IV.
        It uses a 16-element LFSR over GF(2^31-1), two 32-bit memory cells (R1, R2), and two
        nonlinear functions (L1, L2) to generate 32-bit words per clock.

        #### gpu0_baseline — Full ZUC::process, 1 key/thread

        Calls `ZUC::process(pt, out, len, key, iv)` which runs full initialization (state loading
        + 32 dummy rounds) and produces the full keystream, then compares all bytes.

        #### gpu1_optimized — Early-exit keystream match

        Calls `ZUC::match_keystream(key, iv, target, len)` which computes target keystream
        (`pt XOR ct`) and **returns false immediately** at the first byte mismatch. ZUC's
        initialization dominates cost but the early-exit saves keystream comparison for mismatches
        that occur early in the output stream.

        #### gpu2_optimized+ilp — Early-exit + ILP2

        Same `match_keystream` plus **ILP2** (2 keys per thread per loop).

        | Variant | Keystream bytes compared (avg) | Keys/thread/iter |
        |---------|-------------------------------|-----------------|
        | `gpu0_baseline` | All (full message) | 1 |
        | `gpu1_optimized` | ~1–4 (early exit) | 1 |
        | `gpu2_optimized+ilp` | ~1–4 (early exit) | 2 |

        ---

        ### 9. SNOW-V

        SNOW-V is a stream cipher with a 256-bit key and 128-bit IV. It uses a 16-word LFSR over
        GF(2^128) (split into two 8-word LFSRs A and B over GF(2^16)) and an AES-based FSM with
        three 128-bit registers (R1, R2, R3).

        #### gpu0_baseline — SNOW_V::process without T-table (`UseTTable=false`)

        The FSM update uses a direct AES SubBytes + ShiftRows implementation without precomputed
        tables. The MixColumns equivalent in the LFSR update uses GF(2^16) polynomial reduction.

        #### gpu1_optimized — Early-exit + T-table (`SNOW_V::match_keystream<true>`)

        Uses `SNOW_V::match_keystream<true>` which:
        1. Enables the **T-table path** (`UseTTable=true`) — precomputed AES T-tables accelerate
        the FSM's SubBytes/ShiftRows/MixColumns sequence from ~16 S-box lookups per step to
        4 T-table lookups.
        2. Compares keystream bytes against `target = pt XOR ct` with **early exit** on mismatch.

        #### gpu2_optimized+ilp — T-table early-exit + ILP2

        Same `match_keystream<true>` plus **ILP2** (2 keys per thread per loop).

        | Variant | AES FSM path | Keystream match | Keys/thread/iter |
        |---------|-------------|----------------|-----------------|
        | `gpu0_baseline` | No T-table | Full output | 1 |
        | `gpu1_optimized` | T-table (4 lookups/step) | Early exit | 1 |
        | `gpu2_optimized+ilp` | T-table | Early exit | 2 |

        ---

        ### 10. AES-128

        AES-128 is a 10-round SPN block cipher with a 128-bit key and 128-bit block. Each round:
        SubBytes → ShiftRows → MixColumns → AddRoundKey (last round skips MixColumns).

        #### gpu0_baseline — Naive MixColumns (`UseTTable=false`)

        MixColumns is computed using a **general GF(2⁸) multiplication loop**:
        ```
        gmul(a, b):
        result = 0
        for 8 iterations:
        if b & 1: result ^= a
        a = (a << 1) ^ (0x1b if a & 0x80 else 0)
        b >>= 1
        ```
        Each column requires four `gmul` calls. Data-dependent branching and bit-by-bit iteration
        make this slower but algorithmically straightforward.

        #### gpu1_optimized — Fast MixColumns (`UseTTable=true`)

        Replaces the 8-iteration loop with **precomputed GF(2⁸) bit-shift tricks**:
        ```
        gmul2(a) = (a << 1) ^ (0x1b if a & 0x80)   // multiply by 2 in GF(2^8), 2 ops
        gmul3(a) = gmul2(a) ^ a                      // multiply by 3 = 2+1, 3 ops
        ```
        Each MixColumns column becomes 8 XOR/shift operations total — no loop, no data-dependent
        branching. Significantly faster on GPU due to eliminated loop overhead and branch divergence.

        #### gpu2_optimized+ilp — Fast MixColumns + ILP4

        Same fast MixColumns plus **ILP4** (4 keys per thread per loop):
        ```
        for k in [tid, N, stride×4]:
        try_key(base | k)              // AES pipeline 0
        try_key(base | (k+stride))     // AES pipeline 1
        try_key(base | (k+2×stride))   // AES pipeline 2
        try_key(base | (k+3×stride))   // AES pipeline 3
        ```
        The 4 independent AES pipelines overlap each other's S-box and T-table load latencies,
        increasing arithmetic throughput utilization.

        | Variant | MixColumns | Keys/thread/iter | Speedup vs baseline |
        |---------|-----------|-----------------|---------------------|
        | `gpu0_baseline` | Naive (8-iter GF loop) | 1 | 1× |
        | `gpu1_optimized` | Fast (gmul2/gmul3) | 1 | ~2–4× |
        | `gpu2_optimized+ilp` | Fast (gmul2/gmul3) | 4 | Highest |

        ---

        ### 11. Salsa20

        Salsa20 is a 20-round ARX stream cipher with a 256-bit key, 64-bit nonce, and 64-bit
        counter. One block produces 64 bytes from a 4×4 matrix of 32-bit words.

        State layout:
        ```
        σ0   k0   k1   k2      σ0 = 0x61707865  "expa"
        k3   σ1   n0   n1      σ1 = 0x3320646e  "nd 3"
        t0   t1   σ2   k4      σ2 = 0x79622d32  "2-by"
        k5   k6   k7   σ3      σ3 = 0x6b206574  "te k"
        ```

        Quarter-round: `b ^= (a+d)<<<7; c ^= (b+a)<<<9; d ^= (c+b)<<<13; a ^= (d+c)<<<18`

        #### gpu0_baseline — Full 64-byte block, 1 key/thread

        Calls `Salsa20::process(pt, out, len, key, nonce)` which runs 20 rounds and produces
        the full 64-byte keystream block, then compares all bytes.

        #### gpu1_optimized — 16-byte prefix match (block_words4)

        Calls `Salsa20::block_words4(key, counter, nonce, out4)` which runs the full 20-round
        core but **extracts only the first 4 output words (16 bytes)**:
        ```
        out4[i] = (final_state[i] + initial_state[i]) for i in [0..3]
        ```
        `salsa_match_prefix_words4()` compares directly from the `uint32_t[4]` without
        materializing a byte buffer, with early exit on first mismatch byte.

        Savings: ~4× fewer stores, no 48-byte tail computation, early exit.

        #### gpu2_optimized+ilp — 16-byte prefix + ILP2

        Same `block_words4` plus **ILP2** (2 keys per thread per loop).

        | Variant | Bytes computed | Keys/thread/iter |
        |---------|---------------|-----------------|
        | `gpu0_baseline` | 64 (full block) | 1 |
        | `gpu1_optimized` | 16 (first 4 words) | 1 |
        | `gpu2_optimized+ilp` | 16 (first 4 words) | 2 |

        ---

        ### 12. Grain-128AEADv2

        Grain-128AEADv2 is a NIST LWC finalist AEAD cipher with a 128-bit key, 96-bit nonce,
        and 64-bit authentication tag. It uses:
        - **LFSR (128 bits)**: feedback polynomial `x^128 + x^7 + x^38 + x^70 + x^81 + x^96 + 1`
        - **NFSR (128 bits)**: nonlinear feedback with product terms up to 4-way AND
        - **Pre-output h(x)**: nonlinear function of 5 selected LFSR/NFSR taps
        - **Output**: `y_t = h(x) XOR LFSR[93] XOR NFSR[2,15,36,45,64,73,89]`

        **Initialization**: 512 clock cycles:
        1. 320 clocks — key+nonce fed back
        2. 64 clocks — upper key → LFSR, lower key → NFSR
        3. 64 clocks — fill auth_acc (MAC accumulator)
        4. 64 clocks — fill auth_sr (MAC shift register)

        **Encryption**: alternating outputs — even clocks (z_i) XOR with plaintext; odd clocks (z′)
        update the MAC.

        **Authentication**: AD processed through MAC with DER-encoded length prefix before ciphertext.
        64-bit tag extracted from auth_acc at completion.

        #### gpu0_baseline — Full AEAD with CT + tag verification, 1 key/thread

        Calls `Grain128AEADv2::process(pt, ct_out, len, ad, ad_len, tag_out, key, nonce)` which
        runs the complete AEAD (512-clock init + AD + encrypt + tag generation). Both ciphertext
        and tag are compared.

        #### gpu2_optimized+ilp — Full AEAD + ILP2

        Same full `process()` as baseline plus **ILP2** (2 keys per thread per loop). The two
        independent LFSR/NFSR register chains overlap their long sequential clock dependencies.

        #### gpu4_bitsliced — 32-lane bitsliced matching

        Processes **32 candidate keys simultaneously** using `uint32_t` bitsliced words:

        ```
        Step 1: Pack 32 sequential keys into key_bitsliced[128]
                key_bitsliced[b] bit k = bit b of key[k]

        Step 2: Grain128AEADv2_Bitsliced::match_keys(...)
                - All 512+128 = 640 init clocks run bitsliced (32 lanes in parallel)
                - Fast-forward past 128 MAC init clocks + AD clocks
                - For each plaintext bit:
                z_i_mask = clock_all_32_lanes()
                compare against target keystream bit
                match_mask &= (z_i_mask XNOR target_bit_mask)
                if match_mask == 0: all 32 lanes failed → exit

        Step 3: If match_mask != 0: report matching lane(s)
        ```

        **Key optimization**: skips full tag generation — performs keystream-only matching.
        Most wrong keys are rejected after the first 1–2 keystream bytes (8–16 clock pairs),
        saving nearly all of the 512-clock init amortization and all tag computation.

        | Variant | Verification | Keys/thread/iter | Notes |
        |---------|-------------|-----------------|-------|
        | `gpu0_baseline` | Full CT + 64-bit tag | 1 | Complete AEAD per key |
        | `gpu2_optimized+ilp` | Full CT + 64-bit tag | 2 | ILP2 overlaps two AEAD pipelines |
        | `gpu4_bitsliced` | Keystream-only (early reject) | 32 | Best throughput; skips tag |

        ---

        ## CPU Optimizations

        The CPU reference implementation uses the same cipher structs from `ciphers_enhanced.cuh`.
        Key optimizations per cipher category:

        ### Block Ciphers (SIMON, PRESENT, SPECK, AES)

        | Cipher | CPU Optimization |
        |--------|----------------|
        | **SIMON 32/64** | `expand_key_vectorized` packs 4 rounds per `uint64_t` op; precomputed round keys reused across all candidate evaluations at same bit level |
        | **PRESENT-80** | `expand_key` precomputes 32 round keys once; `encrypt` with precomputed keys; no repeated key schedule per candidate |
        | **SPECK 64/128** | `expand_key` precomputes 27 round keys; pure ARX loop — no memory lookups |
        | **AES-128** | `encrypt<true>` uses fast gmul2/gmul3 MixColumns (same as GPU optimized path); no T-table on CPU to avoid L1 thrash |

        ### Stream Ciphers (Grain v1, Trivium, ChaCha20, Salsa20, ZUC, SNOW-V)

        All stream cipher CPU implementations use **early-exit keystream matching**:

        1. Precompute `target[i] = pt[i] XOR ct[i]` once before the key sweep
        2. For each candidate key: init cipher state, then compare keystream byte-by-byte
        3. **Return false immediately** on the first mismatching byte

        This avoids wasting time generating the full keystream for wrong keys. For a 16-byte
        message and random keys, the expected number of bytes compared per wrong key is:
        ```
        E[bytes] = sum_{i=1}^{16} (i × (255/256)^(i-1) × (1/256)) + 16 × (255/256)^16 ≈ 1.004
        ```
        i.e., on average **~1 byte** is compared per wrong key (regardless of message length).

        Additional per-cipher CPU notes:

        | Cipher | Extra CPU optimization |
        |--------|----------------------|
        | **ChaCha20** | `block_words4` computes only first 16 bytes; avoids 48-byte tail |
        | **Salsa20** | `block_words4` computes only first 16 bytes; avoids 48-byte tail |
        | **SNOW-V** | `match_keystream<true>` enables T-table path for faster AES FSM |
        | **Grain v1** | IV copied to local stack before loop to avoid global pointer chase per key |
        | **Trivium** | IV copied to local stack; 1152-clock init unavoidable |
        | **ZUC-128** | IV copied to local stack; match_keystream for early exit |

        ### AEAD Ciphers (TinyJAMBU-128, Grain-128AEADv2)

        Both AEAD cipher CPU implementations also use **early-exit keystream matching** via
        `match_keystream()`:

        1. Run full cipher initialization (unavoidable)
        2. Compare keystream bytes against `pt XOR ct` byte-by-byte
        3. **Skip tag computation entirely** for wrong keys — return false on first byte mismatch
        4. Only compute and verify the tag when all plaintext bytes match (rare: only the correct key)

        This avoids the expensive tag generation (P640 finalization for TinyJAMBU; 64 additional
        MAC clock cycles for Grain-128AEADv2) for the vast majority of candidates.

        ---

        ## Repository Layout

        ```
        lightweight_ciphers_project_GPU_bruteforce/
        ├── CMakeLists.txt
        ├── README.md
        ├── plot_results.py                   # Visualization script (per-cipher + summary plots)
        ├── src/
        │   ├── main.cu                       # Benchmark driver, self-tests, CLI parsing
        │   ├── ciphers_enhanced.cuh          # All 12 cipher implementations (host + device)
        │   ├── bruteforce_gpu_enhanced.cuh   # GPU kernels, CipherType/GpuVariant enums, launch helpers
        │   ├── bruteforce_cpu.hpp            # CPU brute-force reference implementations
        │   ├── present_spbox_tables.inc      # Pre-generated PRESENT-80 SP-box lookup tables
        │   └── util.hpp                      # CUDA error checking, CSV helpers, hex formatting
        └── results_all.csv                   # Full benchmark results (generated by running bench)
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

        ### Build commands

        ```bash
        cd lightweight_ciphers_project_GPU_bruteforce
        mkdir -p build && cd build
        cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 ..
        cmake --build . -j$(nproc)
        # Binary: build/bench
        ```

        ---

        ## Self-Tests

        Verifies all 12 cipher implementations against official test vectors before any benchmarking.

        ```bash
        ./bench --test
        ```

        Expected output (14 tests):
        ```
        SIMON32/64 Self-Test:                  PASS
        PRESENT-80 Self-Test:                  PASS
        SPECK64/128 Self-Test:                 PASS
        Grain v1 Self-Test:                    PASS
        Trivium Self-Test:                     PASS
        ChaCha20 Block Self-Test:              PASS
        TinyJAMBU-128 Self-Test:               PASS
        TinyJAMBU-128 Bitsliced Self-Test:     PASS
        ZUC-128 Self-Test:                     PASS
        SNOW-V Self-Test:                      PASS
        AES-128 Self-Test:                     PASS
        Salsa20 Self-Test:                     PASS
        Grain-128AEADv2 Self-Test:             PASS
        Grain-128AEADv2 Bitsliced Test:        PASS
        ```

        ### Test Vectors Used

        | Cipher | Source |
        |--------|--------|
        | SIMON 32/64 | NSA SIMON/SPECK specification |
        | PRESENT-80 | ISO/IEC 29192-2 |
        | SPECK 64/128 | NSA SIMON/SPECK specification |
        | Grain v1 | eSTREAM project test vectors |
        | Trivium | eSTREAM project test vectors |
        | ChaCha20 | RFC 8439 §2.4.2 |
        | TinyJAMBU-128 | NIST LWC TinyJAMBU v2 specification |
        | ZUC-128 | 3GPP TS 35.223 |
        | SNOW-V | SNOW-V design document |
        | AES-128 | NIST FIPS 197 Appendix B |
        | Salsa20 | eSTREAM Set 1 vector 1 |
        | Grain-128AEADv2 | NIST LWC Grain-128AEAD v2 specification |

        ### Grain-128AEADv2 NIST Test Vector

        - Key: `000102030405060708090a0b0c0d0e0f`
        - Nonce: `000102030405060708090a0b`
        - AD: `0001020304050607`
        - PT: `0001020304050607`
        - CT: `96d1bda7ae11f0ba`
        - Tag: `22b0c12039a20e28`

        ---

        ## Running Experiments

        All commands below assume you are inside the `build/` directory.

        ### Run all 12 ciphers (recommended full benchmark)

        ```bash
        ./bench --cipher all \
                --min_bits 1 --max_bits 30 --step_bits 1 \
                --cpu_repeats 3 --gpu_repeats 10 \
                --blocks 1024 --threads 256 \
                --out ../results_all.csv
        ```

        ### GPU only (skip CPU baselines)

        ```bash
        ./bench --cipher all --gpu_only \
                --min_bits 1 --max_bits 20 --step_bits 1 \
                --out ../results_gpu_only.csv
        ```

        ### CPU only

        ```bash
        ./bench --cipher all --cpu_only \
                --min_bits 1 --max_bits 20 --step_bits 1 \
                --out ../results_cpu_only.csv
        ```

        ---

        ### Individual Cipher Commands

        #### 1. SIMON 32/64

        ```bash
        ./bench --cipher simon --min_bits 1 --max_bits 30 --step_bits 1 --out ../results_simon.csv
        ```

        Run specific variants:
        ```bash
        # ILP4 only
        ./bench --cipher simon --variants optimized_ilp --min_bits 1 --max_bits 30 --out ../results_simon_ilp.csv
        ```

        #### 2. PRESENT-80

        ```bash
        ./bench --cipher present --variants all \
                --min_bits 1 --max_bits 30 --step_bits 1 \
                --out ../results_present.csv
        ```

        Run only the shared-memory variant (fastest):
        ```bash
        ./bench --cipher present --variants shared --min_bits 1 --max_bits 30 --out ../results_present_shared.csv
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

        Run specific variants:
        ```bash
        # Baseline (full 64-byte block)
        ./bench --cipher chacha --variants baseline --min_bits 1 --max_bits 20 --out ../results_chacha_baseline.csv

        # Optimized only (16-byte prefix match)
        ./bench --cipher chacha --variants optimized --min_bits 1 --max_bits 20 --out ../results_chacha_opt.csv

        # ILP2 + prefix match
        ./bench --cipher chacha --variants optimized_ilp --min_bits 1 --max_bits 20 --out ../results_chacha_ilp.csv
        ```

        #### 7. TinyJAMBU-128

        ```bash
        ./bench --cipher tinyjambu --min_bits 1 --max_bits 20 --step_bits 1 --out ../results_tinyjambu.csv
        ```

        Run only the bitsliced kernel (fastest):
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

        Run specific variants (T-table effect):
        ```bash
        # Baseline (no T-table)
        ./bench --cipher snowv --variants baseline --min_bits 1 --max_bits 20 --out ../results_snowv_baseline.csv

        # Optimized (T-table + early exit)
        ./bench --cipher snowv --variants optimized --min_bits 1 --max_bits 20 --out ../results_snowv_opt.csv
        ```

        #### 10. AES-128

        ```bash
        ./bench --cipher aes --min_bits 1 --max_bits 30 --step_bits 1 --out ../results_aes.csv
        ```

        Run specific variants:
        ```bash
        # Baseline (naive GF MixColumns)
        ./bench --cipher aes --variants baseline --min_bits 1 --max_bits 30 --out ../results_aes_baseline.csv

        # Optimized (fast gmul2/gmul3 MixColumns)
        ./bench --cipher aes --variants optimized --min_bits 1 --max_bits 30 --out ../results_aes_opt.csv

        # ILP4 (fast MixColumns + 4 keys/thread)
        ./bench --cipher aes --variants optimized_ilp --min_bits 1 --max_bits 30 --out ../results_aes_ilp.csv
        ```

        #### 11. Salsa20

        ```bash
        ./bench --cipher salsa --min_bits 1 --max_bits 30 --step_bits 1 --out ../results_salsa20.csv
        ```

        Run specific variants:
        ```bash
        # Baseline (full 64-byte block)
        ./bench --cipher salsa --variants baseline --min_bits 1 --max_bits 30 --out ../results_salsa_baseline.csv

        # Optimized (16-byte prefix only)
        ./bench --cipher salsa --variants optimized --min_bits 1 --max_bits 30 --out ../results_salsa_opt.csv

        # ILP2 + prefix
        ./bench --cipher salsa --variants optimized_ilp --min_bits 1 --max_bits 30 --out ../results_salsa_ilp.csv
        ```

        #### 12. Grain-128AEADv2

        ```bash
        ./bench --cipher grain128 --min_bits 1 --max_bits 20 --step_bits 1 --out ../results_grain128.csv
        ```

        Run specific variants:
        ```bash
        # Bitsliced only (fastest — 32 keys/thread + early reject)
        ./bench --cipher grain128 --variants bitsliced \
                --min_bits 1 --max_bits 20 \
                --out ../results_grain128_bs.csv

        # Baseline only (full AEAD, useful for correctness study)
        ./bench --cipher grain128 --variants baseline \
                --min_bits 1 --max_bits 16 \
                --out ../results_grain128_baseline.csv

        # All variants
        ./bench --cipher grain128 --variants all \
                --min_bits 1 --max_bits 16 \
                --out ../results_grain128_all.csv
        ```

        ---

        ### Selecting GPU Variants Manually

        ```bash
        # Baseline only
        ./bench --cipher grain --variants baseline

        # Optimized (early-exit / fast MixColumns / T-table)
        ./bench --cipher trivium --variants optimized

        # ILP (2–4 keys per thread per loop)
        ./bench --cipher chacha --variants optimized_ilp

        # Shared-memory SP-box (PRESENT-80 only)
        ./bench --cipher present --variants shared

        # Bitsliced 32-lane (TinyJAMBU-128 or Grain-128AEADv2 only)
        ./bench --cipher tinyjambu --variants bitsliced
        ./bench --cipher grain128 --variants bitsliced

        # All variants for a cipher
        ./bench --cipher present --variants all
        ./bench --cipher aes --variants all
        ```

        ---

        ## Randomized Correctness Verification

        Run `--verify` to execute a randomized correctness test for all 12 ciphers without benchmarking:

        ```bash
        ./bench --verify
        ```

        Increase the number of unknown bits for a more thorough search:
        ```bash
        ./bench --verify --verify_bits 4    # search space = 2^4 = 16 candidates per key
        ./bench --verify --verify_bits 10   # search space = 2^10 = 1024 candidates per key
        ```

        ### What --verify does

        For each of the 12 ciphers:

        1. **5 random keys** are generated (seeded with `srand(42)` for reproducibility).
        2. Each key encrypts a fixed plaintext (PT1) to produce ciphertext (CT1).
        3. A CPU brute-force with `unknown_bits=verify_bits` is run — the correct key is always
        the last candidate (lowest bits all set to 1) to guarantee worst-case coverage.
        4. The found key is verified against CT1.
        5. **Second PT/CT pair check**: the same found key encrypts PT2 → verifies output matches CT2.
        This eliminates false positives, especially important for SIMON 32/64 (32-bit block).

        ### Failure-case reporting

        If any cipher fails (wrong key, false positive, or no key found), the output lists:
        - Which cipher failed
        - Trial number and what was expected vs what was found
        - Whether the second PT/CT pair check failed

        Example output (all pass):
        ```
        === Randomized Key Recovery Verification ===
        5 random keys per cipher, unknown_bits=1, CPU brute-force
        Each found key is also verified against a second PT/CT pair.

        SIMON 32/64          PASS
        PRESENT-80           PASS
        SPECK 64/128         PASS
        Grain v1             PASS
        Trivium              PASS
        ChaCha20             PASS
        TinyJAMBU-128        PASS
        ZUC-128              PASS
        SNOW-V               PASS
        AES-128              PASS
        Salsa20              PASS
        Grain-128AEADv2      PASS

        Results: 60/60 passed — no failure cases detected.
        ```

        Example output (with a failure):
        ```
        SIMON 32/64          FAIL
        FAIL  SIMON32/64 #2: found=no [second PT/CT mismatch]

        --- Verification FAILED ---
        SIMON32/64 #2 failed.
        ```

        ---

        ## CLI Reference

        ```
        Usage: bench [options]
        --cipher <name|all>
        simon | present | speck | grain | trivium | chacha |
        tinyjambu | zuc | snowv | aes | salsa | grain128 | all
        (default: all)

        --variants <mode>
        baseline        — 1 key/thread, full encrypt
        optimized       — cipher-specific fast path (early-exit, T-table, gmul2)
        optimized_ilp   — optimized + ILP (2–4 keys per thread per loop)
        shared          — shared-memory SP-box (PRESENT-80 only)
        bitsliced       — 32-lane bitsliced (TinyJAMBU-128, Grain-128AEADv2 only;
                                falls back to baseline for other ciphers)
        all             — all applicable variants for the cipher
        auto            — best variants per cipher (default)

        --out <path>            Output CSV file (default: results.csv)
        --min_bits <N>          Minimum unknown bits to sweep (default: 1)
        --max_bits <N>          Maximum unknown bits to sweep (default: 30)
        --step_bits <N>         Step size for bit sweep (default: 1)
        --cpu_repeats <N>       CPU timing repetitions (default: 3)
        --gpu_repeats <N>       GPU timing repetitions (default: 10)
        --blocks <N>            CUDA grid blocks (default: 1024)
        --threads <N>           CUDA threads per block (default: 256)
        --cpu_only              Run CPU benchmarks only
        --gpu_only              Run GPU benchmarks only
        --test                  Run self-tests only (no benchmarks)
        --verify                Randomized correctness verification (no benchmarks)
        --verify_bits <N>       Unknown bits for --verify mode (default: 1)

        # Bonus Advanced Extensions
        --multi_gpu             Multi-GPU scaling benchmark (SIMON 32/64, all available GPUs)
        --key_strategy <mode>   Key bit selection strategy benchmark (SIMON 32/64)
        low         — sweep bits [0..b-1] (default behaviour)
        high        — sweep bits [64-b..63]
        interleaved — every other bit position: 0,2,4,...
        random      — b randomly chosen bit positions (seeded, reproducible)
        ```

        ---

        ## CSV Output Format

        Each row in the output CSV:

        ```
        cipher, platform, variant, unknown_bits, keys_tested, seconds, keys_per_second, found_key_hex
        ```

        - `cipher` — cipher name (e.g. `simon32_64`, `grain128aeadv2`)
        - `platform` — `cpu` or `gpu`
        - `variant` — e.g. `cpu_baseline`, `gpu0_baseline`, `gpu4_bitsliced`
        - `unknown_bits` — N (search space = 2^N keys)
        - `keys_tested` — actual number of keys evaluated in this run
        - `seconds` — wall-clock time for the sweep
        - `keys_per_second` — throughput
        - `found_key_hex` — non-zero only when the correct key falls within the search space

        Example rows:
        ```
        simon32_64,cpu,cpu_baseline,10,1024,0.000012,85333333,0x0000000000000000
        simon32_64,gpu,gpu0_baseline,10,1024,0.0000031,330322580,0x0000000000000000
        simon32_64,gpu,gpu2_optimized+ilp,10,1024,0.0000019,539000000,0x0000000000000000
        present80,gpu,gpu3_optimized+shared,10,1024,0.0000022,465454545,0x0000000000000000
        aes128,cpu,cpu_baseline,10,1024,0.000015,68266666,0x0000000000000000
        aes128,gpu,gpu0_baseline,10,1024,0.0000058,176551724,0x0000000000000000
        aes128,gpu,gpu1_optimized,10,1024,0.0000021,487619047,0x0000000000000000
        aes128,gpu,gpu2_optimized+ilp,10,1024,0.0000014,731428571,0x0000000000000000
        salsa20,gpu,gpu0_baseline,10,1024,0.0000049,208979591,0x0000000000000000
        salsa20,gpu,gpu1_optimized,10,1024,0.0000018,568888888,0x0000000000000000
        salsa20,gpu,gpu2_optimized+ilp,10,1024,0.0000012,853333333,0x0000000000000000
        tinyjambu_128,gpu,gpu4_bitsliced,10,1024,0.0000045,227555555,0xa55a12343fffffff
        grain128aeadv2,cpu,cpu_baseline,10,1024,0.00025,4096000,0x0000000000000000
        grain128aeadv2,gpu,gpu0_baseline,10,1024,0.000031,33032258,0x0000000000000000
        grain128aeadv2,gpu,gpu2_optimized+ilp,10,1024,0.000018,56888888,0x0000000000000000
        grain128aeadv2,gpu,gpu4_bitsliced,10,1024,0.0000052,196923076,0xa55a12343fffffff
        ```

        ---

        ## Plotting Results

        ```bash
        cd ..   # back to lightweight_ciphers_project_GPU_bruteforce/

        # Plot all ciphers from a CSV
        python3 plot_results.py results_all.csv --outdir plots/

        # Plot a single cipher
        python3 plot_results.py results_all.csv --only_cipher grain128aeadv2 --outdir plots/

        # Specify CSV with flag
        python3 plot_results.py --csv results_all.csv --outdir plots/
        ```

        ### Output files

        **Per-cipher plots** (one pair per cipher):
        - `<cipher>__time_vs_bits.png` — wall-clock time vs unknown_bits per variant (log-scale y)
        - `<cipher>__throughput_vs_bits.png` — throughput (keys/s) vs unknown_bits per variant (log-scale y)

        **Summary plots** (generated when plotting all ciphers):
        - `summary__best_gpu_throughput.png` — bar chart: best GPU keys/s for every cipher at max bits
        - `summary__cpu_vs_gpu_speedup.png` — bar chart: GPU speedup factor over CPU per cipher
        - `summary__all_gpu_throughput.png` — all ciphers on one axes: best GPU throughput vs bits
        - `summary__all_gpu_time.png` — all ciphers on one axes: best GPU attack time vs bits

        ---

        ## Bonus: Multi-GPU Scaling

        The `--multi_gpu` flag runs a parallel brute-force search for SIMON 32/64 using all
        available CUDA GPUs simultaneously. The key space `[0, 2^b)` is partitioned evenly:
        GPU g handles the range `[g × ⌈2^b/N⌉, (g+1) × ⌈2^b/N⌉)`. Each GPU runs on its own
        CUDA stream in a dedicated `std::thread`; results are merged after all threads join.

        ### How it works

        ```
        Key space [0, 2^b)
        ├── GPU 0: [0,       2^b/2)   ILP2 kernel, CUDA stream, std::thread
        └── GPU 1: [2^b/2,  2^b)     ILP2 kernel, CUDA stream, std::thread
                ↓ join all threads ↓
        Combined throughput ≈ sum(per-GPU keys/s)
        ```

        ### Example output (RTX A6000 + RTX A5000)

        ```
        unknown_bits=28 (keys=268435456)
        single_gpu (GPU 0):  0.01891 s, 1.42e10 keys/s, found=yes
        multi_gpu  (2 GPUs): combined 2.53e10 keys/s, speedup=1.79x, found=yes
        GPU 0: 1.48e10 keys/s  (RTX A6000, 84 SMs)
        GPU 1: 1.05e10 keys/s  (RTX A5000, 64 SMs)
        ```

        Observed speedup: **~1.7–2.0×** with two heterogeneous GPUs (A6000 + A5000). Perfect 2×
        scaling is not achieved because the A5000 is slower; with identical GPUs exact 2× scaling
        is expected.

        ### Running the multi-GPU benchmark

        ```bash
        # SIMON 32/64, 20–30 unknown bits, both GPUs
        ./bench --multi_gpu --min_bits 20 --max_bits 30 --step_bits 2 \
                --gpu_repeats 5 --out results_multigpu.csv

        # Quick check (8–16 bits)
        ./bench --multi_gpu --min_bits 8 --max_bits 16 --gpu_only
        ```

        CSV output variant labels:
        - `multigpu_single` — single GPU (GPU 0) baseline
        - `multigpu_2x` — combined 2-GPU run

        ---

        ## Bonus: Key Bit Selection Strategies

        The `--key_strategy` flag benchmarks four different strategies for choosing *which* b bits
        of the 64-bit key to treat as unknown and brute-force. Each strategy produces a different
        **bit mask** of the key and uses PDEP (bit-deposit) to map sweep index → key candidate.

        ### PDEP key construction

        ```
        candidate_key = fixed_bits | pdep64(sweep_index, mask)

        pdep64(v, mask):  deposit bits of v into positions where mask has 1s
        e.g. pdep64(0b11, 0b1010) -> 0b1010   (bit0 of v -> bit1, bit1 of v -> bit3)
        ```

        For the default LOW_BITS strategy `mask = (1<<b)-1`, `pdep64(i, mask) = i` — no extra
        cost, identical to the standard `base_key | i` kernel.

        ### The four strategies

        | Strategy | Bit positions attacked | Mask example (b=4) |
        |---|---|---|
        | `low` | Lowest b bits: [0..b-1] | `0x000000000000000F` |
        | `high` | Highest b bits: [64-b..63] | `0xF000000000000000` |
        | `interleaved` | Every other bit: 0, 2, 4, … | `0x0000000000000055` |
        | `random` | b randomly chosen positions (seed=42) | `0x4041220000420000` |

        ### When strategies matter

        In a real attack, the analyst chooses the strategy that aligns with what is actually
        unknown. Examples:
        - **Low bits**: manufacturer reset sets upper key bytes to a known pattern
        - **High bits**: key derivation fixes lower bytes; entropy is in the upper half
        - **Interleaved**: alternating bytes come from two independent sources
        - **Random**: no structure assumed; demonstrates that all strategies find the key at the
        same throughput (the PDEP overhead is negligible — CUDA compiles it to a compact loop)

        ### Example output (b=20, SIMON 32/64)

        ```
        unknown_bits=20 (keys=1048576)
        strategy_low_bits:    0.000105 s, 9.98e9 keys/s, mask=0x00000000000FFFFF, found=yes
        strategy_high_bits:   0.000104 s, 1.01e10 keys/s, mask=0xFFFFF00000000000, found=yes
        strategy_interleaved: 0.000106 s, 9.88e9 keys/s, mask=0x0000000055555555, found=yes
        strategy_random_bits: 0.000105 s, 9.98e9 keys/s, mask=0x5041220000420000, found=yes
        ```

        All four strategies achieve identical throughput — the PDEP loop is resolved at compile
        time for fixed b and adds no measurable overhead vs the default low-bits kernel.

        ### Running the key strategy benchmark

        ```bash
        # All 4 strategies, bits 1–20, SIMON 32/64
        ./bench --key_strategy low --min_bits 1 --max_bits 20 --step_bits 1 \
                --gpu_only --out results_strategies.csv

        # Quick demo (4 bits)
        ./bench --key_strategy low --min_bits 4 --max_bits 4 --gpu_only
        ```

        Note: `--key_strategy` always benchmarks all 4 strategies in one run regardless of which
        value is passed; the argument currently selects the primary strategy (future extension for
        single-strategy mode).

        CSV output variant labels: `strategy_low_bits`, `strategy_high_bits`,
        `strategy_interleaved`, `strategy_random_bits`

        ---

        ## Tuning Tips

        | Situation | Suggested action |
        |-----------|----------------|
        | Low GPU occupancy / slow throughput | Increase `--blocks` (try 2048 or 4096) |
        | Faster benchmark runs | Reduce `--max_bits` and `--gpu_repeats` |
        | Compare GPU variants only | `--gpu_only` |
        | Study PRESENT-80 shared-memory effect | `--cipher present --variants all` |
        | Study TinyJAMBU-128 bitsliced effect | `--cipher tinyjambu --variants all --max_bits 16` |
        | Study Grain-128AEADv2 bitsliced effect | `--cipher grain128 --variants all --max_bits 16` |
        | Study AES-128 MixColumns optimization | `--cipher aes --variants all` |
        | Study SNOW-V T-table effect | `--cipher snowv --variants all` |
        | Compare ARX stream ciphers | `--cipher chacha` then `--cipher salsa` |
        | Compare eSTREAM portfolio | `--cipher grain` then `--cipher trivium` |
        | Compare AEAD ciphers | `--cipher tinyjambu` then `--cipher grain128` |
        | Compare Grain variants | `--cipher grain` then `--cipher grain128` |
        | Verify correctness after code changes | `./bench --verify --verify_bits 4` |
        | Multi-GPU scaling study | `--multi_gpu --min_bits 20 --max_bits 30 --step_bits 2` |
        | Key strategy comparison | `--key_strategy low --min_bits 1 --max_bits 20 --gpu_only` |
        | Random strategy reproducibility check | `--key_strategy random --min_bits 10 --max_bits 10` (run twice, same mask) |
