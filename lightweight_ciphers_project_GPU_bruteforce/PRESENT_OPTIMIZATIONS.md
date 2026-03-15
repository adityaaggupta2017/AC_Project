# PRESENT-80 optimization notes

The updated PRESENT path uses three levels:

## 1. Baseline
- scalar `sBoxLayer`
- scalar `pLayer_opt`
- full round-key expansion into `rk[32]`

## 2. Optimized
- **byte SP-box table**
  - each round is reduced to 8 table lookups on the post-addRoundKey state bytes
  - each table entry already contains the combined result of S-box + pLayer contribution for one input byte position
- **on-the-fly key schedule**
  - avoids storing 32 round keys per candidate key in the GPU thread
  - reduces local memory / register pressure versus expand-then-encrypt

## 3. Optimized + shared
- the same byte SP-box table is copied from constant memory to shared memory once per block
- useful as a comparison variant
- on many GPUs this may or may not beat constant-memory lookup because 64-bit shared-memory table accesses can still experience bank conflicts

## Why this is better than the old scalar PRESENT path
The old path still performed the bit permutation in scalar software every round, which is one of the slowest parts of PRESENT in software. The updated path replaces the round transform with table composition and removes the separate round-key array from the hot path.

## What this is **not**
This is **not** a full bit-sliced PRESENT implementation. Full bit-slicing is a stronger optimization direction, but it is a larger rewrite because multiple blocks must be packed and processed together per thread/warp.
