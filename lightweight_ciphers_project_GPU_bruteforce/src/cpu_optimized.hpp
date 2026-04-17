#pragma once
/**
 * cpu_optimized.hpp  —  Literature-derived CPU optimizations for brute-force search.
 *
 * Technique 1 — TriviumWP: packed-integer (word-parallel) representation for Trivium.
 *   State stored as 3+3+4 uint32_t (A/B/C registers). Eliminates byte-array overhead
 *   (288-byte state fits in 10 registers; no memmove, no modular index arithmetic).
 *   Shift is 3-5 integer instructions vs 288 byte copies. All state in GP registers.
 *   Ref: De Cannière & Preneel, "Trivium Specifications," eSTREAM Phase 3, 2007;
 *        SUPERCOP word-parallel convention; tap-gap parallelism analysis.
 *
 * Technique 2 — GrainV1WP: packed-integer (word-parallel) Grain v1.
 *   LFSR/NFSR stored as uint32_t[3] each (80 bits). Shift is 3 shifts + 2 ORs vs
 *   80 byte copies. All state in 6 GP registers — zero cache pressure.
 *   Ref: Hell, Johansson & Meier, "Grain: a stream cipher for constrained environments,"
 *        IJWMC 2007; SUPERCOP word-parallel optimised code.
 *
 * Technique 3 — Grain128WP: packed-integer Grain-128AEADv2.
 *   LFSR/NFSR stored as uint32_t[4] each (128 bits). Eliminates 127-byte memmove
 *   per clock step × 512 init clocks = 130K byte copies/key eliminated.
 *   Ref: Maximov & Hell, ePrint 2020/659; NIST LWC submission optimized C.
 *
 * Technique 4 — AES128NI: AES-NI hardware round instructions for AES-128 brute-force.
 *   Single AESENC instruction per AES round vs. ~12 software ops (SubBytes+ShiftRows+MixCols).
 *   ILP4: 4 independent AES-NI pipelines overlap 4-cycle AESENC latency on AMD Zen 2.
 *   Ref: Gueron & Krasnov, "Making AES Great Again," ePrint 2018/392.
 *
 * Technique 5 — ILP2: 2 independent keys per loop iteration for all other ciphers.
 *   Out-of-order CPU executes two independent pipelines simultaneously.
 *   Ref: Koo et al., "Parallel Implementations of SIMON and SPECK, Revisited," 2017;
 *        Goll & Gueron, "Vectorization of ChaCha Stream Cipher," ePrint 2013/759.
 */

#include "ciphers_enhanced.cuh"
#include "bruteforce_cpu.hpp"

// AES-NI: only available on host, not device.
// Compile with -maes -msse4.1 (set in CMakeLists).
#if !defined(__CUDA_ARCH__) && defined(__AES__) && defined(__SSE4_1__)
#  include <wmmintrin.h>
#  include <smmintrin.h>
#  define CPU_OPT_HAVE_AESNI 1
#endif

// ============================================================
// Technique 1: TriviumWP — packed-integer (word-parallel) Trivium
// ============================================================
// State: A (93 bits), B (84 bits), C (111 bits) each stored as packed uint32_t.
//   aw[3]: A[0..92] packed as aw[0]=bits0..31, aw[1]=32..63, aw[2]=64..92 (bits0..28)
//   bw[3]: B[0..83] packed as bw[0]=bits0..31, bw[1]=32..63, bw[2]=64..83 (bits0..19)
//   cw[4]: C[0..110] packed as cw[0..2]=bits0..95, cw[3]=96..110 (bits0..14)
// Bit j of register: (S[j>>5] >> (j&31)) & 1
// Shift (discard bit0, insert new_bit at end):
//   shift93:  aw[0..1] cascade, aw[2]=(aw[2]>>1)|(nb<<28)   [bit92=bit28 of aw[2]]
//   shift84:  bw[0..1] cascade, bw[2]=(bw[2]>>1)|(nb<<19)   [bit83=bit19 of bw[2]]
//   shift111: cw[0..2] cascade, cw[3]=(cw[3]>>1)|(nb<<14)   [bit110=bit14 of cw[3]]
// All 10 words fit in GP registers → zero cache pressure vs 288-byte circular buffer.
// Tap mapping (verified against state[] flat indices):
//   t1 = A[65]^A[92]   = (aw[2]>>1)^(aw[2]>>28) masked to bit 0
//   t2 = B[68]^B[83]   = (bw[2]>>4)^(bw[2]>>19) masked to bit 0
//   t3 = C[65]^C[110]  = (cw[2]>>1)^(cw[3]>>14) masked to bit 0
//   AND: A[90]&A[91]=(aw[2]>>26)&(aw[2]>>27); B[81]&B[82]=(bw[2]>>17)&(bw[2]>>18)
//        C[108]&C[109]=(cw[3]>>12)&(cw[3]>>13)
//   Cross: B[77]=(bw[2]>>13), C[86]=(cw[2]>>22), A[68]=(aw[2]>>4)
// ============================================================
struct TriviumWP {
    uint32_t aw[3];  // Register A: 93 bits
    uint32_t bw[3];  // Register B: 84 bits
    uint32_t cw[4];  // Register C: 111 bits

    static inline uint32_t gb(const uint32_t* S, int p) {
        return (S[p >> 5] >> (p & 31)) & 1u;
    }

    // Shift A by 1: discard A[0] (oldest), insert nb at A[92] = bit28 of aw[2]
    static inline void shift93(uint32_t S[3], uint32_t nb) {
        S[0] = (S[0] >> 1) | (S[1] << 31);
        S[1] = (S[1] >> 1) | (S[2] << 31);
        S[2] = (S[2] >> 1) | (nb << 28);
    }
    // Shift B by 1: discard B[0], insert nb at B[83] = bit19 of bw[2]
    static inline void shift84(uint32_t S[3], uint32_t nb) {
        S[0] = (S[0] >> 1) | (S[1] << 31);
        S[1] = (S[1] >> 1) | (S[2] << 31);
        S[2] = (S[2] >> 1) | (nb << 19);
    }
    // Shift C by 1: discard C[0], insert nb at C[110] = bit14 of cw[3]
    static inline void shift111(uint32_t S[4], uint32_t nb) {
        S[0] = (S[0] >> 1) | (S[1] << 31);
        S[1] = (S[1] >> 1) | (S[2] << 31);
        S[2] = (S[2] >> 1) | (S[3] << 31);
        S[3] = (S[3] >> 1) | (nb << 14);
    }

    inline uint8_t clock_step() {
        // Cache high words — all taps are in word 2 or 3
        const uint32_t a2 = aw[2], b2 = bw[2], c2 = cw[2], c3 = cw[3];
        const uint32_t t1_pre = ((a2 >> 1) ^ (a2 >> 28)) & 1u;  // A[65]^A[92]
        const uint32_t t2_pre = ((b2 >> 4) ^ (b2 >> 19)) & 1u;  // B[68]^B[83]
        const uint32_t t3_pre = ((c2 >> 1) ^ (c3 >> 14)) & 1u;  // C[65]^C[110]
        const uint32_t z = t1_pre ^ t2_pre ^ t3_pre;
        // Feedback: t1_fb = t1_pre ^ (A[90]&A[91]) ^ B[77]
        const uint32_t t1_fb = t1_pre ^ ((a2>>26)&(a2>>27)&1u) ^ ((b2>>13)&1u);
        // Feedback: t2_fb = t2_pre ^ (B[81]&B[82]) ^ C[86]
        const uint32_t t2_fb = t2_pre ^ ((b2>>17)&(b2>>18)&1u) ^ ((c2>>22)&1u);
        // Feedback: t3_fb = t3_pre ^ (C[108]&C[109]) ^ A[68]
        const uint32_t t3_fb = t3_pre ^ ((c3>>12)&(c3>>13)&1u) ^ ((a2>> 4)&1u);
        // A gets t3_fb, B gets t1_fb, C gets t2_fb
        shift93(aw, t3_fb);
        shift84(bw, t1_fb);
        shift111(cw, t2_fb);
        return (uint8_t)z;
    }

    inline uint8_t gen_byte() {
        uint8_t b = 0;
        b |= clock_step() << 0; b |= clock_step() << 1;
        b |= clock_step() << 2; b |= clock_step() << 3;
        b |= clock_step() << 4; b |= clock_step() << 5;
        b |= clock_step() << 6; b |= clock_step() << 7;
        return b;
    }

    static inline TriviumWP make(const uint8_t key[10], const uint8_t iv[10]) {
        TriviumWP s{};
        // A[0..79] = key bits (bit j → bit j of aw[j>>5])
        for (int i = 0; i < 80; i++)
            if ((key[i >> 3] >> (i & 7)) & 1u) s.aw[i >> 5] |= (1u << (i & 31));
        // B[0..79] = IV bits
        for (int i = 0; i < 80; i++)
            if ((iv[i >> 3] >> (i & 7)) & 1u) s.bw[i >> 5] |= (1u << (i & 31));
        // C[108..110] = 1 (original state[285..287]=1; C offset=177, so 285-177=108)
        s.cw[3] |= (1u << (108 - 96)) | (1u << (109 - 96)) | (1u << (110 - 96));
        // 1152 warm-up clocks
        for (int i = 0; i < 1152; i++) s.clock_step();
        return s;
    }

    static inline bool match(const uint8_t key[10], const uint8_t iv[10],
                             const uint8_t* target, int length) {
        TriviumWP s = make(key, iv);
        for (int i = 0; i < length; i++)
            if (s.gen_byte() != target[i]) return false;
        return true;
    }
};

// ============================================================
// Technique 2: GrainV1WP — packed-integer (word-parallel) Grain v1
// ============================================================
// State: LFSR[80] and NFSR[80] each packed as uint32_t[3].
//   L[0]=bits0..31, L[1]=32..63, L[2]=64..79 (valid at bits0..15 only)
// Shift (discard oldest bit0, insert nb at bit79 = bit15 of L[2]):
//   L[0] = (L[0]>>1)|(L[1]<<31); L[1] = (L[1]>>1)|(L[2]<<31); L[2] = (L[2]>>1)|(nb<<15)
// All 6 words fit in GP registers → zero cache pressure vs 256-byte circular buffer.
// Tap mapping verified against GrainV1 spec:
//   LFSR fb: L(0)^L(13)^L(23)^L(38)^L(51)^L(62)
//            = (L[0]>>0)^(L[0]>>13)^(L[0]>>23)^(L[1]>>6)^(L[1]>>19)^(L[1]>>30)
//   h inputs: L(3)=(L[0]>>3), L(25)=(L[0]>>25), L(46)=(L[1]>>14), L(64)=(L[2]>>0), N(63)=(N[1]>>31)
//   output extra: N(1)^N(2)^N(4)^N(10)^N(31)^N(43)^N(56)
// Feedback computed from current state BEFORE any shift.
// ============================================================
struct GrainV1WP {
    uint32_t L[3];  // LFSR: L[0]=bits0..31, L[1]=32..63, L[2]=64..79 (bits0..15 valid)
    uint32_t N[3];  // NFSR: same layout

    static inline void shift80(uint32_t S[3], uint32_t nb) {
        S[0] = (S[0] >> 1) | (S[1] << 31);
        S[1] = (S[1] >> 1) | (S[2] << 31);
        S[2] = (S[2] >> 1) | (nb << 15);
    }

    inline uint32_t lfsr_fb_val() const {
        // L(0)^L(13)^L(23)^L(38)^L(51)^L(62)
        const uint32_t l0=L[0], l1=L[1];
        return ((l0>>0)^(l0>>13)^(l0>>23)^(l1>>6)^(l1>>19)^(l1>>30)) & 1u;
    }

    inline uint32_t nfsr_fb_val() const {
        const uint32_t l0=L[0], n0=N[0], n1=N[1];
        // Linear: L(0)^N(0)^N(9)^N(14)^N(21)^N(28)^N(33)^N(37)^N(45)^N(52)^N(60)^N(62)
        uint32_t v = (l0>>0)^(n0>>0)^(n0>>9)^(n0>>14)^(n0>>21)^(n0>>28)
                    ^(n1>>1)^(n1>>5)^(n1>>13)^(n1>>20)^(n1>>28)^(n1>>30);
        // Nonlinear 2-way:
        v ^= (n1>>28)&(n1>>31);               // N(60)&N(63)
        v ^= (n1>>1)&(n1>>5);                 // N(33)&N(37)
        v ^= (n0>>9)&(n0>>15);                // N(9)&N(15)
        // Nonlinear 3-way:
        v ^= (n1>>13)&(n1>>20)&(n1>>28);      // N(45)&N(52)&N(60)
        v ^= (n0>>21)&(n0>>28)&(n1>>1);       // N(21)&N(28)&N(33)
        // Nonlinear 4-way:
        v ^= (n0>>9)&(n0>>28)&(n1>>13)&(n1>>31);    // N(9)&N(28)&N(45)&N(63)
        v ^= (n1>>1)&(n1>>5)&(n1>>20)&(n1>>28);     // N(33)&N(37)&N(52)&N(60)
        v ^= (n0>>15)&(n0>>21)&(n1>>28)&(n1>>31);   // N(15)&N(21)&N(60)&N(63)
        // Nonlinear 5-way:
        v ^= (n1>>5)&(n1>>13)&(n1>>20)&(n1>>28)&(n1>>31);   // N(37)&N(45)&N(52)&N(60)&N(63)
        v ^= (n0>>9)&(n0>>15)&(n0>>21)&(n0>>28)&(n1>>1);    // N(9)&N(15)&N(21)&N(28)&N(33)
        // Nonlinear 6-way:
        v ^= (n0>>21)&(n0>>28)&(n1>>1)&(n1>>5)&(n1>>13)&(n1>>20); // N(21..52)
        return v & 1u;
    }

    inline uint32_t h_func_val() const {
        // h(x0=L(3),x1=L(25),x2=L(46),x3=L(64),x4=N(63))
        const uint32_t x0=(L[0]>>3)&1u, x1=(L[0]>>25)&1u, x2=(L[1]>>14)&1u;
        const uint32_t x3=(L[2]>>0)&1u,  x4=(N[1]>>31)&1u;
        return x0^x4^(x0&x3)^(x2&x3)^(x3&x4)
                   ^(x0&x1&x2)^(x0&x2&x3)
                   ^(x0&x2&x4)^(x1&x2&x4)^(x2&x3&x4);
    }

    inline uint32_t output_bit_val() const {
        // h ^ N(1)^N(2)^N(4)^N(10)^N(31)^N(43)^N(56)
        const uint32_t n0=N[0], n1=N[1];
        const uint32_t extra = ((n0>>1)^(n0>>2)^(n0>>4)^(n0>>10)^(n0>>31)
                                ^(n1>>11)^(n1>>24)) & 1u;
        return h_func_val() ^ extra;
    }

    inline void init_clock() {
        // Compute all from current state, then shift
        const uint32_t out  = output_bit_val();
        const uint32_t lnew = lfsr_fb_val() ^ out;
        const uint32_t nnew = nfsr_fb_val() ^ out;
        shift80(L, lnew);
        shift80(N, nnew);
    }

    inline uint32_t stream_bit() {
        const uint32_t out  = output_bit_val();
        const uint32_t lnew = lfsr_fb_val();
        const uint32_t nnew = nfsr_fb_val();
        shift80(L, lnew);
        shift80(N, nnew);
        return out;
    }

    inline uint8_t gen_byte() {
        uint8_t b = 0;
        b |= (uint8_t)(stream_bit()<<0); b |= (uint8_t)(stream_bit()<<1);
        b |= (uint8_t)(stream_bit()<<2); b |= (uint8_t)(stream_bit()<<3);
        b |= (uint8_t)(stream_bit()<<4); b |= (uint8_t)(stream_bit()<<5);
        b |= (uint8_t)(stream_bit()<<6); b |= (uint8_t)(stream_bit()<<7);
        return b;
    }

    static inline GrainV1WP make(const uint8_t key[10], const uint8_t iv[8]) {
        GrainV1WP s{};
        // NFSR[0..79] = key bits
        for (int i = 0; i < 80; i++)
            if ((key[i>>3]>>(i&7))&1u) s.N[i>>5] |= (1u<<(i&31));
        // LFSR[0..63] = IV bits
        for (int i = 0; i < 64; i++)
            if ((iv[i>>3]>>(i&7))&1u) s.L[i>>5] |= (1u<<(i&31));
        // LFSR[64..79] = 1: bits 0..15 of L[2]
        s.L[2] = 0x0000FFFFu;
        // 160 init clocks
        for (int i = 0; i < 160; i++) s.init_clock();
        return s;
    }

    static inline bool match(const uint8_t key[10], const uint8_t iv[8],
                             const uint8_t* target, int length) {
        GrainV1WP s = make(key, iv);
        for (int i = 0; i < length; i++)
            if (s.gen_byte() != target[i]) return false;
        return true;
    }
};

// ============================================================
// Technique 3: Grain128WP — packed-integer Grain-128AEADv2
// ============================================================
// State: LFSR[128] and NFSR[128] each packed as uint32_t[4].
//   L[0]=bits0..31, L[1]=32..63, L[2]=64..95, L[3]=96..127
// shift128: discard bit0, insert nb at bit127 = bit31 of L[3].
// Auth state (auth_acc, auth_sr) is NOT tracked: skipped for keystream-only matching.
// This eliminates the 127-byte memmove × (320+64+128) init clocks = 66K byte-copies/key.
// AD processing: 16 cipher clocks per byte (2 per bit × 8 bits). AD data does NOT
//   feed back into LFSR/NFSR — only into auth state which we discard.
// All feedback computed from current state BEFORE any shift.
// ============================================================
struct Grain128WP {
    uint32_t L[4];  // LFSR: L[0]=bits0..31, L[1]=32..63, L[2]=64..95, L[3]=96..127
    uint32_t N[4];  // NFSR: same layout

    static inline void shift128(uint32_t S[4], uint32_t nb) {
        S[0] = (S[0]>>1)|(S[1]<<31);
        S[1] = (S[1]>>1)|(S[2]<<31);
        S[2] = (S[2]>>1)|(S[3]<<31);
        S[3] = (S[3]>>1)|(nb<<31);
    }

    inline uint32_t lfsr_fb_val() const {
        // L(0)^L(7)^L(38)^L(70)^L(81)^L(96)
        return ((L[0]>>0)^(L[0]>>7)^(L[1]>>6)^(L[2]>>6)^(L[2]>>17)^(L[3]>>0)) & 1u;
    }

    inline uint32_t nfsr_fb_val() const {
        const uint32_t l0=L[0], n0=N[0], n1=N[1], n2=N[2], n3=N[3];
        // Linear: L(0)^N(0)^N(26)^N(56)^N(91)^N(96)
        uint32_t v = (l0>>0)^(n0>>0)^(n0>>26)^(n1>>24)^(n2>>27)^(n3>>0);
        // 2-way products:
        v ^= (n0>>3)&(n2>>3);       // N(3)&N(67)
        v ^= (n0>>11)&(n0>>13);     // N(11)&N(13)
        v ^= (n0>>17)&(n0>>18);     // N(17)&N(18)
        v ^= (n0>>27)&(n1>>27);     // N(27)&N(59)
        v ^= (n1>>8)&(n1>>16);      // N(40)&N(48)
        v ^= (n1>>29)&(n2>>1);      // N(61)&N(65)
        v ^= (n2>>4)&(n2>>20);      // N(68)&N(84)
        // 3-way products:
        v ^= (n0>>22)&(n0>>24)&(n0>>25); // N(22)&N(24)&N(25)
        v ^= (n2>>6)&(n2>>14)&(n2>>18);  // N(70)&N(78)&N(82)
        // 4-way product:
        v ^= (n2>>24)&(n2>>28)&(n2>>29)&(n2>>31); // N(88)&N(92)&N(93)&N(95)
        return v & 1u;
    }

    inline uint32_t h_func_val() const {
        // (N(12)&L(8))^(L(13)&L(20))^(N(95)&L(42))^(L(60)&L(79))^(N(12)&N(95)&L(94))
        const uint32_t n12=(N[0]>>12)&1u, l8=(L[0]>>8)&1u;
        const uint32_t l13=(L[0]>>13)&1u, l20=(L[0]>>20)&1u;
        const uint32_t n95=(N[2]>>31)&1u, l42=(L[1]>>10)&1u;
        const uint32_t l60=(L[1]>>28)&1u, l79=(L[2]>>15)&1u;
        const uint32_t l94=(L[2]>>30)&1u;
        return (n12&l8)^(l13&l20)^(n95&l42)^(l60&l79)^(n12&n95&l94);
    }

    inline uint32_t output_bit_val() const {
        // h ^ L(93) ^ N(2)^N(15)^N(36)^N(45)^N(64)^N(73)^N(89)
        const uint32_t l93=(L[2]>>29)&1u;
        const uint32_t extra = ((N[0]>>2)^(N[0]>>15)^(N[1]>>4)^(N[1]>>13)
                               ^(N[2]>>0)^(N[2]>>9)^(N[2]>>25)) & 1u;
        return h_func_val() ^ l93 ^ extra;
    }

    inline uint32_t clock_normal() {
        const uint32_t y    = output_bit_val();
        const uint32_t lnew = lfsr_fb_val();
        const uint32_t nnew = nfsr_fb_val();
        shift128(L, lnew);
        shift128(N, nnew);
        return y;
    }

    inline uint32_t clock_init() {
        const uint32_t y    = output_bit_val();
        const uint32_t lnew = lfsr_fb_val() ^ y;
        const uint32_t nnew = nfsr_fb_val() ^ y;
        shift128(L, lnew);
        shift128(N, nnew);
        return y;
    }

    inline void clock_key_inject(uint32_t lfsr_extra, uint32_t nfsr_extra) {
        const uint32_t y    = output_bit_val();
        const uint32_t lnew = lfsr_fb_val() ^ y ^ lfsr_extra;
        const uint32_t nnew = nfsr_fb_val() ^ y ^ nfsr_extra;
        shift128(L, lnew);
        shift128(N, nnew);
    }

    static inline Grain128WP make(const uint8_t key[16], const uint8_t nonce[12]) {
        Grain128WP s{};
        // NFSR[0..127] = key bits
        for (int i = 0; i < 128; i++)
            if ((key[i>>3]>>(i&7))&1u) s.N[i>>5] |= (1u<<(i&31));
        // LFSR[0..95] = nonce bits
        for (int i = 0; i < 96; i++)
            if ((nonce[i>>3]>>(i&7))&1u) s.L[i>>5] |= (1u<<(i&31));
        // LFSR[96..126]=1, LFSR[127]=0 → L[3] = 0x7FFFFFFF
        s.L[3] = 0x7FFFFFFFu;
        // Phase 1: 320 init clocks (with output feedback)
        for (int t = 0; t < 320; t++) s.clock_init();
        // Phase 2: 64 key-inject clocks
        for (int t = 0; t < 64; t++) {
            const uint32_t le = (key[(t+64)>>3]>>((t+64)&7)) & 1u;
            const uint32_t ne = (key[t>>3]>>(t&7)) & 1u;
            s.clock_key_inject(le, ne);
        }
        // Phase 3: 128 auth-init clocks (skip auth tracking — L/N advance identically)
        for (int t = 0; t < 128; t++) s.clock_normal();
        return s;
    }

    // Process AD: 16 clocks per byte (2 per bit × 8 bits). AD bytes do not
    // feed back into LFSR/NFSR, so we run 16 clock_normal()s regardless of byte value.
    inline void process_ad_bytes(const uint8_t* ad, int ad_len) {
        auto clock16 = [&]() {
            for (int b = 0; b < 8; b++) { clock_normal(); clock_normal(); }
        };
        // DER-encode length field (same as match_keystream)
        if (ad_len < 128) { clock16(); }
        else if (ad_len < 256) { clock16(); clock16(); }
        else { clock16(); clock16(); clock16(); }
        for (int i = 0; i < ad_len; i++) clock16();
    }

    static inline bool match(const uint8_t key[16], const uint8_t nonce[12],
                             const uint8_t* ad, int ad_len,
                             const uint8_t* target_ks, int pt_len) {
        Grain128WP s = make(key, nonce);
        s.process_ad_bytes(ad, ad_len);
        // Each plaintext byte: 2 clocks per bit (first = keystream z_i, second = auth stream)
        for (int i = 0; i < pt_len; i++) {
            uint8_t ks = 0;
            for (int b = 0; b < 8; b++) {
                ks |= (uint8_t)(s.clock_normal() << b);
                s.clock_normal();  // auth stream bit — discard
            }
            if (ks != target_ks[i]) return false;
        }
        return true;
    }
};

// ============================================================
// Technique 3: AES128NI — hardware AES-NI encryption
// ============================================================
// Uses AESENC / AESENCLAST instructions (one full AES round each,
// latency 4 cycles on AMD Zen 2) instead of software SubBytes + ShiftRows +
// MixColumns (~12 operations per round).
// ILP4: 4 independent encryption pipelines hide the 4-cycle instruction latency.
// ============================================================
#ifdef CPU_OPT_HAVE_AESNI

static inline __m128i aes_ni_key_assist(__m128i t1, __m128i t2) {
    t2 = _mm_shuffle_epi32(t2, 0xff);
    __m128i t3 = _mm_slli_si128(t1, 4);
    t1 = _mm_xor_si128(t1, t3);
    t3 = _mm_slli_si128(t3, 4);
    t1 = _mm_xor_si128(t1, t3);
    t3 = _mm_slli_si128(t3, 4);
    t1 = _mm_xor_si128(t1, t3);
    return _mm_xor_si128(t1, t2);
}

static inline void aes_ni_expand_key128(const uint8_t key[16], __m128i rk[11]) {
    __m128i t = _mm_loadu_si128((const __m128i*)key);
    rk[0]  = t;
    t = aes_ni_key_assist(t, _mm_aeskeygenassist_si128(t, 0x01)); rk[1]  = t;
    t = aes_ni_key_assist(t, _mm_aeskeygenassist_si128(t, 0x02)); rk[2]  = t;
    t = aes_ni_key_assist(t, _mm_aeskeygenassist_si128(t, 0x04)); rk[3]  = t;
    t = aes_ni_key_assist(t, _mm_aeskeygenassist_si128(t, 0x08)); rk[4]  = t;
    t = aes_ni_key_assist(t, _mm_aeskeygenassist_si128(t, 0x10)); rk[5]  = t;
    t = aes_ni_key_assist(t, _mm_aeskeygenassist_si128(t, 0x20)); rk[6]  = t;
    t = aes_ni_key_assist(t, _mm_aeskeygenassist_si128(t, 0x40)); rk[7]  = t;
    t = aes_ni_key_assist(t, _mm_aeskeygenassist_si128(t, 0x80)); rk[8]  = t;
    t = aes_ni_key_assist(t, _mm_aeskeygenassist_si128(t, 0x1b)); rk[9]  = t;
    t = aes_ni_key_assist(t, _mm_aeskeygenassist_si128(t, 0x36)); rk[10] = t;
}

static inline __m128i aes_ni_encrypt(__m128i pt, const __m128i rk[11]) {
    pt = _mm_xor_si128(pt, rk[0]);
    pt = _mm_aesenc_si128(pt, rk[1]);
    pt = _mm_aesenc_si128(pt, rk[2]);
    pt = _mm_aesenc_si128(pt, rk[3]);
    pt = _mm_aesenc_si128(pt, rk[4]);
    pt = _mm_aesenc_si128(pt, rk[5]);
    pt = _mm_aesenc_si128(pt, rk[6]);
    pt = _mm_aesenc_si128(pt, rk[7]);
    pt = _mm_aesenc_si128(pt, rk[8]);
    pt = _mm_aesenc_si128(pt, rk[9]);
    return _mm_aesenclast_si128(pt, rk[10]);
}
#endif  // CPU_OPT_HAVE_AESNI

// ============================================================
// Brute-force helper macros
// ============================================================
#define CPUOPT_BF_HEADER(N_EXPR)                          \
    CpuBFResult res;                                      \
    const uint64_t N = (N_EXPR);                          \
    res.keys_tested = N;                                  \
    if (N == 0) return res;                               \
    const uint64_t base_key = (known_high << (uint64_t)unknown_bits);

#define CPUOPT_BF_FOOTER(search_fn)                       \
    res.seconds = time_cpu_function(search_fn, repeats);  \
    return res;

// ============================================================
// 1. SIMON 32/64 — ILP2
// 2 independent Feistel pipelines overlap ARX latency.
// ============================================================
inline CpuBFResult brute_force_cpu_opt_simon(uint32_t pt, uint32_t ct,
                                             uint64_t known_high, int unknown_bits,
                                             int repeats) {
    CPUOPT_BF_HEADER(bf_space_size_cpu(unknown_bits))
    auto fn = [&]() {
        bool found = false; uint64_t fk = 0; uint64_t acc = 0;
        uint16_t rk0[SIMON32_64_ROUNDS], rk1[SIMON32_64_ROUNDS];
        uint64_t i = 0;
        for (; i + 1 < N; i += 2) {
            const uint64_t k0 = base_key | i, k1 = base_key | (i+1);
            Simon32_64_Enhanced::expand_key_vectorized(k0, rk0);
            Simon32_64_Enhanced::expand_key_vectorized(k1, rk1);
            const uint32_t c0 = Simon32_64_Enhanced::encrypt_optimized(pt, rk0);
            const uint32_t c1 = Simon32_64_Enhanced::encrypt_optimized(pt, rk1);
            acc ^= ((uint64_t)c0 ^ (uint64_t)c1) << (i & 7);
            if (!found && c0 == ct) { found = true; fk = k0; }
            if (!found && c1 == ct) { found = true; fk = k1; }
        }
        if (i < N) {
            Simon32_64_Enhanced::expand_key_vectorized(base_key | i, rk0);
            uint32_t c0 = Simon32_64_Enhanced::encrypt_optimized(pt, rk0);
            acc ^= c0; if (!found && c0 == ct) { found = true; fk = base_key | i; }
        }
        res.found = found; res.found_key = fk;
        cpu_bf_detail::mix_sink_u64(acc ^ fk);
    };
    CPUOPT_BF_FOOTER(fn)
}

// ============================================================
// 2. PRESENT-80 — ILP2
// ============================================================
inline CpuBFResult brute_force_cpu_opt_present(uint64_t pt, uint64_t ct,
                                               uint64_t known_high, int unknown_bits,
                                               int repeats) {
    CPUOPT_BF_HEADER(bf_space_size_cpu(unknown_bits))
    auto fn = [&]() {
        bool found = false; uint64_t fk = 0; uint64_t acc = 0;
        uint64_t rk0[32], rk1[32];
        uint64_t i = 0;
        for (; i + 1 < N; i += 2) {
            const uint64_t k0 = base_key | i, k1 = base_key | (i+1);
            uint8_t key0[10], key1[10];
            cpu_bf_detail::key64_to_key80_zero_hi16(k0, key0);
            cpu_bf_detail::key64_to_key80_zero_hi16(k1, key1);
            Present80::expand_key(key0, rk0);
            Present80::expand_key(key1, rk1);
            const uint64_t c0 = Present80::encrypt(pt, rk0);
            const uint64_t c1 = Present80::encrypt(pt, rk1);
            acc ^= (c0 ^ c1) << (i & 7);
            if (!found && c0 == ct) { found = true; fk = k0; }
            if (!found && c1 == ct) { found = true; fk = k1; }
        }
        if (i < N) {
            uint8_t key0[10]; cpu_bf_detail::key64_to_key80_zero_hi16(base_key|i, key0);
            Present80::expand_key(key0, rk0);
            uint64_t c0 = Present80::encrypt(pt, rk0);
            acc ^= c0; if (!found && c0 == ct) { found = true; fk = base_key|i; }
        }
        res.found = found; res.found_key = fk;
        cpu_bf_detail::mix_sink_u64(acc ^ fk);
    };
    CPUOPT_BF_FOOTER(fn)
}

// ============================================================
// 3. SPECK 64/128 — ILP2
// ============================================================
inline CpuBFResult brute_force_cpu_opt_speck(uint64_t pt, uint64_t ct,
                                             uint64_t known_high, int unknown_bits,
                                             int repeats) {
    CPUOPT_BF_HEADER(bf_space_size_cpu(unknown_bits))
    auto fn = [&]() {
        bool found = false; uint64_t fk = 0; uint64_t acc = 0;
        uint32_t rk0[SPECK64_128_ROUNDS], rk1[SPECK64_128_ROUNDS];
        uint64_t i = 0;
        for (; i + 1 < N; i += 2) {
            const uint64_t k0 = base_key | i, k1 = base_key | (i+1);
            uint8_t key0[16], key1[16];
            cpu_bf_detail::key64_to_key128_zero_hi64(k0, key0);
            cpu_bf_detail::key64_to_key128_zero_hi64(k1, key1);
            Speck64_128::expand_key(key0, rk0);
            Speck64_128::expand_key(key1, rk1);
            const uint64_t c0 = Speck64_128::encrypt(pt, rk0);
            const uint64_t c1 = Speck64_128::encrypt(pt, rk1);
            acc ^= (c0 ^ c1) << (i & 7);
            if (!found && c0 == ct) { found = true; fk = k0; }
            if (!found && c1 == ct) { found = true; fk = k1; }
        }
        if (i < N) {
            uint8_t key0[16]; cpu_bf_detail::key64_to_key128_zero_hi64(base_key|i, key0);
            Speck64_128::expand_key(key0, rk0);
            uint64_t c0 = Speck64_128::encrypt(pt, rk0);
            acc ^= c0; if (!found && c0 == ct) { found = true; fk = base_key|i; }
        }
        res.found = found; res.found_key = fk;
        cpu_bf_detail::mix_sink_u64(acc ^ fk);
    };
    CPUOPT_BF_FOOTER(fn)
}

// ============================================================
// 4. Grain v1 — Circular-buffer fast implementation + ILP2
// Eliminates 160 byte-copies/clock step in the 160-step init.
// ============================================================
inline CpuBFResult brute_force_cpu_opt_grain(const uint8_t* pt, const uint8_t* ct,
                                              const uint8_t* iv, int length,
                                              uint64_t known_high, int unknown_bits,
                                              int repeats) {
    CPUOPT_BF_HEADER(bf_space_size_cpu(unknown_bits))
    uint8_t target[64] = {};
    cpu_bf_detail::make_target_xor(pt, ct, length, target);
    auto fn = [&]() {
        bool found = false; uint64_t fk = 0; uint64_t acc = 0;
        uint64_t i = 0;
        for (; i + 1 < N; i += 2) {
            const uint64_t k0 = base_key | i, k1 = base_key | (i+1);
            uint8_t key0[10], key1[10];
            cpu_bf_detail::key64_to_key80_zero_hi16(k0, key0);
            cpu_bf_detail::key64_to_key80_zero_hi16(k1, key1);
            const bool m0 = GrainV1WP::match(key0, iv, target, length);
            const bool m1 = GrainV1WP::match(key1, iv, target, length);
            acc ^= (uint64_t)(m0 ? 0x9b : 0x41) << (i & 7);
            if (!found && m0) { found = true; fk = k0; }
            if (!found && m1) { found = true; fk = k1; }
        }
        if (i < N) {
            uint8_t key0[10]; cpu_bf_detail::key64_to_key80_zero_hi16(base_key|i, key0);
            bool m0 = GrainV1WP::match(key0, iv, target, length);
            acc ^= m0; if (!found && m0) { found = true; fk = base_key|i; }
        }
        res.found = found; res.found_key = fk;
        cpu_bf_detail::mix_sink_u64(acc ^ fk);
    };
    CPUOPT_BF_FOOTER(fn)
}

// ============================================================
// 5. Trivium — Circular-buffer fast implementation + ILP2
// Eliminates 864 byte-copies/clock step in the 1152-step init.
// ============================================================
inline CpuBFResult brute_force_cpu_opt_trivium(const uint8_t* pt, const uint8_t* ct,
                                               const uint8_t* iv, int length,
                                               uint64_t known_high, int unknown_bits,
                                               int repeats) {
    CPUOPT_BF_HEADER(bf_space_size_cpu(unknown_bits))
    uint8_t target[64] = {};
    cpu_bf_detail::make_target_xor(pt, ct, length, target);
    auto fn = [&]() {
        bool found = false; uint64_t fk = 0; uint64_t acc = 0;
        uint64_t i = 0;
        for (; i + 1 < N; i += 2) {
            const uint64_t k0 = base_key | i, k1 = base_key | (i+1);
            uint8_t key0[10], key1[10];
            cpu_bf_detail::key64_to_key80_zero_hi16(k0, key0);
            cpu_bf_detail::key64_to_key80_zero_hi16(k1, key1);
            const bool m0 = TriviumWP::match(key0, iv, target, length);
            const bool m1 = TriviumWP::match(key1, iv, target, length);
            acc ^= (uint64_t)(m0 ? 0xd7 : 0x23) << (i & 7);
            if (!found && m0) { found = true; fk = k0; }
            if (!found && m1) { found = true; fk = k1; }
        }
        if (i < N) {
            uint8_t key0[10]; cpu_bf_detail::key64_to_key80_zero_hi16(base_key|i, key0);
            bool m0 = TriviumWP::match(key0, iv, target, length);
            acc ^= m0; if (!found && m0) { found = true; fk = base_key|i; }
        }
        res.found = found; res.found_key = fk;
        cpu_bf_detail::mix_sink_u64(acc ^ fk);
    };
    CPUOPT_BF_FOOTER(fn)
}

// ============================================================
// 6. ChaCha20 — 16-byte prefix match + ILP2
// Already uses block_words4 prefix match; ILP2 adds pipeline overlap.
// ============================================================
inline CpuBFResult brute_force_cpu_opt_chacha20(const uint8_t* pt, const uint8_t* ct,
                                                const uint8_t* nonce12, int length,
                                                uint64_t known_high, int unknown_bits,
                                                int repeats) {
    CPUOPT_BF_HEADER(bf_space_size_cpu(unknown_bits))
    uint8_t target[64] = {};
    cpu_bf_detail::make_target_xor(pt, ct, length, target);
    auto fn = [&]() {
        bool found = false; uint64_t fk = 0; uint64_t acc = 0;
        uint64_t i = 0;
        for (; i + 1 < N; i += 2) {
            const uint64_t k0 = base_key | i, k1 = base_key | (i+1);
            uint8_t key0[32], key1[32];
            cpu_bf_detail::key64_to_key256_zero_hi192(k0, key0);
            cpu_bf_detail::key64_to_key256_zero_hi192(k1, key1);
            uint32_t out0[4] = {}, out1[4] = {};
            ChaCha20::block_words4(key0, 1u, nonce12, out0);
            ChaCha20::block_words4(key1, 1u, nonce12, out1);
            const bool m0 = cpu_bf_detail::chacha_match_prefix_words4_host(out0, target, length);
            const bool m1 = cpu_bf_detail::chacha_match_prefix_words4_host(out1, target, length);
            acc ^= ((uint64_t)out0[0] ^ (uint64_t)out1[0]) << (i & 7);
            if (!found && m0) { found = true; fk = k0; }
            if (!found && m1) { found = true; fk = k1; }
        }
        if (i < N) {
            uint8_t key0[32]; cpu_bf_detail::key64_to_key256_zero_hi192(base_key|i, key0);
            uint32_t out0[4] = {};
            ChaCha20::block_words4(key0, 1u, nonce12, out0);
            bool m0 = cpu_bf_detail::chacha_match_prefix_words4_host(out0, target, length);
            acc ^= out0[0]; if (!found && m0) { found = true; fk = base_key|i; }
        }
        res.found = found; res.found_key = fk;
        cpu_bf_detail::mix_sink_u64(acc ^ fk);
    };
    CPUOPT_BF_FOOTER(fn)
}

// ============================================================
// 7. TinyJAMBU-128 — full AEAD early-exit + ILP2
// TinyJAMBU128 has no match_keystream; use encrypt() + byte-compare.
// ILP2: two independent pipelines reduce stall latency.
// ============================================================
static inline bool tinyjambu_match(const uint8_t* pt, const uint8_t* ct, int length,
                                    const uint8_t* nonce, const uint8_t* ad, int ad_len,
                                    const uint8_t key[16]) {
    uint8_t out_ct[64]; uint8_t out_tag[8];
    TinyJAMBU128::encrypt(pt, out_ct, length, out_tag, key, nonce, ad, ad_len);
    for (int j = 0; j < length; j++) if (out_ct[j] != ct[j]) return false;
    return true;
}

inline CpuBFResult brute_force_cpu_opt_tinyjambu(const uint8_t* pt, const uint8_t* ct,
                                                  const uint8_t* nonce, int length,
                                                  const uint8_t* ad, int ad_len,
                                                  uint64_t known_high, int unknown_bits,
                                                  int repeats) {
    CPUOPT_BF_HEADER(bf_space_size_cpu(unknown_bits))
    auto fn = [&]() {
        bool found = false; uint64_t fk = 0; uint64_t acc = 0;
        uint64_t i = 0;
        for (; i + 1 < N; i += 2) {
            const uint64_t k0 = base_key | i, k1 = base_key | (i+1);
            uint8_t key0[16], key1[16];
            cpu_bf_detail::key64_to_key128_zero_hi64(k0, key0);
            cpu_bf_detail::key64_to_key128_zero_hi64(k1, key1);
            const bool m0 = tinyjambu_match(pt, ct, length, nonce, ad, ad_len, key0);
            const bool m1 = tinyjambu_match(pt, ct, length, nonce, ad, ad_len, key1);
            acc ^= (uint64_t)(m0 ? 0xab : 0x55) << (i & 7);
            if (!found && m0) { found = true; fk = k0; }
            if (!found && m1) { found = true; fk = k1; }
        }
        if (i < N) {
            uint8_t key0[16]; cpu_bf_detail::key64_to_key128_zero_hi64(base_key|i, key0);
            bool m0 = tinyjambu_match(pt, ct, length, nonce, ad, ad_len, key0);
            acc ^= m0; if (!found && m0) { found = true; fk = base_key|i; }
        }
        res.found = found; res.found_key = fk;
        cpu_bf_detail::mix_sink_u64(acc ^ fk);
    };
    CPUOPT_BF_FOOTER(fn)
}

// ============================================================
// 8. ZUC-128 — early-exit keystream match + ILP2
// ============================================================
inline CpuBFResult brute_force_cpu_opt_zuc(const uint8_t* pt, const uint8_t* ct,
                                            const uint8_t* iv, int length,
                                            uint64_t known_high, int unknown_bits,
                                            int repeats) {
    CPUOPT_BF_HEADER(bf_space_size_cpu(unknown_bits))
    uint8_t target[64] = {};
    cpu_bf_detail::make_target_xor(pt, ct, length, target);
    auto fn = [&]() {
        bool found = false; uint64_t fk = 0; uint64_t acc = 0;
        uint64_t i = 0;
        for (; i + 1 < N; i += 2) {
            const uint64_t k0 = base_key | i, k1 = base_key | (i+1);
            uint8_t key0[16], key1[16];
            cpu_bf_detail::key64_to_key128_zero_hi64(k0, key0);
            cpu_bf_detail::key64_to_key128_zero_hi64(k1, key1);
            const bool m0 = ZUC::match_keystream(key0, iv, target, length);
            const bool m1 = ZUC::match_keystream(key1, iv, target, length);
            acc ^= (uint64_t)(m0 ? 0xbe : 0x7f) << (i & 7);
            if (!found && m0) { found = true; fk = k0; }
            if (!found && m1) { found = true; fk = k1; }
        }
        if (i < N) {
            uint8_t key0[16]; cpu_bf_detail::key64_to_key128_zero_hi64(base_key|i, key0);
            bool m0 = ZUC::match_keystream(key0, iv, target, length);
            acc ^= m0; if (!found && m0) { found = true; fk = base_key|i; }
        }
        res.found = found; res.found_key = fk;
        cpu_bf_detail::mix_sink_u64(acc ^ fk);
    };
    CPUOPT_BF_FOOTER(fn)
}

// ============================================================
// 9. SNOW-V — T-table early-exit match + ILP2
// ============================================================
inline CpuBFResult brute_force_cpu_opt_snowv(const uint8_t* pt, const uint8_t* ct,
                                              const uint8_t* iv, int length,
                                              uint64_t known_high, int unknown_bits,
                                              int repeats) {
    CPUOPT_BF_HEADER(bf_space_size_cpu(unknown_bits))
    uint8_t target[64] = {};
    cpu_bf_detail::make_target_xor(pt, ct, length, target);
    auto fn = [&]() {
        bool found = false; uint64_t fk = 0; uint64_t acc = 0;
        uint64_t i = 0;
        for (; i + 1 < N; i += 2) {
            const uint64_t k0 = base_key | i, k1 = base_key | (i+1);
            uint8_t key0[32], key1[32];
            cpu_bf_detail::key64_to_key256_zero_hi192(k0, key0);
            cpu_bf_detail::key64_to_key256_zero_hi192(k1, key1);
            const bool m0 = SNOW_V::match_keystream<true>(key0, iv, target, length);
            const bool m1 = SNOW_V::match_keystream<true>(key1, iv, target, length);
            acc ^= (uint64_t)(m0 ? 0xcc : 0x33) << (i & 7);
            if (!found && m0) { found = true; fk = k0; }
            if (!found && m1) { found = true; fk = k1; }
        }
        if (i < N) {
            uint8_t key0[32]; cpu_bf_detail::key64_to_key256_zero_hi192(base_key|i, key0);
            bool m0 = SNOW_V::match_keystream<true>(key0, iv, target, length);
            acc ^= m0; if (!found && m0) { found = true; fk = base_key|i; }
        }
        res.found = found; res.found_key = fk;
        cpu_bf_detail::mix_sink_u64(acc ^ fk);
    };
    CPUOPT_BF_FOOTER(fn)
}

// ============================================================
// 10. AES-128 — AES-NI ILP4 (if available) else fast gmul2 ILP2
// ============================================================
inline CpuBFResult brute_force_cpu_opt_aes(const uint8_t* pt, const uint8_t* ct,
                                            uint64_t known_high, int unknown_bits,
                                            int repeats) {
    CPUOPT_BF_HEADER(bf_space_size_cpu(unknown_bits))
#ifdef CPU_OPT_HAVE_AESNI
    const __m128i pt_m = _mm_loadu_si128((const __m128i*)pt);
    const __m128i ct_m = _mm_loadu_si128((const __m128i*)ct);
    auto fn = [&]() {
        bool found = false; uint64_t fk = 0;
        __m128i acc_m = _mm_setzero_si128();
        uint64_t i = 0;
        // ILP4: expand 4 keys, encrypt 4 blocks interleaved to hide 4-cycle AESENC latency
        for (; i + 3 < N; i += 4) {
            uint8_t kb0[16], kb1[16], kb2[16], kb3[16];
            cpu_bf_detail::key64_to_key128_zero_hi64(base_key|i,   kb0);
            cpu_bf_detail::key64_to_key128_zero_hi64(base_key|i+1, kb1);
            cpu_bf_detail::key64_to_key128_zero_hi64(base_key|i+2, kb2);
            cpu_bf_detail::key64_to_key128_zero_hi64(base_key|i+3, kb3);
            __m128i rk0[11], rk1[11], rk2[11], rk3[11];
            aes_ni_expand_key128(kb0, rk0); aes_ni_expand_key128(kb1, rk1);
            aes_ni_expand_key128(kb2, rk2); aes_ni_expand_key128(kb3, rk3);
            // Interleaved encrypt — 4 pipelines hide 4-cycle AESENC latency
            __m128i c0 = _mm_xor_si128(pt_m, rk0[0]);
            __m128i c1 = _mm_xor_si128(pt_m, rk1[0]);
            __m128i c2 = _mm_xor_si128(pt_m, rk2[0]);
            __m128i c3 = _mm_xor_si128(pt_m, rk3[0]);
            for (int r = 1; r <= 9; r++) {
                c0 = _mm_aesenc_si128(c0, rk0[r]);
                c1 = _mm_aesenc_si128(c1, rk1[r]);
                c2 = _mm_aesenc_si128(c2, rk2[r]);
                c3 = _mm_aesenc_si128(c3, rk3[r]);
            }
            c0 = _mm_aesenclast_si128(c0, rk0[10]);
            c1 = _mm_aesenclast_si128(c1, rk1[10]);
            c2 = _mm_aesenclast_si128(c2, rk2[10]);
            c3 = _mm_aesenclast_si128(c3, rk3[10]);
            acc_m = _mm_xor_si128(acc_m, _mm_xor_si128(_mm_xor_si128(c0,c1),_mm_xor_si128(c2,c3)));
            if (!found && _mm_movemask_epi8(_mm_cmpeq_epi8(c0, ct_m)) == 0xffff) { found = true; fk = base_key|i;   }
            if (!found && _mm_movemask_epi8(_mm_cmpeq_epi8(c1, ct_m)) == 0xffff) { found = true; fk = base_key|i+1; }
            if (!found && _mm_movemask_epi8(_mm_cmpeq_epi8(c2, ct_m)) == 0xffff) { found = true; fk = base_key|i+2; }
            if (!found && _mm_movemask_epi8(_mm_cmpeq_epi8(c3, ct_m)) == 0xffff) { found = true; fk = base_key|i+3; }
        }
        // Tail
        for (; i < N; i++) {
            uint8_t kb[16]; cpu_bf_detail::key64_to_key128_zero_hi64(base_key|i, kb);
            __m128i rk[11]; aes_ni_expand_key128(kb, rk);
            __m128i c = aes_ni_encrypt(pt_m, rk);
            acc_m = _mm_xor_si128(acc_m, c);
            if (!found && _mm_movemask_epi8(_mm_cmpeq_epi8(c, ct_m)) == 0xffff) { found = true; fk = base_key|i; }
        }
        res.found = found; res.found_key = fk;
        // Fold acc_m into a scalar for the sink
        alignas(16) uint64_t tmp[2];
        _mm_store_si128((__m128i*)tmp, acc_m);
        cpu_bf_detail::mix_sink_u64(tmp[0] ^ tmp[1] ^ fk);
    };
#else
    // Fallback: fast gmul2/gmul3 MixColumns + ILP2
    auto fn = [&]() {
        bool found = false; uint64_t fk = 0; uint64_t acc = 0;
        uint64_t i = 0;
        for (; i + 1 < N; i += 2) {
            const uint64_t k0 = base_key | i, k1 = base_key | (i+1);
            uint8_t key0[16], key1[16];
            cpu_bf_detail::key64_to_key128_zero_hi64(k0, key0);
            cpu_bf_detail::key64_to_key128_zero_hi64(k1, key1);
            uint8_t out0[16] = {}, out1[16] = {};
            AES128::encrypt<true>(pt, key0, out0);
            AES128::encrypt<true>(pt, key1, out1);
            const bool m0 = (__builtin_memcmp(out0, ct, 16) == 0);
            const bool m1 = (__builtin_memcmp(out1, ct, 16) == 0);
            acc ^= ((uint64_t)out0[0] ^ (uint64_t)out1[0]) << (i & 7);
            if (!found && m0) { found = true; fk = k0; }
            if (!found && m1) { found = true; fk = k1; }
        }
        if (i < N) {
            uint8_t key0[16]; cpu_bf_detail::key64_to_key128_zero_hi64(base_key|i, key0);
            uint8_t out0[16] = {}; AES128::encrypt<true>(pt, key0, out0);
            bool m0 = (__builtin_memcmp(out0, ct, 16) == 0);
            acc ^= out0[0]; if (!found && m0) { found = true; fk = base_key|i; }
        }
        res.found = found; res.found_key = fk;
        cpu_bf_detail::mix_sink_u64(acc ^ fk);
    };
#endif
    CPUOPT_BF_FOOTER(fn)
}

// ============================================================
// 11. Salsa20 — 16-byte prefix match + ILP2
// ============================================================
inline CpuBFResult brute_force_cpu_opt_salsa20(const uint8_t* pt, const uint8_t* ct,
                                               const uint8_t* nonce8, int length,
                                               uint64_t known_high, int unknown_bits,
                                               int repeats) {
    CPUOPT_BF_HEADER(bf_space_size_cpu(unknown_bits))
    uint8_t target[64] = {};
    cpu_bf_detail::make_target_xor(pt, ct, length, target);
    auto fn = [&]() {
        bool found = false; uint64_t fk = 0; uint64_t acc = 0;
        uint64_t i = 0;
        for (; i + 1 < N; i += 2) {
            const uint64_t k0 = base_key | i, k1 = base_key | (i+1);
            uint8_t key0[32], key1[32];
            cpu_bf_detail::key64_to_key256_zero_hi192(k0, key0);
            cpu_bf_detail::key64_to_key256_zero_hi192(k1, key1);
            uint32_t out0[4] = {}, out1[4] = {};
            Salsa20::block_words4(key0, 0ULL, nonce8, out0);
            Salsa20::block_words4(key1, 0ULL, nonce8, out1);
            const bool m0 = cpu_bf_detail::chacha_match_prefix_words4_host(out0, target, length);
            const bool m1 = cpu_bf_detail::chacha_match_prefix_words4_host(out1, target, length);
            acc ^= ((uint64_t)out0[0] ^ (uint64_t)out1[0]) << (i & 7);
            if (!found && m0) { found = true; fk = k0; }
            if (!found && m1) { found = true; fk = k1; }
        }
        if (i < N) {
            uint8_t key0[32]; cpu_bf_detail::key64_to_key256_zero_hi192(base_key|i, key0);
            uint32_t out0[4] = {};
            Salsa20::block_words4(key0, 0ULL, nonce8, out0);
            bool m0 = cpu_bf_detail::chacha_match_prefix_words4_host(out0, target, length);
            acc ^= out0[0]; if (!found && m0) { found = true; fk = base_key|i; }
        }
        res.found = found; res.found_key = fk;
        cpu_bf_detail::mix_sink_u64(acc ^ fk);
    };
    CPUOPT_BF_FOOTER(fn)
}

// ============================================================
// 12. Grain-128AEADv2 — Grain128WP word-parallel + early-exit + ILP2
// Grain128WP stores LFSR/NFSR as uint32_t[4] each (128 bits).
// Eliminates 127-byte memmove × (320+64+128) init clocks = 66K copies/key.
// Auth state skipped: AD bytes advance LFSR/NFSR identically without feeding back.
// ============================================================
inline CpuBFResult brute_force_cpu_opt_grain128(const uint8_t* pt, const uint8_t* ct,
                                                const uint8_t* nonce12, int length,
                                                const uint8_t* ad, int ad_len,
                                                uint64_t known_high, int unknown_bits,
                                                int repeats) {
    CPUOPT_BF_HEADER(bf_space_size_cpu(unknown_bits))
    uint8_t target[64] = {};
    cpu_bf_detail::make_target_xor(pt, ct, length, target);
    auto fn = [&]() {
        bool found = false; uint64_t fk = 0; uint64_t acc = 0;
        uint64_t i = 0;
        for (; i + 1 < N; i += 2) {
            const uint64_t k0 = base_key | i, k1 = base_key | (i+1);
            uint8_t key0[16], key1[16];
            cpu_bf_detail::key64_to_key128_zero_hi64(k0, key0);
            cpu_bf_detail::key64_to_key128_zero_hi64(k1, key1);
            const bool m0 = Grain128WP::match(key0, nonce12, ad, ad_len, target, length);
            const bool m1 = Grain128WP::match(key1, nonce12, ad, ad_len, target, length);
            acc ^= (uint64_t)(m0 ? 0xf0 : 0x0f) << (i & 7);
            if (!found && m0) { found = true; fk = k0; }
            if (!found && m1) { found = true; fk = k1; }
        }
        if (i < N) {
            uint8_t key0[16]; cpu_bf_detail::key64_to_key128_zero_hi64(base_key|i, key0);
            bool m0 = Grain128WP::match(key0, nonce12, ad, ad_len, target, length);
            acc ^= m0; if (!found && m0) { found = true; fk = base_key|i; }
        }
        res.found = found; res.found_key = fk;
        cpu_bf_detail::mix_sink_u64(acc ^ fk);
    };
    CPUOPT_BF_FOOTER(fn)
}

// ============================================================
// 13. Rocca — match_first16 early-exit + ILP2
// match_first16 signature: (pt[16], ct[16], key[32], nonce[16])
// Key sweep: low unknown_bits of K0[0..7] LE. K0[8..15]=0, K1=0.
// ============================================================
inline CpuBFResult brute_force_cpu_opt_rocca(const uint8_t* pt, const uint8_t* ct,
                                              const uint8_t* nonce,
                                              uint64_t known_high, int unknown_bits,
                                              int repeats) {
    CPUOPT_BF_HEADER(bf_space_size_cpu(unknown_bits))
    auto fn = [&]() {
        bool found = false; uint64_t fk = 0; uint64_t acc = 0;
        uint64_t i = 0;
        for (; i + 1 < N; i += 2) {
            const uint64_t k0 = base_key | i, k1 = base_key | (i+1);
            uint8_t key0[32] = {}, key1[32] = {};
            for (int j = 0; j < 8; j++) {
                key0[j] = (uint8_t)((k0 >> (8*j)) & 0xFF);
                key1[j] = (uint8_t)((k1 >> (8*j)) & 0xFF);
            }
            const bool m0 = Rocca::match_first16(pt, ct, key0, nonce);
            const bool m1 = Rocca::match_first16(pt, ct, key1, nonce);
            acc ^= (uint64_t)(m0 ? 0xa5 : 0x5a) << (i & 7);
            if (!found && m0) { found = true; fk = k0; }
            if (!found && m1) { found = true; fk = k1; }
        }
        if (i < N) {
            const uint64_t k0 = base_key | i;
            uint8_t key0[32] = {};
            for (int j = 0; j < 8; j++) key0[j] = (uint8_t)((k0 >> (8*j)) & 0xFF);
            bool m0 = Rocca::match_first16(pt, ct, key0, nonce);
            acc ^= m0; if (!found && m0) { found = true; fk = k0; }
        }
        res.found = found; res.found_key = fk;
        cpu_bf_detail::mix_sink_u64(acc ^ fk);
    };
    CPUOPT_BF_FOOTER(fn)
}

// ============================================================
// 14. Rocca-S — match_first16 early-exit + ILP2
// match_first16 signature: (pt[16], ct[16], key[32], nonce[16])
// ============================================================
inline CpuBFResult brute_force_cpu_opt_rocca_s(const uint8_t* pt, const uint8_t* ct,
                                               const uint8_t* nonce,
                                               uint64_t known_high, int unknown_bits,
                                               int repeats) {
    CPUOPT_BF_HEADER(bf_space_size_cpu(unknown_bits))
    auto fn = [&]() {
        bool found = false; uint64_t fk = 0; uint64_t acc = 0;
        uint64_t i = 0;
        for (; i + 1 < N; i += 2) {
            const uint64_t k0 = base_key | i, k1 = base_key | (i+1);
            uint8_t key0[32] = {}, key1[32] = {};
            for (int j = 0; j < 8; j++) {
                key0[j] = (uint8_t)((k0 >> (8*j)) & 0xFF);
                key1[j] = (uint8_t)((k1 >> (8*j)) & 0xFF);
            }
            const bool m0 = Rocca_S::match_first16(pt, ct, key0, nonce);
            const bool m1 = Rocca_S::match_first16(pt, ct, key1, nonce);
            acc ^= (uint64_t)(m0 ? 0xa5 : 0x5a) << (i & 7);
            if (!found && m0) { found = true; fk = k0; }
            if (!found && m1) { found = true; fk = k1; }
        }
        if (i < N) {
            const uint64_t k0 = base_key | i;
            uint8_t key0[32] = {};
            for (int j = 0; j < 8; j++) key0[j] = (uint8_t)((k0 >> (8*j)) & 0xFF);
            bool m0 = Rocca_S::match_first16(pt, ct, key0, nonce);
            acc ^= m0; if (!found && m0) { found = true; fk = k0; }
        }
        res.found = found; res.found_key = fk;
        cpu_bf_detail::mix_sink_u64(acc ^ fk);
    };
    CPUOPT_BF_FOOTER(fn)
}

#undef CPUOPT_BF_HEADER
#undef CPUOPT_BF_FOOTER
