/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

// Shared numerics of the Leiden engine (detail::leiden_engine).
//
// Everything that feeds a decision or an output is either an exact integer or an explicitly
// rounded (`_rn`) IEEE binary64 operation, so the results do not depend on thread order, launch
// geometry or the compiler's contraction choices:
//
//   - edge and vertex weights are quantised once to int64 fixed point (quantize_*);
//   - the modularity penalty lambda * k_v * K_c is an exact floor of a 64 x 64 -> 128-bit
//     product (Mult, pen);
//   - every pseudo-random choice is a counter-based hash of (seed, iteration, level, phase,
//     sweep, vertex) (ctx, sub_round, order_f), never an RNG stream;
//   - the only floating-point reduction (Q) is a fixed-order tree (tree1024_finish).
//
// The section numbers (§x.y) refer to the design notes of the rapids-singlecell native Leiden
// this engine was ported from; the algorithm and its numerics are unchanged, so both produce
// bitwise identical results for identical inputs.

#include <raft/util/cuda_rt_essentials.hpp>

#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstdint>

namespace cugraph::detail::leiden_engine {

// The engine's fixed-width types. 64-bit values are (unsigned) long long rather than
// (u)int64_t (long on LP64) because the CUDA atomics, shuffles and conversion intrinsics the
// kernels use are declared for (unsigned) long long; the representation is identical.
using i64 = long long;
using u64 = unsigned long long;
using u32 = unsigned int;

constexpr int kWarp          = 32;
constexpr unsigned kFullMask = 0xffffffffu;
constexpr int kBlock         = 256;  // default block (8 warps)
constexpr int kWarpsPerBlock = kBlock / kWarp;
constexpr int kTreeThreads   = 1024;       // TREE1024 (§9.3)
constexpr int kMaxReplicas   = 4;          // §4.2.1
constexpr int kMaxLevels     = 64;         // §4.5 level guard
constexpr i64 kMaxVertices   = 1ll << 30;  // R * n < 2^30 (§4.2, §6.1)
constexpr int kNumSubrounds  = 4;          // S (§4.6)

// ---------------------------------------------------------------------------
// Level-0 weight representation (§6.1)
// ---------------------------------------------------------------------------

// F32: the user's fp32 array, quantised on the fly with a runtime scale (2^s
//      at level 0, 1 at coarse levels where every stored value is an exact
//      integer >= 1).
// I64: a materialised int64 array of quantised weights (`W0q`): fp64 input,
//      fp32 input whose scale is outside [-126, 127], or unit weights with
//      explicit zeros. Entries that are not counted hold 0.
// UNIT: every off-diagonal stored entry has the same weight, the quantised
//      value of 1.0 (use_weights=False, or no data; canonical input without
//      explicit zeros). use_weights=False is "weight 1.0 for every entry with
//      w != 0" through the weighted path, so the unit weight is 2^s quanta
//      (bitwise equal to data[:] = 1).
enum class WKind : int { F32 = 0, I64 = 1, UNIT = 2 };

// Quantised weight of a counted entry (§4.3, normative): max(1, rint(w 2^s))
// of the exact real product, rounded once, half to even; 0 if w <= 0 (not
// counted). The diagonal is filtered separately by every caller (u != v).
//
// fp32 on the fly, valid for -126 <= s <= 127: the fp32 product is exact for
// every result >= 2^-126, and smaller products round to 0 and map to 1
// either way.
__device__ __forceinline__ i64 quantize_f32(float w, float scale)
{
  if (!(w > 0.0f)) return 0;
  const i64 q = __float2ll_rn(__fmul_rn(w, scale));
  return q > 0 ? q : 1;
}

// Exact for any s (fp32 data is widened exactly); the result is <= 2^58.
__device__ __forceinline__ i64 quantize_scaled(double w, int s)
{
  if (!(w > 0.0)) return 0;
  const i64 q = __double2ll_rn(scalbn(w, s));
  return q > 0 ? q : 1;
}

// Weight accessors used by every row kernel. `operator()(j)` returns the
// quantised weight of stored entry j, or 0 if the entry is not counted
// (w <= 0). Callers additionally skip u == v.
template <WKind K>
struct EdgeW;

template <>
struct EdgeW<WKind::F32> {
  const float* w;
  float scale;
  __device__ __forceinline__ i64 operator()(i64 j) const { return quantize_f32(w[j], scale); }
};

template <>
struct EdgeW<WKind::I64> {
  const i64* w;
  __device__ __forceinline__ i64 operator()(i64 j) const
  {
    const i64 x = w[j];
    return x > 0 ? x : 0;
  }
};

template <>
struct EdgeW<WKind::UNIT> {
  i64 q;  // quantised unit weight
  __device__ __forceinline__ i64 operator()(i64) const { return q; }
};

// ---------------------------------------------------------------------------
// Fixed-point scale (§4.3)
// ---------------------------------------------------------------------------

constexpr int kScaleBudgetLog2   = 58;    // 2m_hat <= 2^58 (1 + 2^-27)
constexpr int kF32ScaleMin       = -126;  // fp32 on-the-fly quantisation range
constexpr int kF32ScaleMax       = 127;
constexpr double kGammaHeadroom  = 16.0;
constexpr double kGammaMax       = 1048576.0;  // gamma <= 2^20 (§4.1, RA7)
constexpr i64 kMaxCountedEntries = 1ll << 40;  // nnz_c < 2^40 (§4.1, RA7)

// s0 = e - 1 with (f, e) = frexp(2^58 / Wb), Wb = (double)nnz_c (x) wmax and
// nnz_c the number of counted entries (the diagonal and explicit zeros never
// change s; red team RA4). No clamp (RA7.2). Evaluated with split exponents,
// as the golden reference does, so neither Wb nor 2^58 / Wb can overflow or
// lose bits: with wmax = f_m 2^e_m, Wb = fl(nnz_c f_m) 2^e_m = f_w 2^e_w and
// 2^58 / Wb = fl(1 / f_w) 2^(58 - e_w) (power-of-two scaling commutes with
// rounding). Equal to the literal formula wherever that one is finite.
inline int scale_s0(i64 nnz_c, double wmax)
{
  if (!(nnz_c > 0 && wmax > 0.0)) return 0;  // the scale is irrelevant
  int e_m = 0, e_w = 0, e_q = 0;
  const double f_m = std::frexp(wmax, &e_m);
  const double f_w = std::frexp(static_cast<double>(nnz_c) * f_m, &e_w);
  std::frexp(1.0 / f_w, &e_q);  // 1 / f_w in (1, 2]: e_q in {1, 2}
  return kScaleBudgetLog2 - (e_w + e_m) + (e_q - 1);
}

// h(gamma) = 0 for gamma <= 16 (including 0), else ceil(log2(gamma / 16))
// exactly from frexp (gamma / 16 is exact): e - [f == 0.5] (RA7.3).
inline int scale_headroom(double gamma)
{
  if (!(gamma > kGammaHeadroom)) return 0;
  int e          = 0;
  const double f = std::frexp(gamma / kGammaHeadroom, &e);
  return (f == 0.5) ? e - 1 : e;
}

// s(gamma) = s0 - h(gamma) keeps every penalty below 2^63 (§4.3, E15).
inline int scale_for_gamma(int s0, double gamma) { return s0 - scale_headroom(gamma); }

// Quantised weight of 1.0 at scale s: max(1, rint(2^s)). use_weights=False is
// weight 1.0 for every entry with w != 0 through the weighted path, as the
// golden reference (bitwise equal to data[:] = 1, §13.3). For unit weights
// Wb = nnz_c >= 1, so s <= 58 and the shift cannot overflow.
inline i64 unit_weight(int s) { return s >= 0 ? (1ll << (s < 62 ? s : 62)) : 1; }

// ---------------------------------------------------------------------------
// Integer penalty (§4.3): pen(A, B) = floor(B * c(A)), c(A) = lam * A
// ---------------------------------------------------------------------------

struct Mult {
  u64 M;  // c(A) = M * 2^-(64 + sh); M in [2^63, 2^64) or 0
  int sh;
};

__host__ __device__ __forceinline__ Mult make_mult(i64 a, double lam)
{
  Mult m{0ull, 0};
#ifdef __CUDA_ARCH__
  const double c = __dmul_rn(lam, __ll2double_rn(a));
#else
  const double c = lam * static_cast<double>(a);
#endif
  if (c == 0.0) return m;
  int e          = 0;
  const double f = frexp(c, &e);  // exact
#ifdef __CUDA_ARCH__
  m.M = __double2ull_rn(ldexp(f, 64));  // exact integer in [2^63, 2^64)
#else
  m.M = static_cast<u64>(ldexp(f, 64));
#endif
  m.sh = -e;
  return m;
}

// floor(B * M / 2^(64 + sh)) from the exact 128-bit product.
__host__ __device__ __forceinline__ i64 pen(i64 b, Mult m)
{
  const u64 ub = static_cast<u64>(b);
#ifdef __CUDA_ARCH__
  const u64 hi = __umul64hi(ub, m.M);
#else
  const u64 hi = static_cast<u64>((static_cast<unsigned __int128>(ub) * m.M) >> 64);
#endif
  if (m.sh >= 0) return m.sh >= 64 ? 0 : static_cast<i64>(hi >> m.sh);
  const u64 lo = ub * m.M;
  const int l  = -m.sh;  // 0 < l < 64 by the headroom proof
  return static_cast<i64>((hi << l) | (lo >> (64 - l)));
}

// ---------------------------------------------------------------------------
// Hashes and orders (§4.4, bit-exact)
// ---------------------------------------------------------------------------

__host__ __device__ __forceinline__ u64 mix64(u64 x)
{
  x ^= x >> 30;
  x *= 0xBF58476D1CE4E5B9ull;
  x ^= x >> 27;
  x *= 0x94D049BB133111EBull;
  x ^= x >> 31;
  return x;
}

__host__ __device__ __forceinline__ u32 fmix32(u32 x)
{
  x ^= x >> 16;
  x *= 0x85EBCA6Bu;
  x ^= x >> 13;
  x *= 0xC2B2AE35u;
  x ^= x >> 16;
  return x;
}

enum : u32 { kTagMove = 1, kTagOrder = 2, kTagGumbel = 3, kTagSelf = 4 };
enum : u32 { kPhaseDown = 0, kPhaseUp = 1, kPhaseTop = 2 };

__host__ __device__ __forceinline__ u64 seed64(u32 seed)
{
  return mix64(static_cast<u64>(seed) + 0x9E3779B97F4A7C15ull);
}

// Field widths: it < 2^16, l < 2^8, phase < 2^8, sweep < 2^24.
__host__ __device__ __forceinline__ u64 ctx(u64 s64, u32 tag, u32 it, u32 l, u32 phase, u32 sweep)
{
  return mix64(s64 ^ ((static_cast<u64>(tag) << 56) | (static_cast<u64>(it) << 40) |
                      (static_cast<u64>(l) << 32) | (static_cast<u64>(phase) << 24) |
                      static_cast<u64>(sweep)));
}

// Sub-round of v in a sweep: ctx_move = ctx(seed64, MOVE, it, l, phase, sweep).
__host__ __device__ __forceinline__ int sub_round(u64 ctx_move, int v, int S)
{
  return static_cast<int>((mix64(ctx_move ^ static_cast<u64>(v)) >> 32) % static_cast<u64>(S));
}

// Bijective 32-bit order hash: ctx_order = ctx(seed64, ORDER, it, l, 0, 0).
__host__ __device__ __forceinline__ u32 order_f(u64 ctx_order, int v)
{
  return fmix32(static_cast<u32>(v) ^ static_cast<u32>(ctx_order));
}

// Unique int64 order key: ord(v) = flipped ? 2^33 - f(v) : f(v).
__host__ __device__ __forceinline__ i64 order_key(u32 f, bool flipped)
{
  return flipped ? (1ll << 33) - static_cast<i64>(f) : static_cast<i64>(f);
}

// Bit pattern of a weight for the symmetry fingerprint (§4.2 I1).
__device__ __forceinline__ u64 weight_bits(float w) { return static_cast<u64>(__float_as_uint(w)); }
__device__ __forceinline__ u64 weight_bits(double w)
{
  return static_cast<u64>(__double_as_longlong(w));
}
__device__ __forceinline__ u64 weight_bits(i64 w) { return static_cast<u64>(w); }

// One fingerprint term of a counted entry (v, u, bits): forward and reverse.
__device__ __forceinline__ void fingerprint_terms(i64 v, i64 u, u64 bits, u64& fwd, u64& rev)
{
  const u64 hb = mix64(bits);
  fwd += mix64(((static_cast<u64>(v) << 32) | static_cast<u64>(u)) ^ hb);
  rev += mix64(((static_cast<u64>(u) << 32) | static_cast<u64>(v)) ^ hb);
}

// ---------------------------------------------------------------------------
// TREE1024 (§9.3): the fixed-order fp64 sum used for Q
// ---------------------------------------------------------------------------

// Every thread j of a 1024-thread block passes its strided partial
// acc_j = ((0 + t_j) + t_{j+1024}) + ... (ascending, `__dadd_rn`); the
// pairwise tree then runs for w = 512, 256, ..., 1. The sum is returned to
// every thread. `smem` holds 1024 doubles.
__device__ __forceinline__ double tree1024_finish(double acc, double* smem)
{
  const int j = threadIdx.x;
  smem[j]     = acc;
  __syncthreads();
#pragma unroll 1
  for (int w = kTreeThreads / 2; w >= 1; w >>= 1) {
    if (j < w) smem[j] = __dadd_rn(smem[j], smem[j + w]);
    __syncthreads();
  }
  const double r = smem[0];
  __syncthreads();
  return r;
}

// t_c = (K_c / 2m)^2 with explicit rounding (§4.13).
__device__ __forceinline__ double volume_term(i64 k, i64 two_m)
{
  const double x = __ddiv_rn(__ll2double_rn(k), __ll2double_rn(two_m));
  return __dmul_rn(x, x);
}

// ---------------------------------------------------------------------------
// Degree classes for row-centric kernels (§7)
// ---------------------------------------------------------------------------

enum : int { kClassLight = 0, kClassMid = 1, kClassHub = 2, kClassXl = 3 };
constexpr int kNumClasses = 4;

// Runtime thresholds (tests force every class by lowering them). The class
// only selects the kernel mapping; it never changes a result.
struct ClassThresholds {
  int light = 32;    // warp per vertex, registers only
  int mid   = 128;   // warp per vertex, 256-slot stamped table
  int hub   = 1024;  // block per vertex, 2048-slot stamped table
};  // xl: block per vertex with a global window

__host__ __device__ __forceinline__ int degree_class(i64 deg, const ClassThresholds& t)
{
  return deg <= t.light ? kClassLight
         : deg <= t.mid ? kClassMid
         : deg <= t.hub ? kClassHub
                        : kClassXl;
}

// ---------------------------------------------------------------------------
// Warp helpers (sm_75-safe: shuffles only, no __reduce_*_sync)
// ---------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ T warp_sum(T x)
{
#pragma unroll
  for (int o = kWarp / 2; o > 0; o >>= 1)
    x += __shfl_xor_sync(kFullMask, x, o);
  return x;
}

template <typename T>
__device__ __forceinline__ T warp_max(T x)
{
#pragma unroll
  for (int o = kWarp / 2; o > 0; o >>= 1) {
    const T y = __shfl_xor_sync(kFullMask, x, o);
    x         = y > x ? y : x;
  }
  return x;
}

__device__ __forceinline__ u32 warp_or(u32 x)
{
#pragma unroll
  for (int o = kWarp / 2; o > 0; o >>= 1)
    x |= __shfl_xor_sync(kFullMask, x, o);
  return x;
}

// ---------------------------------------------------------------------------
// Launch geometry (per-device cache; multi-GPU safe)
// ---------------------------------------------------------------------------

struct DeviceInfo {
  int device             = -1;
  int sm_count           = 1;
  int max_threads_per_sm = 2048;
  int max_smem_optin     = 48 * 1024;
};

inline const DeviceInfo& device_info()
{
  static thread_local DeviceInfo cached;
  int dev = 0;
  RAFT_CUDA_TRY(cudaGetDevice(&dev));
  if (dev != cached.device) {
    DeviceInfo d;
    d.device = dev;
    int v    = 0;
    RAFT_CUDA_TRY(cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev));
    d.sm_count = v;
    RAFT_CUDA_TRY(cudaDeviceGetAttribute(&v, cudaDevAttrMaxThreadsPerMultiProcessor, dev));
    d.max_threads_per_sm = v;
    RAFT_CUDA_TRY(cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    d.max_smem_optin = v;
    cached           = d;
  }
  return cached;
}

// Grid = min(#SM x resident blocks, ceil(work / items_per_block)); kernels
// use grid-stride loops, so any grid >= 1 is correct.
inline unsigned grid_for(i64 work, i64 items_per_block, int block = kBlock)
{
  const DeviceInfo& d = device_info();
  const i64 resident =
    static_cast<i64>(d.sm_count) *
    static_cast<i64>(d.max_threads_per_sm / block > 0 ? d.max_threads_per_sm / block : 1);
  i64 g = (work + items_per_block - 1) / items_per_block;
  if (g > resident) g = resident;
  if (g < 1) g = 1;
  return static_cast<unsigned>(g);
}

// Warp-per-row kernels: 8 rows per 256-thread block per grid-stride step.
inline unsigned grid_rows(i64 rows) { return grid_for(rows, kWarpsPerBlock); }
inline unsigned grid_items(i64 items) { return grid_for(items, kBlock); }

}  // namespace cugraph::detail::leiden_engine
