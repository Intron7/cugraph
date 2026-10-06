/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

// LOCAL_MOVE + COMPACT (spec §4.6, §4.7, §7 M0-M5, V1).
//
//   M0  move_level_init      K_hat / csize recounted from P (warp-aggregated;
//                            block-privatised for <= kMoveSmemIds ids, RE8)
//   M1  move_bucket          active vertices of a sweep -> buckets [S][class]
//                            (sweep 0 and TOP sweeps: every vertex)
//   M2  move_eval_light      deg <= 32: warp per vertex, registers only
//       move_eval_mid        deg <= 128: warp per vertex, 256-slot table
//       move_eval_block      hub + xl: block per vertex, 2048-slot table in
//                            hash-range passes (no global memory, §7)
//   M3  move_apply           thread per mover: aggregated K_hat / csize
//                            moves, P[v] <- c_v
//       move_activate        warp per 32 movers (rows flattened):
//                            activation against the post-sub-round state;
//                            in-sweep bucket appends
//   M5  move_compact         order-preserving relabel (= np.unique inverse)
//   V1  move_project         P_l[v] <- P_{l+1}[cmap_l[v]]
//
// Determinism: every DECIDE is a function of exact integers read from the
// frozen state of its sub-round, with a total-order argmax (score, -c);
// apply is commutative integer addition (or a store into an empty community,
// which only its mover can enter); activation runs after apply and reads the
// post-sub-round state. Bucket, list and hash-slot orders are never read by a
// decision. No kernel reads a workspace byte this LOCAL_MOVE has not written
// (§6.5; see M0 for the arrays that need no initialisation).
//
// Driver interface: run_local_move<K>, run_compact and run_project on the
// buffers of Layout::move (MoveBufs).

#include "community/detail/leiden/arena.cuh"
#include "community/detail/leiden/kernels_final.cuh"
#include "community/detail/leiden/numerics.cuh"

#include <cugraph/utilities/error.hpp>

#include <raft/util/cuda_rt_essentials.hpp>

#include <cub/device/device_scan.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace cugraph::detail::leiden_engine {

// ---------------------------------------------------------------------------
// Constants and small device helpers
// ---------------------------------------------------------------------------

constexpr int kMoveLightMaxDeg = 32;         // light kernel: one entry per lane
constexpr int kMoveMidMaxDeg   = 128;        // mid kernel: table load <= 0.5
constexpr int kMoveMidSlots    = 256;        // per warp: 3 KB
constexpr int kMoveMidBlock    = 128;        // mid kernel: 4-warp blocks (13 KB)
constexpr int kMoveHubSlots    = 2048;       // per block: 24 KB
constexpr int kMoveXlCap       = 1536;       // distinct keys per pass (load 0.75)
constexpr int kMoveSmemIds     = 4096;       // M0 block-privatised histogram
constexpr i64 kNoScore         = LLONG_MIN;  // "no candidate" (scores > -2^63)
constexpr int kEmptySlot       = -1;         // free hash-table slot

// Error bits of the move counter block (never set on valid input).
enum : int { kMoveErrLabel = 1, kMoveErrBucket = 2 };

// Sum of x over the lanes of match group m (every lane of m calls this with
// the same m; groups execute independently).
__device__ __forceinline__ i64 move_group_sum(i64 x, unsigned m)
{
  i64 s = 0;
  for (unsigned mm = m; mm; mm &= mm - 1)
    s += __shfl_sync(m, x, __ffs(mm) - 1);
  return s;
}

// sub(v) of §4.4 (numerics.cuh sub_round) with a 32-bit modulo: the hashed
// value is < 2^32, so the result is identical.
__device__ __forceinline__ int move_sub_round(u64 ctx_move, int v, int S)
{
  return static_cast<int>(static_cast<u32>(mix64(ctx_move ^ static_cast<u64>(v)) >> 32) %
                          static_cast<u32>(S));
}

// Total order of candidates: larger score, then smaller community id.
__device__ __forceinline__ bool move_better(i64 g1, int c1, i64 g2, int c2)
{
  return g1 > g2 || (g1 == g2 && c1 < c2);
}

__device__ __forceinline__ void move_warp_argmax(i64& g, int& c)
{
#pragma unroll
  for (int o = kWarp / 2; o > 0; o >>= 1) {
    const i64 og = __shfl_xor_sync(kFullMask, g, o);
    const int oc = __shfl_xor_sync(kFullMask, c, o);
    if (move_better(og, oc, g, c)) {
      g = og;
      c = oc;
    }
  }
}

// Table size for at most `keys` distinct keys at load <= 0.5 (>= 32: one slot
// per lane in the scan), capped at `slots`.
__device__ __forceinline__ int move_table_size(i64 keys, int slots)
{
  if (keys <= 16) return 32;
  const int t = 1 << (32 - __clz(static_cast<int>(2 * keys - 1)));
  return t < slots ? t : slots;
}

// Decision code written to dest's low word: the community id, plus
// kMoveFreshBit when v moves into an empty community (own id or n + v). Only
// v can enter that id in this sub-round and nobody leaves it, so M3a stores
// K_hat / csize there instead of adding (the K_hat of an id that was never
// used in this LOCAL_MOVE is not initialised, see M0).
constexpr u32 kMoveFreshBit = 0x80000000u;
constexpr u32 kMoveIdMask   = 0x7fffffffu;

// DECIDE(v) of §4.6 from the row aggregates: own = W(v, d); (bg, bc) = best
// (score, community) over c != d present in the row ((kNoScore, INT_MAX) if
// none). csize is the frozen state (read only for the empty option).
__device__ __forceinline__ u32 move_decide_final(int v,
                                                 int d,
                                                 i64 n,
                                                 i64 own,
                                                 i64 Kd,
                                                 i64 kv,
                                                 Mult mu,
                                                 i64 bg,
                                                 int bc,
                                                 const int* __restrict__ csize)
{
  const i64 rest = Kd - kv;
  const i64 stay = own - pen(rest, mu);
  i64 cs         = stay;
  u32 cc         = static_cast<u32>(d);
  if (bg > stay) {  // never true for kNoScore
    cs = bg;
    cc = static_cast<u32>(bc);
  }
  if (cs < 0 && rest > 0) {  // v not alone: the empty community (score 0)
    if (csize[v] == 0) {
      cs = 0;
      cc = static_cast<u32>(v) | kMoveFreshBit;
    } else if (csize[n + v] == 0) {  // only v may use n + v
      cs = 0;
      cc = static_cast<u32>(n + v) | kMoveFreshBit;
    }
  }
  // strict improvement (cc != d also holds for a fresh id: it was empty)
  return ((cc & kMoveIdMask) != static_cast<u32>(d) && cs > stay) ? cc : static_cast<u32>(d);
}

// ---------------------------------------------------------------------------
// Buckets and counters
// ---------------------------------------------------------------------------

// Per-sweep vertex lists. Row r of `lists` ([S][n]) holds the bucket of
// sub-round r, split into one segment per degree class: segment c starts at
// off[c] and has capacity cap[c] = number of class-c vertices of the level
// (a vertex enters at most one bucket per sweep, §4.6).
struct MoveBuckets {
  int* lists           = nullptr;
  int* counts          = nullptr;  // [S][kNumClasses]
  int* err             = nullptr;
  i64 n                = 0;
  int off[kNumClasses] = {0, 0, 0, 0};
  int cap[kNumClasses] = {0, 0, 0, 0};

  __host__ __device__ int* list(int r, int c) const
  {
    return lists + static_cast<i64>(r) * n + off[c];
  }
  __host__ __device__ int* count(int r, int c) const { return counts + r * kNumClasses + c; }
};

// Counter block in MoveBufs::bucket_count (S * kNumClasses * 4 ints):
//   [0, 4S)        bucket counts per (sub-round, class)
//   [4S, 5S)       movers per sub-round of the current sweep
//   [5S]           error bits (kMoveErr*)
//   [5S+1, 5S+6)   reserved (class counts + label check of the rsc test hooks)
struct MoveCounters {
  int* base = nullptr;
  int S     = kNumSubrounds;
  int* buckets() const { return base; }
  int* movers(int r) const { return base + S * kNumClasses + r; }
  int* err() const { return base + S * (kNumClasses + 1); }
  int* classes() const { return err() + 1; }
  static std::size_t ints(int S) { return static_cast<std::size_t>(S) * (kNumClasses + 1) + 6; }
};
static_assert(kNumClasses * 4 >= kNumClasses + 1 + 6,
              "move counters must fit MoveBufs::bucket_count");

// Warp-aggregated append of v to bucket `key` = r * kNumClasses + c (-1:
// nothing). Every lane of the warp calls this.
__device__ __forceinline__ void move_bucket_append(const MoveBuckets& B, int key, int v)
{
  const unsigned m = __match_any_sync(kFullMask, key);
  if (key < 0) return;
  const int lane   = threadIdx.x & (kWarp - 1);
  const int leader = __ffs(m) - 1;
  int base         = 0;
  if (lane == leader) base = atomicAdd(B.counts + key, __popc(m));
  base          = __shfl_sync(m, base, leader);
  const int pos = base + __popc(m & ((1u << lane) - 1u));
  const int r = key / kNumClasses, c = key - r * kNumClasses;
  if (pos < B.cap[c]) {
    B.list(r, c)[pos] = v;
  } else {
    atomicOr(B.err, kMoveErrBucket);
  }
}

// Warp-aggregated append of the lanes with `mv` to a list (all lanes call).
__device__ __forceinline__ void move_list_append(int* list, int* count, bool mv, int v)
{
  const unsigned m = __ballot_sync(kFullMask, mv);
  if (!m) return;
  const int lane   = threadIdx.x & (kWarp - 1);
  const int leader = __ffs(m) - 1;
  int base         = 0;
  if (lane == leader) base = atomicAdd(count, __popc(m));
  base = __shfl_sync(kFullMask, base, leader);
  if (mv) list[base + __popc(m & ((1u << lane) - 1u))] = v;
}

// ---------------------------------------------------------------------------
// M0: level_init
// ---------------------------------------------------------------------------

// K[c] = sum of k_hat over P[v] == c and csize[c] = |{v : P[v] == c}| for
// c in [0, id_bound) (zeroed by the caller first); csize of every other id of
// [0, 2n) is set to 0 here. K of those ids is not touched: it is read only
// while the id is in use, and its first use in this LOCAL_MOVE is a move into
// an empty community, which stores it (M3a). `dest` needs no initialisation
// either (§6.5 holds: no stale byte is ever read): it is read only for
// vertices evaluated in the same sub-round; the activity bitmaps are cleared
// by M1 (see MoveFlags). kSmem: a block-privatised histogram over
// id_bound <= kMoveSmemIds ids (few communities, e.g. V-cycle levels),
// flushed with one atomic per used id.
template <bool kSmem>
__global__ void __launch_bounds__(kBlock) move_level_init_kernel(const int* __restrict__ P,
                                                                 const i64* __restrict__ khat,
                                                                 i64 n,
                                                                 i64 id_bound,
                                                                 i64* __restrict__ Kc,
                                                                 int* __restrict__ csize,
                                                                 int* __restrict__ err)
{
  extern __shared__ __align__(16) unsigned char move_init_smem[];
  i64* sK = reinterpret_cast<i64*>(move_init_smem);
  int* sC = reinterpret_cast<int*>(sK + (kSmem ? id_bound : 0));
  if constexpr (kSmem) {
    for (i64 c = threadIdx.x; c < id_bound; c += blockDim.x) {
      sK[c] = 0;
      sC[c] = 0;
    }
    __syncthreads();
  }
  const int lane   = threadIdx.x & (kWarp - 1);
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  int bad          = 0;
  for (i64 base = static_cast<i64>(blockIdx.x) * blockDim.x + (threadIdx.x & ~(kWarp - 1));
       base < n;
       base += stride) {
    const i64 v = base + lane;
    int c       = -1;
    i64 k       = 0;
    if (v < n) {
      c = P[v];
      k = khat[v];
      if (c < 0 || c >= id_bound) {
        bad = 1;
        c   = -1;
      }
      csize[n + v] = 0;
      if (v >= id_bound) csize[v] = 0;
    }
    const unsigned m = __match_any_sync(kFullMask, c);
    const i64 sum    = move_group_sum(k, m);
    if (c >= 0 && lane == __ffs(m) - 1) {
      if constexpr (kSmem) {
        atomicAdd(reinterpret_cast<u64*>(&sK[c]), static_cast<u64>(sum));
        atomicAdd(&sC[c], __popc(m));
      } else {
        atomicAdd(reinterpret_cast<u64*>(&Kc[c]), static_cast<u64>(sum));
        atomicAdd(&csize[c], __popc(m));
      }
    }
  }
  if (bad) atomicOr(err, kMoveErrLabel);
  if constexpr (kSmem) {
    __syncthreads();
    for (i64 c = threadIdx.x; c < id_bound; c += blockDim.x) {
      if (sC[c]) {
        atomicAdd(reinterpret_cast<u64*>(&Kc[c]), static_cast<u64>(sK[c]));
        atomicAdd(&csize[c], sC[c]);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Activity flags (§4.6 `active`, as three bitmaps)
// ---------------------------------------------------------------------------

// The `active` flag of §4.6 is held as three bitmaps of one sweep, which
// carry exactly the same information without any M2 write:
//   A   active at the start of the sweep (= R of the previous sweep; sweep 0
//       and TOP sweeps are all-active and do not read it);
//   R   activated with sub(u) <= r in this sweep, i.e. after u's sub-round:
//       active[u] at the end of the sweep is exactly R[u];
//   Ap  appended in this sweep (activated with sub(u) > r).
// During sub-round r, active[u] == 1 iff (sub(u) <= r ? R[u] : A[u] | Ap[u]):
// M2 clears active[u] only at u's own sub-round, and before it only A or an
// append can have set it, after it only an R activation. The bitmaps
// (3 * ceil(n / 32) words) live in MoveBufs::rank, which COMPACT only uses
// after LOCAL_MOVE; they stay cache-resident (62.5 KB per bitmap at 500k).
struct MoveFlags {
  const u32* A = nullptr;  // null: all-active sweep
  u32* R       = nullptr;
  u32* Ap      = nullptr;
};

__device__ __forceinline__ bool move_bit(const u32* bm, int v)
{
  return (bm[v >> 5] >> (v & 31)) & 1u;
}

// ---------------------------------------------------------------------------
// M1: bucket_build
// ---------------------------------------------------------------------------

// Buckets the vertices active at the start of the sweep (F.A == null: all)
// and clears this sweep's R and Ap words (one word per 32 vertices).
__global__ void __launch_bounds__(kBlock) move_bucket_kernel(const i64* __restrict__ indptr,
                                                             i64 n,
                                                             MoveFlags F,
                                                             u64 ctx_move,
                                                             int S,
                                                             ClassThresholds th,
                                                             MoveBuckets B)
{
  const int lane   = threadIdx.x & (kWarp - 1);
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 base = static_cast<i64>(blockIdx.x) * blockDim.x + (threadIdx.x & ~(kWarp - 1));
       base < n;
       base += stride) {
    const i64 v = base + lane;
    if (lane == 0) {
      F.R[base >> 5]  = 0;
      F.Ap[base >> 5] = 0;
    }
    int key = -1;
    if (v < n) {
      if (!F.A || move_bit(F.A, static_cast<int>(v))) {
        const int r = move_sub_round(ctx_move, static_cast<int>(v), S);
        const int c = degree_class(indptr[v + 1] - indptr[v], th);
        key         = r * kNumClasses + c;
      }
    }
    move_bucket_append(B, key, static_cast<int>(v));
  }
}

// ---------------------------------------------------------------------------
// M2: DECIDE per degree class
// ---------------------------------------------------------------------------

// Frozen state of one sub-round plus the outputs of M2.
template <WKind K>
struct MoveEval {
  const i64* indptr;
  const int* indices;
  EdgeW<K> w;
  const i64* khat;
  const int* P;
  const i64* Kc;
  const int* csize;
  i64* dest;
  int* movers;
  int* mover_count;
  i64 n;
  double lam;
  u64 stamp_hi;  // stamp(sweep, r) << 32
};

// Vertices per warp of a group prefetch: spread the list over every warp of
// the grid first, then batch up to 32 (D's adaptive vpw; layout only).
__device__ __forceinline__ int move_vpw(i64 cnt, i64 nwarps)
{
  const i64 v = cnt / nwarps;
  return v < 1 ? 1 : (v > kWarp ? kWarp : static_cast<int>(v));
}

// light (deg <= 32): warp per vertex, one entry per lane, match_any groups;
// a group prefetch (lane i loads the scalars of the group's i-th vertex).
// 6 resident blocks per SM (<= 40 registers) is 1.25x faster than the
// unconstrained 54 registers at brain500k level 0 (the kernel is latency
// bound on random gathers).
template <WKind K>
__global__ void __launch_bounds__(kBlock, 6) move_eval_light_kernel(MoveEval<K> a,
                                                                    const int* __restrict__ list,
                                                                    const int* __restrict__ count,
                                                                    int cap)
{
  __shared__ i64 s_w[kWarpsPerBlock][kWarp];
  const int lane = threadIdx.x & (kWarp - 1), wib = threadIdx.x / kWarp;
  const i64 cnt = min(*count, cap);
  const i64 nw  = static_cast<i64>(gridDim.x) * kWarpsPerBlock;
  const i64 gw  = static_cast<i64>(blockIdx.x) * kWarpsPerBlock + wib;
  const int vpw = move_vpw(cnt, nw);
  for (i64 base = gw * vpw; base < cnt; base += nw * vpw) {
    int pv = -1, pd = 0;
    i64 pb = 0, pe = 0, pk = 0, pKd = 0;
    Mult pm{0ull, 0};
    if (lane < vpw && base + lane < cnt) {
      pv  = list[base + lane];
      pb  = a.indptr[pv];
      pe  = a.indptr[pv + 1];
      pd  = a.P[pv];
      pk  = a.khat[pv];
      pKd = a.Kc[pd];
      pm  = make_mult(pk, a.lam);
    }
    u32 res       = static_cast<u32>(pd);
    unsigned todo = __ballot_sync(kFullMask, pv >= 0);
    while (todo) {
      const int src = __ffs(todo) - 1;
      todo &= todo - 1;
      const int v = __shfl_sync(kFullMask, pv, src);
      const i64 b = __shfl_sync(kFullMask, pb, src);
      const i64 e = __shfl_sync(kFullMask, pe, src);
      const int d = __shfl_sync(kFullMask, pd, src);
      Mult mu;
      mu.M        = __shfl_sync(kFullMask, pm.M, src);
      mu.sh       = __shfl_sync(kFullMask, pm.sh, src);
      int c       = -1;
      i64 x       = 0;
      const i64 j = b + lane;
      if (j < e) {
        const int u = a.indices[j];
        if (u != v) {
          const i64 q = a.w(j);
          if (q > 0) {
            c = a.P[u];
            x = q;
          }
        }
      }
      const unsigned m = __match_any_sync(kFullMask, c);
      s_w[wib][lane]   = x;
      __syncwarp();
      i64 own = 0, bg = kNoScore;
      int bc = INT_MAX;
      if (c >= 0 && lane == __ffs(m) - 1) {
        i64 sum = 0;
        for (unsigned mm = m; mm; mm &= mm - 1)
          sum += s_w[wib][__ffs(mm) - 1];
        if (c == d) {
          own = sum;
        } else {
          bg = sum - pen(a.Kc[c], mu);
          bc = c;
        }
      }
      __syncwarp();
      own = warp_sum(own);
      move_warp_argmax(bg, bc);
      if (lane == src) res = move_decide_final(v, d, a.n, own, pKd, pk, mu, bg, bc, a.csize);
    }
    if (pv >= 0) { a.dest[pv] = a.stamp_hi | static_cast<u64>(res); }
    move_list_append(a.movers, a.mover_count, pv >= 0 && res != static_cast<u32>(pd), pv);
  }
}

// Hash tables (mid: one per warp; hub / xl: one per block) hold int32 keys
// (the community id; kEmptySlot when free) and int64 values. They are
// initialised at kernel start, and the reader of a vertex's table returns
// every used slot to (kEmptySlot, 0) after use, so every vertex starts on an
// empty table (§6.5). 12 bytes per slot instead of a 16-byte stamped key keeps
// more warps resident.

// Insert a pre-aggregated sum into a warp's table: one leader per distinct c
// per 32-entry chunk, so no two lanes hold the same key concurrently; the
// claimer sets the value (stale values are never read).
__device__ __forceinline__ void move_warp_table_add(int* keys, i64* vals, u32 mask, int c, i64 sum)
{
  volatile int* vk = keys;
  u32 h            = fmix32(static_cast<u32>(c)) & mask;
  while (true) {
    const int cur = vk[h];
    if (cur == c) {
      vals[h] += sum;
      return;
    }
    if (cur == kEmptySlot) {
      if (atomicCAS(&keys[h], kEmptySlot, c) == kEmptySlot) {
        vals[h] = sum;
        return;
      }
      continue;  // claimed meanwhile: re-read slot h
    }
    h = (h + 1) & mask;
  }
}

// mid (deg <= 128): warp per vertex with a 256-slot table per warp (D's
// ev2_mid128 with 4-warp blocks, its faster variant).
template <WKind K>
__global__ void __launch_bounds__(kMoveMidBlock) move_eval_mid_kernel(MoveEval<K> a,
                                                                      const int* __restrict__ list,
                                                                      const int* __restrict__ count,
                                                                      int cap)
{
  constexpr int kWarps = kMoveMidBlock / kWarp;
  __shared__ int s_key[kWarps][kMoveMidSlots];
  __shared__ i64 s_val[kWarps][kMoveMidSlots];
  __shared__ i64 s_w[kWarps][kWarp];
  const int lane = threadIdx.x & (kWarp - 1), wib = threadIdx.x / kWarp;
  int* keys = s_key[wib];
  i64* vals = s_val[wib];
  for (int s = lane; s < kMoveMidSlots; s += kWarp) {
    keys[s] = kEmptySlot;
    vals[s] = 0;
  }
  __syncwarp();
  const i64 cnt = min(*count, cap);
  const i64 nw  = static_cast<i64>(gridDim.x) * kWarps;
  const i64 gw  = static_cast<i64>(blockIdx.x) * kWarps + wib;
  const int vpw = move_vpw(cnt, nw);
  for (i64 base = gw * vpw; base < cnt; base += nw * vpw) {
    int pv = -1, pd = 0;
    i64 pb = 0, pe = 0, pk = 0, pKd = 0;
    Mult pm{0ull, 0};
    if (lane < vpw && base + lane < cnt) {
      pv  = list[base + lane];
      pb  = a.indptr[pv];
      pe  = a.indptr[pv + 1];
      pd  = a.P[pv];
      pk  = a.khat[pv];
      pKd = a.Kc[pd];
      pm  = make_mult(pk, a.lam);
    }
    u32 res       = static_cast<u32>(pd);
    unsigned todo = __ballot_sync(kFullMask, pv >= 0);
    while (todo) {
      const int src = __ffs(todo) - 1;
      todo &= todo - 1;
      const int v = __shfl_sync(kFullMask, pv, src);
      const i64 b = __shfl_sync(kFullMask, pb, src);
      const i64 e = __shfl_sync(kFullMask, pe, src);
      const int d = __shfl_sync(kFullMask, pd, src);
      Mult mu;
      mu.M            = __shfl_sync(kFullMask, pm.M, src);
      mu.sh           = __shfl_sync(kFullMask, pm.sh, src);
      const int tsize = move_table_size(e - b, kMoveMidSlots);
      const u32 mask  = static_cast<u32>(tsize - 1);
      for (i64 j0 = b; j0 < e; j0 += kWarp) {
        const i64 j = j0 + lane;
        int c       = -1;
        i64 x       = 0;
        if (j < e) {
          const int u = a.indices[j];
          if (u != v) {
            const i64 q = a.w(j);
            if (q > 0) {
              c = a.P[u];
              x = q;
            }
          }
        }
        const unsigned m = __match_any_sync(kFullMask, c);
        s_w[wib][lane]   = x;
        __syncwarp();
        if (c >= 0 && lane == __ffs(m) - 1) {
          i64 sum = 0;
          for (unsigned mm = m; mm; mm &= mm - 1)
            sum += s_w[wib][__ffs(mm) - 1];
          move_warp_table_add(keys, vals, mask, c, sum);
        }
        __syncwarp();
      }
      i64 own = 0, bg = kNoScore;
      int bc = INT_MAX;
      for (int s = lane; s < tsize; s += kWarp) {
        const int c = keys[s];
        if (c == kEmptySlot) continue;
        const i64 sum = vals[s];
        keys[s]       = kEmptySlot;  // the next vertex starts empty
        if (c == d) {
          own = sum;
        } else {
          const i64 g = sum - pen(a.Kc[c], mu);
          if (move_better(g, c, bg, bc)) {
            bg = g;
            bc = c;
          }
        }
      }
      own = warp_sum(own);
      move_warp_argmax(bg, bc);
      if (lane == src) res = move_decide_final(v, d, a.n, own, pKd, pk, mu, bg, bc, a.csize);
      __syncwarp();  // the scan finishes before the next claims
    }
    if (pv >= 0) { a.dest[pv] = a.stamp_hi | static_cast<u64>(res); }
    move_list_append(a.movers, a.mover_count, pv >= 0 && res != static_cast<u32>(pd), pv);
  }
}

// Insert into the block table (warp leaders, pre-aggregated per 32-entry
// chunk; several warps may hold the same key). Values are 0 on an empty slot,
// so every inserter adds. Returns false if the pass must be split: more than
// `cap` distinct keys were claimed, or no free slot was found within one
// sweep of the table.
__device__ __forceinline__ bool move_block_table_add(
  int* keys, i64* vals, u32 mask, int c, i64 sum, int* claimed, int cap)
{
  volatile int* vk = keys;
  u32 h            = fmix32(static_cast<u32>(c)) & mask;
  for (u32 probes = 0; probes <= mask;) {
    const int cur = vk[h];
    if (cur == c) {
      atomicAdd(reinterpret_cast<u64*>(&vals[h]), static_cast<u64>(sum));
      return true;
    }
    if (cur == kEmptySlot) {
      const int prev = atomicCAS(&keys[h], kEmptySlot, c);
      if (prev == kEmptySlot) {
        atomicAdd(reinterpret_cast<u64*>(&vals[h]), static_cast<u64>(sum));
        return atomicAdd(claimed, 1) < cap;
      }
      continue;  // claimed by another inserter: re-read slot h
    }
    h = (h + 1) & mask;
    ++probes;
  }
  return false;
}

// Hash-range pass of community c among npass passes (fmix32 is a bijection,
// so doubling npass always terminates; passes nest: pass p of npass splits
// into 2p, 2p + 1 of 2 npass).
__device__ __forceinline__ u64 move_pass_of(int c, u64 npass)
{
  return (static_cast<u64>(fmix32(static_cast<u32>(c))) * npass) >> 32;
}

// hub (deg <= 1024) and xl (> 1024): block per vertex with the 2048-slot
// table in shared memory (no global memory, §7). Each vertex runs hash-range
// passes: npass starts at ceil(deg / pass_cap) and doubles whenever a pass
// claims more than pass_cap distinct keys (the aborted pass's slots are
// re-initialised and the vertex resumes at pass 2p); the running argmax over
// (score, -c) and W(v, d) are carried across passes, so the result does not
// depend on npass. Hub vertices (deg <= 1024 < pass_cap) take one pass.
template <WKind K>
__global__ void __launch_bounds__(kBlock) move_eval_block_kernel(MoveEval<K> a,
                                                                 const int* __restrict__ list_hub,
                                                                 const int* __restrict__ count_hub,
                                                                 int cap_hub,
                                                                 const int* __restrict__ list_xl,
                                                                 const int* __restrict__ count_xl,
                                                                 int cap_xl,
                                                                 int pass_cap)
{
  __shared__ int s_key[kMoveHubSlots];
  __shared__ i64 s_val[kMoveHubSlots];
  __shared__ i64 s_w[kWarpsPerBlock][kWarp];
  __shared__ i64 s_g[kWarpsPerBlock];
  __shared__ int s_c[kWarpsPerBlock];
  __shared__ i64 s_own;
  __shared__ int s_claimed;
  __shared__ int s_ovf;
  const int tid  = threadIdx.x;
  const int lane = tid & (kWarp - 1), wib = tid / kWarp;
  for (int s = tid; s < kMoveHubSlots; s += blockDim.x) {
    s_key[s] = kEmptySlot;
    s_val[s] = 0;
  }
  const i64 n1 = min(*count_hub, cap_hub);
  const i64 n2 = min(*count_xl, cap_xl);
  __syncthreads();
  for (i64 i = blockIdx.x; i < n1 + n2; i += gridDim.x) {
    const int v = i < n1 ? list_hub[i] : list_xl[i - n1];
    const i64 b = a.indptr[v], e = a.indptr[v + 1];
    const int d   = a.P[v];
    const i64 kv  = a.khat[v];
    const Mult mu = make_mult(kv, a.lam);
    const i64 deg = e - b;
    const int tsize =
      move_table_size(deg < pass_cap ? deg : static_cast<i64>(pass_cap), kMoveHubSlots);
    const u32 mask = static_cast<u32>(tsize - 1);
    u64 npass      = deg > pass_cap ? static_cast<u64>((deg + pass_cap - 1) / pass_cap) : 1ull;
    u64 p          = 0;
    i64 rg         = kNoScore;  // running argmax (thread 0)
    int rc         = INT_MAX;
    if (tid == 0) s_own = 0;
    while (p < npass) {
      if (tid == 0) {
        s_claimed = 0;
        s_ovf     = 0;
      }
      __syncthreads();
      for (i64 j0 = b + static_cast<i64>(wib) * kWarp; j0 < e; j0 += blockDim.x) {
        const i64 j = j0 + lane;
        int c       = -1;
        i64 x       = 0;
        if (j < e) {
          const int u = a.indices[j];
          if (u != v) {
            const i64 q = a.w(j);
            if (q > 0) {
              const int cu = a.P[u];
              if (npass == 1 || move_pass_of(cu, npass) == p) {
                c = cu;
                x = q;
              }
            }
          }
        }
        const unsigned m = __match_any_sync(kFullMask, c);
        s_w[wib][lane]   = x;
        __syncwarp();
        if (c >= 0 && lane == __ffs(m) - 1) {
          i64 sum = 0;
          for (unsigned mm = m; mm; mm &= mm - 1)
            sum += s_w[wib][__ffs(mm) - 1];
          if (!move_block_table_add(s_key, s_val, mask, c, sum, &s_claimed, pass_cap))
            atomicOr(&s_ovf, 1);  // read after the barrier
        }
        __syncwarp();
      }
      __syncthreads();
      const bool ovf = s_ovf != 0;
      __syncthreads();  // every thread has read s_ovf
      if (ovf) {        // split the pass: clear its slots, resume at 2p of 2
                        // npass
        for (int s = tid; s < tsize; s += blockDim.x) {
          s_key[s] = kEmptySlot;
          s_val[s] = 0;
        }
        npass *= 2;
        p *= 2;
        continue;  // the loop head synchronises before the next pass
      }
      i64 g  = kNoScore;
      int cc = INT_MAX;
      for (int s = tid; s < tsize; s += blockDim.x) {
        const int c = s_key[s];
        if (c == kEmptySlot) continue;
        const i64 sum = s_val[s];
        s_key[s]      = kEmptySlot;  // the next pass / vertex starts empty
        s_val[s]      = 0;
        if (c == d) {
          s_own = sum;
        } else {
          const i64 sc = sum - pen(a.Kc[c], mu);
          if (move_better(sc, c, g, cc)) {
            g  = sc;
            cc = c;
          }
        }
      }
      move_warp_argmax(g, cc);
      if (lane == 0) {
        s_g[wib] = g;
        s_c[wib] = cc;
      }
      __syncthreads();
      if (wib == 0) {
        g  = lane < kWarpsPerBlock ? s_g[lane] : kNoScore;
        cc = lane < kWarpsPerBlock ? s_c[lane] : INT_MAX;
        move_warp_argmax(g, cc);
        if (lane == 0 && move_better(g, cc, rg, rc)) {
          rg = g;
          rc = cc;
        }
      }
      __syncthreads();
      ++p;
    }
    if (tid == 0) {
      const u32 res = move_decide_final(v, d, a.n, s_own, a.Kc[d], kv, mu, rg, rc, a.csize);
      a.dest[v]     = a.stamp_hi | static_cast<u64>(res);
      if (res != static_cast<u32>(d)) a.movers[atomicAdd(a.mover_count, 1)] = v;
    }
    __syncthreads();  // s_own
  }
}

// ---------------------------------------------------------------------------
// M3: apply, then activate
// ---------------------------------------------------------------------------

// M3a, thread per mover: K_hat / csize moves aggregated per source and per
// destination community (match_any + one atomic per group; exact and
// order-free); a move into an empty community (kMoveFreshBit) stores
// K_hat[c] = k_hat_v and csize[c] = 1 (v is its only entrant, nobody leaves
// it); P[v] <- c_v.
__global__ void __launch_bounds__(kBlock) move_apply_kernel(const i64* __restrict__ khat,
                                                            int* __restrict__ P,
                                                            i64* __restrict__ Kc,
                                                            int* __restrict__ csize,
                                                            const i64* __restrict__ dest,
                                                            const int* __restrict__ movers,
                                                            const int* __restrict__ mover_count,
                                                            int mover_cap)
{
  const int lane   = threadIdx.x & (kWarp - 1);
  const i64 cnt    = min(*mover_count, mover_cap);
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 base = static_cast<i64>(blockIdx.x) * blockDim.x + (threadIdx.x & ~(kWarp - 1));
       base < cnt;
       base += stride) {
    const i64 i = base + lane;
    int v = -1, d = -1, c = -1;
    bool fresh = false;
    i64 k      = 0;
    if (i < cnt) {
      v              = movers[i];
      d              = P[v];
      const u32 code = static_cast<u32>(dest[v]);
      c              = static_cast<int>(code & kMoveIdMask);
      fresh          = (code & kMoveFreshBit) != 0;
      k              = khat[v];
    }
    {
      const unsigned m = __match_any_sync(kFullMask, d);
      const i64 s      = move_group_sum(k, m);
      if (d >= 0 && lane == __ffs(m) - 1) {
        atomicAdd(reinterpret_cast<u64*>(&Kc[d]), static_cast<u64>(-s));
        atomicSub(&csize[d], __popc(m));
      }
    }
    {
      const int ca     = fresh ? -1 : c;
      const unsigned m = __match_any_sync(kFullMask, ca);
      const i64 s      = move_group_sum(k, m);
      if (ca >= 0 && lane == __ffs(m) - 1) {
        atomicAdd(reinterpret_cast<u64*>(&Kc[ca]), static_cast<u64>(s));
        atomicAdd(&csize[ca], __popc(m));
      }
    }
    if (fresh) {
      Kc[c]    = k;
      csize[c] = 1;
    }
    if (v >= 0) P[v] = c;
  }
}

// M3b, warp per 32 movers, after M3a (P is the post-sub-round state): for
// every counted (v, u) of a mover v with P[u] != P[v] and u inactive, u is
// activated; if sub(u) > r it is appended to bucket [sub(u)][class(u)] (the
// atomicOr on Ap admits one append per vertex and sweep) and evaluated later
// in this sweep (§4.6; flags as in MoveFlags). Shortcuts that change no
// result:
//   - do_high = false: every vertex was active at the start of the sweep
//     (sweep 0), so a u with sub(u) > r is still unevaluated, hence active;
//   - do_low = false: in the last sweep of the phase, activations with
//     sub(u) <= r (which only feed the next sweep) are unobservable;
//   - an activation flag that is already set is stable within the kernel,
//     so it is tested before the P[u] load.
// (TOP sweeps skip M3b: every TOP sweep restarts all-active.) A warp takes up
// to 32 movers (adaptive, as move_vpw) and processes their rows flattened
// (lane f of a chunk takes the f-th entry of the concatenated rows), so no
// lane idles on short rows.
template <WKind K>
__global__ void __launch_bounds__(kBlock) move_activate_kernel(const i64* __restrict__ indptr,
                                                               const int* __restrict__ indices,
                                                               EdgeW<K> w,
                                                               const int* __restrict__ P,
                                                               MoveFlags F,
                                                               const int* __restrict__ movers,
                                                               const int* __restrict__ mover_count,
                                                               int mover_cap,
                                                               int r,
                                                               u64 ctx_move,
                                                               int S,
                                                               bool do_low,
                                                               bool do_high,
                                                               ClassThresholds th,
                                                               MoveBuckets B)
{
  const int lane = threadIdx.x & (kWarp - 1);
  const i64 cnt  = min(*mover_count, mover_cap);
  const i64 nw   = static_cast<i64>(gridDim.x) * kWarpsPerBlock;
  const i64 gw   = static_cast<i64>(blockIdx.x) * kWarpsPerBlock + threadIdx.x / kWarp;
  // movers per warp: few movers (coarse levels, late sweeps) are spread
  // over every warp first, their long rows would serialise otherwise
  const int mpw = move_vpw(cnt, nw);
  for (i64 base = gw * mpw; base < cnt; base += nw * mpw) {
    const i64 i = base + lane;
    int v = -1, c = -1;
    i64 rb = 0, deg = 0;
    if (lane < mpw && i < cnt) {
      v   = movers[i];
      c   = P[v];
      rb  = indptr[v];
      deg = indptr[v + 1] - rb;
    }
    // inclusive scan of the row lengths over the warp
    i64 off = deg;
#pragma unroll
    for (int o = 1; o < kWarp; o <<= 1) {
      const i64 y = __shfl_up_sync(kFullMask, off, o);
      if (lane >= o) off += y;
    }
    const i64 total = __shfl_sync(kFullMask, off, kWarp - 1);
    for (i64 f0 = 0; f0 < total; f0 += kWarp) {
      const i64 f = f0 + lane;
      // owner: the first lane whose inclusive offset exceeds f
      int pos = 0;
#pragma unroll
      for (int bstep = kWarp / 2; bstep >= 1; bstep >>= 1) {
        const i64 o = __shfl_sync(kFullMask, off, pos + bstep - 1);
        if (o <= f) pos += bstep;
      }
      const i64 o_end = __shfl_sync(kFullMask, off, pos);
      const i64 o_deg = __shfl_sync(kFullMask, deg, pos);
      const i64 o_b   = __shfl_sync(kFullMask, rb, pos);
      const int x     = __shfl_sync(kFullMask, v, pos);
      const int cx    = __shfl_sync(kFullMask, c, pos);
      int key = -1, u = 0;
      if (f < total) {
        const i64 j = o_b + (f - (o_end - o_deg));
        u           = indices[j];
        if (u != x) {
          const int su  = move_sub_round(ctx_move, u, S);
          const u32 bit = 1u << (u & 31);
          if (su <= r) {
            if (do_low && !(F.R[u >> 5] & bit) && w(j) > 0 && P[u] != cx)
              atomicOr(&F.R[u >> 5], bit);
          } else if (do_high && !((F.A[u >> 5] | F.Ap[u >> 5]) & bit) && w(j) > 0 && P[u] != cx &&
                     !(atomicOr(&F.Ap[u >> 5], bit) & bit)) {
            key = su * kNumClasses + degree_class(indptr[u + 1] - indptr[u], th);
          }
        }
      }
      move_bucket_append(B, key, u);
    }
  }
}

// ---------------------------------------------------------------------------
// M5: COMPACT (order-preserving relabel, §4.7) and V1: project
// ---------------------------------------------------------------------------

struct MoveUsedFlag {
  const int* csize;
  i64 n2;
  __host__ __device__ int operator()(i64 i) const { return (i < n2 && csize[i] > 0) ? 1 : 0; }
};

// rank = exclusive_scan(csize > 0) over [0, 2n] (rank[2n] = C); P[v] <-
// rank[P[v]]; KS[rank[c]] <- K[c] for every used c.
__global__ void move_compact_kernel(const int* __restrict__ rank,
                                    const int* __restrict__ csize,
                                    const i64* __restrict__ Kc,
                                    i64 n,
                                    int* __restrict__ P,
                                    i64* __restrict__ KS)
{
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 i = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; i < 2 * n; i += stride) {
    if (csize[i] > 0) KS[rank[i]] = Kc[i];
    if (i < n) P[i] = rank[P[i]];
  }
}

// V1: P_fine[v] = P_coarse[cmap[v]].
__global__ void move_project_kernel(const int* __restrict__ Pc,
                                    const int* __restrict__ cmap,
                                    i64 n_fine,
                                    int* __restrict__ Pf)
{
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 v = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; v < n_fine; v += stride)
    Pf[v] = Pc[cmap[v]];
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

struct MoveParams {
  double lam = 0.0;  // lam_hat = gamma / 2m_hat (one host division)
  u32 seed   = 0;
  u32 it     = 0;
  u32 level  = 0;
  u32 phase  = kPhaseDown;
  u32 sweep0 = 0;  // TOP: TOP sweeps already run at this level (§4.4)
  int cap    = 4;  // DOWN / UP 4, TOP 32
  int S      = kNumSubrounds;
  ClassThresholds th{};
  int xl_cap   = kMoveXlCap;  // distinct keys per hash-range pass
  i64 id_bound = -1;          // P ids in [0, id_bound) on entry; -1: n
};

struct MoveStats {
  i64 moves  = 0;
  int sweeps = 0;
  i64 last   = 0;                 // movers of the last executed sweep
  std::vector<i64> per_sweep;     // [cap]
  std::vector<i64> per_subround;  // [cap * S]
};

// Small D2H readback (a sync point): through the caller's pinned buffer when
// given, else pageable memory (correct, only slower).
inline void move_readback(
  void* host, const void* dev, std::size_t bytes, void* pinned, cudaStream_t s)
{
  void* dst = pinned ? pinned : host;
  RAFT_CUDA_TRY(cudaMemcpyAsync(dst, dev, bytes, cudaMemcpyDeviceToHost, s));
  RAFT_CUDA_TRY(cudaStreamSynchronize(s));
  if (pinned) std::memcpy(host, pinned, bytes);
}

inline void move_check_params(const MoveParams& p)
{
  CUGRAPH_EXPECTS(p.S >= 1 && p.S <= 64, "leiden: subrounds must be in [1, 64].");
  CUGRAPH_EXPECTS(p.cap >= 0 && p.cap <= (1 << 24), "leiden: sweep cap must be in [0, 2^24].");
  CUGRAPH_EXPECTS(static_cast<u64>(p.sweep0) + static_cast<u64>(p.cap) <= (1ull << 24),
                  "leiden: sweep0 + cap must be <= 2^24.");
  CUGRAPH_EXPECTS(p.it < (1u << 16) && p.level < 256u && p.phase <= kPhaseTop,
                  "leiden: iteration, level or phase out of range.");
  CUGRAPH_EXPECTS(p.lam >= 0.0 && std::isfinite(p.lam), "leiden: lambda must be finite and >= 0.");
  CUGRAPH_EXPECTS(p.th.light >= 0 && p.th.light <= kMoveLightMaxDeg && p.th.mid >= p.th.light &&
                    p.th.mid <= kMoveMidMaxDeg && p.th.hub >= p.th.mid,
                  "leiden: class thresholds need 0 <= light <= 32, "
                  "light <= mid <= 128, hub >= mid.");
  CUGRAPH_EXPECTS(p.xl_cap >= 1 && p.xl_cap <= kMoveXlCap, "leiden: xl_cap must be in [1, 1536].");
}

// M0: K_hat and csize of the level.
inline void run_level_init(
  const int* P, const i64* khat, i64 n, i64 id_bound, const MoveBufs& mb, int* err, cudaStream_t s)
{
  RAFT_CUDA_TRY(cudaMemsetAsync(mb.K, 0, id_bound * sizeof(i64), s));
  RAFT_CUDA_TRY(cudaMemsetAsync(mb.csize, 0, id_bound * sizeof(int), s));
  const unsigned g = grid_items(n);
  if (id_bound <= kMoveSmemIds) {
    const std::size_t smem = static_cast<std::size_t>(id_bound) * (sizeof(i64) + sizeof(int));
    move_level_init_kernel<true><<<g, kBlock, smem, s>>>(P, khat, n, id_bound, mb.K, mb.csize, err);
  } else {
    move_level_init_kernel<false><<<g, kBlock, 0, s>>>(P, khat, n, id_bound, mb.K, mb.csize, err);
  }
  RAFT_CHECK_CUDA(s);
}

inline MoveBuckets move_buckets(const MoveBufs& mb,
                                int S,
                                i64 n,
                                const i64 (&class_count)[kNumClasses])
{
  const MoveCounters ctr{mb.bucket_count, S};
  MoveBuckets B;
  B.lists  = mb.buckets;
  B.counts = ctr.buckets();
  B.err    = ctr.err();
  B.n      = n;
  i64 off  = 0;
  for (int c = 0; c < kNumClasses; ++c) {
    B.off[c] = static_cast<int>(off);
    B.cap[c] = static_cast<int>(class_count[c]);
    off += class_count[c];
  }
  CUGRAPH_EXPECTS(off == n, "leiden: class counts must sum to n.");
  return B;
}

// M2 for sub-round r: one kernel per class present at the level (the bucket
// lists and counts are device-resident, so an empty bucket costs at most a
// no-op launch).
template <WKind K>
void run_move_eval(const MoveEval<K>& a,
                   const MoveBuckets& B,
                   int r,
                   const i64 (&cc)[kNumClasses],
                   int xl_cap,
                   cudaStream_t s)
{
  if (cc[kClassLight] > 0) {
    move_eval_light_kernel<K><<<grid_for(cc[kClassLight], kWarpsPerBlock), kBlock, 0, s>>>(
      a, B.list(r, kClassLight), B.count(r, kClassLight), B.cap[kClassLight]);
    RAFT_CHECK_CUDA(s);
  }
  if (cc[kClassMid] > 0) {
    move_eval_mid_kernel<K>
      <<<grid_for(cc[kClassMid], kMoveMidBlock / kWarp, kMoveMidBlock), kMoveMidBlock, 0, s>>>(
        a, B.list(r, kClassMid), B.count(r, kClassMid), B.cap[kClassMid]);
    RAFT_CHECK_CUDA(s);
  }
  const i64 nb = cc[kClassHub] + cc[kClassXl];
  if (nb > 0) {
    move_eval_block_kernel<K><<<grid_for(nb, 1), kBlock, 0, s>>>(a,
                                                                 B.list(r, kClassHub),
                                                                 B.count(r, kClassHub),
                                                                 B.cap[kClassHub],
                                                                 B.list(r, kClassXl),
                                                                 B.count(r, kClassXl),
                                                                 B.cap[kClassXl],
                                                                 xl_cap);
    RAFT_CHECK_CUDA(s);
  }
}

// LOCAL_MOVE(G, k_hat, P, cap, phase) of §4.6 on a level of n vertices with
// P ids in [0, id_bound) on entry; afterwards P holds ids in [0, 2n) (not
// compacted: run_compact does that). class_count: vertices per degree class
// of this level with the thresholds prm.th (S2 / SL2).
// One readback (SW) per sweep.
template <WKind K>
MoveStats run_local_move(const i64* indptr,
                         const int* indices,
                         EdgeW<K> w,
                         const i64* khat,
                         int* P,
                         i64 n,
                         const i64 (&class_count)[kNumClasses],
                         const MoveParams& prm,
                         const MoveBufs& mb,
                         void* pinned,
                         cudaStream_t s)
{
  move_check_params(prm);
  const int S = prm.S;
  MoveStats st;
  st.per_sweep.assign(prm.cap, 0);
  st.per_subround.assign(static_cast<std::size_t>(prm.cap) * S, 0);
  if (n == 0) {  // every sweep is empty: the first one ends the phase
    st.sweeps = prm.cap > 0 ? 1 : 0;
    return st;
  }
  const i64 id_bound = prm.id_bound < 0 ? n : prm.id_bound;
  CUGRAPH_EXPECTS(id_bound >= 1 && id_bound <= n, "leiden: id_bound must be in [1, n].");
  const MoveCounters ctr{mb.bucket_count, S};
  const MoveBuckets B = move_buckets(mb, S, n, class_count);
  RAFT_CUDA_TRY(cudaMemsetAsync(ctr.base, 0, MoveCounters::ints(S) * sizeof(int), s));
  run_level_init(P, khat, n, id_bound, mb, ctr.err(), s);  // M0

  const u64 s64           = seed64(prm.seed);
  const unsigned g_items  = grid_items(n);
  const unsigned g_movers = grid_for(n, kBlock);
  MoveEval<K> a{
    indptr, indices, w, khat, P, mb.K, mb.csize, mb.dest, mb.movers, nullptr, n, prm.lam, 0ull};
  // activity bitmaps (MoveFlags) in the rank buffer: R of sweep s is
  // bm[s % 2], A of sweep s is R of sweep s - 1
  const i64 words = (n + 31) / 32;
  u32* bm[2]      = {reinterpret_cast<u32*>(mb.rank), reinterpret_cast<u32*>(mb.rank) + words};
  u32* ap         = reinterpret_cast<u32*>(mb.rank) + 2 * words;
  std::vector<int> h(S + 1);
  for (int sweep = 0; sweep < prm.cap; ++sweep) {
    const u64 cm =
      ctx(s64, kTagMove, prm.it, prm.level, prm.phase, prm.sweep0 + static_cast<u32>(sweep));
    RAFT_CUDA_TRY(cudaMemsetAsync(
      ctr.base, 0, static_cast<std::size_t>(S) * (kNumClasses + 1) * sizeof(int), s));
    // M3b work that can be observed (see move_activate_kernel): TOP
    // sweeps restart all-active; sweep 0 starts all-active (M0); the last
    // sweep feeds no next sweep.
    const bool top     = prm.phase == kPhaseTop;
    const bool do_low  = !top && sweep + 1 < prm.cap;
    const bool do_high = !top && sweep > 0;
    // sweep 0 and every TOP sweep are all-active (no A)
    MoveFlags F;
    F.A  = (top || sweep == 0) ? nullptr : bm[(sweep + 1) % 2];
    F.R  = bm[sweep % 2];
    F.Ap = ap;
    move_bucket_kernel<<<g_items, kBlock, 0, s>>>(indptr, n, F, cm, S, prm.th, B);
    RAFT_CHECK_CUDA(s);
    for (int r = 0; r < S; ++r) {
      const u32 stamp = 1u + static_cast<u32>(sweep) * S + r;
      a.stamp_hi      = static_cast<u64>(stamp) << 32;
      a.mover_count   = ctr.movers(r);
      run_move_eval<K>(a, B, r, class_count, prm.xl_cap, s);
      move_apply_kernel<<<g_items, kBlock, 0, s>>>(
        khat, P, mb.K, mb.csize, mb.dest, mb.movers, ctr.movers(r), static_cast<int>(n));
      RAFT_CHECK_CUDA(s);
      if (do_low || do_high) {
        move_activate_kernel<K><<<g_movers, kBlock, 0, s>>>(indptr,
                                                            indices,
                                                            w,
                                                            P,
                                                            F,
                                                            mb.movers,
                                                            ctr.movers(r),
                                                            static_cast<int>(n),
                                                            r,
                                                            cm,
                                                            S,
                                                            do_low,
                                                            do_high,
                                                            prm.th,
                                                            B);
        RAFT_CHECK_CUDA(s);
      }
    }
    // SW: movers per sub-round of this sweep + the error bits
    move_readback(h.data(), ctr.movers(0), (S + 1) * sizeof(int), pinned, s);
    CUGRAPH_EXPECTS(!(h[S] & kMoveErrLabel),
                    "leiden: local move: community ids must "
                    "lie in [0, id_bound).");
    CUGRAPH_EXPECTS(!(h[S] & kMoveErrBucket), "leiden: local move: bucket overflow.");
    i64 moves = 0;
    for (int r = 0; r < S; ++r) {
      st.per_subround[static_cast<std::size_t>(sweep) * S + r] = h[r];
      moves += h[r];
    }
    st.per_sweep[sweep] = moves;
    st.moves += moves;
    st.last = moves;
    ++st.sweeps;
    if (moves == 0) break;  // DOWN / UP: converged; TOP: certified
  }
  return st;
}

// COMPACT (§4.7) after LOCAL_MOVE: P <- rank of its id among the used ids of
// [0, 2n) (csize > 0), KS[rank[c]] <- K[c]; returns C (sync SL1).
inline i64 run_compact(int* P,
                       i64 n,
                       const MoveBufs& mb,
                       i64* KS,
                       void* cub,
                       std::size_t cub_bytes,
                       void* pinned,
                       cudaStream_t s)
{
  if (n == 0) return 0;
  const i64 n2 = 2 * n;
  auto flags =
    thrust::make_transform_iterator(thrust::counting_iterator<i64>(0), MoveUsedFlag{mb.csize, n2});
  const int items  = cub_items(n2 + 1, "compact scan");
  std::size_t need = 0;
  RAFT_CUDA_TRY(cub::DeviceScan::ExclusiveSum(nullptr, need, flags, mb.rank, items, s));
  CUGRAPH_EXPECTS(need <= cub_bytes, "leiden: CUB temp storage too small (compact).");
  std::size_t tb = cub_bytes;
  RAFT_CUDA_TRY(cub::DeviceScan::ExclusiveSum(cub, tb, flags, mb.rank, items, s));
  move_compact_kernel<<<grid_items(n2), kBlock, 0, s>>>(mb.rank, mb.csize, mb.K, n, P, KS);
  RAFT_CHECK_CUDA(s);
  int C = 0;
  move_readback(&C, mb.rank + n2, sizeof(int), pinned, s);
  return C;
}

// V1: P_fine[v] <- P_coarse[cmap[v]] (V-cycle projection / flatten).
inline void run_project(
  const int* P_coarse, const int* cmap, i64 n_fine, int* P_fine, cudaStream_t s)
{
  if (n_fine == 0) return;
  move_project_kernel<<<grid_items(n_fine), kBlock, 0, s>>>(P_coarse, cmap, n_fine, P_fine);
  RAFT_CHECK_CUDA(s);
}

}  // namespace cugraph::detail::leiden_engine
