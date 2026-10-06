/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

// PLEIDEN_R refinement (spec §4.8, §7 R1-R9): a spanning forest inside each
// move community S with a deterministic heaviest-modularity-edge choice, then
// the longest valid prefix of every tree (in ord order) is committed.
//
// R1 refine_ext_u: ext_v = w(v, S_v minus v) and the R-test
//    U_v = ext_v >= pen_v(K_S - k_v).
// R2 refine_choose: hn_v = argmax of (key(v, u), -u) over the U neighbours in
//    S with key >= 0, else v; root_v = (hn_v == v ? v : -1), initialised
//    here by v's own thread (red team RA6).
// R3 forest_chase1: a separate launch; chases towards lower f and writes only
//    root_v (its own) and root_now = now, never -1.
// R4 forest_flip: flipped_v = (root_v == -1), the post-R3 snapshot;
//    forest_chase2: flipped vertices chase along ord to a rooted vertex
//    (reading the snapshot, so no read races a root write) and count the
//    tree sizes (R5).
// R5 scan + tree_scatter: members grouped by root (any order inside a tree);
//    r_v = v; trees listed by commit class.
// R7 refine_inner: inner_v = w(v, members of v's tree preceding it in ord).
// R8 tree_commit: the longest valid prefix, per class: thread per tree (<= 8
//    members: sorting network in registers; almost every tree of a kNN
//    level), warp per tree (<= 32: bitonic sort, warp scans, ballot of the
//    first failure), block per tree (<= 1024: BlockRadixSort + BlockScan),
//    larger trees: DeviceSegmentedSort over just those trees, then a chunked
//    block scan per tree.
// R9 host_flags + scan + cmap: coarse id = rank of the host id among hosts.
//
// The row kernels (R1, R2, R7) use a group prefetch: a warp takes 32
// consecutive rows and loads their metadata with one coalesced load per lane.
//
// Every decision input is an exact int64 (fixed point, §4.3); orders are
// bijective hashes (§4.4), so there are no ties and the result depends neither
// on thread interleaving, launch geometry nor the commit class of a tree. All
// arrays are written before they are read within one call (§6.5); counters
// live in the Control block and are zeroed by the kernel that starts their
// accumulation. The coarse node weights k_hat_{l+1} are summed from the members
// by AGGREGATE (red team RA2); R8's k_hat' (kprime) only serves the debug
// assertion and the tests.

#include "community/detail/leiden/arena.cuh"
#include "community/detail/leiden/kernels_final.cuh"
#include "community/detail/leiden/numerics.cuh"

#include <cugraph/utilities/error.hpp>

#include <raft/util/cuda_rt_essentials.hpp>

#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_sort.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace cugraph::detail::leiden_engine {

// Error bit for a violated forest invariant (debug check of §4.8 invariants
// 1-2; never expected). Shares the Control flags word with the ingest bits.
constexpr u32 kFlagForestInvariant = 1u << 24;

// Tree commit classes (R8): runtime thresholds, so tests can force each path.
struct TreeClasses {
  int thread = 8;     // <= 8 members: thread per tree (sorting network)
  int warp   = 32;    // <= 32 members: warp per tree (bitonic sort)
  int block  = 1024;  // <= 1024 members: block per tree (BlockRadixSort)
};  // larger trees: segmented sort + chunked block scan
// Commit-class lists (R5): thread class from the front of tree_list, warp
// class from its back; block class from the back of large_list, larger trees
// from its front. Counts in Control::tree_list_count[class].
enum : int { kTreeThread = 0, kTreeWarp = 1, kTreeBlock = 2, kTreeLarge = 3 };
constexpr int kTinyTree = 8;  // register arrays of the thread class

constexpr int kCommitBlock = 256;                     // threads of the block commit kernels
constexpr int kCommitItems = 4;                       // items per thread: 1024 members per block
constexpr int kOrdBits     = 34;                      // ord(v) <= 2^33
constexpr u64 kOrdPad      = (1ull << kOrdBits) - 1;  // sorts after every ord

// ord(v) of §4.4: flipped ? 2^33 - f(v) : f(v); unique within a level.
__device__ __forceinline__ i64 refine_ord(u64 ctx_order, int v, bool flipped)
{
  return order_key(order_f(ctx_order, v), flipped);
}

// Slot of the next item of a warp-aggregated append (one atomic per warp);
// -1 for lanes with pred false. Every lane of the warp must call it.
__device__ __forceinline__ int refine_list_append(bool pred, int* counter)
{
  const unsigned m = __ballot_sync(kFullMask, pred);
  if (!m) return -1;
  const int lane   = threadIdx.x & (kWarp - 1);
  const int leader = __ffs(m) - 1;
  int base         = 0;
  if (lane == leader) base = atomicAdd(counter, __popc(m));
  base = __shfl_sync(kFullMask, base, leader);
  return pred ? base + __popc(m & ((1u << lane) - 1u)) : -1;
}

// ---------------------------------------------------------------------------
// R1 / R2 / R7: warp per row (sums and per-entry argmaxes; no tables)
// ---------------------------------------------------------------------------
//
// Group prefetch (as D's move kernels): a warp takes 32 consecutive rows;
// lane i loads row base + i's offsets and per-vertex state, then the warp
// walks the rows with that state broadcast by shuffles, so no row's edge loads
// wait on a dependent metadata load, and the per-vertex epilogue (fixed-point
// multiplier, R-test) runs on all lanes at once.

// R1: ext_v and the R-test U_v (>= as igraph leiden.c:410-413). Labels outside
// [0, C) set kFlagBadLabel.
template <WKind K>
__global__ void refine_ext_u_kernel(const i64* __restrict__ indptr,
                                    const int* __restrict__ indices,
                                    EdgeW<K> w,
                                    const i64* __restrict__ khat,
                                    const int* __restrict__ S,
                                    const i64* __restrict__ KS,
                                    i64 n,
                                    i64 C,
                                    double lam,
                                    i64* __restrict__ ext,
                                    unsigned char* __restrict__ U,
                                    Control* ctl)
{
  const int lane   = threadIdx.x & (kWarp - 1);
  const i64 warp0  = static_cast<i64>(blockIdx.x) * kWarpsPerBlock + threadIdx.x / kWarp;
  const i64 nwarps = static_cast<i64>(gridDim.x) * kWarpsPerBlock;
  for (i64 base = warp0 * kWarp; base < n; base += nwarps * kWarp) {
    const i64 v = base + lane;
    i64 b = 0, e = 0;
    int s = -1;
    if (v < n) {
      b = indptr[v];
      e = indptr[v + 1];
      s = S[v];
      if (s < 0 || s >= C) {
        atomicOr(&ctl->flags, kFlagBadLabel);
        s = -1;
      }
    }
    const int rows = n - base < kWarp ? static_cast<int>(n - base) : kWarp;
    i64 mine       = 0;
    for (int q = 0; q < rows; ++q) {
      const int sq = __shfl_sync(kFullMask, s, q);
      const i64 bq = __shfl_sync(kFullMask, b, q);
      const i64 eq = __shfl_sync(kFullMask, e, q);
      const i64 vq = base + q;
      i64 acc      = 0;
      if (sq >= 0) {  // warp-uniform
        for (i64 j = bq + lane; j < eq; j += kWarp) {
          const int u = indices[j];
          if (u == vq) continue;
          const i64 x = w(j);
          if (x > 0 && S[u] == sq) acc += x;
        }
      }
      acc = warp_sum(acc);
      if (lane == q) mine = acc;
    }
    if (v < n) {
      ext[v] = mine;
      if (s < 0) {
        U[v] = 0;
      } else {
        const i64 kv = khat[v];
        U[v]         = mine >= pen(KS[s] - kv, make_mult(kv, lam)) ? 1 : 0;
      }
    }
  }
}

// R2: hn_v = argmax over counted (v, u) with S_u = S_v, U_u and
// key(v, u) = w_vu - pen_v(k_u) >= 0 of (key, -u); v itself if U_v fails or
// there is no such u. root_v is initialised here, by v's own thread.
template <WKind K>
__global__ void refine_choose_kernel(const i64* __restrict__ indptr,
                                     const int* __restrict__ indices,
                                     EdgeW<K> w,
                                     const i64* __restrict__ khat,
                                     const int* __restrict__ S,
                                     const unsigned char* __restrict__ U,
                                     i64 n,
                                     double lam,
                                     int* __restrict__ hn,
                                     int* __restrict__ root)
{
  const int lane   = threadIdx.x & (kWarp - 1);
  const i64 warp0  = static_cast<i64>(blockIdx.x) * kWarpsPerBlock + threadIdx.x / kWarp;
  const i64 nwarps = static_cast<i64>(gridDim.x) * kWarpsPerBlock;
  for (i64 base = warp0 * kWarp; base < n; base += nwarps * kWarp) {
    const i64 v = base + lane;
    i64 b = 0, e = 0;
    int s = -1;  // -1: v fails the R-test (no edge choice)
    Mult mv{0ull, 0};
    if (v < n && U[v]) {
      b  = indptr[v];
      e  = indptr[v + 1];
      s  = S[v];
      mv = make_mult(khat[v], lam);
    }
    const int rows = n - base < kWarp ? static_cast<int>(n - base) : kWarp;
    int mine       = -1;
    for (int q = 0; q < rows; ++q) {
      const int sq = __shfl_sync(kFullMask, s, q);
      if (sq < 0) continue;  // warp-uniform
      const i64 bq = __shfl_sync(kFullMask, b, q);
      const i64 eq = __shfl_sync(kFullMask, e, q);
      const Mult mq{__shfl_sync(kFullMask, mv.M, q), __shfl_sync(kFullMask, mv.sh, q)};
      const i64 vq = base + q;
      i64 bk       = -1;       // best key (valid keys are >= 0)
      int bu       = INT_MAX;  // its vertex
      for (i64 j = bq + lane; j < eq; j += kWarp) {
        const int u = indices[j];
        if (u == vq || S[u] != sq || !U[u]) continue;
        const i64 x = w(j);
        if (x <= 0) continue;  // not counted
        const i64 key = x - pen(khat[u], mq);
        if (key > bk || (key == bk && u < bu)) {
          bk = key;
          bu = u;
        }
      }
#pragma unroll
      for (int o = kWarp / 2; o > 0; o >>= 1) {
        const i64 ok = __shfl_xor_sync(kFullMask, bk, o);
        const int ou = __shfl_xor_sync(kFullMask, bu, o);
        if (ok > bk || (ok == bk && ou < bu)) {
          bk = ok;
          bu = ou;
        }
      }
      if (lane == q && bk >= 0) mine = bu;
    }
    if (v < n) {
      const int h = mine >= 0 ? mine : static_cast<int>(v);
      hn[v]       = h;
      root[v]     = h == v ? static_cast<int>(v) : -1;
    }
  }
}

// R7: inner_v = sum of counted w_vu with root_u = root_v and ord(u) < ord(v)
// (0 for singleton trees; tree_offset gives the tree sizes).
template <WKind K>
__global__ void refine_inner_kernel(const i64* __restrict__ indptr,
                                    const int* __restrict__ indices,
                                    EdgeW<K> w,
                                    const int* __restrict__ root,
                                    const unsigned char* __restrict__ flipped,
                                    const int* __restrict__ tree_offset,
                                    i64 n,
                                    u64 ctx_order,
                                    i64* __restrict__ inner)
{
  const int lane   = threadIdx.x & (kWarp - 1);
  const i64 warp0  = static_cast<i64>(blockIdx.x) * kWarpsPerBlock + threadIdx.x / kWarp;
  const i64 nwarps = static_cast<i64>(gridDim.x) * kWarpsPerBlock;
  for (i64 base = warp0 * kWarp; base < n; base += nwarps * kWarp) {
    const i64 v = base + lane;
    i64 b = 0, e = 0, ov = 0;
    int rv = -1;  // -1: singleton tree (inner = 0)
    if (v < n) {
      const int r0 = root[v];
      if (tree_offset[r0 + 1] - tree_offset[r0] > 1) {
        rv = r0;
        b  = indptr[v];
        e  = indptr[v + 1];
        ov = refine_ord(ctx_order, static_cast<int>(v), flipped[v] != 0);
      }
    }
    const int rows = n - base < kWarp ? static_cast<int>(n - base) : kWarp;
    i64 mine       = 0;
    for (int q = 0; q < rows; ++q) {
      const int rq = __shfl_sync(kFullMask, rv, q);
      if (rq < 0) continue;  // warp-uniform
      const i64 bq = __shfl_sync(kFullMask, b, q);
      const i64 eq = __shfl_sync(kFullMask, e, q);
      const i64 oq = __shfl_sync(kFullMask, ov, q);
      const i64 vq = base + q;
      i64 acc      = 0;
      for (i64 j = bq + lane; j < eq; j += kWarp) {
        const int u = indices[j];
        if (u == vq || root[u] != rq) continue;
        if (refine_ord(ctx_order, u, flipped[u] != 0) >= oq) continue;
        const i64 x = w(j);
        if (x > 0) acc += x;
      }
      acc = warp_sum(acc);
      if (lane == q) mine = acc;
    }
    if (v < n) inner[v] = mine;
  }
}

// ---------------------------------------------------------------------------
// R3 / R4 / R5: forest extraction and grouping (thread per vertex)
// ---------------------------------------------------------------------------

// R3 (separate launch after R2): for hn_v != v chase towards lower f; if the
// chase moved, root_v <- now and root_now <- now. It never reads root and
// never writes -1, and a chase endpoint never moves itself, so every
// interleaving gives the same root.
__global__ void forest_chase1_kernel(const int* __restrict__ hn,
                                     i64 n,
                                     u64 ctx_order,
                                     int* __restrict__ root)
{
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 i = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
    const int v = static_cast<int>(i);
    int nxt     = hn[v];
    if (nxt == v) continue;
    int now  = v;
    u32 fnow = order_f(ctx_order, now), fnxt = order_f(ctx_order, nxt);
    while (fnow > fnxt) {
      now  = nxt;
      fnow = fnxt;
      nxt  = hn[now];
      fnxt = order_f(ctx_order, nxt);
    }
    if (now != v) {
      root[v]   = now;
      root[now] = now;
    }
  }
}

// R4a: flipped_v <- (root_v == -1), the snapshot R4b reads. Also zeroes the
// tree counts [0, n] (R5) and the commit-class counters.
__global__ void forest_flip_kernel(const int* __restrict__ root,
                                   i64 n,
                                   unsigned char* __restrict__ flipped,
                                   int* __restrict__ tree_count,
                                   Control* ctl)
{
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 v = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; v <= n; v += stride) {
    tree_count[v] = 0;
    if (v < n) flipped[v] = root[v] == -1 ? 1 : 0;
  }
  if (blockIdx.x == 0 && threadIdx.x < 4) ctl->tree_list_count[threadIdx.x] = 0;
}

// R4b + R5 count: a flipped v chases while the current vertex is flipped and
// ord decreases; the chase ends at a rooted (non-flipped) vertex, whose root
// is final after R3 and never written here. Then tree_count[root_v] += 1.
__global__ void forest_chase2_kernel(const int* __restrict__ hn,
                                     const unsigned char* __restrict__ flipped,
                                     i64 n,
                                     u64 ctx_order,
                                     int* root,
                                     int* __restrict__ tree_count,
                                     Control* ctl)
{
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 i = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
    const int v = static_cast<int>(i);
    int rv;
    if (flipped[v]) {
      int now = v, nxt = hn[v];
      i64 onow  = refine_ord(ctx_order, now, true);
      bool fnow = true;
      while (fnow) {
        const bool fnxt = flipped[nxt] != 0;
        const i64 onxt  = refine_ord(ctx_order, nxt, fnxt);
        if (onow <= onxt) break;
        now  = nxt;
        onow = onxt;
        fnow = fnxt;
        nxt  = hn[now];
      }
      if (fnow) {  // invariant 1 violated: ended at an unrooted vertex
        atomicOr(&ctl->flags, kFlagForestInvariant);
        rv = v;
      } else {
        rv = root[now];
      }
      root[v] = rv;
    } else {
      rv = root[v];
    }
    atomicAdd(&tree_count[rv], 1);
  }
}

// R5 scatter: members grouped by root (the cursor is the remaining count);
// r_v <- v and kprime_v <- k_v (R8 overwrites the committed ones); roots of
// trees with >= 2 members are listed by commit class; members of large trees
// also write their segmented-sort key (ord) and value.
__global__ void tree_scatter_kernel(const int* __restrict__ root,
                                    const unsigned char* __restrict__ flipped,
                                    const i64* __restrict__ khat,
                                    const int* __restrict__ tree_offset,
                                    i64 n,
                                    u64 ctx_order,
                                    TreeClasses tc,
                                    int* tree_count,
                                    int* __restrict__ members,
                                    int* __restrict__ r,
                                    i64* __restrict__ kprime,
                                    int* __restrict__ tree_list,
                                    int* __restrict__ large_list,
                                    i64* __restrict__ seg_keys,
                                    int* __restrict__ seg_vals,
                                    Control* ctl)
{
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  // whole warps iterate together (refine_list_append uses the full mask)
  for (i64 base = static_cast<i64>(blockIdx.x) * blockDim.x + (threadIdx.x & ~(kWarp - 1));
       base < n;
       base += stride) {
    const i64 v = base + (threadIdx.x & (kWarp - 1));
    int cls     = -1;
    if (v < n) {
      const int rv   = root[v];
      const int o    = tree_offset[rv];
      const int size = tree_offset[rv + 1] - o;
      const int pos  = o + atomicSub(&tree_count[rv], 1) - 1;
      members[pos]   = static_cast<int>(v);
      r[v]           = static_cast<int>(v);
      kprime[v]      = khat[v];
      if (size > tc.block) {
        seg_keys[pos] = refine_ord(ctx_order, static_cast<int>(v), flipped[v] != 0);
        seg_vals[pos] = static_cast<int>(v);
      }
      if (rv == v && size >= 2)
        cls = size <= tc.thread  ? kTreeThread
              : size <= tc.warp  ? kTreeWarp
              : size <= tc.block ? kTreeBlock
                                 : kTreeLarge;
    }
    int* cnt     = ctl->tree_list_count;
    const int it = refine_list_append(cls == kTreeThread, &cnt[kTreeThread]);
    const int iw = refine_list_append(cls == kTreeWarp, &cnt[kTreeWarp]);
    const int ib = refine_list_append(cls == kTreeBlock, &cnt[kTreeBlock]);
    const int il = refine_list_append(cls == kTreeLarge, &cnt[kTreeLarge]);
    if (it >= 0) tree_list[it] = static_cast<int>(v);
    if (iw >= 0) tree_list[n - 1 - iw] = static_cast<int>(v);
    if (ib >= 0) large_list[n - 1 - ib] = static_cast<int>(v);
    if (il >= 0) large_list[il] = static_cast<int>(v);
  }
}

// ---------------------------------------------------------------------------
// R8: longest valid prefix (exact T-test and gain test, §4.3)
// ---------------------------------------------------------------------------

// Member x at sorted position j >= 1 of a tree, with the prefix sums
// (Kpre, EXTpre) over positions < j, joins iff the prefix is well connected
// (EXTpre >= pen(Kpre, K_S - Kpre)) and joining gains (inner_x >= pen_x(Kpre)).
// Because ord(root) is the tree minimum, inner_root = 0, so EXTpre is the
// exclusive prefix sum of ext - 2 inner (the cut of the prefix in S).
__device__ __forceinline__ bool refine_commit_ok(
  i64 kpre, i64 extpre, i64 ks, i64 kx, i64 ix, double lam)
{
  return extpre >= pen(ks - kpre, make_mult(kpre, lam)) && ix >= pen(kpre, make_mult(kx, lam));
}

template <typename T>
__device__ __forceinline__ T refine_warp_scan(T x)
{
  const int lane = threadIdx.x & (kWarp - 1);
  T incl         = x;
#pragma unroll
  for (int o = 1; o < kWarp; o <<= 1) {
    const T y = __shfl_up_sync(kFullMask, incl, o);
    if (lane >= o) incl += y;
  }
  return incl - x;
}

// Ascending bitonic sort of (key, val) across the 32 lanes (keys unique, pads
// carry kOrdPad).
__device__ __forceinline__ void refine_warp_sort(u64& key, int& val)
{
  const int lane = threadIdx.x & (kWarp - 1);
#pragma unroll
  for (int k = 2; k <= kWarp; k <<= 1) {
#pragma unroll
    for (int j = k >> 1; j > 0; j >>= 1) {
      const u64 ok        = __shfl_xor_sync(kFullMask, key, j);
      const int ov        = __shfl_xor_sync(kFullMask, val, j);
      const bool keep_min = ((lane & j) == 0) == ((lane & k) == 0);
      if (keep_min ? ok < key : ok > key) {
        key = ok;
        val = ov;
      }
    }
  }
}

// Thread per tree of up to kTinyTree members (almost every tree of a kNN
// level has 2-4 members): odd-even transposition sort of (ord, member) in
// registers, then the prefix walk in ord order until the first failure.
__global__ void tree_commit_thread_kernel(const int* __restrict__ tree_list,
                                          const int* __restrict__ tree_offset,
                                          const int* __restrict__ members,
                                          const unsigned char* __restrict__ flipped,
                                          const i64* __restrict__ khat,
                                          const i64* __restrict__ ext,
                                          const i64* __restrict__ inner,
                                          const int* __restrict__ S,
                                          const i64* __restrict__ KS,
                                          double lam,
                                          u64 ctx_order,
                                          int* __restrict__ r,
                                          i64* __restrict__ kprime,
                                          Control* ctl)
{
  const i64 count  = ctl->tree_list_count[kTreeThread];
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 i = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += stride) {
    const int t    = tree_list[i];
    const int o    = tree_offset[t];
    const int size = tree_offset[t + 1] - o;
    u64 key[kTinyTree];
    int x[kTinyTree];
#pragma unroll
    for (int k = 0; k < kTinyTree; ++k) {
      key[k] = kOrdPad;
      x[k]   = -1;
      if (k < size) {
        x[k]   = members[o + k];
        key[k] = static_cast<u64>(refine_ord(ctx_order, x[k], flipped[x[k]] != 0));
      }
    }
#pragma unroll
    for (int rnd = 0; rnd < kTinyTree; ++rnd) {
#pragma unroll
      for (int k = rnd & 1; k + 1 < kTinyTree; k += 2) {
        if (key[k] > key[k + 1]) {
          const u64 tk = key[k];
          key[k]       = key[k + 1];
          key[k + 1]   = tk;
          const int tx = x[k];
          x[k]         = x[k + 1];
          x[k + 1]     = tx;
        }
      }
    }
    const int x0 = x[0];
    const i64 ks = KS[S[x0]];
    i64 kpre     = khat[x0];
    i64 extpre   = ext[x0] - 2 * inner[x0];
    bool open    = true;
#pragma unroll
    for (int k = 1; k < kTinyTree; ++k) {
      if (open && k < size) {
        const int xk = x[k];
        const i64 kx = khat[xk], ix = inner[xk];
        if (refine_commit_ok(kpre, extpre, ks, kx, ix, lam)) {
          r[xk] = x0;
          kpre += kx;
          extpre += ext[xk] - 2 * ix;
        } else {
          open = false;
        }
      }
    }
    kprime[x0] = kpre;
    if (x0 != t) atomicOr(&ctl->flags, kFlagForestInvariant);
  }
}

// Warp per tree of up to 32 members.
__global__ void tree_commit_warp_kernel(const int* __restrict__ tree_list,
                                        i64 n,
                                        const int* __restrict__ tree_offset,
                                        const int* __restrict__ members,
                                        const unsigned char* __restrict__ flipped,
                                        const i64* __restrict__ khat,
                                        const i64* __restrict__ ext,
                                        const i64* __restrict__ inner,
                                        const int* __restrict__ S,
                                        const i64* __restrict__ KS,
                                        double lam,
                                        u64 ctx_order,
                                        int* __restrict__ r,
                                        i64* __restrict__ kprime,
                                        Control* ctl)
{
  const int lane   = threadIdx.x & (kWarp - 1);
  const i64 warp0  = static_cast<i64>(blockIdx.x) * kWarpsPerBlock + threadIdx.x / kWarp;
  const i64 nwarps = static_cast<i64>(gridDim.x) * kWarpsPerBlock;
  const i64 count  = ctl->tree_list_count[kTreeWarp];
  for (i64 i = warp0; i < count; i += nwarps) {
    const int t    = tree_list[n - 1 - i];
    const int o    = tree_offset[t];
    const int size = tree_offset[t + 1] - o;
    u64 key        = kOrdPad;
    int x          = -1;
    if (lane < size) {
      x   = members[o + lane];
      key = static_cast<u64>(refine_ord(ctx_order, x, flipped[x] != 0));
    }
    refine_warp_sort(key, x);
    const bool valid = lane < size;
    i64 kx = 0, dx = 0, ix = 0;
    if (valid) {
      kx = khat[x];
      ix = inner[x];
      dx = ext[x] - 2 * ix;
    }
    const i64 kpre      = refine_warp_scan(kx);
    const i64 extpre    = refine_warp_scan(dx);
    const int x0        = __shfl_sync(kFullMask, x, 0);
    const i64 ks        = KS[S[x0]];
    const bool ok       = !valid || lane == 0 || refine_commit_ok(kpre, extpre, ks, kx, ix, lam);
    const unsigned fail = __ballot_sync(kFullMask, !ok);
    const int p         = fail ? __ffs(fail) - 1 : size;
    if (valid && lane >= 1 && lane < p) r[x] = x0;
    // k_hat' of the host x0: the prefix sum at p (the total if no failure)
    const i64 kp = __shfl_sync(kFullMask, kpre + kx, p - 1);
    if (lane == 0) {
      kprime[x0] = kp;
      if (x0 != t) atomicOr(&ctl->flags, kFlagForestInvariant);
    }
  }
}

// Block per tree of up to kCommitBlock * kCommitItems members.
__global__ void __launch_bounds__(kCommitBlock)
  tree_commit_block_kernel(const int* __restrict__ large_list,
                           i64 n,
                           const int* __restrict__ tree_offset,
                           const int* __restrict__ members,
                           const unsigned char* __restrict__ flipped,
                           const i64* __restrict__ khat,
                           const i64* __restrict__ ext,
                           const i64* __restrict__ inner,
                           const int* __restrict__ S,
                           const i64* __restrict__ KS,
                           double lam,
                           u64 ctx_order,
                           int* __restrict__ r,
                           i64* __restrict__ kprime,
                           Control* ctl)
{
  using Sort = cub::BlockRadixSort<u64, kCommitBlock, kCommitItems, int>;
  using Scan = cub::BlockScan<i64, kCommitBlock>;
  __shared__ union {
    typename Sort::TempStorage sort;
    typename Scan::TempStorage scan;
  } tmp;
  __shared__ int s_x0;
  __shared__ int s_fail;
  const int tid   = threadIdx.x;
  const i64 count = ctl->tree_list_count[kTreeBlock];
  for (i64 i = blockIdx.x; i < count; i += gridDim.x) {
    const int t    = large_list[n - 1 - i];
    const int o    = tree_offset[t];
    const int size = tree_offset[t + 1] - o;
    u64 key[kCommitItems];
    int x[kCommitItems];
#pragma unroll
    for (int k = 0; k < kCommitItems; ++k) {
      const int j = tid * kCommitItems + k;
      key[k]      = kOrdPad;
      x[k]        = -1;
      if (j < size) {
        x[k]   = members[o + j];
        key[k] = static_cast<u64>(refine_ord(ctx_order, x[k], flipped[x[k]] != 0));
      }
    }
    if (tid == 0) s_fail = INT_MAX;
    Sort(tmp.sort).Sort(key, x, 0, kOrdBits);
    __syncthreads();
    i64 kx[kCommitItems], dx[kCommitItems], ix[kCommitItems];
#pragma unroll
    for (int k = 0; k < kCommitItems; ++k) {
      const int j = tid * kCommitItems + k;
      kx[k] = dx[k] = ix[k] = 0;
      if (j < size) {
        kx[k] = khat[x[k]];
        ix[k] = inner[x[k]];
        dx[k] = ext[x[k]] - 2 * ix[k];
      }
    }
    if (tid == 0) s_x0 = x[0];
    i64 kpre[kCommitItems], extpre[kCommitItems];
    Scan(tmp.scan).ExclusiveSum(kx, kpre);
    __syncthreads();
    Scan(tmp.scan).ExclusiveSum(dx, extpre);
    __syncthreads();
    const int x0 = s_x0;
    const i64 ks = KS[S[x0]];
#pragma unroll
    for (int k = 0; k < kCommitItems; ++k) {
      const int j = tid * kCommitItems + k;
      if (j >= 1 && j < size && !refine_commit_ok(kpre[k], extpre[k], ks, kx[k], ix[k], lam))
        atomicMin(&s_fail, j);
    }
    __syncthreads();
    const int p = s_fail < size ? s_fail : size;
#pragma unroll
    for (int k = 0; k < kCommitItems; ++k) {
      const int j = tid * kCommitItems + k;
      if (j >= 1 && j < p) r[x[k]] = x0;
      if (j == p - 1) kprime[x0] = kpre[k] + kx[k];
    }
    if (tid == 0 && x0 != t) atomicOr(&ctl->flags, kFlagForestInvariant);
    __syncthreads();
  }
}

// Block per large tree: members already sorted by ord (segmented sort) in
// sorted_vals[tree_offset[t] ...); chunks of kCommitBlock members with carried
// prefix sums until the first failure.
__global__ void __launch_bounds__(kCommitBlock)
  tree_commit_large_kernel(const int* __restrict__ large_list,
                           const int* __restrict__ tree_offset,
                           const int* __restrict__ sorted_vals,
                           const i64* __restrict__ khat,
                           const i64* __restrict__ ext,
                           const i64* __restrict__ inner,
                           const int* __restrict__ S,
                           const i64* __restrict__ KS,
                           double lam,
                           int* __restrict__ r,
                           i64* __restrict__ kprime,
                           Control* ctl)
{
  using Scan = cub::BlockScan<i64, kCommitBlock>;
  __shared__ typename Scan::TempStorage tmp;
  __shared__ int s_fail;
  const int tid   = threadIdx.x;
  const i64 count = ctl->tree_list_count[kTreeLarge];
  for (i64 i = blockIdx.x; i < count; i += gridDim.x) {
    const int t    = large_list[i];
    const int o    = tree_offset[t];
    const int size = tree_offset[t + 1] - o;
    const int x0   = sorted_vals[o];
    const i64 ks   = KS[S[x0]];
    i64 carry_k = 0, carry_d = 0;
    int p = size;
    for (int c0 = 0; c0 < size; c0 += kCommitBlock) {
      const int j      = c0 + tid;
      const bool valid = j < size;
      int x            = -1;
      i64 kx = 0, dx = 0, ix = 0;
      if (valid) {
        x  = sorted_vals[o + j];
        kx = khat[x];
        ix = inner[x];
        dx = ext[x] - 2 * ix;
      }
      if (tid == 0) s_fail = INT_MAX;
      i64 ek, ed, tk, td;
      Scan(tmp).ExclusiveSum(kx, ek, tk);
      __syncthreads();
      Scan(tmp).ExclusiveSum(dx, ed, td);
      const i64 kpre = carry_k + ek, extpre = carry_d + ed;
      if (valid && j >= 1 && !refine_commit_ok(kpre, extpre, ks, kx, ix, lam))
        atomicMin(&s_fail, j);
      __syncthreads();
      const int f = s_fail;
      if (valid && j >= 1 && j < f) r[x] = x0;
      if (j == f) kprime[x0] = kpre;
      __syncthreads();
      if (f != INT_MAX) {  // block-uniform
        p = f;
        break;
      }
      carry_k += tk;
      carry_d += td;
    }
    if (tid == 0) {
      if (p == size) kprime[x0] = carry_k;
      if (x0 != t) atomicOr(&ctl->flags, kFlagForestInvariant);
    }
    __syncthreads();
  }
}

// Segment offsets of the large trees for DeviceSegmentedSort: segment i is
// tree large_list[i] for i < count, empty otherwise (the host knows only the
// bound n / (block + 1) on the number of large trees).
struct LargeSegmentBegin {
  const int* list;
  const int* count;
  const int* offset;
  __host__ __device__ int operator()(int i) const { return i < *count ? offset[list[i]] : 0; }
};
struct LargeSegmentEnd {
  const int* list;
  const int* count;
  const int* offset;
  __host__ __device__ int operator()(int i) const { return i < *count ? offset[list[i] + 1] : 0; }
};

// ---------------------------------------------------------------------------
// R9: hosts, coarse ids, cmap
// ---------------------------------------------------------------------------

__global__ void host_flags_kernel(const int* __restrict__ r, i64 n, int* __restrict__ flags)
{
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 v = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; v <= n; v += stride)
    flags[v] = v < n && r[v] == v ? 1 : 0;
}

// cmap_v = cid[r_v], the rank of v's host among the hosts (P-independent and
// monotone); n_{l+1} = cid[n], device-resident until SL2.
__global__ void cmap_kernel(const int* __restrict__ r,
                            const int* __restrict__ cid,
                            i64 n,
                            int* __restrict__ cmap,
                            Control* ctl)
{
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 v = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; v < n; v += stride)
    cmap[v] = cid[r[v]];
  if (blockIdx.x == 0 && threadIdx.x == 0) ctl->n_next = cid[n];
}

// ---------------------------------------------------------------------------
// Host runner
// ---------------------------------------------------------------------------

struct RefineParams {
  double lam    = 0.0;  // lambda_hat = gamma / 2m_hat
  u64 ctx_order = 0;    // ctx(seed64, ORDER, it, l, 0, 0)
  TreeClasses trees;
};

inline void refine_check_cub(std::size_t need, std::size_t have, const char* what)
{
  CUGRAPH_EXPECTS(need <= have,
                  "leiden: CUB temp storage for %s exceeds the "
                  "carved region.",
                  what);
}

// PLEIDEN_R on a level graph (indptr, indices, w) with n vertices, vertex
// weights khat, compacted move partition S (ids in [0, C)) and its volumes
// KS. Writes cmap (n entries: the caller's level-arena slot) and ctl->n_next
// (device-resident; the driver reads it at SL2). No host synchronisation.
// rb keeps every intermediate array (ext, U, hn, root, flipped, inner, r,
// kprime, ...) until the aggregate phase reuses the bytes.
template <WKind K>
void run_refine(const i64* indptr,
                const int* indices,
                EdgeW<K> w,
                const i64* khat,
                const int* S,
                const i64* KS,
                i64 n,
                i64 C,
                const RefineParams& rp,
                const RefineBufs& rb,
                int* cmap,
                void* cub,
                std::size_t cub_bytes,
                Control* ctl,
                cudaStream_t s)
{
  if (n == 0) {
    RAFT_CUDA_TRY(cudaMemsetAsync(&ctl->n_next, 0, sizeof(i64), s));
    return;
  }
  // row kernels: one warp per 32 rows (group prefetch)
  const unsigned gr = grid_items(n), gi = grid_items(n);
  // R1, R2
  refine_ext_u_kernel<K>
    <<<gr, kBlock, 0, s>>>(indptr, indices, w, khat, S, KS, n, C, rp.lam, rb.ext, rb.U, ctl);
  RAFT_CHECK_CUDA(s);
  refine_choose_kernel<K>
    <<<gr, kBlock, 0, s>>>(indptr, indices, w, khat, S, rb.U, n, rp.lam, rb.hn, rb.root);
  RAFT_CHECK_CUDA(s);
  // R3 (separate launch), R4, R5
  forest_chase1_kernel<<<gi, kBlock, 0, s>>>(rb.hn, n, rp.ctx_order, rb.root);
  RAFT_CHECK_CUDA(s);
  forest_flip_kernel<<<grid_items(n + 1), kBlock, 0, s>>>(
    rb.root, n, rb.flipped, rb.tree_count, ctl);
  RAFT_CHECK_CUDA(s);
  forest_chase2_kernel<<<gi, kBlock, 0, s>>>(
    rb.hn, rb.flipped, n, rp.ctx_order, rb.root, rb.tree_count, ctl);
  RAFT_CHECK_CUDA(s);
  std::size_t tb = cub_bytes;
  RAFT_CUDA_TRY(cub::DeviceScan::ExclusiveSum(
    cub, tb, rb.tree_count, rb.tree_offset, cub_items(n + 1, "tree offsets"), s));
  tree_scatter_kernel<<<gi, kBlock, 0, s>>>(rb.root,
                                            rb.flipped,
                                            khat,
                                            rb.tree_offset,
                                            n,
                                            rp.ctx_order,
                                            rp.trees,
                                            rb.tree_count,
                                            rb.members,
                                            rb.r,
                                            rb.kprime,
                                            rb.tree_list,
                                            rb.large_list,
                                            rb.sort_keys_a,
                                            rb.sort_vals_a,
                                            ctl);
  RAFT_CHECK_CUDA(s);
  // R7
  refine_inner_kernel<K><<<gr, kBlock, 0, s>>>(
    indptr, indices, w, rb.root, rb.flipped, rb.tree_offset, n, rp.ctx_order, rb.inner);
  RAFT_CHECK_CUDA(s);
  // R8: every class runs over a device-resident count (no host sync); a
  // class whose trees cannot exist at this n is not launched.
  const TreeClasses& tc = rp.trees;
  if (n >= 2) {
    tree_commit_thread_kernel<<<grid_items(n / 2 + 1), kBlock, 0, s>>>(rb.tree_list,
                                                                       rb.tree_offset,
                                                                       rb.members,
                                                                       rb.flipped,
                                                                       khat,
                                                                       rb.ext,
                                                                       rb.inner,
                                                                       S,
                                                                       KS,
                                                                       rp.lam,
                                                                       rp.ctx_order,
                                                                       rb.r,
                                                                       rb.kprime,
                                                                       ctl);
    RAFT_CHECK_CUDA(s);
  }
  if (n > tc.thread) {
    tree_commit_warp_kernel<<<grid_rows(n / (tc.thread + 1) + 1), kBlock, 0, s>>>(rb.tree_list,
                                                                                  n,
                                                                                  rb.tree_offset,
                                                                                  rb.members,
                                                                                  rb.flipped,
                                                                                  khat,
                                                                                  rb.ext,
                                                                                  rb.inner,
                                                                                  S,
                                                                                  KS,
                                                                                  rp.lam,
                                                                                  rp.ctx_order,
                                                                                  rb.r,
                                                                                  rb.kprime,
                                                                                  ctl);
    RAFT_CHECK_CUDA(s);
  }
  if (n > tc.warp) {
    const i64 max_trees = n / (tc.warp + 1);
    tree_commit_block_kernel<<<grid_for(max_trees, 1, kCommitBlock), kCommitBlock, 0, s>>>(
      rb.large_list,
      n,
      rb.tree_offset,
      rb.members,
      rb.flipped,
      khat,
      rb.ext,
      rb.inner,
      S,
      KS,
      rp.lam,
      rp.ctx_order,
      rb.r,
      rb.kprime,
      ctl);
    RAFT_CHECK_CUDA(s);
  }
  if (n > tc.block) {
    const int max_large = static_cast<int>(n / (static_cast<i64>(tc.block) + 1));
    const int* cnt      = &ctl->tree_list_count[kTreeLarge];
    auto begin          = thrust::make_transform_iterator(
      thrust::counting_iterator<int>(0), LargeSegmentBegin{rb.large_list, cnt, rb.tree_offset});
    auto end         = thrust::make_transform_iterator(thrust::counting_iterator<int>(0),
                                               LargeSegmentEnd{rb.large_list, cnt, rb.tree_offset});
    const int items  = cub_items(n, "large-tree sort");
    std::size_t need = 0;
    RAFT_CUDA_TRY(cub::DeviceSegmentedSort::SortPairs(nullptr,
                                                      need,
                                                      rb.sort_keys_a,
                                                      rb.sort_keys_b,
                                                      rb.sort_vals_a,
                                                      rb.sort_vals_b,
                                                      items,
                                                      max_large,
                                                      begin,
                                                      end,
                                                      s));
    refine_check_cub(need, cub_bytes, "the large-tree sort");
    tb = cub_bytes;
    RAFT_CUDA_TRY(cub::DeviceSegmentedSort::SortPairs(cub,
                                                      tb,
                                                      rb.sort_keys_a,
                                                      rb.sort_keys_b,
                                                      rb.sort_vals_a,
                                                      rb.sort_vals_b,
                                                      items,
                                                      max_large,
                                                      begin,
                                                      end,
                                                      s));
    tree_commit_large_kernel<<<grid_for(max_large, 1, kCommitBlock), kCommitBlock, 0, s>>>(
      rb.large_list,
      rb.tree_offset,
      rb.sort_vals_b,
      khat,
      rb.ext,
      rb.inner,
      S,
      KS,
      rp.lam,
      rb.r,
      rb.kprime,
      ctl);
    RAFT_CHECK_CUDA(s);
  }
  // R9 (the host flags reuse tree_count: the scatter cursors are spent)
  host_flags_kernel<<<grid_items(n + 1), kBlock, 0, s>>>(rb.r, n, rb.tree_count);
  RAFT_CHECK_CUDA(s);
  tb = cub_bytes;
  RAFT_CUDA_TRY(cub::DeviceScan::ExclusiveSum(
    cub, tb, rb.tree_count, rb.cid, cub_items(n + 1, "coarse ids"), s));
  cmap_kernel<<<gi, kBlock, 0, s>>>(rb.r, rb.cid, n, cmap, ctl);
  RAFT_CHECK_CUDA(s);
}

}  // namespace cugraph::detail::leiden_engine
