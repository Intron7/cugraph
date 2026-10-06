/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

// AGGREGATE (spec §4.9, §6.3, §7 A1-A6): contraction of level l by the
// refined partition (cmap, from R9) into level l + 1.
//
//   A1  agg_init / agg_count   member counts and windows win[c] = sum of the
//                              stored degrees of c's members (zeroed first);
//       2 scans                member and window offsets (int64 accumulators)
//   A2  agg_scatter            members grouped by coarse id; coarse vertices
//                              listed by gather class (window <= 128: warp,
//                              <= 256: wide warp, else block)
//   A3  agg_gather_warp        warp per coarse vertex, members' edges
//                              flattened into 32-edge chunks, stamped table
//                              per warp (stamp = coarse id): 256 slots (warp
//                              class) or 512 slots (wide-warp class)
//       agg_gather_block       block per coarse vertex: window <= 2048 in one
//                              pass (table of nextpow2(win + 1) slots), larger
//                              windows in hash-range passes over the
//                              4096-slot table (npass doubles above 3072
//                              distinct keys; no global memory)
//                              Modes: holey (rows into the arena top at the
//                              window offsets), count (row lengths only) and
//                              write (rows at indptr_{l+1} in the arena
//                              bottom). k_hat_{l+1}[c] = sum of the members'
//                              k_hat in every mode (red team RA2).
//   A4a scan(cdeg) + agg_stats indptr_{l+1}, nnz_{l+1}, degree classes and
//                              max degree of level l + 1  -> SL2
//   A4b agg_compact            holey rows -> arena bottom (one pass)
//   A6  agg_finish             P_{l+1}[cmap_v] = P_l[v]; level metadata
//
// Every coarse row holds the exact int64 sums of its members' counted entries
// towards other coarse vertices (intra-coarse entries are dropped, node
// weights carried; igraph leiden.c:647-664), stored as __ll2float_rn(sum):
// every stored value is an exactly representable integer >= 1. Row order is
// hash-slot order: no consumer reads it (§4.9), so tests compare rows as
// multisets. The per-level layout decision (§6.3) changes only where rows are
// written, never their content.

#include "community/detail/leiden/arena.cuh"
#include "community/detail/leiden/kernels_final.cuh"
#include "community/detail/leiden/kernels_refine.cuh"
#include "community/detail/leiden/numerics.cuh"

#include <raft/util/cuda_rt_essentials.hpp>

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_scan.cuh>

#include <cuda_runtime.h>

#include <climits>
#include <cstddef>
#include <cstdint>
#include <initializer_list>

namespace cugraph::detail::leiden_engine {

// Gather classes by window (sum of member degrees) and the multi-pass
// capacity; runtime values so tests can force each path.
struct GatherClasses {
  int warp          = 128;   // <= 128: warp, 256-slot table (load <= 0.5)
  int wide          = 256;   // <= 256: warp, 512-slot table (load <= 0.5)
  int block         = 2048;  // <= 2048: block, one pass
  int pass_capacity = 3072;  // multi-pass: distinct keys per pass (0.75)
};
// Gather-class lists (A2): warp class from the front of clist, block class
// from its back, wide-warp class in clist_wide; counts in
// Control::gather_list_count[class].
enum : int { kGatherWarp = 0, kGatherBlock = 1, kGatherWide = 2 };

struct AggregateOptions {
  bool low_memory          = false;  // every level two-pass (layout only)
  bool force_two_pass      = false;  // this level two-pass from the start
  bool force_second_gather = false;  // holey gather, compaction "does not fit"
  GatherClasses gather;
  ClassThresholds move;  // degree classes reported for level l + 1 (SL2)
};

// Error bit: the multi-pass gather could not separate a row's keys (never
// expected; shares the Control flags word with the other bits).
constexpr u32 kFlagGatherInvariant = 1u << 25;

constexpr int kAggWarpSlots  = 256;  // warp class: 8 warps per block
constexpr int kAggWideSlots  = 512;  // wide-warp class: 4 warps per block
constexpr int kAggWideWarps  = 4;
constexpr int kAggBlockSlots = 4096;
constexpr std::size_t kAggBlockSmem =
  kAggBlockSlots * (sizeof(u64) + sizeof(int));  // 48 KB dynamic
constexpr u64 kAggEmpty    = ~0ull;              // stamp 0xffffffff is never a coarse id
constexpr u32 kAggPassSalt = 0x2545F491u;

enum : int { kAggCount = 0, kAggHoley = 1, kAggWrite = 2 };

__device__ __forceinline__ unsigned agg_slot(int key) { return fmix32(static_cast<u32>(key)); }

// Pass of a key among npass hash-range passes (any npass >= 1).
__device__ __forceinline__ unsigned agg_pass(int key, unsigned npass)
{
  const u64 h = fmix32(static_cast<u32>(key) ^ kAggPassSalt);
  return static_cast<unsigned>((h * npass) >> 32);
}

// nextpow2(x) for x >= 1 (x <= 2^30).
__device__ __forceinline__ int agg_pow2(i64 x)
{
  return x <= 1 ? 1 : 1 << (32 - __clz(static_cast<int>(x - 1)));
}

// ---------------------------------------------------------------------------
// A1 / A2
// ---------------------------------------------------------------------------

// Zeroes [0, n] of the member counts, windows and coarse row lengths (entries
// past n_{l+1} stay 0, so the scans can run over the host bound n + 1), the
// gather-class counters and the level statistics (§6.5).
__global__ void agg_init_kernel(
  i64 n, int* __restrict__ mcnt, i64* __restrict__ win, i64* __restrict__ cdeg, Control* ctl)
{
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 i = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; i <= n; i += stride) {
    mcnt[i] = 0;
    win[i]  = 0;
    cdeg[i] = 0;
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    for (int k = 0; k < 3; ++k)
      ctl->gather_list_count[k] = 0;
    ctl->nnz_next = 0;
    for (int k = 0; k < kNumClasses; ++k)
      ctl->coarse_class_count[k] = 0;
    ctl->coarse_max_degree = 0;
  }
}

// mcnt[c] += 1 and win[c] += deg_stored(v) for c = cmap_v (ids outside
// [0, n_{l+1}) set kFlagBadLabel and are skipped).
__global__ void agg_count_kernel(const i64* __restrict__ indptr,
                                 const int* __restrict__ cmap,
                                 i64 n,
                                 int* __restrict__ mcnt,
                                 i64* __restrict__ win,
                                 Control* ctl)
{
  const i64 nn     = ctl->n_next;
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 v = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; v < n; v += stride) {
    const int c = cmap[v];
    if (c < 0 || c >= nn) {
      atomicOr(&ctl->flags, kFlagBadLabel);
      continue;
    }
    atomicAdd(&mcnt[c], 1);
    const i64 d = indptr[v + 1] - indptr[v];
    if (d) atomicAdd(reinterpret_cast<u64*>(&win[c]), static_cast<u64>(d));
  }
}

// Members scattered by coarse id (the cursor is the remaining count); coarse
// vertex i < n_{l+1} is listed by its window: warp class from the front of
// clist, wide-warp class in clist_wide, block class from the back of clist.
__global__ void agg_scatter_kernel(const int* __restrict__ cmap,
                                   i64 n,
                                   const i64* __restrict__ mstart,
                                   int* mcnt,
                                   int* __restrict__ members,
                                   const i64* __restrict__ woff,
                                   GatherClasses gc,
                                   int* __restrict__ clist,
                                   int* __restrict__ clist_wide,
                                   Control* ctl)
{
  const i64 nn     = ctl->n_next;
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 base = static_cast<i64>(blockIdx.x) * blockDim.x + (threadIdx.x & ~(kWarp - 1));
       base < n;
       base += stride) {
    const i64 i = base + (threadIdx.x & (kWarp - 1));
    int cls     = -1;
    if (i < n) {
      const int c = cmap[i];
      if (c >= 0 && c < nn) {
        const i64 pos = mstart[c] + atomicSub(&mcnt[c], 1) - 1;
        members[pos]  = static_cast<int>(i);
      }
      if (i < nn) {
        const i64 win = woff[i + 1] - woff[i];
        cls           = win <= gc.warp ? kGatherWarp : win <= gc.wide ? kGatherWide : kGatherBlock;
      }
    }
    int* cnt     = ctl->gather_list_count;
    const int iw = refine_list_append(cls == kGatherWarp, &cnt[kGatherWarp]);
    const int id = refine_list_append(cls == kGatherWide, &cnt[kGatherWide]);
    const int ib = refine_list_append(cls == kGatherBlock, &cnt[kGatherBlock]);
    if (iw >= 0) clist[iw] = static_cast<int>(i);
    if (id >= 0) clist_wide[id] = static_cast<int>(i);
    if (ib >= 0) clist[n - 1 - ib] = static_cast<int>(i);
  }
}

// ---------------------------------------------------------------------------
// A3: gather (warp class)
// ---------------------------------------------------------------------------

// Warp per coarse vertex c of a warp class (window <= kSlots / 2: 128 with
// 256 slots, 256 with 512 slots), listed in list[0 .. count). Up to 32
// members at a time are flattened into chunks of 32 edges (a 5-step search
// finds each lane's member); match_any pre-aggregates equal coarse
// neighbours, and the group leader inserts into the warp's stamped table (u64
// key c << 32 | cu, value set by the claimer; initialised to EMPTY at kernel
// start, §6.5).
template <WKind K, int kMode, int kSlots, int kWarps>
__global__ void __launch_bounds__(kWarps* kWarp)
  agg_gather_warp_kernel(const i64* __restrict__ indptr,
                         const int* __restrict__ indices,
                         EdgeW<K> w,
                         const i64* __restrict__ khat,
                         const int* __restrict__ cmap,
                         const i64* __restrict__ mstart,
                         const int* __restrict__ members,
                         const i64* __restrict__ woff,
                         const int* __restrict__ list,
                         int list_class,
                         const i64* __restrict__ out_off,
                         int* __restrict__ out_idx,
                         float* __restrict__ out_w,
                         i64* __restrict__ cdeg,
                         i64* __restrict__ khat_next,
                         Control* ctl)
{
  __shared__ u64 tkey[kWarps][kSlots];
  __shared__ i64 tval[kWarps][kSlots];
  __shared__ i64 wsum[kWarps][kWarp];
  const int wib = threadIdx.x / kWarp, lane = threadIdx.x & (kWarp - 1);
  volatile u64* vkey = tkey[wib];
  for (int s = lane; s < kSlots; s += kWarp)
    tkey[wib][s] = kAggEmpty;
  __syncwarp();
  const i64 warp0  = static_cast<i64>(blockIdx.x) * kWarps + wib;
  const i64 nwarps = static_cast<i64>(gridDim.x) * kWarps;
  const i64 count  = ctl->gather_list_count[list_class];
  for (i64 i = warp0; i < count; i += nwarps) {
    const int c          = list[i];
    const i64 win        = woff[c + 1] - woff[c];
    int tsize            = agg_pow2(2 * win);
    tsize                = tsize < kWarp ? kWarp : (tsize > kSlots ? kSlots : tsize);
    const unsigned tmask = static_cast<unsigned>(tsize - 1);
    const i64 m0 = mstart[c], m1 = mstart[c + 1];
    const i64 ob = kMode == kAggCount ? 0 : out_off[c];
    i64 ks       = 0;
    for (i64 mb = m0; mb < m1; mb += kWarp) {
      const int nm = static_cast<int>(m1 - mb < kWarp ? m1 - mb : kWarp);
      int v        = -1;
      i64 eb = 0, dg = 0;
      if (lane < nm) {
        v  = members[mb + lane];
        eb = indptr[v];
        dg = indptr[v + 1] - eb;
        ks += khat[v];
      }
      i64 incl = dg;
#pragma unroll
      for (int o = 1; o < kWarp; o <<= 1) {
        const i64 y = __shfl_up_sync(kFullMask, incl, o);
        if (lane >= o) incl += y;
      }
      const i64 total = __shfl_sync(kFullMask, incl, kWarp - 1);
      const i64 excl  = incl - dg;
      for (i64 t0 = 0; t0 < total; t0 += kWarp) {
        const i64 t = t0 + lane;
        int pos     = 0;  // largest member m < nm with excl[m] <= t
#pragma unroll
        for (int step = kWarp / 2; step > 0; step >>= 1) {
          const int cand = pos + step;
          const i64 ex   = __shfl_sync(kFullMask, excl, cand & (kWarp - 1));
          if (cand < nm && ex <= t) pos = cand;
        }
        const i64 ebm = __shfl_sync(kFullMask, eb, pos);
        const i64 exm = __shfl_sync(kFullMask, excl, pos);
        const int vm  = __shfl_sync(kFullMask, v, pos);
        int cu        = -1;
        i64 x         = 0;
        if (t < total) {
          const i64 j = ebm + (t - exm);
          const int u = indices[j];
          if (u != vm) {
            x = w(j);
            if (x > 0) {
              cu = cmap[u];
              if (cu == c) cu = -1;  // intra-coarse: dropped
            }
          }
        }
        if (cu < 0) x = 0;
        const unsigned m = __match_any_sync(kFullMask, cu);
        wsum[wib][lane]  = x;
        __syncwarp();
        if (cu >= 0 && lane == __ffs(m) - 1) {
          i64 sum = 0;
          for (unsigned mm = m; mm; mm &= mm - 1)
            sum += wsum[wib][__ffs(mm) - 1];
          const u64 mine = (static_cast<u64>(c) << 32) | static_cast<u64>(static_cast<u32>(cu));
          unsigned h     = agg_slot(cu) & tmask;
          while (true) {
            const u64 cur = vkey[h];
            if (cur == mine) {
              tval[wib][h] += sum;
              break;
            }
            if (static_cast<u32>(cur >> 32) != static_cast<u32>(c)) {  // empty or stale stamp
              if (atomicCAS(&tkey[wib][h], cur, mine) == cur) {
                tval[wib][h] = sum;
                break;
              }
              continue;  // lost the slot: re-read it
            }
            h = (h + 1) & tmask;
          }
        }
        __syncwarp();
      }
    }
    ks      = warp_sum(ks);
    int out = 0;
    for (int s0 = 0; s0 < tsize; s0 += kWarp) {
      const int s       = s0 + lane;
      const u64 kk      = vkey[s];
      const bool f      = static_cast<u32>(kk >> 32) == static_cast<u32>(c);
      const unsigned bm = __ballot_sync(kFullMask, f);
      if (kMode != kAggCount && f) {
        const i64 o = ob + out + __popc(bm & ((1u << lane) - 1u));
        out_idx[o]  = static_cast<int>(static_cast<u32>(kk));
        out_w[o]    = __ll2float_rn(tval[wib][s]);
      }
      out += __popc(bm);
    }
    if (lane == 0) {
      cdeg[c]      = out;
      khat_next[c] = ks;
    }
    __syncwarp();
  }
}

// ---------------------------------------------------------------------------
// A3: gather (block class, one pass or hash-range multi-pass)
// ---------------------------------------------------------------------------

// Inserts (key, val) into the block table (int keys, -1 = empty; values are
// zeroed with the table, so every inserter adds atomically). With a capacity
// (multi-pass, cap < INT_MAX), a new key is claimed only while fewer than
// `cap` keys are claimed (*cnt counts successful claims); otherwise *ovf is
// set (read only after the next barrier) and the pass is discarded. At most
// one claim per thread is in flight past the check, so the table holds fewer
// than cap + kBlock < slots keys and every probe terminates; *ovf is set only
// when the pass really holds more than cap distinct keys (contention between
// inserters of one key never overflows), so the doubling of npass ends.
__device__ __forceinline__ void agg_block_insert(
  int* tk, u64* tv, unsigned tmask, int key, i64 val, int* cnt, int* ovf, int cap)
{
  volatile int* vk  = tk;
  const bool capped = cap < INT_MAX;
  unsigned h        = agg_slot(key) & tmask;
  while (true) {
    const int cur = vk[h];
    if (cur == key) break;
    if (cur == -1) {
      if (capped && atomicAdd(cnt, 0) >= cap) {
        atomicOr(ovf, 1);
        return;
      }
      const int prev = atomicCAS(&tk[h], -1, key);
      if (prev == -1) {
        if (capped) atomicAdd(cnt, 1);
        break;
      }
      if (prev == key) break;
    }
    h = (h + 1) & tmask;
  }
  atomicAdd(&tv[h], static_cast<u64>(val));
}

// Block per coarse vertex (block class: listed from the back of clist).
// Members are processed in batches of kBlock (block scan of their degrees),
// their edges in chunks of kBlock (binary search of the owning member);
// match_any pre-aggregates within each warp. Windows <= gc.block run one pass
// with nextpow2(win + 1) slots (cannot overflow). Larger windows run
// hash-range passes over the 4096-slot table: pass p takes the keys with
// agg_pass(key, npass) == p; npass starts at 1 and doubles (restarting the
// row) whenever a pass exceeds gc.pass_capacity distinct keys. The table is
// re-initialised before every pass; each pass appends its keys to the row.
template <WKind K, int kMode>
__global__ void __launch_bounds__(kBlock) agg_gather_block_kernel(const i64* __restrict__ indptr,
                                                                  const int* __restrict__ indices,
                                                                  EdgeW<K> w,
                                                                  const i64* __restrict__ khat,
                                                                  const int* __restrict__ cmap,
                                                                  const i64* __restrict__ mstart,
                                                                  const int* __restrict__ members,
                                                                  const i64* __restrict__ woff,
                                                                  const int* __restrict__ clist,
                                                                  i64 n_list,
                                                                  GatherClasses gc,
                                                                  const i64* __restrict__ out_off,
                                                                  int* __restrict__ out_idx,
                                                                  float* __restrict__ out_w,
                                                                  i64* __restrict__ cdeg,
                                                                  i64* __restrict__ khat_next,
                                                                  Control* ctl)
{
  extern __shared__ __align__(16) unsigned char agg_smem[];
  u64* tv      = reinterpret_cast<u64*>(agg_smem);
  int* tk      = reinterpret_cast<int*>(agg_smem + kAggBlockSlots * sizeof(u64));
  using Scan   = cub::BlockScan<i64, kBlock>;
  using Reduce = cub::BlockReduce<i64, kBlock>;
  __shared__ union {
    typename Scan::TempStorage scan;
    typename Reduce::TempStorage reduce;
  } tmp;
  __shared__ i64 s_excl[kBlock];
  __shared__ i64 s_eb[kBlock];
  __shared__ int s_v[kBlock];
  __shared__ i64 wsum[kWarpsPerBlock][kWarp];
  __shared__ int s_cnt, s_ovf, s_out;
  const int tid = threadIdx.x, wib = tid / kWarp, lane = tid & (kWarp - 1);
  const i64 count = ctl->gather_list_count[kGatherBlock];
  for (i64 i = blockIdx.x; i < count; i += gridDim.x) {
    const int c   = clist[n_list - 1 - i];
    const i64 win = woff[c + 1] - woff[c];
    const i64 m0 = mstart[c], m1 = mstart[c + 1];
    i64 ks = 0;
    for (i64 m = m0 + tid; m < m1; m += kBlock)
      ks += khat[members[m]];
    ks = Reduce(tmp.reduce).Sum(ks);
    __syncthreads();
    const bool single    = win <= gc.block;
    const int tsize      = single ? agg_pow2(win + 1) : kAggBlockSlots;
    const unsigned tmask = static_cast<unsigned>(tsize - 1);
    const int cap        = single ? INT_MAX : gc.pass_capacity;
    const i64 ob         = kMode == kAggCount ? 0 : out_off[c];
    unsigned npass       = 1;
    while (true) {  // until a full set of passes ran without overflow
      if (tid == 0) s_out = 0;
      bool overflow = false;
      for (unsigned p = 0; p < npass; ++p) {
        for (int s = tid; s < tsize; s += kBlock) {
          tk[s] = -1;
          tv[s] = 0;
        }
        if (tid == 0) {
          s_cnt = 0;
          s_ovf = 0;
        }
        __syncthreads();
        for (i64 mb = m0; mb < m1; mb += kBlock) {
          const int nm = static_cast<int>(m1 - mb < kBlock ? m1 - mb : kBlock);
          int v        = -1;
          i64 eb = 0, dg = 0;
          if (tid < nm) {
            v  = members[mb + tid];
            eb = indptr[v];
            dg = indptr[v + 1] - eb;
          }
          i64 excl, total;
          Scan(tmp.scan).ExclusiveSum(dg, excl, total);
          s_excl[tid] = excl;
          s_eb[tid]   = eb;
          s_v[tid]    = v;
          __syncthreads();
          for (i64 t0 = 0; t0 < total; t0 += kBlock) {
            const i64 t = t0 + tid;
            int cu      = -1;
            i64 x       = 0;
            if (t < total) {
              int lo = 0, hi = nm - 1;  // largest m: excl <= t
              while (lo < hi) {
                const int mid = (lo + hi + 1) >> 1;
                if (s_excl[mid] <= t)
                  lo = mid;
                else
                  hi = mid - 1;
              }
              const i64 j = s_eb[lo] + (t - s_excl[lo]);
              const int u = indices[j];
              if (u != s_v[lo]) {
                x = w(j);
                if (x > 0) {
                  cu = cmap[u];
                  if (cu == c || (npass > 1 && agg_pass(cu, npass) != p)) cu = -1;
                }
              }
            }
            if (cu < 0) x = 0;
            const unsigned m = __match_any_sync(kFullMask, cu);
            wsum[wib][lane]  = x;
            __syncwarp();
            if (cu >= 0 && lane == __ffs(m) - 1) {
              i64 sum = 0;
              for (unsigned mm = m; mm; mm &= mm - 1)
                sum += wsum[wib][__ffs(mm) - 1];
              agg_block_insert(tk, tv, tmask, cu, sum, &s_cnt, &s_ovf, cap);
            }
            __syncwarp();
          }
          __syncthreads();  // s_excl / s_eb / s_v are rewritten
        }
        if (s_ovf) {  // block-uniform (read after the barrier)
          overflow = true;
          break;
        }
        for (int s0 = 0; s0 < tsize; s0 += kBlock) {
          const int s       = s0 + tid;
          const bool f      = s < tsize && tk[s] != -1;
          const unsigned bm = __ballot_sync(kFullMask, f);
          int base          = 0;
          if (lane == 0 && bm) base = atomicAdd(&s_out, __popc(bm));
          base = __shfl_sync(kFullMask, base, 0);
          if (kMode != kAggCount && f) {
            const i64 o = ob + base + __popc(bm & ((1u << lane) - 1u));
            out_idx[o]  = tk[s];
            out_w[o]    = __ll2float_rn(static_cast<i64>(tv[s]));
          }
        }
        __syncthreads();
      }
      if (!overflow) break;
      if (npass >= (1u << 31)) {  // cannot happen for cap >= 2
        if (tid == 0) atomicOr(&ctl->flags, kFlagGatherInvariant);
        break;
      }
      npass *= 2;
      __syncthreads();
    }
    if (tid == 0) {
      cdeg[c]      = s_out;
      khat_next[c] = ks;
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// A4a / A4b / A6
// ---------------------------------------------------------------------------

// Degree classes (move thresholds) and max degree of level l + 1, and
// nnz_{l+1} = indptr_{l+1}[n_{l+1}] (block-reduced, one atomic per block).
__global__ void agg_stats_kernel(const i64* __restrict__ cdeg,
                                 const i64* __restrict__ indptr_next,
                                 ClassThresholds th,
                                 Control* ctl)
{
  __shared__ i64 s_cls[kWarpsPerBlock][kNumClasses];
  __shared__ i64 s_max[kWarpsPerBlock];
  const i64 nn         = ctl->n_next;
  i64 cls[kNumClasses] = {0, 0, 0, 0};
  i64 mx               = 0;
  const i64 stride     = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 c = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; c < nn; c += stride) {
    const i64 d = cdeg[c];
    cls[degree_class(d, th)] += 1;
    mx = d > mx ? d : mx;
  }
  const int wib = threadIdx.x / kWarp, lane = threadIdx.x & (kWarp - 1);
#pragma unroll
  for (int k = 0; k < kNumClasses; ++k)
    cls[k] = warp_sum(cls[k]);
  mx = warp_max(mx);
  if (lane == 0) {
    for (int k = 0; k < kNumClasses; ++k)
      s_cls[wib][k] = cls[k];
    s_max[wib] = mx;
  }
  __syncthreads();
  if (threadIdx.x < kNumClasses) {
    i64 t = 0;
    for (int q = 0; q < kWarpsPerBlock; ++q)
      t += s_cls[q][threadIdx.x];
    if (t)
      atomicAdd(reinterpret_cast<u64*>(&ctl->coarse_class_count[threadIdx.x]), static_cast<u64>(t));
  }
  if (threadIdx.x == kNumClasses) {
    i64 t = 0;
    for (int q = 0; q < kWarpsPerBlock; ++q)
      t = s_max[q] > t ? s_max[q] : t;
    if (t) atomicMax(reinterpret_cast<u64*>(&ctl->coarse_max_degree), static_cast<u64>(t));
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) ctl->nnz_next = indptr_next[nn];
}

// Holey rows (at the window offsets, arena top) -> compacted rows at
// indptr_{l+1} (arena bottom). A warp takes 32 rows: lane i prefetches row
// i's offsets, then the warp copies the rows one after the other.
__global__ void agg_compact_kernel(const i64* __restrict__ woff,
                                   const i64* __restrict__ cdeg,
                                   const i64* __restrict__ indptr_next,
                                   i64 nn,
                                   const int* __restrict__ h_idx,
                                   const float* __restrict__ h_w,
                                   int* __restrict__ idx,
                                   float* __restrict__ wt)
{
  const int lane   = threadIdx.x & (kWarp - 1);
  const i64 warp0  = static_cast<i64>(blockIdx.x) * kWarpsPerBlock + threadIdx.x / kWarp;
  const i64 nwarps = static_cast<i64>(gridDim.x) * kWarpsPerBlock;
  for (i64 base = warp0 * kWarp; base < nn; base += nwarps * kWarp) {
    const i64 c = base + lane;
    i64 src = 0, dst = 0, d = 0;
    if (c < nn) {
      src = woff[c];
      dst = indptr_next[c];
      d   = cdeg[c];
    }
    const int rows = nn - base < kWarp ? static_cast<int>(nn - base) : kWarp;
    for (int q = 0; q < rows; ++q) {
      const i64 sq = __shfl_sync(kFullMask, src, q);
      const i64 dq = __shfl_sync(kFullMask, dst, q);
      const i64 nq = __shfl_sync(kFullMask, d, q);
      for (i64 t = lane; t < nq; t += kWarp) {
        idx[dq + t] = h_idx[sq + t];
        wt[dq + t]  = h_w[sq + t];
      }
    }
  }
}

// A6 + level metadata: P_{l+1}[cmap_v] = P_l[v] (every member of a coarse
// vertex carries the same move community, so the racing stores are
// identical); indptr_{l+1} and k_hat_{l+1} copied from the phase scratch into
// their exact-size slots in the arena bottom.
__global__ void agg_finish_kernel(i64 n,
                                  i64 nn,
                                  const int* __restrict__ cmap,
                                  const int* __restrict__ P,
                                  int* __restrict__ P_next,
                                  const i64* __restrict__ indptr_scr,
                                  const i64* __restrict__ khat_scr,
                                  i64* __restrict__ indptr_dst,
                                  i64* __restrict__ khat_dst)
{
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  const i64 end    = n > nn + 1 ? n : nn + 1;
  for (i64 i = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; i < end; i += stride) {
    if (P && i < n) P_next[cmap[i]] = P[i];
    if (i <= nn) indptr_dst[i] = indptr_scr[i];
    if (i < nn) khat_dst[i] = khat_scr[i];
  }
}

__global__ void agg_set_n_next_kernel(Control* ctl, i64 n_next) { ctl->n_next = n_next; }

// ---------------------------------------------------------------------------
// Host side: per-level layout decision (§6.3) and launches
// ---------------------------------------------------------------------------

// Bottom offset after pushing arrays of the given byte sizes (each aligned
// like LevelArena::push_bottom).
inline std::size_t agg_bottom_end(std::size_t bottom, std::initializer_list<std::size_t> sizes)
{
  std::size_t b = bottom;
  for (std::size_t x : sizes)
    b = align_up(b) + x;
  return b;
}

// Plan of one level after stage 1 (before SL2).
struct AggregatePlan {
  bool holey = false;    // holey gather into the arena top
  int* h_idx = nullptr;  // holey region (nnz_l entries)
  float* h_w = nullptr;
};

// The coarse level after stage 2 (rows and metadata in the arena bottom).
struct CoarseLevel {
  i64* indptr        = nullptr;  // [n + 1]
  i64* khat          = nullptr;  // [n]
  int* indices       = nullptr;  // [nnz]
  float* weights     = nullptr;
  i64 n              = 0;
  i64 nnz            = 0;
  bool ok            = true;   // false: workspace_too_small (see arena)
  bool two_pass      = false;  // a second gather wrote the rows
  bool second_gather = false;  // holey gather, compaction did not fit
};

template <WKind K, int kMode>
inline void agg_block_smem_optin()
{
  static thread_local int done_device = -1;
  const int dev                       = device_info().device;
  if (done_device != dev) {
    RAFT_CUDA_TRY(cudaFuncSetAttribute(agg_gather_block_kernel<K, kMode>,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       static_cast<int>(kAggBlockSmem)));
    done_device = dev;
  }
}

// Every gather class (warp, wide warp, block) over device-resident class
// counts; the grids use host bounds (n fine vertices, windows) on the number
// of coarse vertices per class.
template <WKind K, int kMode>
void launch_gather(const i64* indptr,
                   const int* indices,
                   EdgeW<K> w,
                   const i64* khat,
                   i64 n,
                   const int* cmap,
                   const AggregateBufs& ab,
                   const GatherClasses& gc,
                   const i64* out_off,
                   int* out_idx,
                   float* out_w,
                   Control* ctl,
                   cudaStream_t s)
{
  agg_gather_warp_kernel<K, kMode, kAggWarpSlots, kWarpsPerBlock>
    <<<grid_rows(n), kBlock, 0, s>>>(indptr,
                                     indices,
                                     w,
                                     khat,
                                     cmap,
                                     ab.member_start,
                                     ab.members,
                                     ab.window_offset,
                                     ab.clist,
                                     kGatherWarp,
                                     out_off,
                                     out_idx,
                                     out_w,
                                     ab.cdeg,
                                     ab.khat_next,
                                     ctl);
  RAFT_CHECK_CUDA(s);
  if (gc.wide > gc.warp) {  // windows > gc.warp: at most nnz / gc.warp ones
    const i64 bound = n / (gc.warp + 1) + 1;
    agg_gather_warp_kernel<K, kMode, kAggWideSlots, kAggWideWarps>
      <<<grid_for(bound, kAggWideWarps, kAggWideWarps * kWarp), kAggWideWarps * kWarp, 0, s>>>(
        indptr,
        indices,
        w,
        khat,
        cmap,
        ab.member_start,
        ab.members,
        ab.window_offset,
        ab.clist_wide,
        kGatherWide,
        out_off,
        out_idx,
        out_w,
        ab.cdeg,
        ab.khat_next,
        ctl);
    RAFT_CHECK_CUDA(s);
  }
  agg_block_smem_optin<K, kMode>();
  const i64 sms = device_info().sm_count;
  const i64 g   = n < 4 * sms ? n : 4 * sms;
  agg_gather_block_kernel<K, kMode>
    <<<static_cast<unsigned>(g > 0 ? g : 1), kBlock, kAggBlockSmem, s>>>(indptr,
                                                                         indices,
                                                                         w,
                                                                         khat,
                                                                         cmap,
                                                                         ab.member_start,
                                                                         ab.members,
                                                                         ab.window_offset,
                                                                         ab.clist,
                                                                         n,
                                                                         gc,
                                                                         out_off,
                                                                         out_idx,
                                                                         out_w,
                                                                         ab.cdeg,
                                                                         ab.khat_next,
                                                                         ctl);
  RAFT_CHECK_CUDA(s);
}

// Stage 1 (A1-A4a), before SL2: level l (n vertices, nnz stored entries,
// cmap with n_{l+1} = ctl->n_next on the device) is gathered in holey mode
// if the holey region (8 nnz bytes, arena top) and the level metadata at its
// upper bound (n_{l+1} <= n) fit, else in count mode (two-pass level). The
// caller then reads the Control block (SL2: n_next, nnz_next, coarse classes,
// max degree) and calls aggregate_finish.
template <WKind K>
AggregatePlan aggregate_begin(const i64* indptr,
                              const int* indices,
                              EdgeW<K> w,
                              const i64* khat,
                              i64 n,
                              i64 nnz,
                              const int* cmap,
                              const AggregateOptions& opt,
                              const AggregateBufs& ab,
                              LevelArena& arena,
                              void* cub,
                              std::size_t cub_bytes,
                              Control* ctl,
                              cudaStream_t s)
{
  AggregatePlan plan;
  const std::size_t nu   = static_cast<std::size_t>(n);
  const std::size_t meta = align_up(8 * (nu + 1)) + align_up(8 * nu);
  const std::size_t holey =
    align_up(4 * static_cast<std::size_t>(nnz)) + align_up(4 * static_cast<std::size_t>(nnz));
  plan.holey = !opt.low_memory && !opt.force_two_pass && arena.fits(meta, holey);
  if (plan.holey) {
    plan.h_idx = arena.push_top<int>(static_cast<std::size_t>(nnz));
    plan.h_w   = arena.push_top<float>(static_cast<std::size_t>(nnz));
  }
  const unsigned gi = grid_items(n + 1);
  agg_init_kernel<<<gi, kBlock, 0, s>>>(n, ab.member_count, ab.window, ab.cdeg, ctl);
  RAFT_CHECK_CUDA(s);
  if (n > 0) {
    agg_count_kernel<<<grid_items(n), kBlock, 0, s>>>(
      indptr, cmap, n, ab.member_count, ab.window, ctl);
    RAFT_CHECK_CUDA(s);
  }
  std::size_t tb = cub_bytes;
  RAFT_CUDA_TRY(scan_offsets64<int>(cub, tb, ab.member_count, ab.member_start, n + 1, s));
  tb = cub_bytes;
  RAFT_CUDA_TRY(scan_offsets64<i64>(cub, tb, ab.window, ab.window_offset, n + 1, s));
  if (n > 0) {
    agg_scatter_kernel<<<grid_items(n), kBlock, 0, s>>>(cmap,
                                                        n,
                                                        ab.member_start,
                                                        ab.member_count,
                                                        ab.members,
                                                        ab.window_offset,
                                                        opt.gather,
                                                        ab.clist,
                                                        ab.clist_wide,
                                                        ctl);
    RAFT_CHECK_CUDA(s);
    if (plan.holey) {
      launch_gather<K, kAggHoley>(indptr,
                                  indices,
                                  w,
                                  khat,
                                  n,
                                  cmap,
                                  ab,
                                  opt.gather,
                                  ab.window_offset,
                                  plan.h_idx,
                                  plan.h_w,
                                  ctl,
                                  s);
    } else {
      launch_gather<K, kAggCount>(
        indptr, indices, w, khat, n, cmap, ab, opt.gather, nullptr, nullptr, nullptr, ctl, s);
    }
  }
  tb = cub_bytes;
  RAFT_CUDA_TRY(scan_offsets64<i64>(cub, tb, ab.cdeg, ab.indptr_next, n + 1, s));
  agg_stats_kernel<<<grid_items(n), kBlock, 0, s>>>(ab.cdeg, ab.indptr_next, opt.move, ctl);
  RAFT_CHECK_CUDA(s);
  return plan;
}

// Stage 2, after SL2 (n_next, nnz_next known on the host): reserves the
// exact metadata (indptr_{l+1}, k_hat_{l+1}) and rows (int32 indices, fp32
// weights) in the arena bottom. A holey level compacts its rows below the
// holey region when they fit there (one pass); otherwise the holey region is
// released and a second gather writes the rows directly (the level becomes
// two-pass). Only if even that does not fit is the result !ok, with
// arena.required set (workspace_too_small). Then P_{l+1} (if P is given) and
// the metadata. Layout never reaches a decision: every path gives the same
// rows as multisets and identical metadata.
template <WKind K>
CoarseLevel aggregate_finish(const i64* indptr,
                             const int* indices,
                             EdgeW<K> w,
                             const i64* khat,
                             i64 n,
                             const int* cmap,
                             const AggregatePlan& plan,
                             i64 n_next,
                             i64 nnz_next,
                             const AggregateOptions& opt,
                             const AggregateBufs& ab,
                             LevelArena& arena,
                             const int* P,
                             int* P_next,
                             Control* ctl,
                             cudaStream_t s)
{
  CoarseLevel lv;
  lv.n                  = n_next;
  lv.nnz                = nnz_next;
  const std::size_t nn  = static_cast<std::size_t>(n_next);
  const std::size_t ne  = static_cast<std::size_t>(nnz_next);
  const std::size_t end = agg_bottom_end(arena.bottom, {8 * (nn + 1), 8 * nn, 4 * ne, 4 * ne});
  auto push             = [&] {
    lv.indptr  = arena.push_bottom<i64>(nn + 1);
    lv.khat    = arena.push_bottom<i64>(nn);
    lv.indices = arena.push_bottom<int>(ne);
    lv.weights = arena.push_bottom<float>(ne);
  };
  if (plan.holey && !opt.force_second_gather && end + arena.top <= arena.cap) {
    push();
    if (n_next > 0) {
      agg_compact_kernel<<<grid_rows(n_next), kBlock, 0, s>>>(ab.window_offset,
                                                              ab.cdeg,
                                                              ab.indptr_next,
                                                              n_next,
                                                              plan.h_idx,
                                                              plan.h_w,
                                                              lv.indices,
                                                              lv.weights);
      RAFT_CHECK_CUDA(s);
    }
    arena.release_top();
  } else {
    if (plan.holey) {
      arena.release_top();
      lv.second_gather = true;
    }
    lv.two_pass = true;
    if (end > arena.cap) {
      arena.note_overflow(end);
      lv.ok = false;
      return lv;
    }
    push();
    if (n > 0)
      launch_gather<K, kAggWrite>(indptr,
                                  indices,
                                  w,
                                  khat,
                                  n,
                                  cmap,
                                  ab,
                                  opt.gather,
                                  ab.indptr_next,
                                  lv.indices,
                                  lv.weights,
                                  ctl,
                                  s);
  }
  agg_finish_kernel<<<grid_items((n > n_next ? n : n_next) + 1), kBlock, 0, s>>>(
    n, n_next, cmap, P, P_next, ab.indptr_next, ab.khat_next, lv.indptr, lv.khat);
  RAFT_CHECK_CUDA(s);
  return lv;
}

}  // namespace cugraph::detail::leiden_engine
