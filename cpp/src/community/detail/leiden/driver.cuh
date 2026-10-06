/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

// Host driver of the deterministic Leiden engine (spec §4.5, §4.10-§4.14, §8.2): LeidenDriver.
// Included by driver.cu only (the engine translation unit; see driver.hpp).
//
// The typed entry point runs the input path first (I1 ingest_check, layout, workspace, I2 (+ I4)
// quantisation of level 0, see kernels_ingest.cuh); run_engine then runs one resolution:
//
//   per iteration:
//     per level: LOCAL_MOVE (DOWN), COMPACT                         -> SL1
//                C == n: all-active TOP sweeps (certify or re-enter)
//                PLEIDEN_R, AGGREGATE stage 1                        -> SL2
//                refinement merged nothing: re-enter TOP (Lemma R,
//                  <= max_reentry times), then the CC-piece last resort
//                AGGREGATE stage 2 (compaction or two-pass level)
//     projection to level 0, with the V-cycle (UP sweeps + COMPACT per
//     level) in every iteration but the last
//     CC split; with replicas: Q_r and the chosen replica's slice
//     (n_iterations = -1: exact Q and the stop rules)
//   FINALIZE: exact Q, size-ordered labels into labels_out
//
// Every kernel runs on the caller's stream; the readbacks (one per sweep,
// SL1, SL2 and a few per iteration) go through the layout's pinned buffer when
// there is one. Nothing is allocated: every array is carved from the workspace,
// and the level stack lives in the checked arena (a level that does not fit
// runs two-pass; run_engine returns false only if even that fails).

#include "community/detail/leiden/arena.cuh"
#include "community/detail/leiden/driver.hpp"
#include "community/detail/leiden/kernels_aggregate.cuh"
#include "community/detail/leiden/kernels_final.cuh"
#include "community/detail/leiden/kernels_move.cuh"
#include "community/detail/leiden/kernels_refine.cuh"
#include "community/detail/leiden/numerics.cuh"

#include <cugraph/utilities/error.hpp>

#include <raft/util/cuda_rt_essentials.hpp>

#include <cub/device/device_scan.cuh>

#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace cugraph::detail::leiden_engine {

constexpr double kDeltaQStop = 1e-6;  // n_iterations = -1, rule (b) (§4.14)

// Knobs of one call (§11.1); the defaults are the spec's.
struct DriverParams {
  int n_iterations          = 2;  // >= 1, or -1 (until stable)
  int move_sweeps           = 4;  // DOWN cap
  int subrounds             = kNumSubrounds;
  int vcycle_sweeps         = 4;  // UP cap; 0 = no V-cycle
  bool vcycle_last          = false;
  int top_sweeps            = 32;  // TOP cap (§4.10)
  int max_levels            = kMaxLevels;
  int max_reentry           = 4;  // TOP re-entries per level (Lemma R)
  int max_iter_until_stable = 20;
  int max_replicas          = kMaxReplicas;
  i64 replica_nnz_budget    = kReplicaNnzBudget;
  double arena_factor       = 1.9;
  bool low_memory           = false;
  int force_replica         = -1;     // tests (_FORCE_REPLICA)
  bool force_two_pass       = false;  // tests: every level two-pass
  bool force_second_gather  = false;  // tests: compaction "does not fit"
  ClassThresholds th{};
  GatherClasses gather{};
  TreeClasses trees{};
};

// ---------------------------------------------------------------------------
// Small driver kernels
// ---------------------------------------------------------------------------

__global__ void drv_iota_kernel(int* __restrict__ P, i64 n)
{
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 v = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; v < n; v += stride)
    P[v] = static_cast<int>(v);
}

// Flags ids outside [0, bound) (the initial membership: compact ids < n).
__global__ void drv_check_labels_kernel(const int* __restrict__ P, i64 n, i64 bound, Control* ctl)
{
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  bool bad         = false;
  for (i64 v = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; v < n; v += stride)
    bad = bad || P[v] < 0 || P[v] >= bound;
  if (__any_sync(kFullMask, bad) && (threadIdx.x & (kWarp - 1)) == 0)
    atomicOr(&ctl->flags, kFlagBadLabel);
}

// Partition of the replica union (§4.2.1): P[r n + v] = P[v] + r n, r >= 1.
__global__ void drv_replicate_partition_kernel(int* __restrict__ P, i64 n, int R)
{
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 v = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; v < n; v += stride) {
    const int p = P[v];
    for (int r = 1; r < R; ++r)
      P[static_cast<i64>(r) * n + v] = p + static_cast<int>(r * n);
  }
}

// COMPACT_U of one replica slice (§4.2.1, RA8): flags of the ids the slice
// uses (domain [0, C), no offset), then P_out[v] = rank[P_in[v]].
__global__ void drv_slice_flags_kernel(
  const int* __restrict__ P, i64 n, i64 C, int* __restrict__ flags, Control* ctl)
{
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 v = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; v < n; v += stride) {
    const int c = P[v];
    if (c < 0 || c >= C)
      atomicOr(&ctl->flags, kFlagBadLabel);
    else
      flags[c] = 1;  // idempotent store
  }
}

__global__ void drv_slice_relabel_kernel(const int* __restrict__ P_in,
                                         i64 n,
                                         i64 C,
                                         const int* __restrict__ rank,
                                         int* __restrict__ P_out,
                                         Control* ctl)
{
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 v = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; v < n; v += stride) {
    const int c = P_in[v];
    P_out[v]    = (c >= 0 && c < C) ? rank[c] : 0;
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) ctl->n_components = rank[C];
}

// ---------------------------------------------------------------------------
// Statistics of a call (iterations and levels feed leiden_result_t)
// ---------------------------------------------------------------------------

struct LevelStat {
  i64 n = 0, nnz = 0;
  i64 C                = -1;  // communities after the last move phase of the level
  i64 moves            = 0;
  int sweeps           = 0;
  int top_phases       = 0;
  int top_sweeps       = 0;
  i64 top_moves        = 0;
  i64 n_next           = -1;
  int reentries        = 0;
  const char* layout   = "";  // "holey" | "two_pass" (aggregated levels)
  bool second_gather   = false;
  const char* fallback = "";  // "" | "cc" | "pieces_are_singletons"
};

struct IterationStat {
  bool vcycle      = false;
  bool moved       = false;
  i64 cc_pieces    = 0;
  int reentries    = 0;
  int fallbacks    = 0;
  bool uncertified = false;
  bool top_cap_hit = false;
  std::vector<double> replica_q;
  int replica_chosen = -1;
  double q           = std::numeric_limits<double>::quiet_NaN();  // T = -1 only
  std::vector<LevelStat> levels;
  std::vector<i64> vcycle_moves;
};

struct ResolutionResult {
  double modularity     = 0.0;
  i64 n_clusters        = 0;
  i64 l_hat             = 0;
  int scale_exponent    = 0;
  i64 two_m_hat         = 0;
  int replicas          = 1;
  const char* stop_rule = "";  // n_iterations = -1: the rule that stopped
  bool trivial          = false;
  std::vector<IterationStat> iters;
};

// ---------------------------------------------------------------------------
// One level of the hierarchy (host view)
// ---------------------------------------------------------------------------

struct LevelGraph {
  const i64* indptr  = nullptr;
  const int* indices = nullptr;
  const float* wf    = nullptr;  // fp32: level 0 scaled by 2^s on the fly,
                                 // coarse levels exact integers (scale 1)
  const i64* wq   = nullptr;     // int64 W0q (level 0)
  i64 unit        = 1;           // UNIT kind: the quantised unit weight
  double scale    = 1.0;
  const i64* khat = nullptr;
  i64 n = 0, nnz = 0;
  i64 cls[kNumClasses] = {0, 0, 0, 0};  // vertices per degree class
  int* cmap            = nullptr;       // to level l + 1 (arena bottom)

  template <typename Fn>
  void with_w(Fn&& fn) const
  {
    dispatch_weights(wf, wq, scale, unit, std::forward<Fn>(fn));
  }
};

// ---------------------------------------------------------------------------
// The driver (P = 1)
// ---------------------------------------------------------------------------

class LeidenDriver {
 public:
  LeidenDriver(Layout& L, const DriverParams& dp, cudaStream_t s) : L_(L), dp_(dp), s_(s) {}

  i64 syncs() const { return syncs_; }

  // Level 0 after I2 (+ I4) for the current scale.
  void set_level0(const QuantizeResult& q, i64 n, i64 nnz)
  {
    q_          = q;
    n_          = n;
    nnz_        = nnz;
    g0_         = LevelGraph{};
    g0_.indptr  = q.indptr;
    g0_.indices = q.indices;
    g0_.wf      = q.wkind == WKind::F32 ? q.wf32 : nullptr;
    g0_.wq      = q.wkind == WKind::I64 ? q.wq : nullptr;
    g0_.unit    = q.wkind == WKind::UNIT ? q.unit_q : 1;
    g0_.scale   = std::ldexp(1.0, q.s);
    g0_.khat    = L_.l0.khat;
    g0_.n       = n;
    g0_.nnz     = nnz;
    for (int c = 0; c < kNumClasses; ++c)
      g0_.cls[c] = q.class_count[c];
  }

  // LEIDEN_CALL body of one resolution (§4.5). Returns false if the arena
  // overflowed even with two-pass levels (workspace too small).
  bool run_resolution(
    double gamma, u32 seed, const int* init, int R, int* labels_out, ResolutionResult& out)
  {
    const i64 n        = n_;
    const i64 two_m    = q_.two_m_hat;
    out.scale_exponent = q_.s;
    out.two_m_hat      = two_m;
    out.replicas       = R;
    if (two_m <= 0) {  // no counted entry: every vertex its own cluster
      out.trivial    = true;
      out.n_clusters = n;
      drv_iota_kernel<<<grid_items(n), kBlock, 0, s_>>>(labels_out, n);
      RAFT_CHECK_CUDA(s_);
      return true;
    }
    lam_    = gamma / static_cast<double>(two_m);  // one IEEE division
    gamma_  = gamma;
    seed_   = seed;
    s64_    = seed64(seed);
    int* P0 = L_.persist.P0;
    // P0 <- the initial membership (compact ids, checked < n here) or
    // the singletons
    if (init) {
      RAFT_CUDA_TRY(cudaMemcpyAsync(P0, init, n * sizeof(int), cudaMemcpyDeviceToDevice, s_));
      drv_check_labels_kernel<<<grid_items(n), kBlock, 0, s_>>>(P0, n, n, L_.ctl);
      RAFT_CHECK_CUDA(s_);
      ++syncs_;
      const Control c = read_control(L_.ctl, s_, L_.pinned);
      CUGRAPH_EXPECTS(!(c.flags & kFlagBadLabel),
                      "leiden: the initial membership must "
                      "hold community ids in [0, n).");
    } else {
      drv_iota_kernel<<<grid_items(n), kBlock, 0, s_>>>(P0, n);
      RAFT_CHECK_CUDA(s_);
    }
    i64 C         = n;  // id bound of P0
    const int T   = dp_.n_iterations;
    const int V   = dp_.vcycle_sweeps;
    int it        = 0;
    double q_prev = -std::numeric_limits<double>::infinity();
    bool stop     = false;
    while (true) {
      const bool final  = (T >= 1 && it == T - 1) || (T == -1 && stop);
      const bool vcycle = V > 0 && (!final || T == 1 || dp_.vcycle_last);
      IterationStat ist;
      ist.vcycle = vcycle;
      if (it == 0 && R > 1) {
        // replica union (§4.2.1): R disjoint copies, one hierarchy
        drv_replicate_partition_kernel<<<grid_items(n), kBlock, 0, s_>>>(P0, n, R);
        RAFT_CHECK_CUDA(s_);
        const LevelGraph gu = union_level(R);
        i64 Cb              = 0;
        if (!leiden_iteration(gu, P0, gu.n, it, vcycle, ist, Cb)) return false;
        const i64 Cu  = cc_split(gu, P0);  // union ranks [0, Cu)
        ist.cc_pieces = Cu - Cb;
        // Q_r of each replica with one copy's 2m_hat (§4.13)
        g0_.with_w([&](auto w) {
          run_quality(g0_.indptr,
                      g0_.indices,
                      w,
                      g0_.khat,
                      P0,
                      n,
                      R,
                      Cu,
                      two_m,
                      gamma,
                      L_.final_,
                      L_.ctl,
                      s_);
        });
        const Control c = readback();
        int best        = 0;
        for (int r = 0; r < R; ++r) {
          ist.replica_q.push_back(c.q[r]);
          if (c.q[r] > c.q[best]) best = r;  // ties -> lowest r
        }
        if (dp_.force_replica >= 0) {
          CUGRAPH_EXPECTS(dp_.force_replica < R,
                          "leiden: force_replica must be "
                          "< the number of replicas.");
          best = dp_.force_replica;
        }
        ist.replica_chosen = best;
        C                  = compact_slice(P0 + static_cast<i64>(best) * n, n, Cu, P0);
      } else {
        i64 Cb = 0;
        if (!leiden_iteration(g0_, P0, C, it, vcycle, ist, Cb)) return false;
        C             = cc_split(g0_, P0);
        ist.cc_pieces = C - Cb;
      }
      ++it;
      if (!final && T == -1) {  // §4.14: is the next iteration final?
        const double q = quality(P0, C).q[0];
        ist.q          = q;
        if (!ist.moved)
          out.stop_rule = "no_moves";
        else if (q - q_prev < kDeltaQStop)
          out.stop_rule = "delta_q";
        else if (it == dp_.max_iter_until_stable)
          out.stop_rule = "max_iterations";
        stop   = out.stop_rule[0] != '\0';
        q_prev = q;
      }
      out.iters.push_back(std::move(ist));
      if (final) break;
    }
    // FINALIZE (§4.13): exact Q, size-ordered labels
    const Control c = quality(P0, C);
    out.modularity  = c.q[0];
    out.l_hat       = c.l_hat[0];
    out.n_clusters  = C;
    run_rank_labels(P0, n, C, nullptr, labels_out, L_.final_, L_.cub, L_.cub_bytes, L_.ctl, s_);
    readback();  // flags of the label kernels
    return true;
  }

 private:
  Layout& L_;
  DriverParams dp_;
  cudaStream_t s_;
  i64 syncs_ = 0;
  QuantizeResult q_;
  LevelGraph g0_;
  i64 n_ = 0, nnz_ = 0;
  double lam_ = 0.0, gamma_ = 0.0;
  u32 seed_ = 0;
  u64 s64_  = 0;

  // Control readback (one sync) with the device invariant checks.
  Control readback()
  {
    ++syncs_;
    const Control c = read_control(L_.ctl, s_, L_.pinned);
    CUGRAPH_EXPECTS(!(c.flags & kFlagForestInvariant),
                    "leiden: internal error: refinement "
                    "forest invariant violated.");
    CUGRAPH_EXPECTS(!(c.flags & kFlagGatherInvariant),
                    "leiden: internal error: multi-pass "
                    "gather failed.");
    CUGRAPH_EXPECTS(!(c.flags & kFlagBadLabel),
                    "leiden: internal error: community id "
                    "out of range.");
    return c;
  }

  // Partition of level l: P0 at level 0, then Pa / Pb alternately (the
  // projection overwrites a level's partition from the level above).
  int* level_partition(int l) const
  {
    if (l == 0) return L_.persist.P0;
    return (l & 1) ? L_.persist.Pa : L_.persist.Pb;
  }

  LevelGraph union_level(int R) const
  {
    LevelGraph g;
    g.indptr  = L_.uni.indptr;
    g.indices = L_.uni.indices;
    g.wf      = q_.wkind == WKind::F32 ? L_.uni.wf32 : nullptr;
    g.wq      = q_.wkind == WKind::I64 ? L_.uni.wq : nullptr;
    g.unit    = g0_.unit;
    g.scale   = g0_.scale;
    g.khat    = L_.uni.khat;
    g.n       = static_cast<i64>(R) * n_;
    g.nnz     = static_cast<i64>(R) * nnz_;
    for (int c = 0; c < kNumClasses; ++c)
      g.cls[c] = R * g0_.cls[c];
    return g;
  }

  // LOCAL_MOVE + COMPACT on level l; P holds ids in [0, id_bound) on entry
  // and the compacted move partition afterwards. Returns C (sync SL1).
  i64 move_and_compact(const LevelGraph& g,
                       int* P,
                       int l,
                       u32 phase,
                       int cap,
                       u32 sweep0,
                       i64 id_bound,
                       u32 it,
                       MoveStats& st)
  {
    MoveParams prm;
    prm.lam      = lam_;
    prm.seed     = seed_;
    prm.it       = it;
    prm.level    = static_cast<u32>(l);
    prm.phase    = phase;
    prm.sweep0   = sweep0;
    prm.cap      = cap;
    prm.S        = dp_.subrounds;
    prm.th       = dp_.th;
    prm.id_bound = id_bound;
    g.with_w([&](auto w) {
      st =
        run_local_move(g.indptr, g.indices, w, g.khat, P, g.n, g.cls, prm, L_.move, L_.pinned, s_);
    });
    syncs_ += st.sweeps + 1;
    return run_compact(P, g.n, L_.move, L_.persist.KS, L_.cub, L_.cub_bytes, L_.pinned, s_);
  }

  // CC_SPLIT (§4.12) of the partition P on graph g; returns C.
  i64 cc_split(const LevelGraph& g, int* P)
  {
    g.with_w([&](auto w) {
      run_cc_split(g.indptr, g.indices, w, P, g.n, L_.final_, L_.cub, L_.cub_bytes, L_.ctl, s_);
    });
    return readback().n_components;
  }

  // Exact Q (§4.13) of the compact single-copy partition P (C ids).
  Control quality(const int* P, i64 C)
  {
    g0_.with_w([&](auto w) {
      run_quality(g0_.indptr,
                  g0_.indices,
                  w,
                  g0_.khat,
                  P,
                  n_,
                  1,
                  C,
                  q_.two_m_hat,
                  gamma_,
                  L_.final_,
                  L_.ctl,
                  s_);
    });
    return readback();
  }

  // COMPACT_U (§4.2.1): rank of each id of P_in[0, n) among the ids used
  // in the slice (domain [0, C)), written to P_out; returns their number.
  i64 compact_slice(const int* P_in, i64 n, i64 C, int* P_out)
  {
    const FinalBufs& fb = L_.final_;
    RAFT_CUDA_TRY(cudaMemsetAsync(fb.flags, 0, (C + 1) * sizeof(int), s_));
    drv_slice_flags_kernel<<<grid_items(n), kBlock, 0, s_>>>(P_in, n, C, fb.flags, L_.ctl);
    RAFT_CHECK_CUDA(s_);
    std::size_t tb = L_.cub_bytes;
    RAFT_CUDA_TRY(cub::DeviceScan::ExclusiveSum(
      L_.cub, tb, fb.flags, fb.rank, cub_items(C + 1, "slice compact"), s_));
    drv_slice_relabel_kernel<<<grid_items(n), kBlock, 0, s_>>>(P_in, n, C, fb.rank, P_out, L_.ctl);
    RAFT_CHECK_CUDA(s_);
    return readback().n_components;
  }

  // LEIDEN_ITERATION (§4.5) on the level-0 graph g0 with partition P0 (ids
  // in [0, id_bound)). Returns false on arena overflow. C_out: communities
  // of the level-0 partition before the CC split.
  bool leiden_iteration(const LevelGraph& g0,
                        int* P0,
                        i64 id_bound,
                        int it,
                        bool vcycle,
                        IterationStat& ist,
                        i64& C_out)
  {
    LevelArena& arena = L_.arena;
    arena.reset();
    std::vector<LevelGraph> G;
    G.reserve(static_cast<std::size_t>(dp_.max_levels) + 1);
    G.push_back(g0);
    const u32 uit = static_cast<u32>(it);
    bool moved    = false;
    int lvl       = 0;
    u32 phase     = kPhaseDown;
    int reentry   = 0;
    u32 top_done  = 0;  // TOP sweeps run at this level (cumulative, §4.4)
    i64 idb       = id_bound;
    LevelStat rec;
    rec.n   = g0.n;
    rec.nnz = g0.nnz;
    AggregateOptions opt;
    opt.low_memory          = dp_.low_memory;
    opt.force_two_pass      = dp_.force_two_pass;
    opt.force_second_gather = dp_.force_second_gather;
    opt.gather              = dp_.gather;
    opt.move                = dp_.th;
    while (true) {
      LevelGraph& g = G[lvl];
      int* P        = level_partition(lvl);
      MoveStats st;
      i64 C = 0;
      if (phase == kPhaseDown) {
        C          = move_and_compact(g, P, lvl, kPhaseDown, dp_.move_sweeps, 0, idb, uit, st);
        rec.moves  = st.moves;
        rec.sweeps = st.sweeps;
      } else {
        C = move_and_compact(g, P, lvl, kPhaseTop, dp_.top_sweeps, top_done, idb, uit, st);
        top_done += static_cast<u32>(st.sweeps);
        rec.top_phases += 1;
        rec.top_moves += st.moves;
        rec.top_sweeps = static_cast<int>(top_done);
        if (st.last > 0) ist.top_cap_hit = true;
      }
      moved = moved || st.moves > 0;
      idb   = C;
      rec.C = C;
      if (C == g.n) {
        if (phase != kPhaseTop) {  // all-active verification (§4.10)
          phase = kPhaseTop;
          continue;
        }
        ist.levels.push_back(rec);  // certified gamma-separated top
        break;
      }
      // PLEIDEN_R + AGGREGATE stage 1 (speculative; SL2 after A4a)
      const std::size_t mark = arena.bottom;
      int* cmap              = arena.push_bottom<int>(static_cast<std::size_t>(g.n));
      if (!cmap) return false;
      RefineParams rp;
      rp.lam       = lam_;
      rp.ctx_order = ctx(s64_, kTagOrder, uit, static_cast<u32>(lvl), 0, 0);
      rp.trees     = dp_.trees;
      AggregatePlan plan;
      g.with_w([&](auto w) {
        run_refine(g.indptr,
                   g.indices,
                   w,
                   g.khat,
                   P,
                   L_.persist.KS,
                   g.n,
                   C,
                   rp,
                   L_.refine,
                   cmap,
                   L_.cub,
                   L_.cub_bytes,
                   L_.ctl,
                   s_);
        plan = aggregate_begin(g.indptr,
                               g.indices,
                               w,
                               g.khat,
                               g.n,
                               g.nnz,
                               cmap,
                               opt,
                               L_.aggregate,
                               arena,
                               L_.cub,
                               L_.cub_bytes,
                               L_.ctl,
                               s_);
      });
      Control c  = readback();  // SL2
      rec.n_next = c.n_next;
      if (c.n_next == g.n) {  // refinement merged nothing (C < n)
        arena.release_top();
        arena.bottom = mark;              // drop the speculative level
        if (reentry < dp_.max_reentry) {  // Lemma R (§4.10)
          ++reentry;
          ist.reentries += 1;
          rec.reentries = reentry;
          phase         = kPhaseTop;
          continue;
        }
        // last resort (never expected): the connected pieces of P
        ist.fallbacks += 1;
        ist.uncertified = true;
        cmap            = arena.push_bottom<int>(static_cast<std::size_t>(g.n));
        if (!cmap) return false;
        RAFT_CUDA_TRY(cudaMemcpyAsync(cmap, P, g.n * sizeof(int), cudaMemcpyDeviceToDevice, s_));
        const i64 pieces = cc_split(g, cmap);
        if (pieces == g.n) {  // output the pieces (uncertified)
          RAFT_CUDA_TRY(cudaMemcpyAsync(P, cmap, g.n * sizeof(int), cudaMemcpyDeviceToDevice, s_));
          arena.bottom = mark;
          idb          = g.n;
          rec.fallback = "pieces_are_singletons";
          ist.levels.push_back(rec);
          break;
        }
        rec.fallback = "cc";
        agg_set_n_next_kernel<<<1, 1, 0, s_>>>(L_.ctl, pieces);
        RAFT_CHECK_CUDA(s_);
        g.with_w([&](auto w) {
          plan = aggregate_begin(g.indptr,
                                 g.indices,
                                 w,
                                 g.khat,
                                 g.n,
                                 g.nnz,
                                 cmap,
                                 opt,
                                 L_.aggregate,
                                 arena,
                                 L_.cub,
                                 L_.cub_bytes,
                                 L_.ctl,
                                 s_);
        });
        c          = readback();
        rec.n_next = c.n_next;
      }
      // AGGREGATE stage 2: rows below the holey region, or two-pass
      CoarseLevel lv;
      g.with_w([&](auto w) {
        lv = aggregate_finish(g.indptr,
                              g.indices,
                              w,
                              g.khat,
                              g.n,
                              cmap,
                              plan,
                              c.n_next,
                              c.nnz_next,
                              opt,
                              L_.aggregate,
                              arena,
                              P,
                              level_partition(lvl + 1),
                              L_.ctl,
                              s_);
      });
      if (!lv.ok) return false;
      rec.layout        = lv.two_pass ? "two_pass" : "holey";
      rec.second_gather = lv.second_gather;
      g.cmap            = cmap;
      ist.levels.push_back(rec);
      LevelGraph ng;
      ng.indptr  = lv.indptr;
      ng.indices = lv.indices;
      ng.wf      = lv.weights;
      ng.khat    = lv.khat;
      ng.n       = lv.n;
      ng.nnz     = lv.nnz;
      for (int k = 0; k < kNumClasses; ++k)
        ng.cls[k] = c.coarse_class_count[k];
      G.push_back(ng);  // (invalidates g)
      // P_{l+1} holds the move partition's ids, < C <= n_{l+1}
      ++lvl;
      phase    = kPhaseDown;
      reentry  = 0;
      top_done = 0;
      rec      = LevelStat{};
      rec.n    = ng.n;
      rec.nnz  = ng.nnz;
      if (lvl == dp_.max_levels) {  // guard; kNN graphs need 5-12
        ist.levels.push_back(rec);
        break;
      }
    }
    // projection to level 0 (§4.5), with the V-cycle's UP sweeps in all
    // but the final iteration (§4.11)
    i64 Ccur = idb;
    for (int lp = lvl - 1; lp >= 0; --lp) {
      const LevelGraph& g = G[lp];
      int* P              = level_partition(lp);
      run_project(level_partition(lp + 1), g.cmap, g.n, P, s_);
      if (vcycle) {
        MoveStats st;
        Ccur  = move_and_compact(g, P, lp, kPhaseUp, dp_.vcycle_sweeps, 0, Ccur, uit, st);
        moved = moved || st.moves > 0;
        ist.vcycle_moves.push_back(st.moves);
      }
    }
    ist.moved = moved;
    C_out     = Ccur;
    return true;
  }
};

}  // namespace cugraph::detail::leiden_engine
