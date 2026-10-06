/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

// Interface between the typed raw-CSR entry point (community/detail/leiden/leiden_csr_impl.cuh:
// input check, canonicalization, layout, workspace, level-0 quantisation) and the
// type-independent engine (community/detail/leiden/driver_common.cu: local moving, refinement,
// aggregation, finalize).
//
// After the level-0 quantisation (kernels_ingest.cuh, I2) every array the engine reads has an
// internal type (int64 offsets, int32 indices, fp32 / int64 / unit weights), independent of the
// (vertex_t, edge_t, weight_t) of the cuGraph graph. The engine kernels are therefore compiled
// once, in driver_common.cu, instead of once per instantiation of cugraph::leiden.

#include "community/detail/leiden/arena.cuh"
#include "community/detail/leiden/numerics.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>

namespace cugraph::detail::leiden_engine {

// Result of the input check I1 (kernels_ingest.cuh, run_ingest_check).
struct IngestInfo {
  u32 flags     = 0;
  u64 wmax_bits = 0;
  double wmax   = 0.0;  // max counted value (1.0 for unit weights)
  i64 n_counted = 0;    // nnz_c: stored entries with u != v and w > 0
  u64 h_fwd = 0, h_rev = 0;
};

// Level-0 weight kind for one scale exponent (§4.3, §6.1).
inline WKind level0_kind(u32 flags, bool weighted, bool data_is_f32, int s)
{
  if (!weighted) return (flags & kFlagUnitZero) ? WKind::I64 : WKind::UNIT;
  if (data_is_f32 && s >= kF32ScaleMin && s <= kF32ScaleMax) return WKind::F32;
  return WKind::I64;
}

// Replicas of iteration 1 (§4.2.1): R = min(max_replicas, max(1, budget / nnz_c)). Small graphs
// run R disjoint copies with different hash contexts and keep the best one.
constexpr i64 kReplicaNnzBudget = i64{1} << 19;

inline int n_replicas(i64 nnz_counted,
                      int max_replicas = kMaxReplicas,
                      i64 budget       = kReplicaNnzBudget)
{
  if (nnz_counted <= 0) return 1;
  i64 const r = budget / nnz_counted;
  return static_cast<int>(std::min<i64>(max_replicas, std::max<i64>(1, r)));
}

// Level 0 after I2 (+ I4), as produced by run_quantize (kernels_ingest.cuh).
struct QuantizeResult {
  bool ok       = true;  // false: the carved layout lacks a representation this input needs
  int s         = 0;     // fixed-point scale exponent s(gamma)
  i64 unit_q    = 0;     // quantised unit weight (WKind::UNIT)
  i64 two_m_hat = 0;     // 2m_hat (exact int64)
  i64 class_count[kNumClasses] = {0, 0, 0, 0};
  i64 max_degree               = 0;
  WKind wkind                  = WKind::F32;
  // !ok: the layout the input needs
  WKind required_wkind       = WKind::F32;
  bool required_idx64_input  = false;
  std::size_t required_bytes = 0;
  // level-0 arrays (device pointers into the workspace or the input graph)
  i64 const* indptr  = nullptr;
  int const* indices = nullptr;
  float const* wf32  = nullptr;
  i64 const* wq      = nullptr;
};

// Knobs of one engine call. Everything else uses the reference defaults (4 DOWN sweeps of 4
// sub-rounds, 4 V-cycle sweeps in every iteration but the last, 32 TOP sweeps, 4 TOP re-entries
// per level, at most 20 iterations for n_iterations = -1).
struct EngineParams {
  int n_iterations = 2;           // >= 1 (< 2^16), or -1: until stable
  int max_levels   = kMaxLevels;  // levels per iteration, [1, 64]
  bool low_memory  = false;       // every level two-pass (layout only)
};

struct EngineResult {
  double modularity  = 0.0;  // exact Q(gamma) of the returned clustering
  i64 n_clusters     = 0;
  i64 l_hat          = 0;  // intra-community weight (quantised)
  i64 num_iterations = 0;
  i64 num_levels     = 0;  // levels of the last iteration
  i64 syncs          = 0;  // host synchronisations of the engine
  bool trivial       = false;
};

/**
 * Runs the Leiden iterations and FINALIZE (exact Q, size-ordered labels) for one resolution on
 * the level-0 graph `q` carved in `L` (run_quantize must have succeeded on `L`). Writes the
 * int32 labels of the n level-0 vertices to `labels_out`, a device array that may be
 * L.persist.labels32. Every kernel runs on `stream`; nothing is allocated.
 *
 * Returns false if the level arena overflowed even with two-pass levels; L.arena.required then
 * holds the demand (see arena_required_factor) and the call can be repeated on a larger
 * workspace with a bitwise identical result.
 */
bool run_engine(Layout& L,
                QuantizeResult const& q,
                EngineParams const& params,
                double gamma,
                u32 seed,
                int* labels_out,
                cudaStream_t stream,
                EngineResult& out);

}  // namespace cugraph::detail::leiden_engine
