/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The type-independent engine of cugraph::leiden (see driver.hpp): local moving, refinement,
// aggregation and FINALIZE on the internal level-0 form. Compiled once into libcugraph_common
// (not per vertex_t / edge_t / weight_t instantiation, and shared by the single-GPU and the
// multi-GPU library) and with --fmad=false (cpp/CMakeLists.txt).

#include "community/detail/leiden/driver.cuh"
#include "community/detail/leiden/driver.hpp"

namespace cugraph::detail::leiden_engine {

bool run_engine(Layout& L,
                QuantizeResult const& q,
                EngineParams const& params,
                double gamma,
                u32 seed,
                int* labels_out,
                cudaStream_t stream,
                EngineResult& out)
{
  CUGRAPH_EXPECTS(q.ok, "leiden: level 0 was not quantised into this layout.");
  CUGRAPH_EXPECTS(
    params.n_iterations == -1 || (params.n_iterations >= 1 && params.n_iterations < (1 << 16)),
    "leiden: n_iterations must be in [1, 2^16) or -1.");
  CUGRAPH_EXPECTS(params.max_levels >= 1 && params.max_levels <= kMaxLevels,
                  "leiden: max_levels must be in [1, 64].");

  DriverParams dp;
  dp.n_iterations = params.n_iterations;
  dp.max_levels   = params.max_levels;
  dp.low_memory   = params.low_memory;
  dp.arena_factor = L.p.arena_factor;
  dp.subrounds    = L.p.subrounds;

  return run_synced(stream, [&] {
    LeidenDriver drv(L, dp, stream);
    drv.set_level0(q, L.p.n, L.p.nnz);
    ResolutionResult r;
    bool const ok = drv.run_resolution(gamma, seed, nullptr, L.p.replicas, labels_out, r);
    out.syncs     = drv.syncs();
    if (!ok) return false;
    out.modularity     = r.modularity;
    out.n_clusters     = r.n_clusters;
    out.l_hat          = r.l_hat;
    out.trivial        = r.trivial;
    out.num_iterations = static_cast<i64>(r.iters.size());
    out.num_levels     = r.iters.empty() ? 0 : static_cast<i64>(r.iters.back().levels.size());
    return true;
  });
}

}  // namespace cugraph::detail::leiden_engine
