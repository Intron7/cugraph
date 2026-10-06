/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

// Workspace layout of the deterministic Leiden engine (spec §6, §8.5).
//
// The caller allocates ONE device buffer per call of `workspace_bytes(...)` bytes; everything the
// algorithm needs is carved out of it here, including CUB temp storage. Nothing is allocated
// inside the algorithm. Regions, in order:
//
//   control   Control block: device scalars, counters, flags (4 KB)
//   cub       CUB temp storage, sized for the largest n-sized scan / sort
//   phase     one region shared by the phase groups (Move, Refine, Aggregate, Final). Groups
//             with disjoint lifetimes alias the same bytes (§6.2), so it is sized for the
//             largest group.
//   persist   per-call vertex state: P0, ping-pong partitions, K_S (+ int32 labels for int64
//             vertex ids)
//   level0    level-0 graph in internal form: int64 offsets, int32 indices copy (int64 input
//             indices only), int64 W0q, k_hat, degree classes
//   union     replica union graph of iteration 1 (R > 1 only, §4.2.1)
//   arena     level metadata + coarse rows (§6.3): levels 1..l grow up from the bottom, the
//             holey contraction output is carved from the top; one checked allocator
//             (overflow -> two-pass level, regrow and rerun only as a last resort).
//
// The layout never reaches a decision of the algorithm: every layout (arena factor, low_memory,
// holey or two-pass levels, CUB temp size) gives bitwise identical results.

#include "community/detail/leiden/numerics.cuh"

#include <cugraph/utilities/error.hpp>

#include <raft/util/cuda_rt_essentials.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_sort.cuh>
#include <thrust/iterator/transform_iterator.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace cugraph::detail::leiden_engine {

constexpr std::size_t kAlign = 256;

inline std::size_t align_up(std::size_t x, std::size_t a = kAlign) { return (x + a - 1) / a * a; }

// ---------------------------------------------------------------------------
// Control block (device scalars; read back through a small host mailbox)
// ---------------------------------------------------------------------------

// Ingest flag bits (Control::flags). The low four bits equal the golden
// reference's FLAG_NEG / FLAG_NONFINITE / FLAG_NONCANONICAL /
// FLAG_ASYMMETRIC (the last is set on the host from the fingerprints).
enum : u32 {
  kFlagNegative     = 1u << 0,  // w < 0 (weighted input)
  kFlagNonFinite    = 1u << 1,  // NaN / Inf (weighted input)
  kFlagNonCanonical = 1u << 2,  // u_j <= u_{j-1} within a row (parallel edges)
  kFlagAsymmetric   = 1u << 3,  // h_fwd != h_rev (host)
  kFlagBadIndex     = 1u << 4,  // column index outside [0, n)
  kFlagBadIndptr    = 1u << 5,  // indptr not monotone / wrong ends
  kFlagUnitZero     = 1u << 6,  // unit weights with off-diagonal zeros
  kFlagRowTooLong   = 1u << 7,  // a row with >= 2^31 stored entries
  kFlagBadLabel     = 1u << 8,  // community id outside [0, C)
};

struct alignas(16) Control {
  // ingest (I1 ingest_check / I2 quantize)
  u32 flags;
  u32 pad0;
  u64 wmax_bits;  // max counted weight, IEEE bits of the input dtype
  u64 h_fwd;      // order-free symmetry fingerprint (u64 wrapping sums)
  u64 h_rev;
  i64 n_counted;  // nnz_c: counted stored entries
  i64 two_m_hat;  // 2m_hat (int64, exact)
  i64 class_count[kNumClasses];
  i64 max_degree;  // largest stored row of level 0
  // finalize (F1-F5)
  i64 n_components;  // C after CC split / compaction
  i64 l_hat[kMaxReplicas];
  double sum_t[kMaxReplicas];
  double q[kMaxReplicas];
  // driver / move / refine / aggregate: counters and flags
  i64 movers;
  i64 n_next;
  i64 arena_overflow;
  i64 arena_required;
  // refine / aggregate: class counters, coarse-level statistics read back at SL2 together with
  // n_next (kernels_refine.cuh, kernels_aggregate.cuh); each is zeroed by the kernel that starts
  // it
  int tree_list_count[4];               // R5: thread / warp / block / large commits
  int gather_list_count[3];             // A2: warp / wide warp / block gathers
  i64 nnz_next;                         // A4a: entries of level l + 1
  i64 coarse_class_count[kNumClasses];  // A4a: degree classes of l + 1
  i64 coarse_max_degree;                // A4a
  i64 reserved[64];
};
static_assert(sizeof(Control) <= 4096, "Control block must fit in 4 KB");
constexpr std::size_t kControlBytes = 4096;

// Host readback of the control block (§8.3): a small D2H copy plus a stream sync. With a pinned
// host buffer (`pinned`, >= kPinnedBytes; e.g. from cugraph::host_staging_buffer_manager) the
// copy goes through it; otherwise to pageable stack memory, which is correct, only slower.
// Nothing is allocated here.
constexpr std::size_t kPinnedBytes = 4096;
static_assert(sizeof(Control) <= kPinnedBytes, "Control must fit kPinnedBytes");

inline Control read_control(Control const* d_ctl, cudaStream_t s, void* pinned = nullptr)
{
  Control h;
  void* dst = pinned ? pinned : static_cast<void*>(&h);
  RAFT_CUDA_TRY(cudaMemcpyAsync(dst, d_ctl, sizeof(Control), cudaMemcpyDeviceToHost, s));
  RAFT_CUDA_TRY(cudaStreamSynchronize(s));
  if (pinned) std::memcpy(&h, pinned, sizeof(Control));
  return h;
}

inline void clear_control(Control* d_ctl, cudaStream_t s)
{
  RAFT_CUDA_TRY(cudaMemsetAsync(d_ctl, 0, sizeof(Control), s));
}

// ---------------------------------------------------------------------------
// Carver: one code path for sizing (base == nullptr) and carving
// ---------------------------------------------------------------------------

struct Carver {
  char* base      = nullptr;  // nullptr: sizing pass
  std::size_t off = 0;

  template <typename T>
  T* take(std::size_t count)
  {
    off  = align_up(off);
    T* p = base ? reinterpret_cast<T*>(base + off) : nullptr;
    off += count * sizeof(T);
    return p;
  }
  char* bytes(std::size_t n) { return take<char>(n); }
  std::size_t used() const { return align_up(off); }
};

// ---------------------------------------------------------------------------
// Layout parameters
// ---------------------------------------------------------------------------

struct LayoutParams {
  i64 n               = 0;  // level-0 vertices (one copy)
  i64 nnz             = 0;  // stored entries of the input graph
  WKind wkind         = WKind::F32;
  bool idx64_input    = false;  // int64 vertex ids: int32 level-0 index copy, int32 labels
  int replicas        = 1;      // R (§4.2.1)
  double arena_factor = 1.9;    // §6.3
  bool low_memory     = false;  // two-pass contraction (layout only)
  int subrounds       = kNumSubrounds;

  i64 n_union() const { return static_cast<i64>(replicas) * n; }
  i64 nnz_union() const { return static_cast<i64>(replicas) * nnz; }

  void validate() const
  {
    CUGRAPH_EXPECTS(n >= 0 && nnz >= 0, "leiden: n and nnz must be >= 0.");
    CUGRAPH_EXPECTS(replicas >= 1 && replicas <= kMaxReplicas,
                    "leiden: replicas must be in [1, 4].");
    CUGRAPH_EXPECTS(n_union() < kMaxVertices,
                    "leiden: graph too large (replicas * n must be < 2^30).");
    CUGRAPH_EXPECTS(arena_factor > 0.0, "leiden: arena_factor must be > 0.");
    CUGRAPH_EXPECTS(subrounds >= 1 && subrounds <= 64, "leiden: subrounds must be in [1, 64].");
  }
};

// ---------------------------------------------------------------------------
// CUB call helpers (red team RE5)
// ---------------------------------------------------------------------------

// Several CUB algorithms take `int num_items`: every 64-bit count passed to CUB is range-checked.
inline int cub_items(i64 n, char const* what)
{
  CUGRAPH_EXPECTS(n >= 0 && n < (1ll << 31), "leiden: %s: CUB item count must be < 2^31.", what);
  return static_cast<int>(n);
}

template <typename T>
struct WidenToI64 {
  __host__ __device__ __forceinline__ i64 operator()(T x) const { return static_cast<i64>(x); }
};

// Exclusive scan of `items` integer counts into int64 offsets. CUB's
// ExclusiveSum accumulates in the *input* value type, so the counts are fed
// through an int64 transform iterator: every indptr-producing scan uses this
// helper (§7). Size query: pass temp = nullptr.
template <typename CountT>
inline cudaError_t scan_offsets64(void* temp,
                                  std::size_t& temp_bytes,
                                  CountT const* counts,
                                  i64* offsets,
                                  i64 items,
                                  cudaStream_t s = 0)
{
  auto in = thrust::make_transform_iterator(counts, WidenToI64<CountT>{});
  return cub::DeviceScan::ExclusiveSum(
    temp, temp_bytes, in, offsets, cub_items(items, "scan_offsets64"), s);
}

// ---------------------------------------------------------------------------
// CUB temp storage (queried on the host; no device work, no allocation)
// ---------------------------------------------------------------------------

// Largest temp storage of any n-sized CUB call of the algorithm: scans over
// community ids [0, 2n] (int32) and vertex windows (int64), the label sort
// (uint64 keys + int32 values), and the segmented sort of large refinement
// trees (int64 order keys). No CUB call of the driver has nnz items.
inline std::size_t cub_temp_bytes(i64 n_u)
{
  i64 const n2     = 2 * n_u + 2;
  std::size_t best = 0, b = 0;
  auto take = [&](cudaError_t e) {
    RAFT_CUDA_TRY(e);
    best = std::max(best, b);
    b    = 0;
  };
  int* i32  = nullptr;
  i64* i64p = nullptr;
  u64* u64p = nullptr;
  take(cub::DeviceScan::ExclusiveSum(nullptr, b, i32, i32, cub_items(n2, "scan query")));
  take(cub::DeviceScan::ExclusiveSum(nullptr, b, i64p, i64p, cub_items(n_u + 2, "scan query")));
  take(scan_offsets64<int>(nullptr, b, i32, i64p, n_u + 2));
  if (n_u > 0) {
    int const items = cub_items(n_u, "sort query");
    take(cub::DeviceRadixSort::SortPairs(nullptr, b, u64p, u64p, i32, i32, items, 0, 64));
    take(cub::DeviceSegmentedSort::SortPairs(
      nullptr, b, i64p, i64p, i32, i32, items, items, i64p, i64p));
  }
  return best;
}

// ---------------------------------------------------------------------------
// Phase groups (aliased in the phase region)
// ---------------------------------------------------------------------------

// LOCAL_MOVE + COMPACT (§4.6, §4.7). Community id space is [0, 2 n_U).
struct MoveBufs {
  i64* K            = nullptr;  // [2 n_U] community volumes
  int* csize        = nullptr;  // [2 n_U] vertices per community id
  int* rank         = nullptr;  // [2 n_U + 2] COMPACT flags / exclusive ranks
  int* active       = nullptr;  // [n_U]
  i64* dest         = nullptr;  // [n_U] (stamp << 32) | community
  int* buckets      = nullptr;  // [S * n_U] per sub-round vertex lists
  int* bucket_count = nullptr;  // [S * kNumClasses * 4] counts / cursors
  int* movers       = nullptr;  // [n_U]
  int* vlist        = nullptr;  // [n_U] class-grouped vertex list

  void carve(Carver& c, i64 n_u, int S)
  {
    K            = c.take<i64>(2 * n_u);
    csize        = c.take<int>(2 * n_u);
    rank         = c.take<int>(2 * n_u + 2);
    active       = c.take<int>(n_u);
    dest         = c.take<i64>(n_u);
    buckets      = c.take<int>(static_cast<std::size_t>(S) * n_u);
    bucket_count = c.take<int>(static_cast<std::size_t>(S) * kNumClasses * 4);
    movers       = c.take<int>(n_u);
    vlist        = c.take<int>(n_u);
  }
};

// PLEIDEN_R (§4.8). R9 must move everything AGGREGATE needs (cmap, k_hat of the next level) into
// the level stack: the aggregate group aliases these bytes.
struct RefineBufs {
  i64* ext               = nullptr;  // [n_U]
  i64* inner             = nullptr;  // [n_U]
  i64* kprime            = nullptr;  // [n_U] k_hat' per host vertex (R8)
  unsigned char* U       = nullptr;  // [n_U]
  unsigned char* flipped = nullptr;  // [n_U]
  int* hn                = nullptr;  // [n_U]
  int* root              = nullptr;  // [n_U]
  int* tree_count        = nullptr;  // [n_U + 2]
  int* tree_offset       = nullptr;  // [n_U + 2]
  int* members           = nullptr;  // [n_U]
  int* r                 = nullptr;  // [n_U]
  int* cid               = nullptr;  // [n_U + 2]
  i64* sort_keys_a       = nullptr;  // [n_U] segmented path (trees > 1024)
  i64* sort_keys_b       = nullptr;
  int* sort_vals_a       = nullptr;
  int* sort_vals_b       = nullptr;
  // R8 commit-class lists of tree roots
  int* tree_list  = nullptr;  // [n_U] thread class front, warp class back
  int* large_list = nullptr;  // [n_U] larger trees front, block class back

  void carve(Carver& c, i64 n_u)
  {
    tree_list   = c.take<int>(n_u);
    large_list  = c.take<int>(n_u);
    ext         = c.take<i64>(n_u);
    inner       = c.take<i64>(n_u);
    kprime      = c.take<i64>(n_u);
    U           = c.take<unsigned char>(n_u);
    flipped     = c.take<unsigned char>(n_u);
    hn          = c.take<int>(n_u);
    root        = c.take<int>(n_u);
    tree_count  = c.take<int>(n_u + 2);
    tree_offset = c.take<int>(n_u + 2);
    members     = c.take<int>(n_u);
    r           = c.take<int>(n_u);
    cid         = c.take<int>(n_u + 2);
    sort_keys_a = c.take<i64>(n_u);
    sort_keys_b = c.take<i64>(n_u);
    sort_vals_a = c.take<int>(n_u);
    sort_vals_b = c.take<int>(n_u);
  }
};

// AGGREGATE (§4.9).
struct AggregateBufs {
  int* member_count  = nullptr;  // [n_U + 2]
  i64* member_start  = nullptr;  // [n_U + 2]
  int* members       = nullptr;  // [n_U]
  i64* window        = nullptr;  // [n_U + 2] sum of member degrees
  i64* window_offset = nullptr;  // [n_U + 2]
  i64* cdeg          = nullptr;  // [n_U + 2] coarse row lengths
  // gather-class lists and the level-(l + 1) metadata before it is copied into its exact-size
  // arena slots after SL2
  int* clist       = nullptr;  // [n_U] warp class front, block class back
  int* clist_wide  = nullptr;  // [n_U] wide-warp class
  i64* khat_next   = nullptr;  // [n_U + 2] member sums of k_hat
  i64* indptr_next = nullptr;  // [n_U + 2] scan of cdeg

  void carve(Carver& c, i64 n_u)
  {
    member_count  = c.take<int>(n_u + 2);
    member_start  = c.take<i64>(n_u + 2);
    members       = c.take<int>(n_u);
    window        = c.take<i64>(n_u + 2);
    window_offset = c.take<i64>(n_u + 2);
    cdeg          = c.take<i64>(n_u + 2);
    clist         = c.take<int>(n_u);
    clist_wide    = c.take<int>(n_u);
    khat_next     = c.take<i64>(n_u + 2);
    indptr_next   = c.take<i64>(n_u + 2);
  }
};

// FINALIZE: CC split, Q, labels (§4.12, §4.13; F1-F5).
struct FinalBufs {
  int* parent   = nullptr;  // [n_U] union-find forest (F1/F2)
  int* flags    = nullptr;  // [n_U + 2] COMPACT flags
  int* rank     = nullptr;  // [n_U + 2] COMPACT ranks
  i64* K        = nullptr;  // [n_U] community volumes (F3)
  double* t     = nullptr;  // [n_U] (K_c / 2m)^2 (F4)
  int* rep      = nullptr;  // [n_U] replica mask of each community (R > 1)
  int* size     = nullptr;  // [n_U] community sizes (F5)
  int* minv     = nullptr;  // [n_U] minimum vertex id (F5)
  u64* keys_in  = nullptr;  // [n_U] label keys (F5)
  u64* keys_out = nullptr;
  int* vals_in  = nullptr;
  int* vals_out = nullptr;
  int* label_of = nullptr;  // [n_U] community -> label

  void carve(Carver& c, i64 n_u)
  {
    parent   = c.take<int>(n_u);
    flags    = c.take<int>(n_u + 2);
    rank     = c.take<int>(n_u + 2);
    K        = c.take<i64>(n_u);
    t        = c.take<double>(n_u);
    rep      = c.take<int>(n_u);
    size     = c.take<int>(n_u);
    minv     = c.take<int>(n_u);
    keys_in  = c.take<u64>(n_u);
    keys_out = c.take<u64>(n_u);
    vals_in  = c.take<int>(n_u);
    vals_out = c.take<int>(n_u);
    label_of = c.take<int>(n_u);
  }
};

// ---------------------------------------------------------------------------
// Persistent regions
// ---------------------------------------------------------------------------

struct PersistBufs {
  int* P0       = nullptr;  // [n_U] partition carried across iterations
  int* Pa       = nullptr;  // [n_U] ping-pong partitions of the current /
  int* Pb       = nullptr;  //       next level (projection, coarse init)
  i64* KS       = nullptr;  // [n_U] K_S volumes after COMPACT
  int* labels32 = nullptr;  // [n] int32 labels, widened by the caller (int64 vertex ids only)

  void carve(Carver& c, LayoutParams const& p)
  {
    i64 const n_u = p.n_union();
    P0            = c.take<int>(n_u);
    Pa            = c.take<int>(n_u);
    Pb            = c.take<int>(n_u);
    KS            = c.take<i64>(n_u);
    labels32      = p.idx64_input ? c.take<int>(p.n) : nullptr;
  }
};

struct Level0Bufs {
  i64* indptr           = nullptr;  // [n + 1] internal int64 offsets
  int* indices          = nullptr;  // [nnz] int32 copy (null: int32 input read in place)
  i64* wq               = nullptr;  // [nnz] W0q (WKind::I64)
  i64* khat             = nullptr;  // [n]
  unsigned char* vclass = nullptr;  // [n] degree class of level 0

  void carve(Carver& c, LayoutParams const& p)
  {
    indptr  = c.take<i64>(p.n + 1);
    indices = p.idx64_input ? c.take<int>(p.nnz) : nullptr;
    wq      = p.wkind == WKind::I64 ? c.take<i64>(p.nnz) : nullptr;
    khat    = c.take<i64>(p.n);
    vclass  = c.take<unsigned char>(p.n);
  }
};

// Replica union graph (§4.2.1): vertex r n + v, edges shifted by r n.
struct UnionBufs {
  i64* indptr           = nullptr;  // [R n + 1]
  int* indices          = nullptr;  // [R nnz]
  float* wf32           = nullptr;  // [R nnz] (F32)
  i64* wq               = nullptr;  // [R nnz] (I64)
  i64* khat             = nullptr;  // [R n]
  unsigned char* vclass = nullptr;  // [R n]

  // An I64 layout also serves the F32 kind (it is a superset), so its union
  // carries both weight arrays; unions are small (R > 1 only for
  // nnz_c <= 2^18, §4.2.1).
  void carve(Carver& c, LayoutParams const& p)
  {
    if (p.replicas <= 1) return;
    i64 const nu = p.n_union(), eu = p.nnz_union();
    bool const i64w = p.wkind == WKind::I64;
    indptr          = c.take<i64>(nu + 1);
    indices         = c.take<int>(eu);
    wf32            = (i64w || p.wkind == WKind::F32) ? c.take<float>(eu) : nullptr;
    wq              = i64w ? c.take<i64>(eu) : nullptr;
    khat            = c.take<i64>(nu);
    vclass          = c.take<unsigned char>(nu);
  }
};

// ---------------------------------------------------------------------------
// Level stack + edge arena (§6.3)
// ---------------------------------------------------------------------------

// One checked byte region for the level metadata and the coarse rows (§6.3,
// red team RE2): 8 * arena_factor * nnz_0 + 24 * n_0 bytes (union sizes when
// R > 1), plus slack for the per-array alignment of up to kMaxLevels levels.
// 8 B per entry: int32 index + fp32 weight; 24 B per vertex: int64 offsets,
// int64 k_hat and int32 cmap of a coarse level (20 B) + the finer cmap (4 B).
constexpr std::size_t kArenaEntryBytes  = 8;
constexpr std::size_t kArenaVertexBytes = 24;
constexpr std::size_t kArenaSlack       = kMaxLevels * 8 * kAlign;

inline std::size_t arena_vertex_bytes(LayoutParams const& p)
{
  return kArenaVertexBytes * static_cast<std::size_t>(p.n_union()) + kArenaSlack;
}

inline std::size_t arena_bytes(LayoutParams const& p)
{
  double const rows = std::ceil(p.arena_factor * static_cast<double>(kArenaEntryBytes) *
                                static_cast<double>(p.nnz_union()));
  return align_up(static_cast<std::size_t>(rows) + arena_vertex_bytes(p));
}

// Arena factor that makes a demand of `needed` arena bytes fit (reported with a
// workspace_too_small status; the caller frees the old workspace and reallocates with
// max(2 x factor, 1.25 x required)).
inline double arena_required_factor(LayoutParams const& p, std::size_t needed)
{
  double const per = static_cast<double>(kArenaEntryBytes) * static_cast<double>(p.nnz_union());
  if (per <= 0.0) return p.arena_factor;
  double const f = (static_cast<double>(needed) - static_cast<double>(arena_vertex_bytes(p))) / per;
  return std::max(f, p.arena_factor);
}

// Host-side bookkeeping of the arena. The bottom grows with each level's
// metadata and compacted rows (bump pointer, reset per iteration); the top
// holds only the holey output of the level being contracted (no hash
// windows: §6.3, §6.5). Every carve is checked; on overflow the carve returns
// nullptr, `overflow` is set and `required` records the demand, so the driver
// can switch the level to two-pass contraction or report the bytes needed.
struct LevelArena {
  char* base           = nullptr;
  std::size_t cap      = 0;
  std::size_t bottom   = 0;  // bytes used from the bottom
  std::size_t top      = 0;  // bytes reserved from the top
  std::size_t required = 0;
  bool overflow        = false;

  void reset()
  {
    bottom = 0;
    top    = 0;
  }
  std::size_t free_bytes() const { return cap > bottom + top ? cap - bottom - top : 0; }
  bool fits(std::size_t extra_bottom, std::size_t extra_top) const
  {
    return align_up(bottom) + extra_bottom + align_up(top) + extra_top <= cap;
  }

  template <typename T>
  T* push_bottom(std::size_t count)
  {
    std::size_t const b = align_up(bottom), nb = count * sizeof(T);
    if (b + nb + top > cap) {
      note_overflow(b + nb + top);
      return nullptr;
    }
    bottom = b + nb;
    return reinterpret_cast<T*>(base + b);
  }

  // Reserve `count` T from the top (below any earlier top reservation).
  template <typename T>
  T* push_top(std::size_t count)
  {
    std::size_t const nt = align_up(top + count * sizeof(T));
    if (align_up(bottom) + nt > cap) {
      note_overflow(align_up(bottom) + nt);
      return nullptr;
    }
    top = nt;
    return reinterpret_cast<T*>(base + cap - nt);
  }
  void release_top() { top = 0; }

  void note_overflow(std::size_t need)
  {
    overflow = true;
    required = std::max(required, need);
  }
};

// ---------------------------------------------------------------------------
// The full layout
// ---------------------------------------------------------------------------

struct Region {
  std::size_t offset = 0;
  std::size_t bytes  = 0;
};

struct Layout {
  LayoutParams p;
  Control* ctl          = nullptr;
  void* pinned          = nullptr;  // pinned host readback buffer (or null: pageable)
  void* cub             = nullptr;
  std::size_t cub_bytes = 0;
  // Phase groups: all start at the same address (aliased).
  MoveBufs move;
  RefineBufs refine;
  AggregateBufs aggregate;
  FinalBufs final_;
  PersistBufs persist;
  Level0Bufs l0;
  UnionBufs uni;
  LevelArena arena;

  Region r_control, r_cub, r_phase, r_persist, r_level0, r_union, r_arena;
  std::size_t total = 0;
};

// Control, cub and the aliased phase groups for n_u = R * n vertices.
inline void carve_prefix(Layout& L, Carver& c, i64 n_u, int S)
{
  L.r_control.offset = c.used();
  L.ctl              = reinterpret_cast<Control*>(c.bytes(kControlBytes));
  L.r_control.bytes  = c.used() - L.r_control.offset;

  L.r_cub.offset = c.used();
  L.cub_bytes    = cub_temp_bytes(n_u);
  L.cub          = c.bytes(L.cub_bytes);
  L.r_cub.bytes  = c.used() - L.r_cub.offset;

  L.r_phase.offset  = c.used();
  std::size_t phase = 0;
  auto group        = [&](auto&& fn) {
    Carver g{c.base, c.used()};
    fn(g);
    phase = std::max(phase, g.used() - L.r_phase.offset);
  };
  group([&](Carver& g) { L.move.carve(g, n_u, S); });
  group([&](Carver& g) { L.refine.carve(g, n_u); });
  group([&](Carver& g) { L.aggregate.carve(g, n_u); });
  group([&](Carver& g) { L.final_.carve(g, n_u); });
  c.off           = L.r_phase.offset + phase;
  L.r_phase.bytes = c.used() - L.r_phase.offset;
}

// Carves the layout of `p` from `base` (nullptr: sizing pass only).
inline Layout make_layout(LayoutParams const& p, void* base)
{
  p.validate();
  Layout L;
  L.p = p;
  Carver c{static_cast<char*>(base), 0};
  carve_prefix(L, c, p.n_union(), p.subrounds);

  L.r_persist.offset = c.used();
  L.persist.carve(c, p);
  L.r_persist.bytes = c.used() - L.r_persist.offset;

  L.r_level0.offset = c.used();
  L.l0.carve(c, p);
  L.r_level0.bytes = c.used() - L.r_level0.offset;

  L.r_union.offset = c.used();
  L.uni.carve(c, p);
  L.r_union.bytes = c.used() - L.r_union.offset;

  L.r_arena.offset     = c.used();
  std::size_t const ab = arena_bytes(p);
  L.arena.base         = c.bytes(ab);
  L.arena.cap          = ab;
  L.r_arena.bytes      = c.used() - L.r_arena.offset;

  L.total = c.used();
  return L;
}

inline std::size_t workspace_bytes(LayoutParams const& p) { return make_layout(p, nullptr).total; }

// Runs `fn` (device work on `stream`); if it throws, the stream is synchronised before the
// exception propagates, so no queued kernel can touch a workspace the caller frees while
// unwinding (§8.6).
template <typename Fn>
auto run_synced(cudaStream_t stream, Fn&& fn) -> decltype(fn())
{
  try {
    return fn();
  } catch (...) {
    cudaStreamSynchronize(stream);
    cudaGetLastError();  // the original error is the one reported
    throw;
  }
}

}  // namespace cugraph::detail::leiden_engine
