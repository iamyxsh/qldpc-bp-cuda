// Normalized min-sum belief propagation for one syndrome, on the CPU or as a CUDA device function.
// Must stay bit-identical to qldpc_bp/bp.py: float32, additions in edge order, no FMA.
#pragma once

#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#define QBP_FN __host__ __device__ inline
#else
#define QBP_FN inline
#endif

struct QbpGraph {
  int32_t n;  // qubits
  int32_t m;  // checks
  int32_t E;  // edges
  const int32_t* chk_ptr;    // m + 1: the edges of check c are [chk_ptr[c], chk_ptr[c + 1])
  const int32_t* edge_var;   // E: the qubit each edge touches
  const int32_t* var_ptr;    // n + 1: var_edges[var_ptr[v] .. var_ptr[v + 1]) are qubit v's edges
  const int32_t* var_edges;  // E: edge ids grouped by qubit, checks ascending
};

// Writes the n-bit error estimate to `hard`; `v2c` and `c2v` are scratch for E floats each.
// Returns the 1-based iteration whose estimate first reproduced the syndrome, or -1 if none did.
QBP_FN int32_t qbp_min_sum(const QbpGraph& g, const float* prior, const uint8_t* syndrome,
                           float alpha, int32_t max_iters, float* v2c, float* c2v,
                           uint8_t* hard) {
  for (int32_t e = 0; e < g.E; ++e) c2v[e] = 0.0f;
  for (int32_t v = 0; v < g.n; ++v) hard[v] = 0;

  for (int32_t it = 1; it <= max_iters; ++it) {
    // Qubit to check: the prior plus every other check's message, in edge order.
    for (int32_t v = 0; v < g.n; ++v) {
      const int32_t lo = g.var_ptr[v], hi = g.var_ptr[v + 1];
      for (int32_t i = lo; i < hi; ++i) {
        float acc = prior[v];
        for (int32_t j = lo; j < hi; ++j) {
          if (j != i) acc = acc + c2v[g.var_edges[j]];
        }
        v2c[g.var_edges[i]] = acc;
      }
    }

    // Check to qubit: alpha times the smallest magnitude among the other edges, signed by their
    // parity and by whether the check fired. The two smallest magnitudes cover every edge.
    for (int32_t c = 0; c < g.m; ++c) {
      const int32_t lo = g.chk_ptr[c], hi = g.chk_ptr[c + 1];
      float min1 = INFINITY, min2 = INFINITY;
      int32_t argmin = lo;
      bool odd = syndrome[c] != 0;
      for (int32_t e = lo; e < hi; ++e) {
        const float x = v2c[e];
        const float ax = fabsf(x);
        if (ax < min1) {
          min2 = min1;
          min1 = ax;
          argmin = e;
        } else if (ax < min2) {
          min2 = ax;
        }
        if (x < 0.0f) odd = !odd;
      }
      for (int32_t e = lo; e < hi; ++e) {
        const float val = alpha * (e == argmin ? min2 : min1);
        const bool neg = odd != (v2c[e] < 0.0f);
        c2v[e] = neg ? -val : val;
      }
    }

    // A qubit is flipped only when its posterior is strictly negative.
    for (int32_t v = 0; v < g.n; ++v) {
      float acc = prior[v];
      for (int32_t j = g.var_ptr[v]; j < g.var_ptr[v + 1]; ++j) acc = acc + c2v[g.var_edges[j]];
      hard[v] = acc < 0.0f ? 1 : 0;
    }

    bool matched = true;
    for (int32_t c = 0; c < g.m && matched; ++c) {
      uint8_t parity = syndrome[c] != 0 ? 1 : 0;
      for (int32_t e = g.chk_ptr[c]; e < g.chk_ptr[c + 1]; ++e) parity ^= hard[g.edge_var[e]];
      matched = parity == 0;
    }
    if (matched) return it;
  }
  return -1;
}
