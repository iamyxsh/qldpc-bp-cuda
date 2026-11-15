// CPU driver: decodes every syndrome in a problem file on one thread.
// Usage: bp_cpu <problem.bin> <result.bin> [repeats]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <vector>

#include "bp_core.h"
#include "problem_io.h"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <problem.bin> <result.bin> [repeats]\n", argv[0]);
    return 2;
  }
  try {
    const qbp_io::Problem p = qbp_io::read_problem(argv[1]);
    const int repeats = argc > 3 ? std::max(1, std::atoi(argv[3])) : 1;
    const QbpGraph g{p.n, p.m, p.E, p.chk_ptr.data(), p.edge_var.data(), p.var_ptr.data(),
                     p.var_edges.data()};
    std::vector<uint8_t> hard(static_cast<size_t>(p.batch) * p.n);
    std::vector<int32_t> iters(p.batch);
    std::vector<float> v2c(p.E), c2v(p.E);

    double best = std::numeric_limits<double>::infinity();
    for (int r = 0; r < repeats; ++r) {
      const auto t0 = std::chrono::steady_clock::now();
      for (int32_t s = 0; s < p.batch; ++s) {
        iters[s] = qbp_min_sum(g, p.prior.data(), p.syndromes.data() + static_cast<size_t>(s) * p.m,
                               p.alpha, p.max_iters, v2c.data(), c2v.data(),
                               hard.data() + static_cast<size_t>(s) * p.n);
      }
      const std::chrono::duration<double> dt = std::chrono::steady_clock::now() - t0;
      best = std::min(best, dt.count());
    }
    qbp_io::write_result(argv[2], p, hard, iters);
    // One JSON line so the Python side can read the timing.
    std::printf("{\"backend\": \"cpu\", \"batch\": %d, \"decode_seconds\": %.9f}\n", p.batch, best);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
