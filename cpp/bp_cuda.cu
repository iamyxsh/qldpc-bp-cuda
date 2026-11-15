// CUDA driver: one GPU thread per syndrome, the whole BP loop inside a single kernel launch.
// Usage: bp_cuda <problem.bin> <result.bin> [repeats]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "bp_core.h"
#include "problem_io.h"

static void cuda_check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
  }
}

static double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

template <typename T>
struct DeviceArray {
  T* ptr = nullptr;
  explicit DeviceArray(size_t count) {
    void* raw = nullptr;
    cuda_check(cudaMalloc(&raw, std::max<size_t>(count, 1) * sizeof(T)), "cudaMalloc");
    ptr = static_cast<T*>(raw);
  }
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
  ~DeviceArray() { cudaFree(ptr); }
  void upload(const std::vector<T>& host) {
    if (host.empty()) return;
    cuda_check(cudaMemcpy(ptr, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice),
               "upload");
  }
  void download(std::vector<T>& host) const {
    if (host.empty()) return;
    cuda_check(cudaMemcpy(host.data(), ptr, host.size() * sizeof(T), cudaMemcpyDeviceToHost),
               "download");
  }
};

__global__ void qbp_kernel(QbpGraph g, const float* prior, const uint8_t* syndromes, float alpha,
                           int32_t max_iters, float* v2c, float* c2v, uint8_t* hard,
                           int32_t* iters, int32_t batch) {
  const int64_t s = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (s >= batch) return;
  iters[s] = qbp_min_sum(g, prior, syndromes + s * g.m, alpha, max_iters, v2c + s * g.E,
                         c2v + s * g.E, hard + s * g.n);
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <problem.bin> <result.bin> [repeats]\n", argv[0]);
    return 2;
  }
  try {
    const qbp_io::Problem p = qbp_io::read_problem(argv[1]);
    const int repeats = argc > 3 ? std::max(1, std::atoi(argv[3])) : 1;
    const size_t batch = static_cast<size_t>(p.batch);

    // Allocation comes first: it creates the CUDA context, which total_seconds leaves out.
    DeviceArray<int32_t> d_chk_ptr(p.chk_ptr.size());
    DeviceArray<int32_t> d_edge_var(p.edge_var.size());
    DeviceArray<int32_t> d_var_ptr(p.var_ptr.size());
    DeviceArray<int32_t> d_var_edges(p.var_edges.size());
    DeviceArray<float> d_prior(p.prior.size());
    DeviceArray<uint8_t> d_syn(p.syndromes.size());
    DeviceArray<uint8_t> d_hard(batch * p.n);
    DeviceArray<float> d_v2c(batch * p.E);
    DeviceArray<float> d_c2v(batch * p.E);
    DeviceArray<int32_t> d_iters(batch);

    const auto t_start = std::chrono::steady_clock::now();
    d_chk_ptr.upload(p.chk_ptr);
    d_edge_var.upload(p.edge_var);
    d_var_ptr.upload(p.var_ptr);
    d_var_edges.upload(p.var_edges);
    d_prior.upload(p.prior);
    d_syn.upload(p.syndromes);

    const QbpGraph g{p.n, p.m, p.E, d_chk_ptr.ptr, d_edge_var.ptr, d_var_ptr.ptr, d_var_edges.ptr};
    const int threads = 128;
    const int blocks = static_cast<int>((batch + threads - 1) / threads);

    // The first launch also loads the kernel, so it is timed apart from the warmed ones.
    double first = 0.0;
    double best = std::numeric_limits<double>::infinity();
    for (int r = 0; r <= repeats; ++r) {
      const auto t0 = std::chrono::steady_clock::now();
      if (blocks > 0) {
        qbp_kernel<<<blocks, threads>>>(g, d_prior.ptr, d_syn.ptr, p.alpha, p.max_iters, d_v2c.ptr,
                                        d_c2v.ptr, d_hard.ptr, d_iters.ptr, p.batch);
      }
      cuda_check(cudaGetLastError(), "kernel launch");
      cuda_check(cudaDeviceSynchronize(), "kernel run");
      const double dt = seconds_since(t0);
      if (r == 0) {
        first = dt;
      } else {
        best = std::min(best, dt);
      }
    }

    std::vector<uint8_t> hard(batch * p.n);
    std::vector<int32_t> iters(batch);
    d_hard.download(hard);
    d_iters.download(iters);
    const double total = seconds_since(t_start);
    qbp_io::write_result(argv[2], p, hard, iters);

    int device = 0;
    cuda_check(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp prop{};
    cuda_check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    // decode_seconds is the best warmed launch; total_seconds spans uploads, launches, downloads.
    std::printf("{\"backend\": \"cuda\", \"device\": \"%s\", \"batch\": %d, "
                "\"decode_seconds\": %.9f, \"first_launch_seconds\": %.9f, "
                "\"total_seconds\": %.9f}\n",
                prop.name, p.batch, best, first, total);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
