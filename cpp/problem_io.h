// Binary problem and result files shared by the CPU and CUDA drivers.
// qldpc_bp/native.py documents the layout and writes and reads the same files.
#pragma once

#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace qbp_io {

struct Problem {
  int32_t n = 0, m = 0, E = 0, batch = 0, max_iters = 0;
  float alpha = 0.0f;
  std::vector<int32_t> chk_ptr, edge_var, var_ptr, var_edges;
  std::vector<float> prior;
  std::vector<uint8_t> syndromes;  // batch * m
};

struct CloseFile {
  void operator()(std::FILE* f) const { std::fclose(f); }
};
using File = std::unique_ptr<std::FILE, CloseFile>;

inline File open_file(const std::string& path, const char* mode) {
  File f(std::fopen(path.c_str(), mode));
  if (!f) throw std::runtime_error("cannot open " + path);
  return f;
}

template <typename T>
void read_exact(std::FILE* f, T* dst, size_t count) {
  if (count != 0 && std::fread(dst, sizeof(T), count, f) != count) {
    throw std::runtime_error("problem file is truncated");
  }
}

template <typename T>
void write_exact(std::FILE* f, const T* src, size_t count) {
  if (count != 0 && std::fwrite(src, sizeof(T), count, f) != count) {
    throw std::runtime_error("could not write result file");
  }
}

inline void check_index(const std::vector<int32_t>& values, int32_t limit, const char* what) {
  for (int32_t v : values) {
    if (v < 0 || v >= limit) throw std::runtime_error(std::string("index out of range in ") + what);
  }
}

inline void check_offsets(const std::vector<int32_t>& ptr, int32_t total, const char* what) {
  if (ptr.empty() || ptr.front() != 0 || ptr.back() != total) {
    throw std::runtime_error(std::string("bad offsets in ") + what);
  }
  for (size_t i = 1; i < ptr.size(); ++i) {
    if (ptr[i] < ptr[i - 1]) throw std::runtime_error(std::string("bad offsets in ") + what);
  }
}

inline Problem read_problem(const std::string& path) {
  const File f = open_file(path, "rb");
  char magic[4];
  read_exact(f.get(), magic, 4);
  if (std::string(magic, 4) != "QBP1") throw std::runtime_error("not a QBP1 problem file");
  int32_t header[5];
  read_exact(f.get(), header, 5);
  Problem p;
  p.n = header[0];
  p.m = header[1];
  p.E = header[2];
  p.batch = header[3];
  p.max_iters = header[4];
  read_exact(f.get(), &p.alpha, 1);
  const int32_t limit = 1 << 24;
  if (p.n <= 0 || p.m <= 0 || p.E <= 0 || p.batch < 0 || p.max_iters < 0 || p.n > limit ||
      p.m > limit || p.E > limit) {
    throw std::runtime_error("problem header out of range");
  }
  p.chk_ptr.resize(p.m + 1);
  p.edge_var.resize(p.E);
  p.var_ptr.resize(p.n + 1);
  p.var_edges.resize(p.E);
  p.prior.resize(p.n);
  p.syndromes.resize(static_cast<size_t>(p.batch) * p.m);
  read_exact(f.get(), p.chk_ptr.data(), p.chk_ptr.size());
  read_exact(f.get(), p.edge_var.data(), p.edge_var.size());
  read_exact(f.get(), p.var_ptr.data(), p.var_ptr.size());
  read_exact(f.get(), p.var_edges.data(), p.var_edges.size());
  read_exact(f.get(), p.prior.data(), p.prior.size());
  read_exact(f.get(), p.syndromes.data(), p.syndromes.size());
  // The decoder indexes with these arrays, so a bad file is refused here.
  check_offsets(p.chk_ptr, p.E, "chk_ptr");
  check_offsets(p.var_ptr, p.E, "var_ptr");
  check_index(p.edge_var, p.n, "edge_var");
  check_index(p.var_edges, p.E, "var_edges");
  return p;
}

inline void write_result(const std::string& path, const Problem& p,
                         const std::vector<uint8_t>& hard, const std::vector<int32_t>& iters) {
  File f = open_file(path, "wb");
  write_exact(f.get(), "QBO1", 4);
  const int32_t header[2] = {p.batch, p.n};
  write_exact(f.get(), header, 2);
  write_exact(f.get(), hard.data(), hard.size());
  write_exact(f.get(), iters.data(), iters.size());
  if (std::fclose(f.release()) != 0) throw std::runtime_error("could not write result file");
}

}  // namespace qbp_io
