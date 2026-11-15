CXX ?= c++
NVCC ?= nvcc
PYTHON ?= python3
BUILD := build

# No fused multiply-add on either side: the outputs are compared bit for bit with the reference.
CXXFLAGS := -std=c++17 -O2 -ffp-contract=off -Wall -Wextra
NVCCFLAGS := -std=c++17 -O2 --fmad=false

.PHONY: cpu cuda test clean

cpu: $(BUILD)/bp_cpu
cuda: $(BUILD)/bp_cuda

$(BUILD)/bp_cpu: cpp/bp_cpu.cpp cpp/bp_core.h cpp/problem_io.h
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ cpp/bp_cpu.cpp

$(BUILD)/bp_cuda: cpp/bp_cuda.cu cpp/bp_core.h cpp/problem_io.h
	@mkdir -p $(BUILD)
	$(NVCC) $(NVCCFLAGS) -o $@ cpp/bp_cuda.cu

test: cpu
	$(PYTHON) -m pytest -q

clean:
	rm -rf $(BUILD)
