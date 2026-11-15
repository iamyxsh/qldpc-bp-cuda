# qldpc-bp-cuda

Normalized min-sum belief propagation (BP) decoder for quantum LDPC codes. One C++ function,
built for the CPU and as a CUDA kernel, held bit for bit to a NumPy reference.

The Backline paper ([arXiv 2609.09270](https://arxiv.org/abs/2609.09270), Sec. V-C2) calls its
BP-OSD decoder a naive implementation and names hardware-friendly normalized min-sum as a
possible optimization. This repo takes that step on two small codes and measures what changes.

## Status

| Part | State |
| --- | --- |
| [[13,1,3]] and [[72,12,6]] codes, NumPy reference decoders | done, tested |
| C++ core, CPU build | bit-identical to the reference on every shot compared (32,000 in the tests, 110,101 in the timing run) |
| CUDA driver | compiles with nvcc 12.4, **not run on a GPU yet** |
| GPU numbers, circuit-level noise, running inside Backline | not done |

## Results

Code-capacity noise: independent X flips with probability p, perfect syndrome measurement,
X sector only. These are correctness checks on small codes, not circuit-level results.

### [[13,1,3]], every error pattern

All 8,192 patterns weighted by probability, at most 10 iterations. No sampling, so the failure
probabilities are exact.

| Decoder | p = 0.01 | p = 0.05 | p = 0.10 | Weight-2 patterns that fail (of 78) |
| --- | --- | --- | --- | --- |
| Sum-product, 10 iterations, `postprocess="osd"` | 0.00616 | 0.1166 | 0.3309 | 66 |
| Sum-product, 10 iterations, `postprocess="hard"` | 0.00461 | 0.0902 | 0.2645 | 49 |
| Min-sum, 10 iterations, scaling 0.625 | 0.00534 | 0.1027 | 0.2959 | 57 |
| Min-sum, 10 iterations, scaling 0.75, 0.9 or 1.0 | 0.00461 | 0.0902 | 0.2645 | 49 |
| Min-sum, stop at first match, any of the four scalings | 0.00387 | 0.0765 | 0.2269 | 41 |
| Min-sum 0.75, then OSD-0 (`ldpc`) | 0.00239 | 0.0491 | 0.1518 | 25 |
| Sum-product, then OSD-0 (`ldpc`) | 0.00238 | 0.0485 | 0.1490 | 25 |

The first two rows are a NumPy reproduction of `qp.backline.css_bp_decoder`, written from the
PennyLane source as read on 2026-11-15. Backline itself was not run.

- Over the same 10 iterations, min-sum with scaling 0.75 or more has exactly the failure
  probability of sum-product.
- Stopping at the first estimate that reproduces the syndrome beats always running 10
  iterations: 0.0765 against 0.0902 at p = 0.05.
- `postprocess="osd"` flips one qubit, the one with the lowest posterior, for any non-zero
  syndrome. With two errors that does worse than the hard decision.
- OSD-0 after either BP gives the lowest failure probability, about 0.049 at p = 0.05.

### [[72,12,6]], sampled

100,000 shots per point, at most 50 iterations, stopping at the first match. Block failure rate:
syndrome not reproduced, or any of the 12 logical qubits flipped. "Jittered" scales each qubit's
p by a fixed factor within 0.1% of 1.

| p | Scaling 0.625 | 0.75 | 0.9 | 1.0 | 0.9, jittered | 1.0, jittered | 0.75, then OSD-0 (`ldpc`) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.01 | 0.00195 | 0.00188 | 0.00171 | 0.00498 | 0.00146 | 0.00143 | 0.00169 |
| 0.02 | 0.01501 | 0.01416 | 0.01302 | 0.04622 | 0.01217 | 0.01241 | 0.01215 |
| 0.04 | 0.1109 | 0.1031 | 0.0966 | 0.3098 | 0.0946 | 0.0986 | 0.0933 |
| 0.06 | 0.3019 | 0.2837 | 0.2740 | 0.6030 | 0.2719 | 0.2821 | 0.2622 |

- The penalty at scaling 1.0 is a tie artifact, not overconfidence. With a uniform prior every
  message is a whole multiple of the prior, posteriors land on exactly 0.0, and the strict
  `< 0` rule leaves those qubits unflipped. At p = 0.02, 3,655 of the 4,289 shots that never
  reproduce the syndrome end on an exactly-zero posterior.
- With the ties broken, scaling 1.0 and 0.9 are within a few percent of each other.
- OSD-0 on top of scaling 0.75 lowers the failure rate by 8 to 14%.

Counts and 95% intervals for every cell are in `results/monte_carlo.json`.

### Throughput, CPU only

[[72,12,6]], p = 0.02, scaling 0.75, one thread on an arm64 Mac, best of 5 passes.

| Decoder | Batch | Shots per second |
| --- | --- | --- |
| This repo, C++ core | 100 | 542,000 |
| This repo, C++ core | 100,000 | 412,000 |
| `ldpc` BP, one Python call per shot | 100,000 | 187,000 |

This is throughput, not the latency of one shot.

## Run

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
make cpu
.venv/bin/python -m pytest -q
.venv/bin/python -m qldpc_bp.study
```

With an NVIDIA GPU and the CUDA toolkit, `make cuda` builds the GPU driver. The same tests then
compare it with the reference bit for bit, and `python -m qldpc_bp.study --only throughput`
times it.

## How it works

- `qbp_min_sum` in `cpp/bp_core.h` decodes one syndrome: flooding schedule, one pass per check
  that keeps the two smallest magnitudes, a flip only on a strictly negative posterior, and a
  stop at the first estimate that reproduces the syndrome.
- The CPU driver calls it in a loop. The CUDA driver gives each shot one GPU thread and runs the
  whole BP loop inside a single kernel launch.
- `qldpc_bp/bp.py` is the specification: float32, additions in edge order. Both builds turn off
  fused multiply-add, so outputs are compared for equality and not within a tolerance.
- The prior is a per-qubit array, so the same kernel can take a circuit-level check matrix.

## Limits

- [[13,1,3]] is the distance-3 unrotated surface code, a fixture for exact checks.
  [[72,12,6]] is small too.
- Flooding schedule only.
- OSD runs on the CPU through `ldpc`, restarted from the syndrome. It does not reuse this
  decoder's posteriors.

## Next

1. Run the CUDA driver: shots per second against batch size, and the batch size where the GPU
   overtakes the CPU. The per-shot scratch is shot-major today; measure an edge-major layout,
   where a warp's reads coalesce.
2. Circuit-level noise from a Stim detector error model.
3. A GPU post-processing step for shots that do not converge.

## References

- Backline: [arXiv 2609.09270](https://arxiv.org/abs/2609.09270),
  [PennyLaneAI/backline](https://github.com/PennyLaneAI/backline)
- Bivariate bicycle codes: Bravyi et al., "High-threshold and low-overhead fault-tolerant
  quantum memory", Nature 2024
- BP+OSD and `ldpc`: Roffe et al., "Decoding across the quantum LDPC code landscape",
  Phys. Rev. Research 2020
