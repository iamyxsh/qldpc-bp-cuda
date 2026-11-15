"""Problem and result files for the C++ / CUDA decoders, and a runner for their binaries.

Problem file (little endian):
    "QBP1"
    int32   n, m, E, batch, max_iters
    float32 alpha
    int32   chk_ptr[m + 1], edge_var[E], var_ptr[n + 1], var_edges[E]
    float32 prior[n]
    uint8   syndromes[batch * m]

Result file:
    "QBO1"
    int32   batch, n
    uint8   hard[batch * n]
    int32   iters[batch]      (-1: the estimate never reproduced the syndrome)
"""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
BINARIES = {"cpu": REPO / "build" / "bp_cpu", "cuda": REPO / "build" / "bp_cuda"}


def write_problem(path, tanner, syndromes, prior, *, alpha, max_iters):
    syndromes = np.atleast_2d(np.asarray(syndromes, dtype=np.uint8))
    if syndromes.shape[1] != tanner.m:
        raise ValueError("syndrome length does not match the number of checks")
    with open(path, "wb") as f:
        f.write(b"QBP1")
        header = [tanner.n, tanner.m, tanner.E, syndromes.shape[0], max_iters]
        f.write(np.array(header, dtype="<i4").tobytes())
        f.write(np.array([alpha], dtype="<f4").tobytes())
        for arr in (tanner.chk_ptr, tanner.edge_var, tanner.var_ptr, tanner.var_edges):
            f.write(np.asarray(arr, dtype="<i4").tobytes())
        f.write(np.asarray(prior, dtype="<f4").tobytes())
        f.write(np.ascontiguousarray(syndromes).tobytes())


def read_result(path):
    data = Path(path).read_bytes()
    if data[:4] != b"QBO1":
        raise ValueError("not a QBO1 result file")
    batch, n = np.frombuffer(data, dtype="<i4", count=2, offset=4)
    hard = np.frombuffer(data, dtype=np.uint8, count=batch * n, offset=12).reshape(batch, n)
    iters = np.frombuffer(data, dtype="<i4", count=batch, offset=12 + batch * n)
    return hard.copy(), iters.astype(np.int32)


def run(backend, tanner, syndromes, prior, *, alpha, max_iters, repeats=1):
    """Decode with a compiled driver. Returns ``(hard, iters, timing)``."""
    binary = BINARIES[backend]
    if not binary.exists():
        raise FileNotFoundError(f"{binary} is not built; run `make {backend}`")
    with tempfile.TemporaryDirectory() as tmp:
        problem, result = Path(tmp) / "problem.bin", Path(tmp) / "result.bin"
        write_problem(problem, tanner, syndromes, prior, alpha=alpha, max_iters=max_iters)
        proc = subprocess.run(
            [str(binary), str(problem), str(result), str(repeats)],
            check=True,
            capture_output=True,
            text=True,
        )
        hard, iters = read_result(result)
    return hard, iters, json.loads(proc.stdout.strip().splitlines()[-1])
