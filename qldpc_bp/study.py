"""The measurements reported in the README. Run with ``python -m qldpc_bp.study``.

Noise model throughout: independent X flips on data qubits with probability p, perfect
syndrome measurement (code capacity). Only the X-error sector is decoded, with the Z checks.
"""

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

from . import bp, codes, evaluate, native

RESULTS = Path(__file__).resolve().parent.parent / "results"
ALPHAS = (0.625, 0.75, 0.9, 1.0)


def exhaustive_hgp13(ps=(0.01, 0.05, 0.10), alphas=ALPHAS, max_iters=10):
    """Every one of the 2^13 X-error patterns on the [[13,1,3]] code, weighted by its probability.

    No sampling: the failure probability of each decoder at each p is exact.
    """
    code = codes.hgp13()
    t = bp.Tanner(code.hz)
    errors = np.array(list(itertools.product([0, 1], repeat=code.n)), dtype=np.uint8)
    weights = errors.sum(axis=1)
    syndromes = t.syndrome(errors)
    rows = []
    for p in ps:
        prob = p**weights * (1 - p) ** (code.n - weights)
        prior = bp.prior_llr(p, code.n)
        decoders = {
            f'sum-product, postprocess="{post}"': bp.backline_like(
                t, syndromes, p, num_iters=max_iters, postprocess=post
            )
            for post in ("osd", "hard")
        }
        for alpha in alphas:
            for early_stop, rule in ((False, "fixed iterations"), (True, "stop at first match")):
                decoders[f"min-sum a={alpha}, {rule}"] = bp.decode(
                    t, syndromes, prior, alpha=alpha, max_iters=max_iters, early_stop=early_stop
                )[0]
        for method, label in (("minimum_sum", "min-sum a=0.75"), ("product_sum", "sum-product")):
            decoders[f"{label} + OSD-0 (ldpc)"] = evaluate.ldpc_decode(
                code.hz, syndromes, p, bp_method=method, alpha=0.75, max_iters=max_iters, osd=True
            )[0]
        for label, corr in decoders.items():
            unmatched, logical = evaluate.judge(code.hz, code.lz, errors, corr)
            failed = unmatched | logical
            rows.append(
                {
                    "p": p,
                    "decoder": label,
                    "p_unmatched": float(prob[unmatched].sum()),
                    "p_logical": float(prob[logical].sum()),
                    "p_fail": float(prob[failed].sum()),
                    "failing_weight1": int((failed & (weights == 1)).sum()),
                    "failing_weight2": int((failed & (weights == 2)).sum()),
                }
            )
    return rows


def monte_carlo_bb72(ps=(0.01, 0.02, 0.04, 0.06), alphas=ALPHAS, shots=100_000, max_iters=50,
                     seed=2026, jitter=1e-3):
    """Sampled errors on the [[72,12,6]] code: failure rates of min-sum BP.

    The jittered rows scale each qubit's p by a fixed factor within ``jitter`` of 1. That breaks
    the exactly-zero posteriors a uniform prior produces at scaling 1.0.
    """
    code = codes.bb72()
    t = bp.Tanner(code.hz)
    rng = np.random.default_rng(seed)
    spread = 1 + jitter * np.random.default_rng(seed + 1).uniform(-1, 1, code.n)
    rows = []
    for p in ps:
        errors = (rng.random((shots, code.n)) < p).astype(np.uint8)
        syndromes = t.syndrome(errors)
        uniform, jittered = bp.prior_llr(p, code.n), bp.prior_llr(p * spread, code.n)
        runs = [(alpha, "", uniform) for alpha in alphas]
        runs += [(alpha, ", prior jittered", jittered) for alpha in (0.9, 1.0)]
        for alpha, tag, prior in runs:
            hard, iters, post = bp.decode(t, syndromes, prior, alpha=alpha, max_iters=max_iters)
            rows.append(_row(code, errors, hard, p, f"min-sum a={alpha}{tag}", iters, post))
        corr, _ = evaluate.ldpc_decode(
            code.hz, syndromes, p, bp_method="minimum_sum", alpha=0.75, max_iters=max_iters,
            osd=True,
        )
        rows.append(_row(code, errors, corr, p, "min-sum a=0.75 + OSD-0 (ldpc)"))
    return rows


def _row(code, errors, corrections, p, label, iters=None, post=None):
    unmatched, logical = evaluate.judge(code.hz, code.lz, errors, corrections)
    shots = errors.shape[0]
    failures = int((unmatched | logical).sum())
    row = {
        "p": p,
        "decoder": label,
        "shots": shots,
        "unmatched": int(unmatched.sum()),
        "logical": int(logical.sum()),
        "failures": failures,
        "fail_rate": failures / shots,
        "fail_rate_95": list(evaluate.wilson(failures, shots)),
    }
    if iters is not None:
        row["mean_iters_converged"] = float(iters[iters > 0].mean()) if (iters > 0).any() else None
        row["unmatched_with_zero_posterior"] = int((unmatched & (post == 0).any(axis=1)).sum())
    return row


def throughput_bb72(p=0.02, alpha=0.75, max_iters=50, batches=(1, 100, 10_000, 100_000), seed=7,
                    repeats=5):
    """Shots per second by batch size, for each backend that is built, and for `ldpc` BP.

    Every figure is the best of ``repeats`` passes. Batch time divided by batch size is a
    throughput cost. It is not the latency of one shot.
    """
    code = codes.bb72()
    t = bp.Tanner(code.hz)
    rng = np.random.default_rng(seed)
    prior = bp.prior_llr(p, code.n)
    rows = []
    for batch in batches:
        errors = (rng.random((batch, code.n)) < p).astype(np.uint8)
        syndromes = t.syndrome(errors)
        ref_hard, ref_iters, _ = bp.decode(t, syndromes, prior, alpha=alpha, max_iters=max_iters)
        for backend, binary in native.BINARIES.items():
            if not binary.exists():
                continue
            hard, iters, timing = native.run(
                backend, t, syndromes, prior, alpha=alpha, max_iters=max_iters, repeats=repeats
            )
            rows.append(
                {
                    "backend": backend,
                    "device": timing.get("device", "cpu, one thread"),
                    "batch": batch,
                    "decode_seconds": timing["decode_seconds"],
                    "shots_per_second": batch / timing["decode_seconds"],
                    "total_seconds": timing.get("total_seconds"),
                    "matches_reference": bool(
                        np.array_equal(hard, ref_hard) and np.array_equal(iters, ref_iters)
                    ),
                }
            )
        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            evaluate.ldpc_decode(code.hz, syndromes, p, bp_method="minimum_sum", alpha=alpha,
                                 max_iters=max_iters, osd=False)
            best = min(best, time.perf_counter() - t0)
        rows.append(
            {
                "backend": "ldpc",
                "device": "cpu, one thread, one Python call per shot",
                "batch": batch,
                "decode_seconds": best,
                "shots_per_second": batch / best,
                "total_seconds": None,
                "matches_reference": None,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=100_000)
    parser.add_argument("--only", choices=["exhaustive", "monte_carlo", "throughput"])
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    parts = {
        "exhaustive": exhaustive_hgp13,
        "monte_carlo": lambda: monte_carlo_bb72(shots=args.shots),
        "throughput": throughput_bb72,
    }
    for name, fn in parts.items():
        if args.only and args.only != name:
            continue
        rows = fn()
        (RESULTS / f"{name}.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(f"== {name}")
        for row in rows:
            print(json.dumps(row))


if __name__ == "__main__":
    main()
