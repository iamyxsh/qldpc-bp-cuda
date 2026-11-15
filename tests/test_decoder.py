import itertools

import numpy as np
import pytest

from qldpc_bp import bp, codes, evaluate, gf2, native


def test_gf2_rank_and_nullspace():
    mat = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 1, 0]], dtype=np.uint8)
    assert gf2.rank(mat) == 2
    null = gf2.nullspace(mat)
    assert null.shape == (2, 4)
    assert not ((mat.astype(int) @ null.T) % 2).any()
    assert gf2.rank(null) == 2


@pytest.mark.parametrize("name,n,k", [("hgp13", 13, 1), ("bb72", 72, 12)])
def test_code_parameters(name, n, k):
    code = codes.CODES[name]()
    assert (code.n, code.k) == (n, k)
    assert n - gf2.rank(code.hx) - gf2.rank(code.hz) == k
    assert not ((code.hx.astype(int) @ code.hz.T) % 2).any()
    # Logical Z operators commute with every X check and are not products of Z checks.
    assert not ((code.hx.astype(int) @ code.lz.T) % 2).any()
    assert gf2.rank(np.vstack([code.hz, code.lz])) == gf2.rank(code.hz) + k
    # Each logical Z anticommutes with at least one logical X.
    assert gf2.rank((code.lx.astype(int) @ code.lz.T) % 2) == k


def test_hgp13_distance_is_3():
    code = codes.hgp13()
    words = np.array(list(itertools.product([0, 1], repeat=13)), dtype=np.int64)
    weight = words.sum(axis=1)
    # A logical operator is missed by every check of the other type and flips the stored qubit.
    for checks, partner in [(code.hz, code.lz), (code.hx, code.lx)]:
        undetected = ~((words @ checks.T) % 2).any(axis=1)
        flips = ((words @ partner.T) % 2).any(axis=1)
        assert weight[undetected & flips].min() == 3


def test_demo_operator_is_a_logical_z_of_hgp13():
    code = codes.hgp13()
    demo_lz = np.zeros(13, dtype=np.uint8)
    demo_lz[[2, 5, 8]] = 1
    assert not ((code.hx.astype(int) @ demo_lz) % 2).any()
    assert gf2.rank(np.vstack([code.hz, demo_lz])) == gf2.rank(code.hz) + 1


def test_judge_separates_the_three_outcomes():
    code = codes.hgp13()
    zero = np.zeros((1, 13), dtype=np.uint8)
    stabilizer = code.hx[:1]  # an X-type stabilizer acts trivially on the code
    logical_x = code.lx[:1]
    single = np.eye(13, dtype=np.uint8)[:1]
    for residual, expected in [(stabilizer, (False, False)), (logical_x, (False, True)),
                               (single, (True, False))]:
        unmatched, logical = evaluate.judge(code.hz, code.lz, residual, zero)
        assert (bool(unmatched[0]), bool(logical[0])) == expected


@pytest.mark.parametrize("method", ["min_sum", "sum_product"])
def test_single_errors_are_corrected_on_hgp13(method):
    code = codes.hgp13()
    t = bp.Tanner(code.hz)
    errors = np.eye(13, dtype=np.uint8)
    hard, iters, _ = bp.decode(t, t.syndrome(errors), bp.prior_llr(0.05, 13), method=method)
    unmatched, logical = evaluate.judge(code.hz, code.lz, errors, hard)
    assert (iters > 0).all()
    assert not unmatched.any() and not logical.any()


def test_zero_syndrome_gives_no_correction():
    t = bp.Tanner(codes.bb72().hz)
    hard, iters, _ = bp.decode(t, np.zeros((3, t.m), dtype=np.uint8), bp.prior_llr(0.02, t.n))
    assert not hard.any()
    assert (iters == 1).all()


def test_a_frozen_shot_does_not_depend_on_the_rest_of_the_batch():
    t = bp.Tanner(codes.bb72().hz)
    rng = np.random.default_rng(3)
    syndromes = t.syndrome((rng.random((64, t.n)) < 0.06).astype(np.uint8))
    prior = bp.prior_llr(0.06, t.n)
    together = bp.decode(t, syndromes, prior, max_iters=20)
    for i in (0, 17, 63):
        alone = bp.decode(t, syndromes[i : i + 1], prior, max_iters=20)
        assert np.array_equal(alone[0][0], together[0][i])
        assert alone[1][0] == together[1][i]


def test_backline_like_failures_on_two_errors():
    code = codes.hgp13()
    t = bp.Tanner(code.hz)
    pairs = np.zeros((78, 13), dtype=np.uint8)
    for row, (a, b) in enumerate(itertools.combinations(range(13), 2)):
        pairs[row, [a, b]] = 1
    failures = {}
    for postprocess in ("osd", "hard"):
        corr = bp.backline_like(t, t.syndrome(pairs), 0.05, postprocess=postprocess)
        unmatched, logical = evaluate.judge(code.hz, code.lz, pairs, corr)
        failures[postprocess] = int((unmatched | logical).sum())
        if postprocess == "osd":
            assert (corr.sum(axis=1) == 1).all()
    assert failures == {"osd": 66, "hard": 49}
    zero = bp.backline_like(t, np.zeros((1, 6), dtype=np.uint8), 0.05, postprocess="osd")
    assert not zero.any()


def test_early_stop_fails_fewer_two_error_patterns_than_fixed_iterations():
    code = codes.hgp13()
    t = bp.Tanner(code.hz)
    pairs = np.zeros((78, 13), dtype=np.uint8)
    for row, (a, b) in enumerate(itertools.combinations(range(13), 2)):
        pairs[row, [a, b]] = 1
    failures = []
    for early_stop in (False, True):
        hard = bp.decode(t, t.syndrome(pairs), bp.prior_llr(0.05, 13), max_iters=10,
                         early_stop=early_stop)[0]
        unmatched, logical = evaluate.judge(code.hz, code.lz, pairs, hard)
        failures.append(int((unmatched | logical).sum()))
    assert failures == [49, 41]


def test_agrees_with_ldpc_min_sum():
    code = codes.bb72()
    t = bp.Tanner(code.hz)
    rng = np.random.default_rng(11)
    errors = (rng.random((2000, 72)) < 0.03).astype(np.uint8)
    syndromes = t.syndrome(errors)
    hard, iters, _ = bp.decode(t, syndromes, bp.prior_llr(0.03, 72), alpha=0.75, max_iters=30)
    corr, converged = evaluate.ldpc_decode(
        code.hz, syndromes, 0.03, bp_method="minimum_sum", alpha=0.75, max_iters=30, osd=False
    )
    # Two independent implementations: the same shots converge, to the same corrections, up to
    # rare float ties.
    assert np.mean((iters > 0) != converged) < 0.005
    both = (iters > 0) & converged
    assert np.mean((hard[both] != corr[both]).any(axis=1)) < 0.005


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
@pytest.mark.parametrize("name", ["hgp13", "bb72"])
def test_native_matches_reference_bit_for_bit(backend, name):
    if not native.BINARIES[backend].exists():
        pytest.skip(f"build/bp_{backend} is not built")
    code = codes.CODES[name]()
    t = bp.Tanner(code.hz)
    rng = np.random.default_rng(5)
    # The last case gives every qubit its own error rate.
    for p, alpha, spread in [(0.01, 0.75, 0), (0.05, 0.625, 0), (0.12, 1.0, 0), (0.05, 0.9, 0.2)]:
        syndromes = t.syndrome((rng.random((4000, code.n)) < p).astype(np.uint8))
        prior = bp.prior_llr(p * (1 + spread * rng.uniform(-1, 1, code.n)), code.n)
        ref_hard, ref_iters, _ = bp.decode(t, syndromes, prior, alpha=alpha, max_iters=30)
        hard, iters, _ = native.run(backend, t, syndromes, prior, alpha=alpha, max_iters=30)
        assert np.array_equal(hard, ref_hard)
        assert np.array_equal(iters, ref_iters)
