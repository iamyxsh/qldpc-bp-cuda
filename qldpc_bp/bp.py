"""Reference belief-propagation decoders, batched over shots.

This file is the specification the C++ / CUDA decoder in ``cpp/`` must match bit for bit:
float32 throughout, the same order of additions, the same sign and tie rules.

Decoding problem: given a parity-check matrix H and a syndrome s, find an error e with
H e = s (mod 2). Messages are log-likelihood ratios (LLRs); positive means "no error".
"""

import numpy as np


class Tanner:
    """Edge lists of the Tanner graph of H.

    Edges are numbered check by check, variables ascending inside a check, so the edges of
    check c are the contiguous range ``chk_ptr[c] : chk_ptr[c + 1]``. ``var_edges`` lists
    edge ids grouped by variable, checks ascending inside a variable.
    """

    def __init__(self, h):
        h = np.array(h, dtype=np.uint8) & 1
        self.h = h
        self.m, self.n = h.shape
        chk, var = np.nonzero(h)
        self.E = int(chk.size)
        self.edge_var = var.astype(np.int32)
        self.chk_ptr = np.concatenate([[0], np.cumsum(h.sum(axis=1))]).astype(np.int32)
        self.var_edges = np.argsort(var, kind="stable").astype(np.int32)
        self.var_ptr = np.concatenate([[0], np.cumsum(h.sum(axis=0))]).astype(np.int32)
        if (np.diff(self.chk_ptr) < 2).any():
            raise ValueError("every check must touch at least two qubits")

    def var_edge_list(self, v):
        return self.var_edges[self.var_ptr[v] : self.var_ptr[v + 1]]

    def syndrome(self, errors):
        errors = np.asarray(errors, dtype=np.int64)
        return ((errors @ self.h.T.astype(np.int64)) % 2).astype(np.uint8)


def prior_llr(p, n):
    """Per-qubit prior LLR for an independent flip probability ``p`` (scalar or length n)."""
    p = np.broadcast_to(np.asarray(p, dtype=np.float64), (n,))
    return (np.log1p(-p) - np.log(p)).astype(np.float32)


def _var_to_check(t, prior, c2v):
    v2c = np.empty_like(c2v)
    for v in range(t.n):
        edges = t.var_edge_list(v)
        for e in edges:
            acc = np.full(c2v.shape[0], prior[v], dtype=np.float32)
            for e2 in edges:
                if e2 != e:
                    acc = acc + c2v[:, e2]
            v2c[:, e] = acc
    return v2c


def _check_to_var_min_sum(t, syn, v2c, alpha):
    c2v = np.empty_like(v2c)
    alpha = np.float32(alpha)
    for c in range(t.m):
        lo, hi = t.chk_ptr[c], t.chk_ptr[c + 1]
        for e in range(lo, hi):
            mag = None
            neg = syn[:, c].copy()
            for e2 in range(lo, hi):
                if e2 == e:
                    continue
                x = v2c[:, e2]
                ax = np.abs(x)
                mag = ax if mag is None else np.minimum(mag, ax)
                neg ^= x < 0
            val = alpha * mag
            c2v[:, e] = np.where(neg, -val, val)
    return c2v


def _tanh_half(x):
    exp_abs = np.exp(-np.abs(x))
    return np.where(x >= 0, (1 - exp_abs) / (1 + exp_abs), (exp_abs - 1) / (exp_abs + 1)).astype(
        np.float32
    )


def _check_to_var_sum_product(t, syn, v2c):
    c2v = np.empty_like(v2c)
    th = _tanh_half(v2c)
    limit = np.float32(1.0 - 1e-6)
    for c in range(t.m):
        lo, hi = t.chk_ptr[c], t.chk_ptr[c + 1]
        sign = np.where(syn[:, c], np.float32(-1.0), np.float32(1.0))
        for e in range(lo, hi):
            prod = np.ones(v2c.shape[0], dtype=np.float32)
            for e2 in range(lo, hi):
                if e2 != e:
                    prod = prod * th[:, e2]
            prod = np.clip(prod, -limit, limit)
            c2v[:, e] = sign * np.log((1 + prod) / (1 - prod))
    return c2v


def _posterior(t, prior, c2v):
    post = np.empty((c2v.shape[0], t.n), dtype=np.float32)
    for v in range(t.n):
        acc = np.full(c2v.shape[0], prior[v], dtype=np.float32)
        for e in t.var_edge_list(v):
            acc = acc + c2v[:, e]
        post[:, v] = acc
    return post


def decode(t, syndromes, prior, *, method="min_sum", alpha=0.75, max_iters=50, early_stop=True):
    """Flooding-schedule BP on a batch of syndromes.

    Returns ``(hard, iters, posterior)``:
      hard      (B, n) uint8, the error estimate (bit set where the posterior LLR is negative)
      iters     (B,) int32, the 1-based iteration at which the estimate first reproduced the
                syndrome, or -1 if it never did
      posterior (B, n) float32 posterior LLRs behind ``hard``

    With ``early_stop`` a shot is frozen at the first iteration whose estimate reproduces its
    syndrome. Without it every shot runs ``max_iters`` iterations and ``iters`` is ``max_iters``
    when the final estimate reproduces the syndrome.
    """
    if method not in ("min_sum", "sum_product"):
        raise ValueError(f"unknown method {method!r}")
    syndromes = np.atleast_2d(np.asarray(syndromes, dtype=np.uint8))
    prior = np.asarray(prior, dtype=np.float32)
    batch = syndromes.shape[0]
    out_hard = np.zeros((batch, t.n), dtype=np.uint8)
    out_iters = np.full(batch, -1, dtype=np.int32)
    out_post = np.tile(prior, (batch, 1))

    idx = np.arange(batch)
    syn = syndromes.astype(bool)
    c2v = np.zeros((batch, t.E), dtype=np.float32)
    h = t.h.astype(np.int64)

    for it in range(1, max_iters + 1):
        v2c = _var_to_check(t, prior, c2v)
        if method == "min_sum":
            c2v = _check_to_var_min_sum(t, syn, v2c, alpha)
        else:
            c2v = _check_to_var_sum_product(t, syn, v2c)
        post = _posterior(t, prior, c2v)
        hard = (post < 0).astype(np.uint8)
        matched = (((hard.astype(np.int64) @ h.T) % 2).astype(bool) == syn).all(axis=1)

        if it == max_iters:
            done = np.ones(idx.size, dtype=bool)
        elif early_stop:
            done = matched
        else:
            continue
        if done.any():
            sel = idx[done]
            out_hard[sel] = hard[done]
            out_post[sel] = post[done]
            out_iters[sel] = np.where(matched[done], it, -1)
            keep = ~done
            idx, syn, c2v = idx[keep], syn[keep], c2v[keep]
        if idx.size == 0:
            break
    return out_hard, out_iters, out_post


def backline_like(t, syndromes, p, *, num_iters=10, postprocess="osd"):
    """The decoder semantics of ``qp.backline.css_bp_decoder``, as read from its source.

    Sum-product BP with a uniform prior and a fixed iteration count (no early stop), then
    one of two post-processing rules:
      "hard": flip every qubit whose posterior LLR is negative
      "osd":  for a non-zero syndrome flip exactly one qubit, the one with the lowest
              posterior LLR; for a zero syndrome flip nothing
    """
    syndromes = np.atleast_2d(np.asarray(syndromes, dtype=np.uint8))
    hard, _, post = decode(
        t, syndromes, prior_llr(p, t.n), method="sum_product", max_iters=num_iters, early_stop=False
    )
    if postprocess == "hard":
        return hard
    if postprocess != "osd":
        raise ValueError(f"unknown postprocess {postprocess!r}")
    corr = np.zeros_like(hard)
    nonzero = syndromes.any(axis=1)
    best = np.argmin(post, axis=1)
    corr[nonzero, best[nonzero]] = 1
    return corr
