"""Linear algebra over GF(2) for small parity-check matrices."""

import numpy as np


def row_echelon(mat):
    """Reduced row echelon form of ``mat`` over GF(2), and its pivot columns."""
    a = (np.array(mat, dtype=np.uint8) & 1).copy()
    if a.ndim != 2:
        raise ValueError("expected a 2D matrix")
    rows, cols = a.shape
    pivots = []
    r = 0
    for c in range(cols):
        if r == rows:
            break
        hits = np.nonzero(a[r:, c])[0]
        if hits.size == 0:
            continue
        p = r + hits[0]
        if p != r:
            a[[r, p]] = a[[p, r]]
        others = np.nonzero(a[:, c])[0]
        others = others[others != r]
        a[others] ^= a[r]
        pivots.append(c)
        r += 1
    return a, pivots


def rank(mat):
    return len(row_echelon(mat)[1])


def nullspace(mat):
    """Basis of ``{x : mat @ x = 0 (mod 2)}``, one vector per row."""
    a, pivots = row_echelon(mat)
    cols = a.shape[1]
    pivot_set = set(pivots)
    free = [c for c in range(cols) if c not in pivot_set]
    basis = np.zeros((len(free), cols), dtype=np.uint8)
    for i, f in enumerate(free):
        basis[i, f] = 1
        for r, pc in enumerate(pivots):
            if a[r, f]:
                basis[i, pc] = 1
    return basis


def logical_basis(h_commute, h_stabilizer):
    """Operators that commute with ``h_commute`` and are not products of ``h_stabilizer`` rows.

    Returns a basis of ``ker(h_commute)`` modulo ``rowspace(h_stabilizer)``, one vector per row.
    For a CSS code, ``logical_basis(Hx, Hz)`` is a basis of the logical Z operators.
    """
    stack = (np.array(h_stabilizer, dtype=np.uint8) & 1).copy()
    r = rank(stack)
    chosen = []
    for v in nullspace(h_commute):
        candidate = np.vstack([stack, v])
        if rank(candidate) > r:
            stack = candidate
            r += 1
            chosen.append(v)
    cols = np.asarray(h_commute).shape[1]
    return np.array(chosen, dtype=np.uint8).reshape(len(chosen), cols)
