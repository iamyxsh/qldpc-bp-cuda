"""The CSS codes this repo decodes, with their logical operators."""

from dataclasses import dataclass

import numpy as np

from . import gf2


@dataclass(frozen=True)
class CssCode:
    name: str
    hx: np.ndarray  # X-type checks, one per row; they detect Z errors
    hz: np.ndarray  # Z-type checks, one per row; they detect X errors
    lx: np.ndarray  # logical X operators, one per row
    lz: np.ndarray  # logical Z operators, one per row

    @property
    def n(self):
        return self.hx.shape[1]

    @property
    def k(self):
        return self.lz.shape[0]


def make_css(name, hx, hz):
    hx = np.array(hx, dtype=np.uint8) & 1
    hz = np.array(hz, dtype=np.uint8) & 1
    if hx.shape[1] != hz.shape[1]:
        raise ValueError("Hx and Hz act on different numbers of qubits")
    if ((hx.astype(np.int64) @ hz.T.astype(np.int64)) % 2).any():
        raise ValueError("Hx and Hz do not commute")
    return CssCode(name, hx, hz, lx=gf2.logical_basis(hz, hx), lz=gf2.logical_basis(hx, hz))


def hgp13():
    """The [[13,1,3]] hypergraph-product code, with the qubit order the Backline demos use.

    It is the distance-3 unrotated surface code: small enough to enumerate every error.
    """
    hx = [
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    ]
    hz = [
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
    ]
    return make_css("hgp13", hx, hz)


def _shift(size, power):
    return np.roll(np.eye(size, dtype=np.uint8), power % size, axis=1)


def bivariate_bicycle(name, l, m, a_terms, b_terms):
    """Bivariate bicycle code: Hx = [A | B], Hz = [B^T | A^T].

    A and B are sums of monomials in x = S_l (x) I_m and y = I_l (x) S_m, where S is a cyclic
    shift. Each term is ``("x", power)`` or ``("y", power)``.
    """

    def monomial(var, power):
        if var == "x":
            return np.kron(_shift(l, power), np.eye(m, dtype=np.uint8))
        if var == "y":
            return np.kron(np.eye(l, dtype=np.uint8), _shift(m, power))
        raise ValueError(f"unknown variable {var!r}")

    def poly(terms):
        total = np.zeros((l * m, l * m), dtype=np.uint8)
        for var, power in terms:
            total ^= monomial(var, power)
        return total

    a, b = poly(a_terms), poly(b_terms)
    return make_css(name, np.hstack([a, b]), np.hstack([b.T, a.T]))


def bb72():
    """The [[72,12,6]] bivariate bicycle code: A = x^3 + y + y^2, B = y^3 + x + x^2, l = m = 6."""
    return bivariate_bicycle(
        "bb72", 6, 6, [("x", 3), ("y", 1), ("y", 2)], [("y", 3), ("x", 1), ("x", 2)]
    )


CODES = {"hgp13": hgp13, "bb72": bb72}
