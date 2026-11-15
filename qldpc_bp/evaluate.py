"""Judging a correction, and the reference decoders from the `ldpc` package."""

import numpy as np
from ldpc import BpDecoder, BpOsdDecoder


def judge(h, logicals, errors, corrections):
    """Classify each shot's residual ``error XOR correction``.

    Returns ``(unmatched, logical)``, two boolean arrays:
      unmatched  the correction does not reproduce the syndrome, so the decoder knows it failed
      logical    the syndrome is reproduced but the residual flips a logical operator; this is
                 the silent failure

    A residual that is a product of stabilizers is a success even though the correction differs
    from the error that was injected.
    """
    residual = np.asarray(errors, dtype=np.int64) ^ np.asarray(corrections, dtype=np.int64)
    unmatched = ((residual @ np.asarray(h, dtype=np.int64).T) % 2).any(axis=1)
    flipped = ((residual @ np.asarray(logicals, dtype=np.int64).T) % 2).any(axis=1)
    return unmatched, flipped & ~unmatched


def wilson(failures, shots, z=1.96):
    """Wilson score interval for a failure rate."""
    if shots == 0:
        return 0.0, 1.0
    rate = failures / shots
    denom = 1 + z * z / shots
    centre = (rate + z * z / (2 * shots)) / denom
    half = z * np.sqrt(rate * (1 - rate) / shots + z * z / (4 * shots * shots)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def ldpc_decode(h, syndromes, p, *, bp_method, alpha, max_iters, osd):
    """Decode with `ldpc`: plain BP (``osd=False``) or BP followed by OSD-0 (``osd=True``).

    Returns ``(corrections, bp_converged)``. With OSD the correction always reproduces the
    syndrome; ``bp_converged`` still says whether BP got there without it.
    """
    common = dict(
        error_rate=float(p),
        max_iter=int(max_iters),
        bp_method=bp_method,
        ms_scaling_factor=float(alpha),
        schedule="parallel",
    )
    h = np.asarray(h, dtype=np.uint8)
    if osd:
        decoder = BpOsdDecoder(h, osd_method="OSD_0", osd_order=0, **common)
    else:
        decoder = BpDecoder(h, **common)
    syndromes = np.atleast_2d(np.asarray(syndromes, dtype=np.uint8))
    out = np.zeros((syndromes.shape[0], h.shape[1]), dtype=np.uint8)
    converged = np.zeros(syndromes.shape[0], dtype=bool)
    for i, s in enumerate(syndromes):
        out[i] = decoder.decode(s)
        converged[i] = bool(decoder.converge)
    return out, converged
