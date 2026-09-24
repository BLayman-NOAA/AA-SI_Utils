# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Stage 5 dynamic-programming bottom line and Stage 8 confidence.

Candidates are sparse local maxima of the score per ping. The Viterbi pass
finds the minimum-cost line under a Huber transition whose break-point
scales with the distance travelled between pings; a forward-backward pass
over the same graph gives a posterior confidence and a margin per ping.
"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.special import logsumexp


@dataclass
class Candidates:
    """Top-N score maxima per ping.

    Attributes:
        idx: Crop-relative sample index, shape (P, N), -1 where unused.
        score: Lambda at each candidate, NaN where unused.
        range_m: Range of each candidate in metres, NaN where unused.
        valid: Whether the ping has at least one candidate, shape (P,).
    """

    idx: np.ndarray
    score: np.ndarray
    range_m: np.ndarray
    valid: np.ndarray

    @property
    def scores_flat(self):
        return self.score[np.isfinite(self.score)]


@dataclass
class TransitionParams:
    """Per-ping break-point delta and the global scales lambda, beta, T."""

    delta: np.ndarray
    lam: float
    beta: float
    temperature: float
    kind: str = "huber"


@dataclass
class DPResult:
    """Viterbi line and confidence.

    Attributes:
        r_star_idx: Crop-relative sample index of the pick per ping, NaN
            where the ping was skipped.
        chosen: Candidate column chosen per ping, -1 where skipped.
        total_cost: Minimum total path cost V*.
        confidence: Posterior mass within the tolerance of the pick per
            ping, NaN if skipped.
        margin: Cost of the best path forced through a candidate outside
            the tolerance at the ping minus V*; inf when there is none,
            NaN if skipped.
    """

    r_star_idx: np.ndarray
    chosen: np.ndarray
    total_cost: float
    confidence: np.ndarray
    margin: np.ndarray


def select_candidates(score, geom, n_candidates=5, min_score=0.0):
    """Local maxima of the score inside the search window, top N per ping.

    Maxima closer than one pulse length are suppressed in favour of the
    stronger one, so the line can jump between genuinely distinct picks.

    Args:
        score: Lambda on the cropped block, shape (P, S).
        geom: ``PingGeometry``.
        n_candidates: Candidates kept per ping.
        min_score: Candidates below this are dropped; None keeps any finite
            maximum. Pings with none are skipped by the DP.

    Returns:
        Candidates: Sparse candidate set.
    """
    n_ping = score.shape[0]
    sep = geom.pulse_samples
    masked = np.where(geom.window_mask() & np.isfinite(score), score, -np.inf)
    local_max = masked >= ndimage.maximum_filter1d(masked, size=2 * sep + 1, axis=1, mode="nearest")
    floor = -np.inf if min_score is None else float(min_score)
    local_max &= np.isfinite(masked) & (masked >= floor)

    idx = np.full((n_ping, n_candidates), -1, dtype=int)
    sc = np.full((n_ping, n_candidates), np.nan)
    for p in range(n_ping):
        cols = np.nonzero(local_max[p])[0]
        if cols.size == 0:
            continue
        order = cols[np.argsort(-masked[p, cols], kind="stable")]
        chosen = []
        for c in order:
            if all(abs(c - k) > sep for k in chosen):
                chosen.append(c)
                if len(chosen) == n_candidates:
                    break
        idx[p, : len(chosen)] = chosen
        sc[p, : len(chosen)] = masked[p, chosen]
    range_m = np.where(idx >= 0, geom.range0[:, None] + (idx + geom.i_lo) * geom.dr, np.nan)
    return Candidates(idx, sc, range_m, (idx >= 0).any(axis=1))


def transition_params(cand, geom, max_slope_deg=30.0, alpha=1.0, beta_fraction=0.1, kind="huber"):
    """Break-points and cost scales set from the data, without labels.

    Args:
        cand: Candidates from :func:`select_candidates`.
        geom: ``PingGeometry``; supplies along-track distance and pulse length.
        max_slope_deg: Maximum-slope prior phi_max.
        alpha: Transition strength multiplier in [0.5, 2].
        beta_fraction: Shallow-first prior as a fraction of IQR(Lambda);
            0 disables it.
        kind: ``"huber"`` or ``"truncated"`` (quadratic capped at delta).

    Returns:
        TransitionParams: delta per ping, lambda, beta and the temperature.
    """
    dx = np.where(np.isfinite(geom.dx_ping), geom.dx_ping, 0.0)
    delta = np.maximum(dx * np.tan(np.radians(max_slope_deg)), geom.pulse_length_m)
    scores = cand.scores_flat
    iqr = float(np.subtract(*np.percentile(scores, [75, 25]))) if scores.size > 1 else 0.0
    scale = iqr if iqr > 0 else 1.0
    lam = alpha * scale / float(np.median(delta))
    return TransitionParams(delta, lam, beta_fraction * scale, scale / 2.0, kind)


def transition_cost(dr_abs, delta, lam, kind="huber"):
    """Cost of a depth change ``dr_abs`` (m) with break-point ``delta``."""
    if kind == "truncated":
        return np.minimum(lam * dr_abs**2, lam * delta**2)
    if kind != "huber":
        raise ValueError(f"unknown transition kind {kind!r}")
    return np.where(dr_abs <= delta, lam * dr_abs**2 / (2.0 * delta), lam * (dr_abs - delta / 2.0))


def emission_cost(cand, geom, beta):
    """Negative score plus a weak shallow-first penalty, inf where unused."""
    span = np.maximum(geom.r_max - geom.r_min, geom.dr)
    frac = (cand.range_m - geom.r_min[:, None]) / span[:, None]
    cost = -cand.score + beta * frac
    return np.where(cand.idx >= 0, cost, np.inf)


def _pair_cost(cand, prev, p, params):
    """Transition cost matrix (N_prev, N_p) across the pings between."""
    gap_delta = float(params.delta[prev:p].sum())
    dr_abs = np.abs(cand.range_m[p][None, :] - cand.range_m[prev][:, None])
    dr_abs = np.where(np.isfinite(dr_abs), dr_abs, np.inf)
    return transition_cost(dr_abs, gap_delta, params.lam, params.kind)


def _loud_run(row, index):
    """Inclusive bounds of the contiguous True run of ``row`` holding ``index``."""
    lo = hi = int(index)
    if not row[lo]:
        return lo, hi
    while lo > 0 and row[lo - 1]:
        lo -= 1
    while hi < row.size - 1 and row[hi + 1]:
        hi += 1
    return lo, hi


def run_dp(cand, geom, params, tolerance_pulses=2.0, loud=None):
    """Viterbi line plus forward-backward confidence and margin.

    Skipped pings (no candidate) are bridged; the break-point across a gap
    is the sum of the per-ping deltas inside it. Candidates inside the same
    seabed echo as the pick count as the same pick: a seabed echo carries
    more than one local maximum (its tail runs tens of metres in deep
    water), and a posterior split between them says nothing about whether
    the seabed was found. Same echo means within ``tolerance_pulses`` pulse
    lengths plus the beam-geometry spread r (1 - cos(beamwidth / 2)), or,
    when ``loud`` is given, inside the same contiguous run of loud samples.

    Args:
        cand: Candidates.
        geom: ``PingGeometry``.
        params: ``TransitionParams``.
        tolerance_pulses: Pulse lengths added to the beam-geometry spread
            to form the range tolerance.
        loud: Optional boolean (P, S) mask of samples above the seabed
            echo threshold on the cropped block.

    Returns:
        DPResult: Picks, cost, confidence and margin.
    """
    n_ping, n_cand = cand.idx.shape
    valid = np.nonzero(cand.valid)[0]
    emission = emission_cost(cand, geom, params.beta)
    r_star = np.full(n_ping, np.nan)
    chosen = np.full(n_ping, -1, dtype=int)
    confidence = np.full(n_ping, np.nan)
    margin = np.full(n_ping, np.nan)
    if valid.size == 0:
        return DPResult(r_star, chosen, np.inf, confidence, margin)

    temp = params.temperature
    fwd = np.full((n_ping, n_cand), np.inf)
    back = np.full((n_ping, n_cand), -1, dtype=int)
    log_a = np.full((n_ping, n_cand), -np.inf)
    prev = None
    for p in valid:
        if prev is None:
            fwd[p] = emission[p]
            log_a[p] = -emission[p] / temp
        else:
            cost = _pair_cost(cand, prev, p, params)
            total = fwd[prev][:, None] + cost
            back[p] = np.argmin(total, axis=0)
            fwd[p] = emission[p] + total[back[p], np.arange(n_cand)]
            log_a[p] = -emission[p] / temp + logsumexp(log_a[prev][:, None] - cost / temp, axis=0)
        prev = p

    last = valid[-1]
    j = int(np.argmin(fwd[last]))
    total_cost = float(fwd[last, j])
    for p in valid[::-1]:
        chosen[p] = j
        r_star[p] = cand.idx[p, j]
        j = back[p, j]

    bwd = np.full((n_ping, n_cand), np.inf)
    log_b = np.full((n_ping, n_cand), -np.inf)
    bwd[last] = np.where(cand.idx[last] >= 0, 0.0, np.inf)
    log_b[last] = np.where(cand.idx[last] >= 0, 0.0, -np.inf)
    nxt = last
    for p in valid[-2::-1]:
        cost = _pair_cost(cand, p, nxt, params)
        through = cost + (emission[nxt] + bwd[nxt])[None, :]
        bwd[p] = np.min(through, axis=1)
        log_b[p] = logsumexp(-cost / temp - emission[nxt][None, :] / temp + log_b[nxt][None, :], axis=1)
        nxt = p

    spread = 1.0 - np.cos(np.radians(geom.beamwidth_deg / 2.0))
    for p in valid:
        pick_m = cand.range_m[p, chosen[p]]
        tolerance_m = tolerance_pulses * geom.pulse_length_m + max(pick_m - geom.range0[p], 0.0) * spread
        near = np.abs(cand.range_m[p] - pick_m) <= tolerance_m
        if loud is not None:
            lo, hi = _loud_run(loud[p], cand.idx[p, chosen[p]])
            near |= (cand.idx[p] >= lo) & (cand.idx[p] <= hi)
        near &= cand.idx[p] >= 0
        post = log_a[p] + log_b[p]
        post = post - logsumexp(post)
        confidence[p] = float(np.clip(np.exp(logsumexp(post[near])), 0.0, 1.0))
        others = fwd[p] + bwd[p]
        others[near] = np.inf
        margin[p] = float(np.min(others) - total_cost)
    return DPResult(r_star, chosen, total_cost, confidence, margin)
