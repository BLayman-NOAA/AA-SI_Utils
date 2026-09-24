# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Synthetic split-beam echograms for the seabed detector tests.

The dataset mirrors what echopype's compute_Sv, add_depth, add_splitbeam_angle
and add_location leave on an Sv dataset, with a seabed whose leading edge is
known per ping, so accuracy can be asserted in pulse lengths.
"""

import numpy as np
import pandas as pd
import xarray as xr

SOUND_SPEED = 1500.0
BEAMWIDTH_DEG = 7.0
TRANSDUCER_DEPTH_M = 5.0
WATER_SV_DB = -90.0
SEABED_SV_DB = -30.0


def make_synthetic_sv(
    n_ping=60,
    n_sample=400,
    dr=0.5,
    seabed_depth_m=100.0,
    pulse_length_m=2.0,
    ping_interval_s=1.0,
    vessel_speed_m_s=5.0,
    school=None,
    multiple=False,
    alias=None,
    angles=True,
    gps=True,
    depth=True,
    frequencies_hz=(38000,),
    seed=0,
):
    """Build a synthetic Sv dataset with a known seabed.

    Args:
        n_ping: Number of pings.
        n_sample: Number of range samples.
        dr: Range sample spacing in metres.
        seabed_depth_m: Seabed depth below the surface: a scalar or an
            array of shape (n_ping,) for slopes and steps.
        pulse_length_m: Pulse extent in range, c tau / 2.
        ping_interval_s: Seconds between pings.
        vessel_speed_m_s: Speed used to lay out GPS positions.
        school: Optional dict with ``ping_start``, ``ping_end``, ``top_m``,
            ``bottom_m`` and optional ``sv_db`` (default -45) and
            ``angle_spread_deg`` (default 5.0) describing a dense
            aggregation above the seabed.
        multiple: Add a bottom multiple 15 dB weaker at twice the range.
        alias: Optional dict like ``school`` describing an aliased-seabed
            patch: loud, with angles scattered across +/- 20 degrees.
        angles: Include split-beam angle variables.
        gps: Include latitude and longitude on ping_time.
        depth: Include the surface-referenced ``depth`` variable.
        frequencies_hz: One entry per channel; every channel carries the
            same scene with independent noise.
        seed: Random seed.

    Returns:
        tuple: ``(ds_Sv, truth)`` where ``truth`` has ``edge_idx`` (the
        first sample within 10 dB of the seabed peak, shape (n_ping,), -1
        where the seabed lies beyond the last sample),
        ``edge_m`` (its depth), ``pulse_samples`` and ``transducer_depth_m``.
    """
    rng = np.random.default_rng(seed)
    n_channel = len(frequencies_hz)
    seabed = np.broadcast_to(np.asarray(seabed_depth_m, dtype=float), (n_ping,)).copy()
    echo_range_1d = np.arange(n_sample, dtype=float) * dr
    n_k = max(1, int(np.ceil(pulse_length_m / dr)))

    sv = np.full((n_channel, n_ping, n_sample), WATER_SV_DB)
    sv += rng.normal(0.0, 3.0, sv.shape)
    # Low-SNR angle estimates scatter beyond the beam: HB1603 water column
    # showed about 30 deg2 of summed variance, which +/- one beamwidth gives.
    theta = rng.uniform(-BEAMWIDTH_DEG, BEAMWIDTH_DEG, sv.shape)
    phi = rng.uniform(-BEAMWIDTH_DEG, BEAMWIDTH_DEG, sv.shape)

    def paint_block(spec, sv_db, spread_deg, uniform_angles):
        p0, p1 = int(spec["ping_start"]), int(spec["ping_end"])
        s0 = int(round((spec["top_m"] - TRANSDUCER_DEPTH_M) / dr))
        s1 = int(round((spec["bottom_m"] - TRANSDUCER_DEPTH_M) / dr))
        shape = (n_channel, p1 - p0, s1 - s0)
        sv[:, p0:p1, s0:s1] = sv_db + rng.normal(0.0, 3.0, shape)
        if uniform_angles:
            theta[:, p0:p1, s0:s1] = rng.uniform(-spread_deg, spread_deg, shape)
            phi[:, p0:p1, s0:s1] = rng.uniform(-spread_deg, spread_deg, shape)
        else:
            theta[:, p0:p1, s0:s1] = rng.normal(0.0, spread_deg, shape)
            phi[:, p0:p1, s0:s1] = rng.normal(0.0, spread_deg, shape)

    if school is not None:
        paint_block(
            school,
            school.get("sv_db", -45.0),
            school.get("angle_spread_deg", 5.0),
            uniform_angles=True,
        )
    if alias is not None:
        paint_block(alias, alias.get("sv_db", -40.0), 20.0, uniform_angles=True)

    edge_idx = np.zeros(n_ping, dtype=int)
    rise = np.linspace(WATER_SV_DB, SEABED_SV_DB, n_k + 1)[1:]
    for p in range(n_ping):
        top = int(round((seabed[p] - TRANSDUCER_DEPTH_M) / dr))
        _paint_seabed(sv, theta, phi, p, top, rise, n_k, n_sample, rng, 0.0)
        if multiple:
            _paint_seabed(sv, theta, phi, p, 2 * top, rise - 15.0, n_k, n_sample, rng, 15.0)
        if top >= n_sample:
            edge_idx[p] = -1
            continue
        row = sv[0, p]
        peak = row[top : top + 3 * n_k].max()
        start = max(0, top - n_k)
        edge_idx[p] = start + int(np.argmax(row[start : top + 3 * n_k] >= peak - 10.0))

    ping_time = pd.date_range("2024-01-01", periods=n_ping, freq=f"{ping_interval_s}s").values
    coords = {
        "channel": np.array([str(f) for f in frequencies_hz]),
        "ping_time": ping_time,
        "range_sample": np.arange(n_sample),
    }
    echo_range = np.broadcast_to(echo_range_1d, sv.shape).copy()
    data = {
        "Sv": (("channel", "ping_time", "range_sample"), sv),
        "echo_range": (("channel", "ping_time", "range_sample"), echo_range),
        "frequency_nominal": (("channel",), np.array(frequencies_hz, dtype=float)),
        "tau_effective": (("channel",), np.full(n_channel, 2.0 * pulse_length_m / SOUND_SPEED)),
        "sound_speed": (("channel",), np.full(n_channel, SOUND_SPEED)),
        "beamwidth_alongship": (("channel",), np.full(n_channel, BEAMWIDTH_DEG)),
        "beamwidth_athwartship": (("channel",), np.full(n_channel, BEAMWIDTH_DEG)),
    }
    if depth:
        data["depth"] = (("channel", "ping_time", "range_sample"), echo_range + TRANSDUCER_DEPTH_M)
    if angles:
        data["angle_alongship"] = (("channel", "ping_time", "range_sample"), theta)
        data["angle_athwartship"] = (("channel", "ping_time", "range_sample"), phi)
    if gps:
        along = np.arange(n_ping) * vessel_speed_m_s * ping_interval_s
        data["latitude"] = (("ping_time",), 42.0 + along / 111_000.0)
        data["longitude"] = (("ping_time",), np.full(n_ping, -70.0))
    ds = xr.Dataset(data, coords=coords)

    offset = TRANSDUCER_DEPTH_M if depth else 0.0
    truth = {
        "edge_idx": edge_idx,
        "edge_m": edge_idx * dr + offset,
        "pulse_samples": n_k,
        "transducer_depth_m": TRANSDUCER_DEPTH_M,
        "seabed_depth_m": seabed,
    }
    return ds, truth


def _paint_seabed(sv, theta, phi, p, top, rise, n_k, n_sample, rng, weaken_db):
    """Paint one ping's seabed echo: rise, plateau, decaying tail."""
    if top >= n_sample:
        return
    start = max(0, top - len(rise))
    seg = rise[len(rise) - (top - start) :]
    sv[:, p, start:top] = seg[None, :] + rng.normal(0.0, 1.0, (sv.shape[0], top - start))
    plateau_end = min(n_sample, top + 3 * n_k)
    n_plateau = plateau_end - top
    sv[:, p, top:plateau_end] = SEABED_SV_DB - weaken_db + rng.normal(0.0, 1.0, (sv.shape[0], n_plateau))
    tail = np.arange(n_sample - plateau_end, dtype=float)
    tail_db = SEABED_SV_DB - weaken_db - 3.0 * tail / n_k
    tail_db = np.maximum(tail_db, -60.0 - weaken_db)
    sv[:, p, plateau_end:] = tail_db[None, :] + rng.normal(0.0, 1.0, (sv.shape[0], tail.size))
    spread = np.concatenate([np.full(n_plateau, 0.3), 0.3 + 0.7 * np.minimum(tail / (6 * n_k), 1.0)])
    theta[:, p, top:] = rng.normal(0.0, 1.0, (sv.shape[0], n_sample - top)) * spread[None, :]
    phi[:, p, top:] = rng.normal(0.0, 1.0, (sv.shape[0], n_sample - top)) * spread[None, :]
    theta[:, p, start:top] = rng.normal(0.0, 0.5, (sv.shape[0], top - start))
    phi[:, p, start:top] = rng.normal(0.0, 0.5, (sv.shape[0], top - start))
