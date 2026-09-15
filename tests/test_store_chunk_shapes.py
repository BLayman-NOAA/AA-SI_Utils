# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Squaring up echopype's swap-path chunks before a per-file zarr store write.

Zarr rejects any dimension whose interior chunks differ, or whose final chunk is
larger than the first.  echopype's swap path assembles each group from the
blocks it spilled, so a file whose datagrams did not arrive in equal runs comes
back ragged along ``ping_time`` -- ``(169, 1, 1)`` on one HB1603 file, which
failed the survey 41 minutes into ``read_raw``.  Whether swap runs at all is
decided per file from live memory, so the same file can write cleanly on one run
and fail on the next; these tests pin the normalization rather than the weather.
"""

from __future__ import annotations

from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from aa_si_utils import utils

RS = 64  # stand-in for the real 15355 range samples; chunk *shape* is the point


def _ragged_dataset(ping_chunks=(169, 1, 1), n_channel=5):
    """A Beam group chunked the way the failing file was."""
    n_ping = sum(ping_chunks)
    arr = da.zeros(
        (n_channel, n_ping, RS),
        chunks=((1,) * n_channel, tuple(ping_chunks), (RS,)),
        dtype="f8",
    )
    return xr.Dataset(
        {"backscatter_r": (("channel", "ping_time", "range_sample"), arr)},
        coords={"ping_time": np.arange(n_ping)},
    )


class _StubEchoData:
    """EchoData stand-in whose to_zarr is a real, failing-if-ragged xarray write."""

    def __init__(self, groups: dict[str, xr.Dataset]) -> None:
        self._groups = dict(groups)

    @property
    def group_paths(self):
        return list(self._groups)

    def __getitem__(self, key):
        return self._groups[key]

    def __setitem__(self, key, value):
        self._groups[key] = value

    def chunks_of(self, group, var="backscatter_r"):
        return self._groups[group][var].chunks

    def to_zarr(self, save_path, zarr_format=2, compress=True, **kwargs):
        for index, (path, ds) in enumerate(self._groups.items()):
            ds.to_zarr(
                str(save_path), group=path,
                mode="w" if index == 0 else "a", zarr_format=zarr_format,
            )

    def cleanup_swap_files(self):
        pass


def _zarr_writable(chunks) -> bool:
    return all(c == chunks[0] for c in chunks[:-1]) and chunks[-1] <= chunks[0]


# ---------------------------------------------------------------------------
# _zarr_writable_chunks
# ---------------------------------------------------------------------------


def test_the_reported_ragged_ping_time_is_squared_up():
    ed = _StubEchoData({"Sonar/Beam_group1": _ragged_dataset()})

    utils._zarr_writable_chunks(ed)

    channel, ping_time, _ = ed.chunks_of("Sonar/Beam_group1")
    assert _zarr_writable(ping_time)
    assert sum(ping_time) == 171, "no data may be gained or lost"
    # Already-legal per-channel blocking must survive: collapsing it would
    # multiply the per-chunk footprint by the channel count during the write.
    assert channel == (1, 1, 1, 1, 1)


def test_legal_chunks_are_left_untouched():
    ds = _ragged_dataset(ping_chunks=(100, 100, 50))
    ed = _StubEchoData({"Sonar/Beam_group1": ds})
    before = ed.chunks_of("Sonar/Beam_group1")

    utils._zarr_writable_chunks(ed)

    assert ed.chunks_of("Sonar/Beam_group1") == before


def test_a_final_chunk_larger_than_the_first_is_also_caught():
    """The other half of zarr's rule, which a max()-only check would miss."""
    ed = _StubEchoData({"Sonar/Beam_group1": _ragged_dataset(ping_chunks=(10, 10, 99))})

    utils._zarr_writable_chunks(ed)

    assert _zarr_writable(ed.chunks_of("Sonar/Beam_group1")[1])


def test_ping_time_is_capped_so_one_block_cannot_swallow_the_axis():
    big = utils.DEFAULT_PING_TIME_CHUNK * 3
    ed = _StubEchoData({"Sonar/Beam_group1": _ragged_dataset(ping_chunks=(big, 1, 1))})

    utils._zarr_writable_chunks(ed)

    ping_time = ed.chunks_of("Sonar/Beam_group1")[1]
    assert ping_time[0] == utils.DEFAULT_PING_TIME_CHUNK
    assert _zarr_writable(ping_time)


def test_numpy_backed_groups_are_a_no_op():
    """Without swap the arrays carry no chunks at all."""
    ds = xr.Dataset({"backscatter_r": (("ping_time",), np.zeros(171))})
    ed = _StubEchoData({"Sonar/Beam_group1": ds})

    utils._zarr_writable_chunks(ed)

    assert ed.chunks_of("Sonar/Beam_group1") is None


def test_a_group_that_is_none_is_skipped():
    ed = _StubEchoData({"Provenance": None, "Sonar/Beam_group1": _ragged_dataset()})

    utils._zarr_writable_chunks(ed)  # must not raise

    assert _zarr_writable(ed.chunks_of("Sonar/Beam_group1")[1])


# ---------------------------------------------------------------------------
# End to end through _open_and_store
# ---------------------------------------------------------------------------


def test_a_ragged_file_writes_its_store(tmp_path, monkeypatch):
    """The failure that killed the survey: a real zarr write of ragged chunks."""
    ed = _StubEchoData({"Sonar/Beam_group1": _ragged_dataset()})
    monkeypatch.setattr(utils.ep, "open_raw", lambda *a, **k: ed)

    raw = tmp_path / "HB1603_L1-D20160627-T212739.raw"
    raw.write_bytes(b"\x00" * 64)
    store_dir = tmp_path / "exe_temp"
    store_dir.mkdir()

    out = utils._open_and_store(
        raw, "EK60", False, "auto", "zarr", store_dir, None, False
    )

    assert Path(out).exists()
    back = xr.open_zarr(out, group="Sonar/Beam_group1")
    assert back["backscatter_r"].shape == (5, 171, RS)
