# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Tests for last-ping extraction and the boundary check it feeds.

Synthetic raw files are built from real Simrad datagram frames, so the readers
are exercised against the actual binary layout rather than a mock.
"""

from __future__ import annotations

from datetime import datetime
from struct import pack

import pytest

from aa_si_utils.data_retrieval import filter_paths_by_file_time
from aa_si_utils.raw_file_times import (
    _last_ping_from_scan,
    _last_ping_from_tail,
    last_ping_time,
)


_EPOCH_1601 = datetime(1601, 1, 1)


def _nt_timestamp(when: datetime) -> int:
    """Encode a datetime as a 64-bit NT timestamp (100 ns ticks since 1601)."""
    delta = when - _EPOCH_1601
    microseconds = (delta.days * 86400 + delta.seconds) * 10**6 + delta.microseconds
    return microseconds * 10


def _record(dg_type: bytes, when: datetime, body: bytes = b"") -> bytes:
    """One Simrad datagram: length word, type, timestamp, body, length word."""
    dg_size = 12 + len(body)
    return (
        pack("=I", dg_size)
        + dg_type
        + pack("=Q", _nt_timestamp(when))
        + body
        + pack("=I", dg_size)
    )


def _write_raw(path, records: bytes) -> str:
    path.write_bytes(records)
    return str(path)


def _ek80_file(tmp_path, name, ping_times, trailing_nmea=True):
    """An EK80-shaped file: XML0 config, RAW3 pings, optional trailing NME0."""
    data = _record(b"XML0", ping_times[0], b"<Configuration/>")
    for when in ping_times:
        data += _record(b"RAW3", when, b"\x00" * 32)
    if trailing_nmea:
        data += _record(b"NME0", ping_times[-1], b"$GPGGA,,,,")
    return _write_raw(tmp_path / name, data)


# ---------------------------------------------------------------------------
# last_ping_time
# ---------------------------------------------------------------------------

def test_tail_read_finds_last_raw3(tmp_path):
    pings = [
        datetime(2024, 10, 12, 4, 56, 13),
        datetime(2024, 10, 12, 5, 10, 0),
        datetime(2024, 10, 12, 5, 30, 45, 500000),
    ]
    path = _ek80_file(tmp_path, "D20241012-T045613.raw", pings)
    assert last_ping_time(path) == pings[-1]


def test_tail_read_walks_past_trailing_nmea(tmp_path):
    pings = [datetime(2024, 10, 12, 5, 0, 0), datetime(2024, 10, 12, 5, 15, 0)]
    path = _ek80_file(tmp_path, "D20241012-T050000.raw", pings, trailing_nmea=True)
    with open(path, "rb") as fh:
        size = tmp_path.joinpath("D20241012-T050000.raw").stat().st_size
        assert _last_ping_from_tail(fh, size) == pings[-1]


def test_ek60_raw0_pings(tmp_path):
    pings = [datetime(2016, 7, 25, 20, 58, 32), datetime(2016, 7, 25, 21, 4, 59)]
    data = _record(b"CON0", pings[0], b"config")
    for when in pings:
        data += _record(b"RAW0", when, b"\x00" * 16)
    path = _write_raw(tmp_path / "D20160725-T205832.raw", data)
    assert last_ping_time(path) == pings[-1]


def test_truncated_tail_falls_back_to_forward_scan(tmp_path):
    pings = [
        datetime(2024, 10, 12, 5, 0, 0),
        datetime(2024, 10, 12, 5, 10, 0),
        datetime(2024, 10, 12, 5, 20, 0),
    ]
    data = _record(b"XML0", pings[0], b"<Configuration/>")
    for when in pings[:-1]:
        data += _record(b"RAW3", when, b"\x00" * 32)
    # Last record's header survives but its body is cut short and its trailing
    # length word is gone, which is what an interrupted download leaves behind.
    data += _record(b"RAW3", pings[-1], b"\xff" * 64)[:-20]
    path = _write_raw(tmp_path / "D20241012-T050000.raw", data)

    with open(path, "rb") as fh:
        fh.seek(0, 2)
        assert _last_ping_from_tail(fh, fh.tell()) is None
        assert _last_ping_from_scan(fh) == pings[-1]

    assert last_ping_time(path) == pings[-1]


def test_unreadable_and_empty_files_return_none(tmp_path):
    assert last_ping_time(tmp_path / "missing.raw") is None
    assert last_ping_time(_write_raw(tmp_path / "empty.raw", b"")) is None
    assert last_ping_time(_write_raw(tmp_path / "junk.raw", b"not a raw file")) is None


def test_file_without_ping_datagrams_returns_none(tmp_path):
    data = _record(b"XML0", datetime(2024, 10, 12, 5, 0, 0), b"<Configuration/>")
    assert last_ping_time(_write_raw(tmp_path / "D20241012-T050000.raw", data)) is None


# ---------------------------------------------------------------------------
# filter_paths_by_file_time boundary verification
# ---------------------------------------------------------------------------

_WINDOW_START = "2024-10-12T00:15"


def test_boundary_file_before_a_recording_gap_is_excluded(tmp_path):
    """The HB2407 case: a file 17 days stale is not rescued by the next stamp."""
    stale = _ek80_file(
        tmp_path,
        "D20240924-T232256.raw",
        [datetime(2024, 9, 24, 23, 22, 56), datetime(2024, 9, 25, 0, 18, 25)],
    )
    in_range = _ek80_file(
        tmp_path,
        "D20241012-T045613.raw",
        [datetime(2024, 10, 12, 4, 56, 13), datetime(2024, 10, 12, 5, 30, 0)],
    )

    kept = filter_paths_by_file_time([stale, in_range], _WINDOW_START, None)
    assert kept == [in_range]

    # Name-only filtering is what let the stale file through.
    name_only = filter_paths_by_file_time(
        [stale, in_range], _WINDOW_START, None, verify_boundary=False
    )
    assert name_only == [stale, in_range]


def test_boundary_file_actually_recording_into_window_is_kept(tmp_path):
    straddling = _ek80_file(
        tmp_path,
        "D20241012-T000000.raw",
        [datetime(2024, 10, 12, 0, 0, 0), datetime(2024, 10, 12, 1, 0, 0)],
    )
    later = _ek80_file(
        tmp_path,
        "D20241012-T010000.raw",
        [datetime(2024, 10, 12, 1, 0, 0), datetime(2024, 10, 12, 2, 0, 0)],
    )

    kept = filter_paths_by_file_time([straddling, later], _WINDOW_START, None)
    assert kept == [straddling, later]


def test_last_file_with_no_next_stamp_is_rescued_by_its_bytes(tmp_path):
    """Names cannot bound the final file's end, so only its bytes can."""
    only = _ek80_file(
        tmp_path,
        "D20241012-T000000.raw",
        [datetime(2024, 10, 12, 0, 0, 0), datetime(2024, 10, 12, 3, 0, 0)],
    )
    assert filter_paths_by_file_time([only], _WINDOW_START, None) == [only]

    # Name-only filtering drops it: no next stamp means no inferred end.
    assert filter_paths_by_file_time(
        [only], _WINDOW_START, None, verify_boundary=False
    ) == []


def test_last_file_ending_before_window_stays_excluded(tmp_path):
    only = _ek80_file(
        tmp_path,
        "D20241012-T000000.raw",
        [datetime(2024, 10, 12, 0, 0, 0), datetime(2024, 10, 12, 0, 10, 0)],
    )
    assert filter_paths_by_file_time([only], _WINDOW_START, None) == []


def test_unreadable_boundary_file_keeps_name_based_verdict(tmp_path, capsys):
    missing = str(tmp_path / "D20241012-T000000.raw")
    later = _ek80_file(
        tmp_path,
        "D20241012-T010000.raw",
        [datetime(2024, 10, 12, 1, 0, 0), datetime(2024, 10, 12, 2, 0, 0)],
    )

    kept = filter_paths_by_file_time([missing, later], _WINDOW_START, None)
    assert kept == [missing, later]
    assert "Could not read last ping" in capsys.readouterr().out


def test_verbose_false_suppresses_the_unreadable_boundary_note(tmp_path, capsys):
    """Mirrors aa_si_calibration._storage.filter_paths_by_file_time, which
    takes the same flag; the two are kept diff-able."""
    missing = str(tmp_path / "D20241012-T000000.raw")
    later = _ek80_file(
        tmp_path,
        "D20241012-T010000.raw",
        [datetime(2024, 10, 12, 1, 0, 0), datetime(2024, 10, 12, 2, 0, 0)],
    )

    kept = filter_paths_by_file_time(
        [missing, later], _WINDOW_START, None, verbose=False
    )
    assert kept == [missing, later]
    assert "Could not read last ping" not in capsys.readouterr().out


def test_only_the_boundary_file_is_opened(tmp_path, monkeypatch):
    paths = [
        _ek80_file(tmp_path, "D20241012-T000000.raw",
                   [datetime(2024, 10, 12, 0, 0, 0), datetime(2024, 10, 12, 0, 30, 0)]),
        _ek80_file(tmp_path, "D20241012-T010000.raw",
                   [datetime(2024, 10, 12, 1, 0, 0)]),
        _ek80_file(tmp_path, "D20241012-T020000.raw",
                   [datetime(2024, 10, 12, 2, 0, 0)]),
    ]
    opened = []
    import aa_si_utils.data_retrieval as dr

    def _spy(path, storage_options=None):
        opened.append(path)
        return datetime(2024, 10, 12, 0, 30, 0)

    monkeypatch.setattr(dr, "last_ping_time", _spy)
    filter_paths_by_file_time(paths, _WINDOW_START, None)
    assert opened == [paths[0]]


def test_no_start_bound_skips_verification(tmp_path, monkeypatch):
    import aa_si_utils.data_retrieval as dr

    monkeypatch.setattr(
        dr, "last_ping_time", lambda *a, **k: pytest.fail("should not be called")
    )
    paths = ["/data/D20241012-T000000.raw", "/data/D20241012-T010000.raw"]
    assert filter_paths_by_file_time(paths, None, "2024-10-12T02:00") == paths
