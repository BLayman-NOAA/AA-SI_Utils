# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Last-ping-time extraction from Simrad EK60/EK80 raw files.

Reads datagram envelopes only, never sample payloads, so a file's true
recording end can be established without loading it. This settles the one case
the filename-time heuristic cannot decide: a file whose name stamp falls before
a requested window, where only the bytes say whether it records into it.

NOTE: mirrored in ``aa_si_calibration.raw_file_times``. ``aa_si_calibration``
intentionally depends on neither ``aa_si_utils`` nor ``aa_recipe_manager`` (see
that package's ``_storage`` module), so this small helper is duplicated rather
than shared. The Simrad datagram envelope is a fixed binary layout, so the two
copies are not expected to drift; keep them in sync if one is changed.
"""

import os
from datetime import datetime, timedelta, timezone
from struct import error as StructError, unpack

from aa_si_utils import _storage


_NT_EPOCH = datetime(1601, 1, 1, tzinfo=timezone.utc)

# Sample datagrams: EK60 writes RAW0, EK80 writes RAW3.
_PING_TYPES = (b"RAW0", b"RAW3")

# Leading length word, 4-byte type code, 8-byte NT timestamp.
_HEADER_FORMAT = "=I4sQ"
_HEADER_LEN = 16

# A record is framed by a 4-byte length word at each end, and the length itself
# counts only the type, timestamp, and body. So the bytes on disk for one record
# are its length plus these 8.
_FRAME_EXTRA = 8

# Smallest length word: a type and a timestamp with an empty body.
_MIN_DG_SIZE = 12

# Smallest possible record: both length words, a type, and a timestamp.
_MIN_RECORD = _HEADER_LEN + 4

# How far back from EOF to look for a ping datagram before giving up. Trailing
# NME0/XML0 runs are short, so a ping is normally within a handful of records.
_MAX_TAIL_RECORDS = 500


def _nt_to_naive_utc(nt_timestamp):
    """Convert a 64-bit NT timestamp to a naive UTC datetime.

    Naive to match ``parse_datetime_from_filename``, so ping times and name
    stamps compare directly against the same window bounds.
    """
    converted = _NT_EPOCH + timedelta(microseconds=nt_timestamp // 10)
    return converted.replace(tzinfo=None)


def _last_ping_from_tail(fh, size):
    """Walk records backwards from EOF, returning the last ping time.

    The trailing length word gives a record's start and the leading word
    confirms it, so a mismatch means the frame is broken and the walk stops.

    Returns:
        datetime | None: None when the frame does not check out (a truncated
        file) or no ping datagram sits near the end.
    """
    pos = size
    for _ in range(_MAX_TAIL_RECORDS):
        if pos < _MIN_RECORD:
            return None

        fh.seek(pos - 4)
        trailing = unpack("=I", fh.read(4))[0]
        record_start = pos - trailing - _FRAME_EXTRA
        if record_start < 0:
            return None

        fh.seek(record_start)
        header = fh.read(_HEADER_LEN)
        if len(header) < _HEADER_LEN:
            return None

        leading, dg_type, nt_timestamp = unpack(_HEADER_FORMAT, header)
        if leading != trailing:
            return None
        if dg_type in _PING_TYPES:
            return _nt_to_naive_utc(nt_timestamp)

        pos = record_start
    return None


def _last_ping_from_scan(fh):
    """Walk records forward from the start, returning the last ping time.

    Costs a pass over the file but tolerates a broken tail: it reports the last
    complete ping datagram instead of failing. Only 16-byte headers are read;
    sample payloads are seeked past.

    Returns:
        datetime | None: None when the file holds no readable ping datagram.
    """
    fh.seek(0)
    last = None
    while True:
        header = fh.read(_HEADER_LEN)
        if len(header) < _HEADER_LEN:
            return last

        dg_size, dg_type, nt_timestamp = unpack(_HEADER_FORMAT, header)
        if dg_size < _MIN_DG_SIZE:
            return last
        if dg_type in _PING_TYPES:
            last = _nt_to_naive_utc(nt_timestamp)

        fh.seek(dg_size - _FRAME_EXTRA, 1)


def last_ping_time(path, storage_options=None):
    """Time of the last ping recorded in a Simrad raw file.

    Tries a backwards walk from EOF first, which reads a few hundred bytes. For
    local files it falls back to a forward header scan, which survives the
    truncated tail that is the backwards walk's usual failure. Remote paths skip
    that fallback: a forward walk over object storage costs thousands of small
    range requests, too steep for a file that is probably damaged.

    Args:
        path: Local path or remote URL of a ``.raw`` file.
        storage_options: fsspec options applied to remote URLs.

    Returns:
        datetime | None: Naive UTC time of the last RAW0/RAW3 datagram, or None
        when it cannot be determined.
    """
    try:
        if _storage.is_remote(path):
            fs = _storage.get_fs(path, storage_options)
            with fs.open(str(path), "rb") as fh:
                return _last_ping_from_tail(fh, fs.size(str(path)))

        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            from_tail = _last_ping_from_tail(fh, size)
            return from_tail if from_tail is not None else _last_ping_from_scan(fh)
    except (OSError, ValueError, OverflowError, StructError):
        return None
