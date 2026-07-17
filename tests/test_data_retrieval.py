# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Tests for query/download helpers in data_retrieval.

Network-touching paths are exercised via a fake `requests` module so the
tests run offline. Only the labeling, subfolder, and skip-existing logic
is covered here; the live ArcGIS / S3 paths are intentionally not tested.
"""

from __future__ import annotations

import io
import json
import tarfile
from datetime import datetime
from pathlib import Path

import pytest

# Import the module directly to avoid triggering aa_si_utils/__init__.py,
# which pulls in heavy optional deps (echopype) unrelated to these tests.
import importlib.util
import sys

_DR_PATH = (
    Path(__file__).resolve().parent.parent / "src" / "aa_si_utils" / "data_retrieval.py"
)
_spec = importlib.util.spec_from_file_location("aa_si_utils_data_retrieval", _DR_PATH)
dr = importlib.util.module_from_spec(_spec)
sys.modules["aa_si_utils_data_retrieval"] = dr
_spec.loader.exec_module(dr)


# ---------------------------------------------------------------------------
# build_query_label
# ---------------------------------------------------------------------------


def test_build_query_label_cruise_and_dates():
    label = dr.build_query_label(
        file_time_start="2016-07-25T20:58",
        file_time_end="2016-07-25T21:45",
        filters={"CRUISE_NAME": "HB1603"},
    )
    assert label == "HB1603_2016-07-25_2058-2145"


def test_build_query_label_dataset_only():
    label = dr.build_query_label(dataset_name="DY2207_EK80")
    assert label == "DY2207_EK80"


def test_build_query_label_falls_back_to_query():
    assert dr.build_query_label() == "query"


def test_build_query_label_includes_collection_dates():
    label1 = dr.build_query_label(
        dataset_name="DY2207_EK80",
        collection_start="2022-06-04",
        collection_end="2022-06-04",
    )
    label2 = dr.build_query_label(
        dataset_name="DY2207_EK80",
        collection_start="2022-06-05",
        collection_end="2022-06-05",
    )

    assert label1 != label2
    assert "2022-06-04" in label1
    assert "2022-06-05" in label2


def test_build_query_label_distinguishes_other_query_filters():
    label1 = dr.build_query_label(
        filters={"CRUISE_NAME": "HB1603", "PLATFORM_NAME": "Henry B. Bigelow"},
    )
    label2 = dr.build_query_label(
        filters={"CRUISE_NAME": "HB1603", "PLATFORM_NAME": "Oscar Dyson"},
    )

    assert label1 != label2


def test_build_query_label_truncates_when_too_long():
    # Force an overlong label by stacking frequencies
    label = dr.build_query_label(
        filters={"CRUISE_NAME": "VERY_LONG_CRUISE_NAME"},
        file_time_start="2016-07-25T20:58",
        file_time_end="2016-07-26T21:45",
        frequencies=["18WKHZ", "38WKHZ", "70WKHZ", "120WKHZ", "200WKHZ"],
    )
    assert len(label) <= dr._LABEL_MAX_LEN + 9  # trunc + "_" + 8-char hash
    # Hash suffix is deterministic, so same inputs → same label
    label2 = dr.build_query_label(
        filters={"CRUISE_NAME": "VERY_LONG_CRUISE_NAME"},
        file_time_start="2016-07-25T20:58",
        file_time_end="2016-07-26T21:45",
        frequencies=["18WKHZ", "38WKHZ", "70WKHZ", "120WKHZ", "200WKHZ"],
    )
    assert label == label2
    # Different params → different hash suffix
    label3 = dr.build_query_label(
        filters={"CRUISE_NAME": "VERY_LONG_CRUISE_NAME"},
        file_time_start="2016-07-25T20:58",
        file_time_end="2016-07-27T21:45",  # changed
        frequencies=["18WKHZ", "38WKHZ", "70WKHZ", "120WKHZ", "200WKHZ"],
    )
    assert label != label3


def test_build_query_label_sanitizes_unsafe_chars():
    label = dr.build_query_label(
        filters={"CRUISE_NAME": "HB/16:03 (test)"},
    )
    # Slashes, colons, spaces, parens all collapsed to underscores
    assert "/" not in label
    assert ":" not in label
    assert " " not in label


# ---------------------------------------------------------------------------
# download_ncei_data — per-query subfolder & skip-existing
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, body: bytes, status_code: int = 200):
        self._body = body
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size: int = 0):
        yield self._body


def _install_fake_network(monkeypatch, body: bytes, head_ext: str = ".raw"):
    """Patch the requests module used inside data_retrieval."""

    def fake_head(url, timeout=30, allow_redirects=True):
        # Respond OK for any extension probe so the function picks one and
        # proceeds. Return the requested extension's URL as-is.
        return _FakeResponse(b"", status_code=200)

    def fake_get(url, stream=False, timeout=300, **_kw):
        return _FakeResponse(body)

    monkeypatch.setattr(dr.requests, "get", fake_get)
    monkeypatch.setattr(dr.requests, "head", fake_head)


def _make_record(filename: str) -> dict:
    return {
        "CLOUD_PATH": f"s3://noaa-wcsd-pds/data/raw/HB1603/EK60/{filename}",
        "FILE_NAME": filename,
    }


def test_query_returns_json_safe_file_datetime(monkeypatch):
    monkeypatch.setattr(
        dr,
        "_fetch_all_pages",
        lambda params: [_make_record("D20160725-T210000.raw")],
    )

    result = dr.query_ncei_data(file_time_start="2016-07-25T20:00")

    file_datetime = result["records"][0]["FILE_DATETIME"]
    assert file_datetime == "2016-07-25T21:00:00"
    json.dumps(result)


def test_query_includes_file_straddling_window_start(monkeypatch):
    monkeypatch.setattr(
        dr,
        "_fetch_all_pages",
        lambda params: [
            _make_record("D20160725-T130000.raw"),
            _make_record("D20160725-T150000.raw"),
        ],
    )

    result = dr.query_ncei_data(
        file_time_start="2016-07-25T14:00",
        file_time_end="2016-07-25T18:00",
    )

    # The 13:00 file records until the next file starts (15:00), so it has
    # in-window data and is kept despite starting before the window.
    names = [r["FILE_NAME"] for r in result["records"]]
    assert names == ["D20160725-T130000.raw", "D20160725-T150000.raw"]


def test_query_last_file_end_not_borrowed_across_datasets(monkeypatch):
    def rec(filename, dataset):
        record = _make_record(filename)
        record["DATASET_NAME"] = dataset
        return record

    monkeypatch.setattr(
        dr,
        "_fetch_all_pages",
        lambda params: [
            rec("D20160725-T120000.raw", "dsA"),
            rec("D20160725-T130000.raw", "dsA"),
            rec("D20160725-T150000.raw", "dsB"),
        ],
    )

    result = dr.query_ncei_data(
        file_time_start="2016-07-25T14:00",
        file_time_end="2016-07-25T18:00",
    )

    # dsA's 13:00 file is the last of its dataset: with no same-dataset
    # successor its end is unknown, so it falls back to the own-stamp rule
    # and is excluded — dsB's 15:00 stamp must not act as its end time.
    names = [r["FILE_NAME"] for r in result["records"]]
    assert names == ["D20160725-T150000.raw"]


def test_query_widens_derived_collection_start(monkeypatch):
    captured = {}

    def fake_fetch(params):
        captured["where"] = params["where"]
        return []

    monkeypatch.setattr(dr, "_fetch_all_pages", fake_fetch)

    dr.query_ncei_data(file_time_start="2016-07-25T00:30")
    # Widened one day so a previous-day file recording past midnight is
    # fetched as a candidate for the overlap filter.
    assert "COLLECTION_DATE >= DATE '2016-07-24'" in captured["where"]

    dr.query_ncei_data(
        file_time_start="2016-07-25T00:30", collection_start="2016-07-25"
    )
    # An explicit collection_start is respected untouched.
    assert "COLLECTION_DATE >= DATE '2016-07-25'" in captured["where"]


def test_download_creates_per_query_subfolder(monkeypatch, tmp_path):
    _install_fake_network(monkeypatch, body=b"fake raw bytes")

    result = dr.download_ncei_data(
        {
            "records": [_make_record("D20160725-T210000.raw")],
            "query_label": "HB1603_2016-07-25_2100-2200",
        },
        output_dir=tmp_path,
    )

    expected_dir = tmp_path / "HB1603_2016-07-25_2100-2200"
    assert result["download_dir"] == expected_dir.as_posix()
    assert expected_dir.is_dir()
    assert (expected_dir / "D20160725-T210000.raw").is_file()
    assert result["downloaded_paths"] == [
        (expected_dir / "D20160725-T210000.raw").as_posix()
    ]
    json.dumps(result)


def test_download_query_id_overrides_label(monkeypatch, tmp_path):
    _install_fake_network(monkeypatch, body=b"fake raw bytes")

    result = dr.download_ncei_data(
        {"records": [_make_record("D20160725-T210000.raw")], "query_label": "auto"},
        output_dir=tmp_path,
        query_id="my_chosen_name",
    )

    assert (tmp_path / "my_chosen_name").is_dir()
    assert not (tmp_path / "auto").exists()
    assert result["download_dir"] == (tmp_path / "my_chosen_name").as_posix()


def test_download_query_id_dotdot_cannot_escape_output_dir(monkeypatch, tmp_path):
    _install_fake_network(monkeypatch, body=b"fake raw bytes")

    result = dr.download_ncei_data(
        {"records": [_make_record("D20160725-T210000.raw")], "query_label": "auto"},
        output_dir=tmp_path,
        query_id="..",
    )

    assert result["download_dir"] == (tmp_path / "downloads").as_posix()
    assert (tmp_path / "downloads" / "D20160725-T210000.raw").is_file()
    assert not (tmp_path.parent / "D20160725-T210000.raw").exists()


def test_download_skips_existing_file(monkeypatch, tmp_path):
    call_count = {"n": 0}

    def counting_get(url, stream=False, timeout=300, **_kw):
        call_count["n"] += 1
        return _FakeResponse(b"fake")

    def fake_head(url, timeout=30, allow_redirects=True):
        return _FakeResponse(b"", status_code=200)

    monkeypatch.setattr(dr.requests, "get", counting_get)
    monkeypatch.setattr(dr.requests, "head", fake_head)

    payload = {
        "records": [_make_record("D20160725-T210000.raw")],
        "query_label": "label_one",
    }

    dr.download_ncei_data(payload, output_dir=tmp_path)
    assert call_count["n"] == 1
    # Second call with same query should hit the skip path
    dr.download_ncei_data(payload, output_dir=tmp_path)
    assert call_count["n"] == 1, "Existing file should not be re-downloaded"


def test_download_accepts_bare_list_with_query_label_kwarg(monkeypatch, tmp_path):
    _install_fake_network(monkeypatch, body=b"fake")

    result = dr.download_ncei_data(
        [_make_record("a.raw")],
        output_dir=tmp_path,
        query_label="from_kwarg",
    )
    assert result["download_dir"] == (tmp_path / "from_kwarg").as_posix()


def test_download_skips_existing_tar_extraction(monkeypatch, tmp_path):
    # Build a real tar file in memory and serve it via fake_get
    inner = b"hello"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        info = tarfile.TarInfo(name="inner.raw")
        info.size = len(inner)
        tf.addfile(info, io.BytesIO(inner))
    tar_bytes = buf.getvalue()

    _install_fake_network(monkeypatch, body=tar_bytes)

    rec = {
        "CLOUD_PATH": "s3://noaa-wcsd-pds/data/raw/HB1603/EK60/archive",
        "FILE_NAME": "archive",  # no extension → triggers probe + ".tar"
    }
    payload = {"records": [rec], "query_label": "tarq"}

    out1 = dr.download_ncei_data(payload, output_dir=tmp_path)
    extracted = tmp_path / "tarq" / "archive"
    assert extracted.is_dir()
    assert (extracted / "inner.raw").is_file()

    # Second invocation should reuse the extracted dir, no new download
    call_count = {"n": 0}

    def counting_get(url, stream=False, timeout=300, **_kw):
        call_count["n"] += 1
        return _FakeResponse(tar_bytes)

    monkeypatch.setattr(dr.requests, "get", counting_get)
    out2 = dr.download_ncei_data(payload, output_dir=tmp_path)
    assert call_count["n"] == 0
    assert out2["download_dir"] == (tmp_path / "tarq").as_posix()
    assert any(Path(p).name == "inner.raw" for p in out2["downloaded_paths"])
