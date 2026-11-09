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


# ---------------------------------------------------------------------------
# file_names: keep only the named files
# ---------------------------------------------------------------------------


def _catalogue():
    return [
        _make_record("HB1603_L1-D20160707-T192446.raw"),
        _make_record("HB1603_L1-D20160707-T194743.raw"),
        _make_record("D20160725-T205832.raw"),
        _make_record("D20160725-T212129.raw"),
    ]


def test_file_names_keeps_named_files_in_catalogue_order_matching_either_form(monkeypatch):
    monkeypatch.setattr(dr, "_fetch_all_pages", lambda params: _catalogue())

    result = dr.query_ncei_data(
        file_time_start="2016-07-01",
        file_time_end="2016-08-01",
        file_names=["D20160725-T212129.tar", "D20160707-T192446.raw"],
    )

    names = [r["FILE_NAME"] for r in result["records"]]
    assert names == ["HB1603_L1-D20160707-T192446.raw", "D20160725-T212129.raw"]
    assert [u.rsplit("/", 1)[-1] for u in result["raw_urls"]] == names


def test_file_names_raises_when_a_named_file_is_missing(monkeypatch):
    monkeypatch.setattr(dr, "_fetch_all_pages", lambda params: _catalogue())

    with pytest.raises(ValueError, match="1 of 2 named file"):
        dr.query_ncei_data(
            file_time_start="2016-07-01",
            file_names=["D20160725-T212129.raw", "D20160801-T000000.raw"],
        )


def test_file_names_rejects_a_name_without_a_stamp(monkeypatch):
    monkeypatch.setattr(dr, "_fetch_all_pages", lambda params: _catalogue())

    with pytest.raises(ValueError, match="no D"):
        dr.query_ncei_data(file_time_start="2016-07-01", file_names=["notes.raw"])


def test_empty_file_names_keeps_everything(monkeypatch):
    monkeypatch.setattr(dr, "_fetch_all_pages", lambda params: _catalogue())

    result = dr.query_ncei_data(file_time_start="2016-07-01", file_names=[])

    assert len(result["records"]) == 4


def test_file_names_applies_after_the_time_window(monkeypatch):
    """A named file outside the window is a miss, not a way around the window."""
    monkeypatch.setattr(dr, "_fetch_all_pages", lambda params: _catalogue())

    with pytest.raises(ValueError, match="not in the query result"):
        dr.query_ncei_data(
            file_time_start="2016-07-20",
            file_time_end="2016-08-01",
            file_names=["D20160707-T192446.raw"],
        )


# ---------------------------------------------------------------------------
# listing: the archive's own file listing
# ---------------------------------------------------------------------------

_FOLDER = "s3://noaa-wcsd-pds/data/raw/HB1603/EK60/"
_DAY = [
    "D20160814-T130000",
    "D20160814-T140000",
    "D20160814-T150000",
    "D20160814-T160000",
    "D20160814-T170000",
]


def _listing_xml(keys, token=None):
    contents = "".join(f"<Contents><Key>{key}</Key><Size>1</Size></Contents>" for key in keys)
    next_token = f"<NextContinuationToken>{token}</NextContinuationToken>" if token else ""
    return (
        '<?xml version="1.0"?>'
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        f"{contents}{next_token}</ListBucketResult>"
    ).encode()


class _ListingResponse:
    def __init__(self, content, status_code=200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise dr.requests.exceptions.HTTPError(f"HTTP {self.status_code}")


def _row(stem, extension="", cruise="HB1603", folder=_FOLDER):
    return {
        "CLOUD_PATH": f"{folder}{stem}{extension}",
        "FILE_NAME": f"{stem}.tar",
        "DATASET_NAME": f"{cruise}_EK60",
        "CRUISE_NAME": cruise,
        "TYPE_NAME": "WCSD Raw",
    }


def _archive(monkeypatch, listed, bucket=None):
    """Fake the catalogue holding *listed* and the bucket holding every file of _DAY."""
    calls = {"bucket": 0, "catalogue": []}
    bucket = _DAY if bucket is None else bucket
    monkeypatch.setattr(dr, "_LISTING_RETRY_SECONDS", 0, raising=False)

    def fake_fetch(params):
        calls["catalogue"].append(params)
        return [_row(stem) for stem in listed]

    def fake_get(url, params=None, timeout=None):
        calls["bucket"] += 1
        prefix = params["prefix"]
        keys = [f"{prefix}{stem}{ext}" for stem in bucket for ext in (".raw", ".bot", ".idx")]
        return _ListingResponse(_listing_xml(keys))

    monkeypatch.setattr(dr, "_fetch_all_pages", fake_fetch)
    monkeypatch.setattr(dr.requests, "get", fake_get)
    return calls


def _stems(result):
    return [u.rsplit("/", 1)[-1].removesuffix(".raw") for u in result["raw_urls"]]


def _query(**kwargs):
    kwargs.setdefault("filters", {"CRUISE_NAME": "HB1603"})
    kwargs.setdefault("file_time_start", "2016-08-14T14:30")
    kwargs.setdefault("file_time_end", "2016-08-14T15:30")
    return dr.query_ncei_data(**kwargs)


def test_list_bucket_follows_continuation_and_keeps_raw_files(monkeypatch):
    pages = [
        _listing_xml(["data/a/D1.raw", "data/a/D1.bot"], token="next+page"),
        _listing_xml(["data/a/D2.raw", "data/a/D2.idx"]),
    ]
    seen = []

    def fake_get(url, params=None, timeout=None):
        seen.append(dict(params))
        return _ListingResponse(pages[len(seen) - 1])

    monkeypatch.setattr(dr.requests, "get", fake_get)

    keys = dr._list_bucket_raw_files("s3://noaa-wcsd-pds/data/a/")

    assert keys == ["s3://noaa-wcsd-pds/data/a/D1", "s3://noaa-wcsd-pds/data/a/D2"]
    assert seen[0]["prefix"] == "data/a/"
    assert seen[0]["delimiter"] == "/"
    assert "continuation-token" not in seen[0]
    assert seen[1]["continuation-token"] == "next+page"


def test_auto_keeps_the_catalogues_result_when_it_is_complete(monkeypatch, capsys):
    _archive(monkeypatch, _DAY)

    expected = _query(listing="catalog")
    result = _query()

    assert result == expected
    assert _stems(result) == ["D20160814-T140000", "D20160814-T150000"]
    assert "Warning" not in capsys.readouterr().out


def test_auto_reads_the_listing_when_the_catalogue_is_short(monkeypatch, capsys):
    _archive(monkeypatch, ["D20160814-T130000", "D20160814-T170000"])

    result = _query()

    assert _stems(result) == ["D20160814-T140000", "D20160814-T150000"]
    assert "using the listing" in capsys.readouterr().out
    json.dumps(result)


def test_listing_gives_a_file_missing_from_the_catalogue_a_minimal_record(monkeypatch):
    _archive(monkeypatch, ["D20160814-T150000"])

    records = _query()["records"]

    assert records[0] == {
        "FILE_NAME": "D20160814-T140000.raw",
        "CLOUD_PATH": f"{_FOLDER}D20160814-T140000",
        "FILE_DATETIME": "2016-08-14T14:00:00",
        "DATASET_NAME": "HB1603_EK60",
        "CRUISE_NAME": "HB1603",
    }
    # A file the catalogue has keeps its full record.
    assert records[1]["TYPE_NAME"] == "WCSD Raw"


def test_listing_drops_the_neighbour_a_short_catalogue_overincludes(monkeypatch):
    # The catalogue lacks 14:00, so its 13:00 file seems to record until 15:00.
    _archive(monkeypatch, ["D20160814-T130000", "D20160814-T150000"])

    result = _query()

    assert "D20160814-T130000" not in _stems(result)


def test_listing_ignores_a_file_before_a_long_recording_gap(monkeypatch, capsys):
    # The catalogue's date clause leaves out the 10 August file, whose inferred
    # end would otherwise reach across the gap into the window.
    stems = ["D20160810-T120000", "D20160814-T150000", "D20160814-T160000"]
    _archive(monkeypatch, stems[1:], bucket=stems)

    result = _query()

    assert _stems(result) == ["D20160814-T150000"]
    assert "using the listing" not in capsys.readouterr().out


def test_file_names_apply_to_files_from_the_listing(monkeypatch):
    _archive(monkeypatch, ["D20160814-T150000"])

    result = _query(file_names=["D20160814-T140000.raw"])

    assert _stems(result) == ["D20160814-T140000"]


def test_a_per_file_filter_stays_on_the_catalogue_and_warns_it_is_short(monkeypatch, capsys):
    calls = _archive(monkeypatch, ["D20160814-T150000"])

    result = _query(filters={"CRUISE_NAME": "HB1603", "FREQUENCY": "38kHz"})

    assert _stems(result) == ["D20160814-T150000"]
    out = capsys.readouterr().out
    assert "1 of the 5 raw files" in out
    assert "FREQUENCY" in out
    # The folder lookup drops the per-file filter and the dates.
    undated = calls["catalogue"][-1]["where"]
    assert "CRUISE_NAME = 'HB1603'" in undated
    assert "FREQUENCY" not in undated
    assert "COLLECTION_DATE" not in undated


def test_a_per_file_filter_says_nothing_when_the_catalogue_is_complete(monkeypatch, capsys):
    _archive(monkeypatch, _DAY)

    _query(frequencies=["38kHz"])

    assert "may be missing files" not in capsys.readouterr().out


def test_a_query_naming_no_cruise_never_reads_the_archive(monkeypatch):
    calls = _archive(monkeypatch, ["D20160814-T150000"])

    _query(filters={"PLATFORM_NAME": "Henry B. Bigelow (HB)"})

    assert calls["bucket"] == 0
    assert len(calls["catalogue"]) == 1


def test_a_folder_stored_as_tar_stays_on_the_catalogue(monkeypatch, capsys):
    _archive(monkeypatch, ["D20160814-T150000"], bucket=[])

    result = _query()

    assert _stems(result) == ["D20160814-T150000"]
    assert "using the listing" not in capsys.readouterr().out


def test_catalog_mode_never_reads_the_archive(monkeypatch):
    calls = _archive(monkeypatch, ["D20160814-T150000"])

    result = _query(listing="catalog")

    assert _stems(result) == ["D20160814-T150000"]
    assert calls["bucket"] == 0


def test_bucket_mode_uses_the_listing(monkeypatch):
    _archive(monkeypatch, ["D20160814-T150000"])

    assert _stems(_query(listing="bucket")) == ["D20160814-T140000", "D20160814-T150000"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"filters": {"CRUISE_NAME": "HB1603", "FREQUENCY": "38kHz"}},
        {"frequencies": ["38kHz"]},
        {"calibration_state": 4},
        {"collection_start": "2016-08-14"},
        {"bbox": (-70, 38, -69, 39)},
        {"filters": {"PLATFORM_NAME": "Henry B. Bigelow (HB)"}},
    ],
)
def test_bucket_mode_refuses_what_a_listing_cannot_answer(monkeypatch, kwargs):
    _archive(monkeypatch, _DAY)

    with pytest.raises(ValueError, match="listing='bucket'"):
        _query(listing="bucket", **kwargs)


def test_bucket_mode_raises_when_the_archive_holds_no_raw_files(monkeypatch):
    _archive(monkeypatch, _DAY, bucket=[])

    with pytest.raises(ValueError, match="no raw files"):
        _query(listing="bucket")


def _failing_get(url, params=None, timeout=None):
    return _ListingResponse(b"", status_code=503)


def test_auto_fails_when_the_archive_cannot_be_read(monkeypatch):
    # A short list would be checkpointed under the step's normal hash.
    _archive(monkeypatch, ["D20160814-T150000"])
    monkeypatch.setattr(dr.requests, "get", _failing_get)

    with pytest.raises(RuntimeError, match="listing='catalog'"):
        _query()


def test_bucket_mode_fails_when_the_archive_cannot_be_read(monkeypatch):
    _archive(monkeypatch, ["D20160814-T150000"])
    monkeypatch.setattr(dr.requests, "get", _failing_get)

    with pytest.raises(RuntimeError, match="listing='catalog'"):
        _query(listing="bucket")


def test_catalog_mode_gets_through_an_archive_outage(monkeypatch):
    _archive(monkeypatch, ["D20160814-T150000"])
    monkeypatch.setattr(dr.requests, "get", _failing_get)

    assert _stems(_query(listing="catalog")) == ["D20160814-T150000"]


def test_a_failed_listing_page_is_tried_again(monkeypatch):
    _archive(monkeypatch, ["D20160814-T150000"])
    answers = [
        _ListingResponse(b"", status_code=503),
        _ListingResponse(b"<not a listing"),
        _ListingResponse(_listing_xml([f"data/raw/HB1603/EK60/{s}.raw" for s in _DAY])),
    ]
    monkeypatch.setattr(dr.requests, "get", lambda url, params=None, timeout=None: answers.pop(0))

    assert _stems(_query()) == ["D20160814-T140000", "D20160814-T150000"]
    assert answers == []


def test_a_per_file_filter_still_answers_when_the_archive_cannot_be_read(monkeypatch, capsys):
    # The comparison only informs a warning here, so it cannot shorten the result.
    _archive(monkeypatch, ["D20160814-T150000"])
    monkeypatch.setattr(dr.requests, "get", _failing_get)

    result = _query(frequencies=["38kHz"])

    assert _stems(result) == ["D20160814-T150000"]
    assert "could not compare" in capsys.readouterr().out


def test_one_catalogue_row_missing_from_the_archive_does_not_stop_the_fallback(monkeypatch, capsys):
    _archive(monkeypatch, ["D20160814-T150000", "D20160814-T145900"])

    result = _query()

    assert _stems(result) == ["D20160814-T140000", "D20160814-T150000"]
    out = capsys.readouterr().out
    assert "using the listing" in out
    assert "1 file(s) the catalogue lists are not in it" in out


def test_a_per_file_filter_warns_though_a_catalogue_row_is_missing_from_the_archive(monkeypatch, capsys):
    _archive(monkeypatch, ["D20160814-T150000", "D20160814-T145900"])

    _query(frequencies=["38kHz"])

    assert "1 of the 5 raw files" in capsys.readouterr().out


def test_a_folder_sharing_no_file_with_the_catalogue_stays_on_it_and_warns(monkeypatch, capsys):
    _archive(monkeypatch, ["D20160814-T150000"], bucket=["D20160901-T000000"])

    result = _query()

    assert _stems(result) == ["D20160814-T150000"]
    assert "share no files" in capsys.readouterr().out


def test_a_cruise_with_no_catalogue_rows_is_reported(monkeypatch, capsys):
    _archive(monkeypatch, [])

    result = _query()

    assert result["raw_urls"] == []
    assert "names no S3 folder" in capsys.readouterr().out


def test_bucket_mode_raises_for_a_cruise_with_no_catalogue_rows(monkeypatch):
    _archive(monkeypatch, [])

    with pytest.raises(ValueError, match="no S3 folder"):
        _query(listing="bucket")


def _failing_fetch(params):
    raise dr.requests.exceptions.ConnectionError("catalogue down")


@pytest.mark.parametrize("listing", ["auto", "bucket"])
def test_a_named_cruise_fails_when_the_catalogue_cannot_be_read(monkeypatch, listing):
    monkeypatch.setattr(dr, "_fetch_all_pages", _failing_fetch)

    with pytest.raises(RuntimeError, match="catalogue down"):
        _query(listing=listing)


def test_catalog_mode_keeps_its_empty_answer_when_the_catalogue_cannot_be_read(monkeypatch):
    monkeypatch.setattr(dr, "_fetch_all_pages", _failing_fetch)

    assert _query(listing="catalog")["raw_urls"] == []


def test_an_arcgis_error_in_a_good_response_counts_as_a_failed_read(monkeypatch):
    class Answer:
        def raise_for_status(self):
            pass

        def json(self):
            return {"error": {"code": 500, "message": "Unable to complete operation"}}

    monkeypatch.setattr(dr.requests, "get", lambda url, params=None, timeout=None: Answer())

    with pytest.raises(RuntimeError, match="Unable to complete"):
        _query()
    assert _query(listing="catalog")["raw_urls"] == []


def test_a_malformed_stamp_in_the_archive_is_passed_over(monkeypatch):
    _archive(monkeypatch, ["D20160814-T150000"], bucket=["D20160814-T150000", "D20161301-T000000"])

    assert _stems(_query()) == ["D20160814-T150000"]


def test_a_stamp_that_is_no_real_date_parses_to_none():
    assert dr.parse_datetime_from_filename("D20161301-T000000.raw") is None
    assert dr.parse_datetime_from_filename("D20160814-T150000.raw") == datetime(2016, 8, 14, 15)


def test_a_query_without_filters_keeps_its_label(monkeypatch):
    monkeypatch.setattr(dr, "_fetch_all_pages", lambda params: [])
    kwargs = {
        "dataset_name": "HB1603_EK60",
        "file_time_start": "2016-08-14T14:30",
        "file_time_end": "2016-08-15T15:30",
    }

    result = dr.query_ncei_data(listing="catalog", **kwargs)

    assert result["query_label"] == dr.build_query_label(**kwargs)


def test_an_unknown_listing_mode_is_refused():
    with pytest.raises(ValueError, match="listing must be one of"):
        dr.query_ncei_data(listing="s3")
