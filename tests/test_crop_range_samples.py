# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Unit tests for trailing range-sample cropping."""

import numpy as np
import pytest
import xarray as xr

from aa_si_utils import utils


def _make_ds_sv(n_range=10, deepest_with_data=5, n_channels=2, n_pings=4):
    """Sv dataset where range samples past *deepest_with_data* are all NaN."""
    echo_range = np.broadcast_to(
        np.arange(n_range, dtype=float) * 2.0, (n_channels, n_pings, n_range)
    ).copy()
    sv = np.full((n_channels, n_pings, n_range), np.nan)
    sv[:, :, : deepest_with_data + 1] = 1.0
    return xr.Dataset(
        data_vars={
            "Sv": (("channel", "ping_time", "range_sample"), sv),
            "echo_range": (("channel", "ping_time", "range_sample"), echo_range),
        },
        coords={
            "channel": np.array([f"ch{i}" for i in range(n_channels)]),
            "ping_time": np.datetime64("2024-01-01T00:00:00")
            + np.arange(n_pings, dtype="timedelta64[s]"),
            "range_sample": np.arange(n_range),
        },
    )


def test_crops_to_deepest_finite_sample():
    ds = _make_ds_sv(n_range=10, deepest_with_data=5)
    out = utils.crop_range_samples(ds)
    assert out.sizes["range_sample"] == 6


def test_crop_is_lossless():
    ds = _make_ds_sv(n_range=10, deepest_with_data=5)
    out = utils.crop_range_samples(ds)
    assert int(np.isfinite(out["Sv"]).sum()) == int(np.isfinite(ds["Sv"]).sum())


def test_one_deep_finite_sample_holds_the_crop():
    """A single finite value at depth must keep the whole axis: the crop is
    driven by the deepest sample holding data anywhere, not by most pings."""
    ds = _make_ds_sv(n_range=10, deepest_with_data=2)
    ds["Sv"].values[0, 0, 9] = 1.0
    out = utils.crop_range_samples(ds)
    assert out.sizes["range_sample"] == 10


def test_nothing_to_crop_returns_input_unchanged():
    ds = _make_ds_sv(n_range=10, deepest_with_data=9)
    out = utils.crop_range_samples(ds)
    assert out.sizes["range_sample"] == 10


def test_max_range_m_caps_at_depth():
    ds = _make_ds_sv(n_range=10, deepest_with_data=9)
    # echo_range is 0, 2, 4, ... so a 7 m cap keeps samples at 0-6 m.
    out = utils.crop_range_samples(ds, max_range_m=7.0)
    assert out.sizes["range_sample"] == 4
    assert float(out["echo_range"].values[..., -1].max()) == 6.0


def test_max_range_m_wins_when_it_is_the_shallower_cut():
    ds = _make_ds_sv(n_range=10, deepest_with_data=9)
    out = utils.crop_range_samples(ds, max_range_m=5.0, drop_trailing_all_nan=True)
    assert out.sizes["range_sample"] == 3
    assert int(np.isfinite(out["Sv"]).sum()) < int(np.isfinite(ds["Sv"]).sum())


def test_both_modes_off_is_a_no_op():
    ds = _make_ds_sv(n_range=10, deepest_with_data=5)
    out = utils.crop_range_samples(ds, drop_trailing_all_nan=False)
    assert out.sizes["range_sample"] == 10


def test_all_nan_raises():
    ds = _make_ds_sv(n_range=10, deepest_with_data=5)
    ds["Sv"].values[:] = np.nan
    with pytest.raises(ValueError, match="no finite values"):
        utils.crop_range_samples(ds)


def test_max_range_below_every_sample_raises():
    ds = _make_ds_sv(n_range=10, deepest_with_data=5)
    ds["echo_range"].values[:] += 100.0
    with pytest.raises(ValueError, match="excludes every range sample"):
        utils.crop_range_samples(ds, max_range_m=1.0)


def test_dask_input_stays_lazy():
    ds = _make_ds_sv(n_range=10, deepest_with_data=5).chunk({"ping_time": 2})
    out = utils.crop_range_samples(ds)
    assert out["Sv"].chunks is not None
    assert out.sizes["range_sample"] == 6


def test_dask_and_numpy_agree():
    ds = _make_ds_sv(n_range=12, deepest_with_data=7)
    eager = utils.crop_range_samples(ds)
    lazy = utils.crop_range_samples(ds.chunk({"ping_time": 2}))
    assert eager.sizes["range_sample"] == lazy.sizes["range_sample"]


def _make_ds_with_outlier(n_range=40, normal_deepest=10, outlier_deepest=35,
                          n_pings=200, n_outliers=1, jitter=2, seed=0):
    """Most pings reach *normal_deepest* (plus jitter); a few reach far deeper."""
    rng = np.random.default_rng(seed)
    ds = _make_ds_sv(n_range=n_range, deepest_with_data=0, n_channels=2,
                     n_pings=n_pings)
    sv = np.full((2, n_pings, n_range), np.nan)
    for p in range(n_pings):
        last = normal_deepest + int(rng.integers(-jitter, jitter + 1))
        sv[:, p, : last + 1] = 1.0
    for p in range(n_outliers):
        sv[:, p, : outlier_deepest + 1] = 1.0
    ds["Sv"] = (("channel", "ping_time", "range_sample"), sv)
    return ds


def test_outlier_rejection_crops_to_deepest_retained_ping():
    ds = _make_ds_with_outlier(n_range=40, normal_deepest=10, outlier_deepest=35)
    out = utils.crop_range_samples(ds, outlier_sigma=3.0, outlier_margin=1.5)
    # The retained pings reach at most normal_deepest + jitter = 12.
    assert out.sizes["range_sample"] == 13
    # Lossless mode would have been held open by the single outlier.
    assert utils.crop_range_samples(ds).sizes["range_sample"] == 36


def test_outlier_rejection_keeps_uncommon_but_real_depths():
    """A minority of genuinely deeper pings must survive; only wild outliers go."""
    ds = _make_ds_with_outlier(n_range=60, normal_deepest=10, outlier_deepest=55,
                               n_pings=200, n_outliers=1)
    # 20 pings (10%) legitimately reach 25.
    ds["Sv"].values[:, 100:120, : 26] = 1.0
    out = utils.crop_range_samples(ds, outlier_sigma=3.0, outlier_margin=1.5)
    assert out.sizes["range_sample"] == 26


def test_outlier_margin_controls_sensitivity():
    ds = _make_ds_with_outlier(n_range=40, normal_deepest=10, outlier_deepest=35)
    loose = utils.crop_range_samples(ds, outlier_sigma=3.0, outlier_margin=100.0)
    tight = utils.crop_range_samples(ds, outlier_sigma=3.0, outlier_margin=1.5)
    assert loose.sizes["range_sample"] > tight.sizes["range_sample"]


def test_outlier_rejection_reports_flagged_pings(capsys):
    ds = _make_ds_with_outlier(n_range=40, normal_deepest=10, outlier_deepest=35,
                               n_outliers=3)
    utils.crop_range_samples(ds, outlier_sigma=3.0, outlier_margin=1.5)
    out = capsys.readouterr().out
    assert "flagged 3 of 200 pings" in out
    assert "dropped" in out and "finite sample" in out


def test_outlier_rejection_takes_precedence_over_lossless():
    ds = _make_ds_with_outlier(n_range=40, normal_deepest=10, outlier_deepest=35)
    out = utils.crop_range_samples(ds, outlier_sigma=3.0,
                                   drop_trailing_all_nan=True)
    assert out.sizes["range_sample"] == 13


def test_max_range_m_composes_as_a_ceiling():
    """Both criteria active: the shallower cut wins, either way round."""
    ds = _make_ds_with_outlier(n_range=40, normal_deepest=10, outlier_deepest=35)
    # echo_range is 0, 2, 4, ...; a 10 m ceiling keeps samples 0-5.
    tighter_cap = utils.crop_range_samples(ds, outlier_sigma=3.0, max_range_m=10.0)
    assert tighter_cap.sizes["range_sample"] == 6
    looser_cap = utils.crop_range_samples(ds, outlier_sigma=3.0, max_range_m=200.0)
    assert looser_cap.sizes["range_sample"] == 13


def test_outlier_rejection_on_dask_matches_numpy():
    ds = _make_ds_with_outlier(n_range=40, normal_deepest=10, outlier_deepest=35)
    eager = utils.crop_range_samples(ds, outlier_sigma=3.0)
    lazy = utils.crop_range_samples(ds.chunk({"ping_time": 32}), outlier_sigma=3.0)
    assert eager.sizes["range_sample"] == lazy.sizes["range_sample"]
    assert lazy["Sv"].chunks is not None


def test_all_nan_pings_do_not_drag_the_statistics():
    """A ping with no finite sample must not count as 'deepest == 0'."""
    ds = _make_ds_with_outlier(n_range=40, normal_deepest=10, outlier_deepest=35)
    ds["Sv"].values[:, 50:150, :] = np.nan
    out = utils.crop_range_samples(ds, outlier_sigma=3.0, outlier_margin=1.5)
    assert out.sizes["range_sample"] == 13


def test_outlier_rejection_flagging_everything_raises():
    ds = _make_ds_with_outlier(n_range=40, normal_deepest=10, outlier_deepest=35)
    with pytest.raises(ValueError, match="flags every ping"):
        utils.crop_range_samples(ds, outlier_sigma=-100.0, outlier_margin=1.5)


def test_dataset_without_range_sample_is_returned_unchanged():
    ds = xr.Dataset({"Sv": (("ping_time",), np.array([1.0, 2.0]))})
    assert utils.crop_range_samples(ds) is ds


def test_untouched_variables_survive_the_slice():
    ds = _make_ds_sv(n_range=10, deepest_with_data=5)
    ds["frequency_nominal"] = (("channel",), np.array([38000.0, 120000.0]))
    out = utils.crop_range_samples(ds)
    np.testing.assert_array_equal(
        out["frequency_nominal"].values, ds["frequency_nominal"].values
    )
