import os

import numpy as np
import pandas as pd
import pytest

import pywrdrb
from pywrdrb.path_manager import get_pn_object
from pywrdrb.utils.results_sets import obs_results_set_opts

NYC_RESERVOIRS = ["cannonsville", "pepacton", "neversink"]
DRBC_START = pd.Timestamp("1999-12-01")
DRBC_END = pd.Timestamp("2021-11-30")


def _load_obs_storage_df():
    """Helper: load merged res_storage observations as a single DataFrame."""
    data = pywrdrb.Data(results_sets=["res_storage"], print_status=False)
    data.load_observations()
    df = data.res_storage["obs"][0]
    df.index = pd.to_datetime(df.index)
    return df


def test_data_load_observations():

    data = pywrdrb.Data(results_sets=obs_results_set_opts,
                        print_status=True)
    data.load_observations()

    # Make sure that each results_set is stored as an attribute
    # of the data object
    for results_set in obs_results_set_opts:
        assert hasattr(data, results_set)

        # Make sure that the 'obs' key exists and is a list
        results_data = getattr(data, results_set)
        assert isinstance(results_data, dict)
        assert "obs" in results_data
        assert isinstance(results_data["obs"], dict)

        # Make sure the scenario of 'obs' is a DataFrame
        assert isinstance(results_data["obs"][0], pd.DataFrame)


def test_nyc_storage_extends_back_to_drbc_start():
    """NYC reservoirs should have non-null observations at or before 1999-12-01
    after the DRBC merge."""
    df = _load_obs_storage_df()
    for r in NYC_RESERVOIRS:
        assert r in df.columns, f"missing NYC column {r}"
        first_valid = df[r].first_valid_index()
        assert first_valid is not None, f"{r} has no observations"
        assert first_valid <= DRBC_START, (
            f"{r} first valid index {first_valid} is later than DRBC start {DRBC_START}"
        )


def test_nyc_storage_matches_drbc_first_row():
    """First-row DRBC values must survive into the merged loaded data."""
    df = _load_obs_storage_df()
    expected = {
        "pepacton": 100636.0,
        "cannonsville": 60488.0,
        "neversink": 15968.0,
    }
    for r, val in expected.items():
        assert df.loc[DRBC_START, r] == pytest.approx(val, abs=1.0), (
            f"{r} on {DRBC_START.date()} = {df.loc[DRBC_START, r]}, expected {val}"
        )


def test_nyc_storage_magnitude_bounds():
    """Sanity check that no NYC values are negative or absurdly large.
    Reservoir capacities (approx, MG): cannonsville 96k, pepacton 140k,
    neversink 35k. Bound at 200k MG to catch unit errors."""
    df = _load_obs_storage_df()
    for r in NYC_RESERVOIRS:
        col = df[r].dropna()
        assert (col >= 0).all(), f"{r} has negative values"
        assert (col <= 200_000).all(), f"{r} has values > 200,000 MG (unit error?)"


def test_nyc_storage_splice_continuity():
    """The 2021-11-30 -> 2021-12-01 splice (DRBC -> USGS) should not introduce
    a > 10% jump for any NYC reservoir."""
    df = _load_obs_storage_df()
    for r in NYC_RESERVOIRS:
        v_pre = df.loc[DRBC_END, r]
        v_post = df.loc[DRBC_END + pd.Timedelta(days=1), r]
        assert pd.notna(v_pre) and pd.notna(v_post), (
            f"{r} splice values missing: pre={v_pre}, post={v_post}"
        )
        rel = abs(v_pre - v_post) / v_pre
        assert rel < 0.10, (
            f"{r} splice jump {rel:.2%} > 10% (pre={v_pre}, post={v_post}). "
            "DRBC vs USGS may be diverging."
        )


def test_prompton_pre_1990_storage_dropped():
    """USGS 01428900 has anomalous prompton elevation 1986-1990 followed by a
    long gap; per ``STORAGE_VALID_FROM`` we drop everything before 1990."""
    df = _load_obs_storage_df()
    pre_1990 = df.loc[df.index < pd.Timestamp("1990-01-01"), "prompton"]
    assert pre_1990.notna().sum() == 0, (
        f"prompton has {pre_1990.notna().sum()} non-null values before 1990-01-01; "
        "STORAGE_VALID_FROM should have dropped them."
    )


def test_storage_index_well_formed():
    df = _load_obs_storage_df()
    assert df.index.is_monotonic_increasing
    assert df.index.is_unique


def test_usgs_nyc_storage_audit_csv_exists():
    """The pre-merge USGS NYC storage should be persisted to _raw/ for audit."""
    pn = get_pn_object()
    raw_dir = pn.observations.get_str() + os.sep + "_raw"
    audit_path = os.path.join(raw_dir, "usgs_nyc_storage_mg.csv")
    if not os.path.exists(audit_path):
        pytest.skip(
            "usgs_nyc_storage_mg.csv not yet generated; "
            "run `python -m pywrdrb.pre.obs_data_retrieval` to produce it."
        )
    audit = pd.read_csv(audit_path, index_col=0, parse_dates=True)
    for r in NYC_RESERVOIRS:
        assert r in audit.columns, f"audit CSV missing {r}"
