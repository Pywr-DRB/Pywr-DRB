from pathlib import Path

import pandas as pd

from scripts.compare_starfit_parametric import (
    _build_diagnostics_timeseries,
    _plot_reservoir_diagnostics,
    _summarize_release_differences,
)


def test_diagnostics_timeseries_and_summary_columns():
    idx = pd.date_range("2001-01-01", periods=5, freq="D")
    legacy = pd.DataFrame(
        {
            "fewalter": [10, 20, 30, 40, 50],
            "blueMarsh": [11, 21, 31, 41, 51],
        },
        index=idx,
    )
    parametric = pd.DataFrame(
        {
            "fewalter": [10, 22, 31, 41, 52],
            "blueMarsh": [10, 20, 32, 42, 52],
        },
        index=idx,
    )
    reservoirs = ["fewalter", "blueMarsh"]

    summary = _summarize_release_differences(legacy, parametric, reservoirs)
    ts = _build_diagnostics_timeseries(legacy, parametric, reservoirs)

    expected_summary_cols = {
        "reservoir",
        "n_steps",
        "mean_legacy_release_mgd",
        "mean_parametric_release_mgd",
        "bias_mgd_param_minus_legacy",
        "mean_abs_diff_mgd",
        "max_abs_diff_mgd",
        "rmse_mgd",
    }
    assert expected_summary_cols.issubset(set(summary.columns))
    assert "fewalter_diff_mgd" in ts.columns
    assert "blueMarsh_parametric_mgd" in ts.columns
    assert abs(ts["fewalter_diff_mgd"].iloc[1] - 2.0) < 1e-12


def test_plot_reservoir_diagnostics_writes_file(tmp_path: Path):
    idx = pd.date_range("2001-01-01", periods=6, freq="D")
    ts = pd.DataFrame(
        {
            "fewalter_legacy_mgd": [10, 12, 14, 13, 15, 16],
            "fewalter_parametric_mgd": [10, 11, 15, 13, 16, 16],
            "fewalter_diff_mgd": [0, -1, 1, 0, 1, 0],
        },
        index=idx,
    )
    out = tmp_path / "fewalter_plot.png"
    _plot_reservoir_diagnostics(ts, "fewalter", out)
    assert out.exists()
    assert out.stat().st_size > 0
