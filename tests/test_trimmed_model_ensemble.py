"""
Tests for ensemble-mode trimmed-model integration.

Covers two pieces:
- ModelBuilder routing: when use_trimmed_model=True AND inflow_ensemble_indices
  is set, the trimmed-mode release parameter must be emitted as
  PresimulatedReleaseEnsemble (not the single-trace `dataframe` type).
- Validation: missing ensemble HDF5 raises a clear FileNotFoundError pointing
  the user at STARFITReleaseEnsemblePreprocessor.
"""
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from pywrdrb import ModelBuilder
from pywrdrb.utils.lists import (
    independent_starfit_reservoirs,
    starfit_reservoir_list,
)


def _write_synthetic_presim_hdf5(path, start_date, end_date, realizations):
    """Write a node-first STARFIT release HDF5 + JSON sidecar that satisfies
    _validate_trimmed_model_config in ensemble mode."""
    dates = pd.date_range(start_date, end_date)
    str_dtype = h5py.string_dtype(encoding="utf-8")
    date_strings = np.asarray(
        [d.strftime("%Y-%m-%d") for d in dates], dtype=object
    )
    column_labels = np.asarray([str(r) for r in realizations], dtype=object)
    rng = np.random.default_rng(seed=3)

    with h5py.File(path, "w") as hf:
        for reservoir in starfit_reservoir_list:
            grp = hf.create_group(reservoir)
            grp.attrs.create(
                "column_labels", column_labels, dtype=str_dtype
            )
            for rid in realizations:
                grp.create_dataset(
                    str(rid),
                    data=rng.uniform(5, 50, size=len(dates)).astype(np.float64),
                )
            grp.create_dataset("date", data=date_strings, dtype=str_dtype)

    sidecar = str(path).replace(".hdf5", "_metadata.json")
    with open(sidecar, "w") as f:
        json.dump(
            {
                "inflow_type": "nhmv10_withObsScaled",
                "start_date": str(dates[0].date()),
                "end_date": str(dates[-1].date()),
                "reservoirs": list(starfit_reservoir_list),
                "realization_ids": [str(r) for r in realizations],
                "initial_volume_frac": 0.8,
                "source": "STARFITReleaseEnsemblePreprocessor",
            },
            f,
        )


def test_ensemble_trimmed_emits_presimulated_release_ensemble_parameter(tmp_path):
    """When use_trimmed_model=True AND inflow_ensemble_indices is set, the
    release parameters must dispatch to PresimulatedReleaseEnsemble."""
    realizations = [0, 1]
    presim_path = tmp_path / "presimulated_releases_mgd.hdf5"
    _write_synthetic_presim_hdf5(
        presim_path, "1999-12-01", "2001-01-31", realizations
    )

    builder = ModelBuilder(
        start_date="2000-01-01",
        end_date="2000-03-31",
        inflow_type="nhmv10_withObsScaled",
        options={
            "use_trimmed_model": True,
            "presimulated_releases_file": str(presim_path),
            "inflow_ensemble_indices": realizations,
        },
    )
    builder.make_model()

    presim_params = {
        name: p
        for name, p in builder.model_dict["parameters"].items()
        if isinstance(p, dict)
        and name.startswith("presimulated_release_")
    }
    assert presim_params, "no presimulated_release_* parameters emitted"
    # One presim parameter per independent STARFIT reservoir.
    assert len(presim_params) == len(independent_starfit_reservoirs)

    for name, p in presim_params.items():
        assert p["type"] == "PresimulatedReleaseEnsemble", (
            f"{name} emitted type {p['type']!r} instead of "
            "PresimulatedReleaseEnsemble"
        )
        assert p["inflow_ensemble_indices"] == realizations
        # Must reference the HDF5, not the CSV.
        assert p["presim_filename"].endswith(".hdf5")
        # Reservoir name must match the parameter key suffix.
        suffix = name.removeprefix("presimulated_release_")
        assert p["node"] == suffix
        assert suffix in independent_starfit_reservoirs


def test_single_trace_trimmed_path_unchanged_without_ensemble(tmp_path):
    """When inflow_ensemble_indices is None we keep emitting type=dataframe."""
    csv_path = tmp_path / "presimulated_releases_mgd.csv"
    # Write a CSV covering the simulation period with one column per reservoir.
    dates = pd.date_range("1999-12-01", "2001-01-31")
    df = pd.DataFrame(
        {res: 10.0 for res in starfit_reservoir_list},
        index=dates,
    )
    df.index.name = "datetime"
    df.to_csv(csv_path)

    sidecar = tmp_path / "presimulated_releases_mgd_metadata.json"
    with open(sidecar, "w") as f:
        json.dump(
            {
                "inflow_type": "nhmv10_withObsScaled",
                "start_date": str(dates[0].date()),
                "end_date": str(dates[-1].date()),
                "reservoirs": list(starfit_reservoir_list),
                "initial_volume_frac": 0.8,
                "source": "STARFITOfflineSimulator",
            },
            f,
        )

    builder = ModelBuilder(
        start_date="2000-01-01",
        end_date="2000-03-31",
        inflow_type="nhmv10_withObsScaled",
        options={
            "use_trimmed_model": True,
            "presimulated_releases_file": str(csv_path),
        },
    )
    builder.make_model()

    presim_params = [
        p
        for name, p in builder.model_dict["parameters"].items()
        if isinstance(p, dict)
        and name.startswith("presimulated_release_")
    ]
    assert presim_params
    for p in presim_params:
        assert p["type"] == "dataframe"


def test_ensemble_trimmed_missing_hdf5_error_message():
    """The validation error in ensemble mode must point at
    STARFITReleaseEnsemblePreprocessor (not STARFITOfflineSimulator)."""
    fake_path = "/no/such/dir/presimulated_releases_mgd.hdf5"
    with pytest.raises(FileNotFoundError) as exc_info:
        ModelBuilder(
            start_date="2000-01-01",
            end_date="2000-03-31",
            inflow_type="nhmv10_withObsScaled",
            options={
                "use_trimmed_model": True,
                "presimulated_releases_file": fake_path,
                "inflow_ensemble_indices": [0, 1],
            },
        )
    msg = str(exc_info.value)
    assert "STARFITReleaseEnsemblePreprocessor" in msg
    assert fake_path in msg


def test_ensemble_trimmed_missing_realization_error(tmp_path):
    """If a requested realization isn't in the artifact metadata, validation
    must raise ValueError naming the missing IDs."""
    presim_path = tmp_path / "presimulated_releases_mgd.hdf5"
    _write_synthetic_presim_hdf5(
        presim_path, "1999-12-01", "2001-01-31", realizations=[0, 1]
    )

    with pytest.raises(ValueError, match="missing realization"):
        ModelBuilder(
            start_date="2000-01-01",
            end_date="2000-03-31",
            inflow_type="nhmv10_withObsScaled",
            options={
                "use_trimmed_model": True,
                "presimulated_releases_file": str(presim_path),
                "inflow_ensemble_indices": [0, 1, 5],  # 5 is not in the file
            },
        )
