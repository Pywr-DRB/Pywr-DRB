"""
Tests for the "eager" STARFIT-release consumption in PredictedInflowEnsemblePreprocessor.

perfect_foresight no longer recomputes STARFIT inline — it requires
``presimulated_releases_mgd.hdf5`` produced by STARFITReleaseEnsemblePreprocessor.
These tests verify:
  * the FileNotFoundError message points at STARFITReleaseEnsemblePreprocessor;
  * with the artifact present, load() preloads per-realization release frames
    and process()'s starfit_releases attribute matches the HDF5 columns.
"""
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from pywrdrb.pre import predict_inflows
from pywrdrb.pre.predict_inflows import PredictedInflowEnsemblePreprocessor
from pywrdrb.utils.lists import starfit_reservoir_list


N_DAYS = 10
REALIZATIONS = ["0", "1"]


@pytest.fixture
def synthetic_inflow_hdf5(tmp_path):
    """
    Build a tiny ensemble inflow HDF5 with every node referenced by
    ``pywrdrb_all_nodes`` so PredictedInflowEnsemblePreprocessor.load()
    can extract a realization slice.
    """
    from pywrdrb.pre.predict_inflows import pywrdrb_all_nodes

    str_dtype = h5py.string_dtype(encoding="utf-8")
    dates = (
        pd.date_range("2020-01-01", periods=N_DAYS)
        .strftime("%Y-%m-%d")
        .tolist()
    )
    rng = np.random.default_rng(seed=5)

    flows_dir = tmp_path / "flows" / "test_inflow"
    flows_dir.mkdir(parents=True)
    inflow_path = flows_dir / "catchment_inflow_mgd.hdf5"

    with h5py.File(inflow_path, "w") as hf:
        for node in pywrdrb_all_nodes:
            grp = hf.create_group(node)
            grp.attrs.create(
                "column_labels",
                np.asarray(REALIZATIONS, dtype=object),
                dtype=str_dtype,
            )
            for rid in REALIZATIONS:
                grp.create_dataset(
                    rid, data=rng.uniform(10, 100, size=N_DAYS).astype(np.float64)
                )
            grp.create_dataset(
                "date",
                data=np.asarray(dates, dtype=object),
                dtype=str_dtype,
            )

    return {"flows_dir": flows_dir, "inflow_path": inflow_path, "dates": dates}


def _write_synthetic_release_hdf5(flows_dir, dates):
    """Write a node-first STARFIT release HDF5 covering every reservoir."""
    str_dtype = h5py.string_dtype(encoding="utf-8")
    date_strings = np.asarray(dates, dtype=object)
    column_labels = np.asarray(REALIZATIONS, dtype=object)
    rng = np.random.default_rng(seed=17)
    release_path = flows_dir / "presimulated_releases_mgd.hdf5"

    fixture = {rid: {} for rid in REALIZATIONS}
    with h5py.File(release_path, "w") as hf:
        for reservoir in starfit_reservoir_list:
            grp = hf.create_group(reservoir)
            grp.attrs.create("column_labels", column_labels, dtype=str_dtype)
            for rid in REALIZATIONS:
                arr = rng.uniform(1, 25, size=len(dates)).astype(np.float64)
                grp.create_dataset(rid, data=arr)
                fixture[rid][reservoir] = arr
            grp.create_dataset("date", data=date_strings, dtype=str_dtype)

    return release_path, fixture


class _StubFlowsPathNavigator:
    """Stand-in for the parts of pn that PredictedInflowEnsemblePreprocessor uses."""

    def __init__(self, flows_dir):
        self._flows_dir = Path(flows_dir)
        self.flows = self  # so pn.flows.get_str(...) reaches our get_str
        self.sc = self  # so pn.sc.get(...) reaches our get
        self.catchment_withdrawals = self  # for sw_avg_wateruse_pywrdrb_catchments_mgd.csv

    def get_str(self, key):
        return str(self._flows_dir)

    def get(self, key):
        # Both pn.sc.get('flows/<type>') and pn.catchment_withdrawals.get(<csv>)
        # land here. For the consumption CSV, return a real package file.
        if isinstance(key, str) and key.endswith(".csv"):
            from pywrdrb.path_manager import get_pn_object

            return get_pn_object().catchment_withdrawals.get(key)
        return self._flows_dir


def _patch_pn(monkeypatch, flows_dir):
    monkeypatch.setattr(
        predict_inflows, "pn", _StubFlowsPathNavigator(flows_dir),
        raising=False,
    )


def test_ensemble_pf_raises_when_artifact_missing(synthetic_inflow_hdf5, monkeypatch):
    """perfect_foresight + missing presimulated_releases_mgd.hdf5 must raise
    FileNotFoundError pointing at STARFITReleaseEnsemblePreprocessor."""
    flows_dir = synthetic_inflow_hdf5["flows_dir"]

    # The preprocessor uses self.pn (set in DataPreprocessor base via get_pn_object).
    # We can't intercept that easily, so we rely on the real pn for the consumption
    # CSV, and override only the per-instance flow_type lookup. The simplest path
    # is to instantiate, override the path attributes, and call load().
    pp = PredictedInflowEnsemblePreprocessor(
        flow_type="nhmv10",
        ensemble_hdf5_file=str(synthetic_inflow_hdf5["inflow_path"]),
        realization_ids=[int(r) for r in REALIZATIONS],
        modes=("perfect_foresight",),
    )
    # Reroute the only flows-dir lookup that happens inside load() — the eager
    # presim HDF5 path — to our tmp dir (where we have NOT written the file).
    pp.pn = _StubFlowsPathNavigator(flows_dir)

    with pytest.raises(FileNotFoundError) as exc_info:
        pp.load()
    assert "STARFITReleaseEnsemblePreprocessor" in str(exc_info.value)
    assert "perfect_foresight" in str(exc_info.value)


def test_ensemble_pf_loads_releases_when_artifact_present(
    synthetic_inflow_hdf5, monkeypatch
):
    """When the release artifact exists, load() must populate
    _starfit_release_data with one DataFrame per realization keyed by str(rid)."""
    flows_dir = synthetic_inflow_hdf5["flows_dir"]
    dates = synthetic_inflow_hdf5["dates"]
    release_path, fixture = _write_synthetic_release_hdf5(flows_dir, dates)

    pp = PredictedInflowEnsemblePreprocessor(
        flow_type="nhmv10",
        ensemble_hdf5_file=str(synthetic_inflow_hdf5["inflow_path"]),
        realization_ids=[int(r) for r in REALIZATIONS],
        modes=("perfect_foresight",),
    )
    pp.pn = _StubFlowsPathNavigator(flows_dir)
    pp.load()

    # Both realizations must be preloaded.
    assert set(pp._starfit_release_data.keys()) == set(REALIZATIONS)

    for rid in REALIZATIONS:
        df = pp._starfit_release_data[rid]
        # Every STARFIT reservoir is present as a column.
        for reservoir in starfit_reservoir_list:
            assert reservoir in df.columns
            np.testing.assert_allclose(
                df[reservoir].to_numpy(),
                fixture[rid][reservoir],
                rtol=0,
                atol=0,
            )
        # Index is the canonical date axis.
        assert list(df.index.strftime("%Y-%m-%d")) == dates


def test_ensemble_pf_raises_when_realization_missing(
    synthetic_inflow_hdf5,
):
    """If a requested realization is not in the artifact's column_labels,
    load() must raise a clear ValueError."""
    flows_dir = synthetic_inflow_hdf5["flows_dir"]
    dates = synthetic_inflow_hdf5["dates"]
    _write_synthetic_release_hdf5(flows_dir, dates)

    # Request realization "5" which is not in REALIZATIONS=["0","1"]. The
    # inflow HDF5 also doesn't contain "5", so the inflow load step would
    # fail first; we instead request "0" and a missing one to pass inflow
    # extraction up to the STARFIT validation. Adjust by writing a third
    # realization to the inflow file.
    # Simpler: request only "0" but doctor the release file to drop it.
    # Cleanest: write the release artifact without "1" but request both.
    str_dtype = h5py.string_dtype(encoding="utf-8")
    release_path = flows_dir / "presimulated_releases_mgd.hdf5"
    with h5py.File(release_path, "w") as hf:
        # Only realization "0" present.
        for reservoir in starfit_reservoir_list:
            grp = hf.create_group(reservoir)
            grp.attrs.create(
                "column_labels",
                np.asarray(["0"], dtype=object),
                dtype=str_dtype,
            )
            grp.create_dataset(
                "0", data=np.ones(len(dates), dtype=np.float64) * 5.0
            )
            grp.create_dataset(
                "date",
                data=np.asarray(dates, dtype=object),
                dtype=str_dtype,
            )

    pp = PredictedInflowEnsemblePreprocessor(
        flow_type="nhmv10",
        ensemble_hdf5_file=str(synthetic_inflow_hdf5["inflow_path"]),
        realization_ids=[0, 1],
        modes=("perfect_foresight",),
    )
    pp.pn = _StubFlowsPathNavigator(flows_dir)

    with pytest.raises(ValueError, match="not present in"):
        pp.load()
