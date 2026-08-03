"""
Tests for STARFITReleaseEnsemblePreprocessor and PresimulatedReleaseEnsemble.

Builds a small synthetic catchment-inflow HDF5 in tmp_path matching the
node-first schema produced by the upstream stochastic-inflow generator, runs
the ensemble STARFIT preprocessor with use_mpi=False, and verifies:
  * output schema (node-first, column_labels, date axis, metadata sidecar)
  * per-realization equivalence vs. STARFITOfflineSimulator on the same inputs
  * idempotency / force-rebuild
  * subset realizations
  * PresimulatedReleaseEnsemble.value() round-trips against the artifact
"""
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from pywrdrb.parameters import ensemble as ensemble_module
from pywrdrb.pre.generate_presimulated_releases import (
    OUTPUT_HDF5_NAME,
    OUTPUT_METADATA_NAME,
    INPUT_HDF5_NAME,
    STARFITOfflineSimulator,
    STARFITReleaseEnsemblePreprocessor,
)
from pywrdrb.utils.lists import starfit_reservoir_list


# A small donor list: every STARFIT reservoir must appear as a column in the
# synthetic catchment-inflow HDF5 because simulate_all() validates presence.
SYNTHETIC_RESERVOIRS = list(starfit_reservoir_list)

# A handful of additional non-reservoir nodes that show up in the real input
# HDF5; included so we exercise the "ignore extra groups" code path.
EXTRA_NODES = ["delLordville", "delMontague", "delTrenton"]


@pytest.fixture
def synthetic_inflow_hdf5(tmp_path):
    """
    Write a tiny ensemble inflow HDF5 in tmp_path with the node-first schema:
        /<node>/.attrs['column_labels'] = [b'0', b'1', b'2']
        /<node>/<rid>: 1D float array
        /<node>/date:  1D string array
    """
    n_days = 60  # ~2 months — enough to cross one harmonic cycle slice
    realizations = ["0", "1", "2"]
    dates = (
        pd.date_range("2020-01-01", periods=n_days)
        .strftime("%Y-%m-%d")
        .tolist()
    )

    rng = np.random.default_rng(seed=11)
    str_dtype = h5py.string_dtype(encoding="utf-8")

    flows_dir = tmp_path / "flows" / "test_inflow"
    flows_dir.mkdir(parents=True)
    input_path = flows_dir / INPUT_HDF5_NAME

    # Per-realization, per-node fixture so tests can rebuild the same inputs
    # and run the engine inline for equivalence assertions.
    fixture = {rid: {} for rid in realizations}

    with h5py.File(input_path, "w") as hf:
        for node in SYNTHETIC_RESERVOIRS + EXTRA_NODES:
            grp = hf.create_group(node)
            grp.attrs.create(
                "column_labels",
                np.asarray(realizations, dtype=object),
                dtype=str_dtype,
            )
            for rid in realizations:
                # Order-of-magnitude inflows (MGD); exact values don't matter
                # so long as they're positive and similar to real DRB scales.
                arr = rng.uniform(20, 200, size=n_days).astype(np.float64)
                grp.create_dataset(rid, data=arr)
                fixture[rid][node] = arr
            grp.create_dataset(
                "date",
                data=np.asarray(dates, dtype=object),
                dtype=str_dtype,
            )

    return {
        "path": str(input_path),
        "flows_dir": str(flows_dir),
        "realizations": realizations,
        "dates": dates,
        "fixture": fixture,
    }


def _make_preprocessor(synthetic_inflow_hdf5, **kwargs):
    """Build the preprocessor and reroute its I/O paths to tmp_path."""
    pp = STARFITReleaseEnsemblePreprocessor(inflow_type="nhmv10", **kwargs)
    pp._input_path = synthetic_inflow_hdf5["path"]
    pp._output_path = os.path.join(
        synthetic_inflow_hdf5["flows_dir"], OUTPUT_HDF5_NAME
    )
    pp._metadata_path = os.path.join(
        synthetic_inflow_hdf5["flows_dir"], OUTPUT_METADATA_NAME
    )
    return pp


def test_run_produces_node_first_hdf5(synthetic_inflow_hdf5):
    pp = _make_preprocessor(synthetic_inflow_hdf5)
    pp.run()

    assert os.path.exists(pp._output_path)
    assert os.path.exists(pp._metadata_path)

    with h5py.File(pp._output_path, "r") as f:
        # One group per STARFIT reservoir (no extras, no NYC).
        assert sorted(f.keys()) == sorted(SYNTHETIC_RESERVOIRS)

        for node in SYNTHETIC_RESERVOIRS:
            grp = f[node]
            labels = [
                lbl.decode() if isinstance(lbl, bytes) else str(lbl)
                for lbl in grp.attrs["column_labels"]
            ]
            assert labels == synthetic_inflow_hdf5["realizations"]
            for rid in synthetic_inflow_hdf5["realizations"]:
                assert rid in grp
                assert grp[rid].shape == (len(synthetic_inflow_hdf5["dates"]),)
                # Releases are non-negative floats.
                arr = grp[rid][:]
                assert np.all(np.isfinite(arr))
                assert (arr >= 0).all()
            stored_dates = [
                d.decode() if isinstance(d, bytes) else str(d)
                for d in grp["date"][:]
            ]
            assert stored_dates == synthetic_inflow_hdf5["dates"]


def test_metadata_sidecar_shape(synthetic_inflow_hdf5):
    pp = _make_preprocessor(synthetic_inflow_hdf5)
    pp.run()

    with open(pp._metadata_path) as f:
        meta = json.load(f)

    assert meta["inflow_type"] == "nhmv10"
    assert meta["start_date"] == "2020-01-01"
    assert meta["end_date"] == synthetic_inflow_hdf5["dates"][-1]
    assert sorted(meta["reservoirs"]) == sorted(SYNTHETIC_RESERVOIRS)
    assert meta["realization_ids"] == synthetic_inflow_hdf5["realizations"]
    assert meta["initial_volume_frac"] == pytest.approx(0.8)
    assert meta["source"] == "STARFITReleaseEnsemblePreprocessor"


def test_per_realization_equivalence_with_offline_simulator(synthetic_inflow_hdf5):
    """For every realization, the HDF5 release values must equal the values
    produced by STARFITOfflineSimulator.simulate_all() on the same inputs."""
    pp = _make_preprocessor(synthetic_inflow_hdf5)
    pp.run()

    sim = STARFITOfflineSimulator(initial_volume_frac=0.8)
    sim.load_parameters()

    index = pd.to_datetime(pd.Index(synthetic_inflow_hdf5["dates"]))

    with h5py.File(pp._output_path, "r") as f:
        for rid in synthetic_inflow_hdf5["realizations"]:
            # Rebuild the same per-realization inflow DataFrame the
            # preprocessor saw and run the engine inline.
            fixture = synthetic_inflow_hdf5["fixture"][rid]
            df_in = pd.DataFrame(
                {n: fixture[n] for n in fixture}, index=index
            )
            expected = sim.simulate_all(df_in, reservoir_list=SYNTHETIC_RESERVOIRS)

            for reservoir in SYNTHETIC_RESERVOIRS:
                got = f[reservoir][rid][:]
                np.testing.assert_allclose(
                    got, expected[reservoir].to_numpy(),
                    rtol=0, atol=0,
                    err_msg=f"mismatch for realization {rid}, reservoir {reservoir}",
                )


def test_idempotent_when_output_present(synthetic_inflow_hdf5):
    pp = _make_preprocessor(synthetic_inflow_hdf5)
    pp.run()
    first_mtime = os.path.getmtime(pp._output_path)

    time.sleep(0.05)

    pp2 = _make_preprocessor(synthetic_inflow_hdf5)
    pp2.run()  # short-circuits in process()

    assert os.path.getmtime(pp._output_path) == first_mtime


def test_force_rebuild(synthetic_inflow_hdf5):
    pp = _make_preprocessor(synthetic_inflow_hdf5)
    pp.run()
    first_mtime = os.path.getmtime(pp._output_path)

    time.sleep(0.05)

    pp2 = _make_preprocessor(synthetic_inflow_hdf5, force=True)
    pp2.run()

    assert os.path.getmtime(pp._output_path) > first_mtime


def test_subset_realizations(synthetic_inflow_hdf5):
    """When realization_ids is a subset, only those are written."""
    subset = ["0", "2"]
    pp = _make_preprocessor(synthetic_inflow_hdf5, realization_ids=subset)
    pp.run()

    with h5py.File(pp._output_path, "r") as f:
        for reservoir in SYNTHETIC_RESERVOIRS:
            labels = [
                lbl.decode() if isinstance(lbl, bytes) else str(lbl)
                for lbl in f[reservoir].attrs["column_labels"]
            ]
            assert labels == subset
            for rid in subset:
                assert rid in f[reservoir]
            assert "1" not in f[reservoir]


def test_subset_reservoirs(synthetic_inflow_hdf5):
    """When reservoir_list is a subset, only those reservoirs are written."""
    subset_res = ["wallenpaupack", "prompton", "blueMarsh"]
    pp = _make_preprocessor(synthetic_inflow_hdf5, reservoir_list=subset_res)
    pp.run()

    with h5py.File(pp._output_path, "r") as f:
        assert sorted(f.keys()) == sorted(subset_res)


def test_missing_input_raises(tmp_path):
    pp = STARFITReleaseEnsemblePreprocessor(inflow_type="nhmv10")
    pp._input_path = str(tmp_path / "does_not_exist.hdf5")
    pp._output_path = str(tmp_path / "out.hdf5")
    pp._metadata_path = str(tmp_path / "out_metadata.json")
    with pytest.raises(FileNotFoundError):
        pp.load()


# ---------------------------------------------------------------------------
# PresimulatedReleaseEnsemble round-trip with the produced artifact.
# ---------------------------------------------------------------------------

class _StubPathNavigator:
    """Minimal pn replacement that returns a fixed tmp directory."""

    def __init__(self, flows_dir):
        self._flows_dir = Path(flows_dir)
        self.sc = self  # so pn.sc.get(...) reaches our `get`

    def get(self, key):
        return self._flows_dir


def test_presimulated_release_ensemble_value_round_trip(
    synthetic_inflow_hdf5, monkeypatch
):
    """PresimulatedReleaseEnsemble.value() must return the same numbers
    that STARFITReleaseEnsemblePreprocessor wrote to the HDF5."""
    pp = _make_preprocessor(synthetic_inflow_hdf5)
    pp.run()

    # Move pn used inside ensemble.py to point at our tmp flows dir.
    monkeypatch.setattr(
        ensemble_module,
        "pn",
        _StubPathNavigator(synthetic_inflow_hdf5["flows_dir"]),
    )

    import pywr.core
    model = pywr.core.Model()

    rids = [int(r) for r in synthetic_inflow_hdf5["realizations"]]

    # Pick a couple of representative reservoirs.
    for reservoir in ("wallenpaupack", "blueMarsh"):
        param = ensemble_module.PresimulatedReleaseEnsemble(
            model,
            name=reservoir,
            inflow_type="synthetic",
            inflow_ensemble_indices=rids,
        )

        # Shape: (n_days, n_realizations).
        assert param.release_ensemble.shape == (
            len(synthetic_inflow_hdf5["dates"]),
            len(rids),
        )

        # Spot-check first/last day for every realization against the HDF5.
        with h5py.File(pp._output_path, "r") as f:
            grp = f[reservoir]
            for rid in synthetic_inflow_hdf5["realizations"]:
                expected_first = grp[rid][0]
                expected_last = grp[rid][-1]
                first_date = pd.to_datetime(synthetic_inflow_hdf5["dates"][0])
                last_date = pd.to_datetime(synthetic_inflow_hdf5["dates"][-1])
                assert (
                    param.release_ensemble.loc[first_date, rid]
                    == pytest.approx(expected_first)
                )
                assert (
                    param.release_ensemble.loc[last_date, rid]
                    == pytest.approx(expected_last)
                )


def test_presimulated_release_ensemble_load_classmethod(
    synthetic_inflow_hdf5, monkeypatch
):
    """The pywr `load` classmethod must accept the dispatch dict shape that
    ModelBuilder emits in ensemble trimmed-model mode."""
    pp = _make_preprocessor(synthetic_inflow_hdf5)
    pp.run()

    monkeypatch.setattr(
        ensemble_module,
        "pn",
        _StubPathNavigator(synthetic_inflow_hdf5["flows_dir"]),
    )

    import pywr.core
    model = pywr.core.Model()

    data = {
        "node": "wallenpaupack",
        "inflow_type": "synthetic",
        "inflow_ensemble_indices": [0, 1],
        "presim_filename": OUTPUT_HDF5_NAME,
    }
    param = ensemble_module.PresimulatedReleaseEnsemble.load(model, data)
    assert param.release_ensemble.shape[1] == 2
    assert list(param.release_ensemble.columns) == ["0", "1"]
