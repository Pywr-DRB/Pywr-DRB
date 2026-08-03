"""
Tests for FloodNodeInflowEnsemblePreprocessor.

Builds a small synthetic input HDF5 in tmp_path that mirrors the node-first
schema produced by the upstream stochastic-inflow generator, runs the
ensemble preprocessor with use_mpi=False, and verifies the output schema,
per-realization mass balance, idempotency, and force-rebuild.
"""
import os
import time

import h5py
import numpy as np
import pandas as pd
import pytest

from pywrdrb.pre.flood_node_inflows import (
    FLOOD_NODE_IDS,
    INPUT_HDF5_NAME,
    OUTPUT_HDF5_NAME,
    FloodNodeInflowEnsemblePreprocessor,
)


# Donor nodes the helpers reference. The synthetic HDF5 must include all of
# these so the per-realization computation has the data it needs.
SYNTHETIC_NODES = [
    "cannonsville",
    "01425000",
    "pepacton",
    "01417000",
    "neversink",
    "01436000",
    "prompton",
    "delLordville",
    "delMontague",
]

# A handful of "other" catchment nodes that appear in the real HDF5 — included
# so we exercise the "rewrite all input groups verbatim" code path.
EXTRA_NODES = ["mongaupeCombined", "wallenpaupack", "delDRCanal", "delTrenton"]


@pytest.fixture
def synthetic_input_hdf5(tmp_path):
    """
    Write a tiny ensemble input HDF5 in tmp_path with the node-first schema:
        /<node>/.attrs['column_labels'] = [b'0', b'1', b'2']
        /<node>/<rid>: 1D float array
        /<node>/date: 1D string array
    """
    n_days = 5
    n_realizations = 3
    realizations = [str(i) for i in range(n_realizations)]
    dates = pd.date_range("2020-01-01", periods=n_days).strftime("%Y-%m-%d").tolist()

    rng = np.random.default_rng(seed=42)

    flows_dir = tmp_path / "flows" / "test_inflow"
    flows_dir.mkdir(parents=True)
    input_path = flows_dir / INPUT_HDF5_NAME

    nodes = SYNTHETIC_NODES + EXTRA_NODES

    # Pre-build per-realization inflow tables so we can later assert mass
    # balance against the inputs.
    fixture = {}
    str_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(input_path, "w") as hf:
        for node in nodes:
            grp = hf.create_group(node)
            grp.attrs.create(
                "column_labels", np.asarray(realizations, dtype=object), dtype=str_dtype
            )
            for rid in realizations:
                if node == "delLordville":
                    arr = rng.uniform(250, 400, size=n_days)
                elif node == "delMontague":
                    arr = rng.uniform(150, 250, size=n_days)
                elif node in ("cannonsville", "pepacton"):
                    arr = rng.uniform(80, 150, size=n_days)
                elif node in ("01425000", "01417000"):
                    arr = rng.uniform(8, 20, size=n_days)
                elif node == "neversink":
                    arr = rng.uniform(30, 60, size=n_days)
                elif node == "01436000":
                    arr = rng.uniform(3, 10, size=n_days)
                elif node == "prompton":
                    arr = rng.uniform(2, 5, size=n_days)
                else:
                    arr = rng.uniform(10, 50, size=n_days)
                grp.create_dataset(rid, data=arr.astype(np.float64))
                fixture.setdefault(rid, {})[node] = arr.astype(np.float64)
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
        "nodes": nodes,
        "fixture": fixture,
    }


def _make_preprocessor(synthetic_input_hdf5, **kwargs):
    """Construct a preprocessor and reroute its I/O paths to tmp_path."""
    pp = FloodNodeInflowEnsemblePreprocessor(inflow_type="nhmv10", **kwargs)
    pp._input_path = synthetic_input_hdf5["path"]
    pp._output_path = os.path.join(synthetic_input_hdf5["flows_dir"], OUTPUT_HDF5_NAME)
    return pp


def test_run_produces_node_first_hdf5(synthetic_input_hdf5):
    pp = _make_preprocessor(synthetic_input_hdf5)
    pp.load()
    pp.process()
    pp.save()

    out_path = pp._output_path
    assert os.path.exists(out_path)

    with h5py.File(out_path, "r") as f:
        # Every input node group is present, plus the three flood-gauge groups.
        for node in synthetic_input_hdf5["nodes"]:
            assert node in f, f"missing input node group {node}"
        for fid in FLOOD_NODE_IDS:
            assert fid in f, f"missing flood-gauge group {fid}"

        # Each group has the expected schema.
        for node in synthetic_input_hdf5["nodes"] + list(FLOOD_NODE_IDS):
            grp = f[node]
            labels = [
                lbl.decode() if isinstance(lbl, bytes) else str(lbl)
                for lbl in grp.attrs["column_labels"]
            ]
            assert labels == synthetic_input_hdf5["realizations"]
            for rid in synthetic_input_hdf5["realizations"]:
                assert rid in grp, f"missing realization dataset /{node}/{rid}"
                assert grp[rid].shape == (len(synthetic_input_hdf5["dates"]),)
            assert "date" in grp
            stored_dates = [
                d.decode() if isinstance(d, bytes) else str(d) for d in grp["date"][:]
            ]
            assert stored_dates == synthetic_input_hdf5["dates"]


def test_per_realization_lordville_mass_balance(synthetic_input_hdf5):
    """For each realization, original delLordville == new delLordville
    + new 01426500 + new 01421000 (no clamping for this synthetic data)."""
    pp = _make_preprocessor(synthetic_input_hdf5)
    pp.load()
    pp.process()
    pp.save()

    fixture = synthetic_input_hdf5["fixture"]
    with h5py.File(pp._output_path, "r") as f:
        for rid in synthetic_input_hdf5["realizations"]:
            new_lordville = f["delLordville"][rid][:]
            new_hale = f["01426500"][rid][:]
            new_fish = f["01421000"][rid][:]
            recovered = new_lordville + new_hale + new_fish
            np.testing.assert_allclose(
                recovered, fixture[rid]["delLordville"], rtol=1e-12, atol=1e-9
            )


def test_per_realization_montague_mass_balance(synthetic_input_hdf5):
    pp = _make_preprocessor(synthetic_input_hdf5)
    pp.load()
    pp.process()
    pp.save()

    fixture = synthetic_input_hdf5["fixture"]
    with h5py.File(pp._output_path, "r") as f:
        for rid in synthetic_input_hdf5["realizations"]:
            new_montague = f["delMontague"][rid][:]
            new_bridge = f["01436690"][rid][:]
            recovered = new_montague + new_bridge
            np.testing.assert_allclose(
                recovered, fixture[rid]["delMontague"], rtol=1e-12, atol=1e-9
            )


def test_idempotent_when_output_present(synthetic_input_hdf5):
    pp = _make_preprocessor(synthetic_input_hdf5)
    pp.load()
    pp.process()
    pp.save()
    first_mtime = os.path.getmtime(pp._output_path)

    # Wait a beat so any rebuild would advance mtime perceptibly.
    time.sleep(0.05)

    pp2 = _make_preprocessor(synthetic_input_hdf5)
    pp2.load()
    pp2.process()  # should short-circuit
    pp2.save()  # no-op because process() left augmented_inflows empty

    assert os.path.getmtime(pp._output_path) == first_mtime


def test_force_rebuild(synthetic_input_hdf5):
    pp = _make_preprocessor(synthetic_input_hdf5)
    pp.load()
    pp.process()
    pp.save()
    first_mtime = os.path.getmtime(pp._output_path)

    time.sleep(0.05)

    pp2 = _make_preprocessor(synthetic_input_hdf5, force=True)
    pp2.load()
    pp2.process()
    pp2.save()

    assert os.path.getmtime(pp._output_path) > first_mtime


def test_subset_realizations(synthetic_input_hdf5):
    """When realization_ids is a subset, only those are written."""
    subset = ["0", "2"]
    pp = _make_preprocessor(synthetic_input_hdf5, realization_ids=subset)
    pp.load()
    pp.process()
    pp.save()

    with h5py.File(pp._output_path, "r") as f:
        labels = [
            lbl.decode() if isinstance(lbl, bytes) else str(lbl)
            for lbl in f["delLordville"].attrs["column_labels"]
        ]
        assert labels == subset
        for rid in subset:
            assert rid in f["delLordville"]
        # Realization 1 must NOT be present in the output.
        assert "1" not in f["delLordville"] or set(f["delLordville"].keys()) - {
            "date", "0", "2"
        } == set()
