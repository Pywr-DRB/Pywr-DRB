"""
Integration tests covering the routing path:
  ModelBuilder (ensemble + flood ops) -> FlowEnsemble parameter dict
  FlowEnsemble.__init__ honoring inflow_filename
  FloodNodeInflowEnsemblePreprocessor output consumed by FlowEnsemble

The end-to-end ``pywrdrb.Model.load`` + ``model.run()`` against a synthetic
ensemble HDF5 requires a much larger fixture (consumption CSVs, reservoir
release tables, etc.); those are exercised by the existing
``test_flood_operations.py`` and ``test_sample_run_all_datasets.py`` suites
on the CSV path. The tests here cover the new plumbing in isolation.
"""
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from pywrdrb.model_builder import ModelBuilder
from pywrdrb.parameters import ensemble as ensemble_module
from pywrdrb.pre.flood_node_inflows import (
    FLOOD_NODE_IDS,
    INPUT_HDF5_NAME,
    OUTPUT_HDF5_NAME,
    FloodNodeInflowEnsemblePreprocessor,
)


# ---------------------------------------------------------------------------
# ModelBuilder routing (no I/O — just inspect the emitted model_dict).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("flood_ops", [True, False])
def test_model_builder_ensemble_routes_to_correct_filename(flood_ops):
    """When inflow_ensemble_indices is set, every FlowEnsemble parameter
    must declare the inflow_filename matching the flood_ops setting."""
    builder = ModelBuilder(
        start_date="2000-01-01",
        end_date="2000-01-02",
        inflow_type="nhmv10",
        options={
            "enable_nyc_flood_operations": flood_ops,
            "inflow_ensemble_indices": [0, 1],
        },
    )
    builder.make_model()

    expected = (
        "catchment_inflow_with_flood_nodes_mgd.hdf5"
        if flood_ops
        else "catchment_inflow_mgd.hdf5"
    )

    flow_ensemble_params = [
        v for v in builder.model_dict["parameters"].values()
        if isinstance(v, dict) and v.get("type") == "FlowEnsemble"
    ]
    assert flow_ensemble_params, "no FlowEnsemble parameters emitted"
    for p in flow_ensemble_params:
        assert p.get("inflow_filename") == expected, (
            f"FlowEnsemble for node {p.get('node')!r} has "
            f"inflow_filename={p.get('inflow_filename')!r}, expected {expected!r}"
        )


def test_model_builder_csv_branch_unchanged_without_ensemble():
    """When inflow_ensemble_indices is None we must keep using the CSV path
    and must not emit FlowEnsemble parameters."""
    builder = ModelBuilder(
        start_date="2000-01-01",
        end_date="2000-01-02",
        inflow_type="nhmv10",
        options={"enable_nyc_flood_operations": True},
    )
    builder.make_model()

    flow_ensemble_params = [
        v for v in builder.model_dict["parameters"].values()
        if isinstance(v, dict) and v.get("type") == "FlowEnsemble"
    ]
    assert flow_ensemble_params == []


# ---------------------------------------------------------------------------
# FlowEnsemble round-trip with a fabricated augmented HDF5.
# ---------------------------------------------------------------------------

class _StubPathNavigator:
    """Minimal pn replacement that returns a fixed tmp directory."""
    def __init__(self, flows_dir):
        self._flows_dir = Path(flows_dir)
        self.sc = self  # so pn.sc.get(...) reaches _StubPathNavigator.get

    def get(self, key):
        # FlowEnsemble calls pn.sc.get(f"flows/{inflow_type}").
        # Return the same fixed dir regardless of inflow_type — tests pass
        # one preprocessor's output through one FlowEnsemble.
        return self._flows_dir


@pytest.fixture
def augmented_hdf5(tmp_path):
    """Run FloodNodeInflowEnsemblePreprocessor on a synthetic input and
    return the augmented HDF5 path (alongside its containing dir)."""
    flows_dir = tmp_path / "flows" / "synthetic"
    flows_dir.mkdir(parents=True)
    input_path = flows_dir / INPUT_HDF5_NAME

    realizations = ["0", "1"]
    dates = pd.date_range("2020-01-01", periods=4).strftime("%Y-%m-%d").tolist()
    rng = np.random.default_rng(seed=7)
    str_dtype = h5py.string_dtype(encoding="utf-8")

    nodes = [
        "cannonsville", "01425000", "pepacton", "01417000",
        "neversink", "01436000", "prompton",
        "delLordville", "delMontague",
        # A few extra non-flood-related nodes to exercise the full rewrite.
        "wallenpaupack", "delDRCanal",
    ]

    with h5py.File(input_path, "w") as hf:
        for node in nodes:
            grp = hf.create_group(node)
            grp.attrs.create(
                "column_labels",
                np.asarray(realizations, dtype=object),
                dtype=str_dtype,
            )
            for rid in realizations:
                grp.create_dataset(
                    rid, data=rng.uniform(50, 300, size=len(dates))
                )
            grp.create_dataset(
                "date",
                data=np.asarray(dates, dtype=object),
                dtype=str_dtype,
            )

    pp = FloodNodeInflowEnsemblePreprocessor(inflow_type="nhmv10")
    pp._input_path = str(input_path)
    pp._output_path = str(flows_dir / OUTPUT_HDF5_NAME)
    pp.load()
    pp.process()
    pp.save()

    return {
        "flows_dir": str(flows_dir),
        "output_path": str(flows_dir / OUTPUT_HDF5_NAME),
        "realizations": realizations,
    }


def test_flow_ensemble_loads_augmented_hdf5(augmented_hdf5, monkeypatch):
    """FlowEnsemble with inflow_filename pointing at the augmented HDF5
    should successfully load a flood-gauge node group."""
    # Swap pn used inside FlowEnsemble for one that points at our tmp dir.
    monkeypatch.setattr(
        ensemble_module, "pn",
        _StubPathNavigator(augmented_hdf5["flows_dir"]),
    )

    # Use a minimal pywr model so FlowEnsemble.__init__ can call super().
    import pywr.core
    model = pywr.core.Model()

    rids = [int(r) for r in augmented_hdf5["realizations"]]
    fe = ensemble_module.FlowEnsemble(
        model,
        name="01426500",
        inflow_type="synthetic",
        inflow_ensemble_indices=rids,
        inflow_filename=OUTPUT_HDF5_NAME,
    )

    # Should have loaded a DataFrame with one column per realization.
    assert fe.inflow_ensemble.shape[1] == len(rids)
    # Values are non-negative (helpers floor at zero) and finite.
    assert np.all(np.isfinite(fe.inflow_ensemble.values))
    assert (fe.inflow_ensemble.values >= 0).all()


def test_flow_ensemble_default_filename_unchanged(monkeypatch, tmp_path):
    """A model dict that does not specify inflow_filename must still load
    catchment_inflow_mgd.hdf5 — backwards compatibility check."""
    flows_dir = tmp_path / "flows" / "synthetic"
    flows_dir.mkdir(parents=True)

    # Write an empty-but-valid HDF5 with one node so the backwards-compat
    # path is exercised at the filename-choice level.
    realizations = ["0"]
    dates = pd.date_range("2020-01-01", periods=2).strftime("%Y-%m-%d").tolist()
    str_dtype = h5py.string_dtype(encoding="utf-8")
    default_path = flows_dir / "catchment_inflow_mgd.hdf5"
    with h5py.File(default_path, "w") as hf:
        grp = hf.create_group("cannonsville")
        grp.attrs.create(
            "column_labels", np.asarray(realizations, dtype=object), dtype=str_dtype
        )
        grp.create_dataset("0", data=np.array([1.0, 2.0]))
        grp.create_dataset(
            "date", data=np.asarray(dates, dtype=object), dtype=str_dtype
        )

    monkeypatch.setattr(
        ensemble_module, "pn", _StubPathNavigator(str(flows_dir))
    )

    import pywr.core
    model = pywr.core.Model()
    fe = ensemble_module.FlowEnsemble.load(
        model,
        {
            "node": "cannonsville",
            "inflow_type": "synthetic",
            "inflow_ensemble_indices": [0],
        },
    )

    assert fe.inflow_ensemble.shape == (2, 1)


# ---------------------------------------------------------------------------
# End-to-end: preprocessor output is structurally compatible with the
# existing FlowEnsemble loader for a flood-gauge node.
# ---------------------------------------------------------------------------

def test_preprocessor_output_satisfies_flow_ensemble_for_all_flood_nodes(
    augmented_hdf5, monkeypatch
):
    """All three flood-gauge nodes must be loadable via FlowEnsemble against
    the augmented HDF5 that the preprocessor just produced."""
    monkeypatch.setattr(
        ensemble_module, "pn",
        _StubPathNavigator(augmented_hdf5["flows_dir"]),
    )

    import pywr.core
    model = pywr.core.Model()
    rids = [int(r) for r in augmented_hdf5["realizations"]]

    for fid in FLOOD_NODE_IDS:
        fe = ensemble_module.FlowEnsemble(
            model,
            name=fid,
            inflow_type="synthetic",
            inflow_ensemble_indices=rids,
            inflow_filename=OUTPUT_HDF5_NAME,
        )
        assert fe.inflow_ensemble.shape[1] == len(rids)
