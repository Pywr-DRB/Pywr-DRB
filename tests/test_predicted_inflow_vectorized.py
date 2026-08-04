"""
Bit-identity tests for the vectorized perfect-foresight kernel in
PredictedInflowEnsemblePreprocessor.

The vectorized path (`_vectorize_perfect_foresight = True`, the default) must
reproduce the scalar reference path (`PredictedInflowPreprocessor._predict_value`
per day) EXACTLY — np.testing.assert_array_equal, no tolerance — including:
  * the .iloc[-1] out-of-range fallback in BOTH directions (negative lags occur:
    Trenton lag 1 with travel time 4 gives lag -3, so the series start reads the
    record's LAST value);
  * float32 inflow data (the staged ensembles are float32) mixed with float64
    catchment_wc constants and float64 STARFIT releases;
  * interior start_date/end_date prediction windows;
  * the raw-inflow fallback when a STARFIT reservoir has no release column.

Fixtures mirror tests/test_predicted_inflow_ensemble_eager.py but write float32
inflows to match production.
"""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from pywrdrb.pre import predict_inflows
from pywrdrb.pre.predict_inflows import PredictedInflowEnsemblePreprocessor
from pywrdrb.utils.lists import starfit_reservoir_list


N_DAYS = 730  # two years: long enough for lag +/- shifts, short enough for the scalar path
REALIZATIONS = ["0", "1"]


@pytest.fixture
def synthetic_ensemble(tmp_path):
    """Tiny ensemble inflow + STARFIT release HDF5 pair (float32 inflows)."""
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
                    rid, data=rng.uniform(10, 100, size=N_DAYS).astype(np.float32)
                )
            grp.create_dataset(
                "date", data=np.asarray(dates, dtype=object), dtype=str_dtype
            )

    release_path = flows_dir / "presimulated_releases_mgd.hdf5"
    rng_rel = np.random.default_rng(seed=17)
    with h5py.File(release_path, "w") as hf:
        for reservoir in starfit_reservoir_list:
            grp = hf.create_group(reservoir)
            grp.attrs.create(
                "column_labels",
                np.asarray(REALIZATIONS, dtype=object),
                dtype=str_dtype,
            )
            for rid in REALIZATIONS:
                grp.create_dataset(
                    rid,
                    data=rng_rel.uniform(1, 25, size=N_DAYS).astype(np.float64),
                )
            grp.create_dataset(
                "date", data=np.asarray(dates, dtype=object), dtype=str_dtype
            )

    return {"flows_dir": flows_dir, "inflow_path": inflow_path, "dates": dates}


class _StubFlowsPathNavigator:
    """Stand-in for the parts of pn that PredictedInflowEnsemblePreprocessor uses."""

    def __init__(self, flows_dir):
        self._flows_dir = Path(flows_dir)
        self.flows = self
        self.sc = self
        self.catchment_withdrawals = self

    def get_str(self, key):
        return str(self._flows_dir)

    def get(self, key):
        if isinstance(key, str) and key.endswith(".csv"):
            from pywrdrb.path_manager import get_pn_object

            return get_pn_object().catchment_withdrawals.get(key)
        return self._flows_dir


def _build(synthetic_ensemble, *, vectorize, start_date=None, end_date=None):
    # flow_type must resolve in the real path navigator at __init__ (as in
    # test_predicted_inflow_ensemble_eager.py); load()-time lookups are then
    # rerouted to the stub.
    pp = PredictedInflowEnsemblePreprocessor(
        flow_type="nhmv10",
        ensemble_hdf5_file=str(synthetic_ensemble["inflow_path"]),
        realization_ids=[int(r) for r in REALIZATIONS],
        start_date=start_date,
        end_date=end_date,
        modes=("perfect_foresight",),
    )
    pp.pn = _StubFlowsPathNavigator(synthetic_ensemble["flows_dir"])
    pp._vectorize_perfect_foresight = vectorize
    return pp


def _assert_predictions_equal(vec, ref):
    assert set(vec.ensemble_predictions) == set(ref.ensemble_predictions)
    for rid in ref.ensemble_predictions:
        df_vec = vec.ensemble_predictions[rid]
        df_ref = ref.ensemble_predictions[rid]
        assert list(df_vec.columns) == list(df_ref.columns)
        assert (df_vec["datetime"].values == df_ref["datetime"].values).all()
        for col in df_ref.columns:
            if col == "datetime":
                continue
            np.testing.assert_array_equal(
                df_vec[col].to_numpy(),
                df_ref[col].to_numpy(),
                err_msg=f"realization {rid}, column {col}",
            )


def test_vectorized_matches_scalar_full_window(synthetic_ensemble):
    """Full record: both out-of-range directions hit the .iloc[-1] fill."""
    ref = _build(synthetic_ensemble, vectorize=False)
    ref.load()
    ref.process()

    vec = _build(synthetic_ensemble, vectorize=True)
    vec.load()
    vec.process()

    _assert_predictions_equal(vec, ref)


def test_vectorized_matches_scalar_interior_window(synthetic_ensemble):
    """Interior start/end dates: shifted targets exist in the (longer) data
    index but not in the prediction index — exercises subset alignment."""
    kwargs = dict(start_date="2020-06-01", end_date="2021-05-31")
    ref = _build(synthetic_ensemble, vectorize=False, **kwargs)
    ref.load()
    ref.process()

    vec = _build(synthetic_ensemble, vectorize=True, **kwargs)
    vec.load()
    vec.process()

    _assert_predictions_equal(vec, ref)


def test_vectorized_matches_scalar_missing_release_column(synthetic_ensemble):
    """A STARFIT reservoir without a release column takes the raw-inflow
    fallback branch — both paths must agree there too."""
    dropped = starfit_reservoir_list[0]

    ref = _build(synthetic_ensemble, vectorize=False)
    ref.load()
    for rid in list(ref._starfit_release_data):
        ref._starfit_release_data[rid] = ref._starfit_release_data[rid].drop(
            columns=[dropped]
        )
    ref.process()

    vec = _build(synthetic_ensemble, vectorize=True)
    vec.load()
    for rid in list(vec._starfit_release_data):
        vec._starfit_release_data[rid] = vec._starfit_release_data[rid].drop(
            columns=[dropped]
        )
    vec.process()

    _assert_predictions_equal(vec, ref)
