"""
Preprocessor for generating inflow predictions at nodes in the pywrdrb model.

Overview:
This class generates lag-based inflow predictions/forecasts at Montague and Trenton,
which are used to determine NYC and lower basin reservoir operations while accounting for travel time.
It uses regression models, trained on historical data, to predict flows 1-4 days ahead
based on catchment-level data and travel times between nodes. The output data has
multiple different columns corresponding to different prediction nodes, lead time lags, and
regression modes.

Prediction Modes:
- "regression_disagg": AR regression predictions using catchment inflows (realistic forecasts)
- "perfect_foresight": Pre-simulated STARFIT releases + actual catchment inflows with lag routing
  (best retrospective analysis, accounts for reservoir operations)

Technical Notes:
- Extends PredictedTimeseriesPreprocessor with specific inflow prediction logic
- Incorporates travel times to properly account for flow routing
- Adjusts predictions for water consumption in each catchment
- Outputs predictions for 1-4 days ahead at Montague and Trenton

Example Usage:
from pywrdrb.pre import PredictedInflowPreprocessor
inflow_predictor = PredictedInflowPreprocessor(flow_type="nhmv10")
inflow_predictor.process()
inflow_predictor.save()

Links:
- See SI for Hamilton et al. (2024) for more details on the method formulation.

Change Log:
TJA, 2025-05-07, Minor fixes + docstrings
TJA, 2025-10, Fixed bug where nodes with lag < 0 were not being included in predictions
TJA, 2026-03, Added STARFIT-aware perfect_foresight mode; renamed old PF to gage_flow
TJA, 2026-07, Removed legacy gage_flow mode
TJA, 2026-08, Ensemble: per-rank HDF5 reads + point-to-point gather; PF ensemble reads presimulated releases artifact
"""
import io
import os
import time
import h5py
import numpy as np
import pandas as pd
from pywrdrb.pre.predict_timeseries import PredictedTimeseriesPreprocessor
from pywrdrb.pre._mpi_utils import bcast_with_error, point_to_point_gather
from pywrdrb.utils.hdf5 import extract_realization_from_hdf5
from pywrdrb.utils.lists import (
    starfit_reservoir_list,
    reservoir_list_nyc,
    reservoir_list,
    majorflow_list,
)

# Import pywrdrb_all_nodes for optimized HDF5 extraction
from pywrdrb.pywr_drb_node_data import immediate_downstream_nodes_dict

pywrdrb_all_nodes = list(immediate_downstream_nodes_dict.keys())

__all__ = ["PredictedInflowPreprocessor", "PredictedInflowEnsemblePreprocessor"]


class PredictedInflowPreprocessor(PredictedTimeseriesPreprocessor):
    """
    Predicts catchment inflows at Montague and Trenton using specified modes
    (e.g., regression, perfect foresight).

    Examples
    --------
    >>> from pywrdrb.pre import PredictedInflowPreprocessor
    >>> inflow_predictor = PredictedInflowPreprocessor(flow_type="nhmv10")
    >>> inflow_predictor.process()
    >>> inflow_predictor.save()
    """

    def __init__(
        self,
        flow_type,
        start_date=None,
        end_date=None,
        modes=("regression_disagg",),
        use_log=True,
        remove_zeros=False,
        use_const=False,
    ):
        """
        Initialize the PredictedInflowPreprocessor.

        Args:
            flow_type (str): Label for the dataset.
            start_date (bool, None): Start date for the time series. If None, match the input data.
            end_date (bool, None): End date for the time series. If None, match the input data.
            modes (tuple): Modes to use for prediction. Default is ('regression_disagg',).
                Options: "regression_disagg", "perfect_foresight"
            use_log (bool): Whether to use log transformation. Default is True.
            remove_zeros (bool): Whether to remove zero values. Default is False.
            use_const (bool): Whether to use a constant in regression. Default is False.

        Returns:
            None
        """
        # Initialize the PredictedTimeseriesPreprocessor
        super().__init__(
            flow_type, start_date, end_date, use_log, remove_zeros, use_const
        )

        # Valid prediction modes
        self.regression_mode_options = [
            "regression_disagg",
            "perfect_foresight",
        ]

        # Modes being used; check validity
        self.modes = modes
        for mode in self.modes:
            assert (
                mode in self.regression_mode_options
            ), f"Invalid regression mode: {mode}. Must be one of {self.regression_mode_options}."

        # Input files used for prediction
        self.input_dirs = {
            "sw_avg_wateruse_pywrdrb_catchments_mgd.csv": self.pn.catchment_withdrawals.get(
                "sw_avg_wateruse_pywrdrb_catchments_mgd.csv"
            ),
            "catchment_inflow_mgd.csv": self.pn.sc.get(f"flows/{self.flow_type}")
            / "catchment_inflow_mgd.csv",
        }

        # Output locations for predicted timeseries
        self.output_dirs = {
            "predicted_inflows_mgd.csv": self.pn.sc.get(f"flows/{self.flow_type}")
            / "predicted_inflows_mgd.csv",
        }

        # Dictionary with (node, travel_time) pairs
        # travel_time is time from each node to Trenton
        self.node_to_trenton_travel_time = {
            "01425000": 4,
            "01417000": 4,
            "delLordville": 4,
            "01436000": 3,
            "wallenpaupack": 3,
            "prompton": 3,
            "shoholaMarsh": 3,
            "mongaupeCombined": 2,
            "01433500": 2,
            "delMontague": 2,
            "beltzvilleCombined": 2,
            "01447800": 2,
            "fewalter": 2,
            "01449800": 2,
            "hopatcong": 1,
            "merrillCreek": 1,
            "nockamixon": 1,
            "delDRCanal": 0,
        }

        # travel_time is time from each node to Montague
        self.node_to_montague_travel_time = {
            "01425000": 2,
            "01417000": 2,
            "delLordville": 2,
            "01436000": 1,
            "wallenpaupack": 1,
            "prompton": 1,
            "shoholaMarsh": 1,
            "mongaupeCombined": 0,
            "01433500": 0,
            "delMontague": 0,
        }

        # STARFIT releases (populated during load() for perfect_foresight mode)
        self.starfit_releases = None

    def load(self):
        """
        Loads catchment inflows, STARFIT releases, and water consumption data.

        For regression modes: Uses catchment_inflow_mgd.csv (marginal/incremental flows)
        For perfect_foresight mode: Uses catchment_inflow_mgd.csv + pre-simulated STARFIT releases
        """
        has_perfect_foresight = "perfect_foresight" in self.modes

        # Load catchment inflow data
        # used to predict inflows at Montague and Trenton via aggregation
        fname = self.input_dirs["catchment_inflow_mgd.csv"]
        self.timeseries_data = pd.read_csv(fname, index_col=0, parse_dates=True)
        self.timeseries_data.index = pd.DatetimeIndex(self.timeseries_data.index)

        # Pre-simulate STARFIT releases for perfect_foresight mode
        if has_perfect_foresight:
            from pywrdrb.pre.generate_presimulated_releases import (
                STARFITOfflineSimulator,
            )

            simulator = STARFITOfflineSimulator(initial_volume_frac=0.8)
            simulator.load_parameters()
            self.starfit_releases = simulator.simulate_all(self.timeseries_data)

        # Load average water consumption data
        # used to adjust the inflow prediction, accounting for water use
        fname = self.input_dirs["sw_avg_wateruse_pywrdrb_catchments_mgd.csv"]
        wc = pd.read_csv(fname)
        wc.index = wc["node"]
        self.catchment_wc = wc

    def save(self):
        """
        Save predicted timeseries to CSV.
        """
        # Make sure the predictions are done successfully
        assert (
            self.predicted_timeseries is not None
        ), "Predicted timeseries is None. Cannot save."

        # Save
        fname = self.output_dirs["predicted_inflows_mgd.csv"]
        self.predicted_timeseries.to_csv(fname, index=False)

    def process(self):
        """Run full prediction workflow."""
        # Ensure data is loaded
        if self.timeseries_data is None:
            self.load()

        # Train regression models for all node-lag combinations
        regressions = self.train_regressions()

        # Generate predictions using the trained regression models
        self.predicted_timeseries = self.make_predictions(regressions)

    def get_prediction_node_lag_combinations(self):
        """
        Return dict of {column_label: [((node, lag), mode)]} across all modes.
        This defines the structure used in make_predictions().

        For regression modes: Aggregates upstream catchments with travel time adjustments
        For perfect_foresight: Aggregates upstream catchments with travel times + STARFIT releases
        """

        # Dictionary to hold regression combination
        # keys are strings of the form "target_lag_mode"
        # values are lists of tuples (node, lag)
        combos = {}

        # Montague predictions
        for lag in [1, 2]:
            for mode in self.modes:
                col = f"delMontague_lag{lag}_{mode}"

                # Aggregate upstream catchments with travel times
                combos[col] = []
                for node, travel_time in self.node_to_montague_travel_time.items():
                    combos[col].append(((node, lag - travel_time), mode))

        # Trenton predictions
        for lag in [1, 2, 3, 4]:
            for mode in self.modes:
                col = f"delTrenton_lag{lag}_{mode}"

                # Aggregate upstream catchments with travel times
                combos[col] = []
                for node, travel_time in self.node_to_trenton_travel_time.items():
                    combos[col].append(((node, lag - travel_time), mode))

        return combos

    def _predict_value(self, idx, date_t, node, lag, mode, regressions):
        """
        Generate a single prediction value for a given time index, node, lag, and mode.

        Overrides the base class to handle the new "perfect_foresight" mode
        which uses pre-simulated STARFIT releases for reservoir nodes.

        For "perfect_foresight" mode:
        - STARFIT reservoir nodes: returns pre-simulated release (no WC adjustment)
        - NYC reservoir nodes: returns 0.0 (NYC releases are the control variable)
        - Non-reservoir nodes: returns catchment inflow (with WC adjustment)

        For other modes: delegates to base class.
        """
        if mode != "perfect_foresight":
            return super()._predict_value(idx, date_t, node, lag, mode, regressions)

        # === perfect_foresight mode ===
        date_lag = date_t + pd.Timedelta(days=lag)

        # NYC reservoir nodes: return 0 (inflow stored, release is control variable)
        if node in reservoir_list_nyc:
            return 0.0

        # STARFIT reservoir nodes: return pre-simulated release
        if node in starfit_reservoir_list:
            if self.starfit_releases is not None and node in self.starfit_releases.columns:
                if date_lag in self.starfit_releases.index:
                    return self.starfit_releases.loc[date_lag, node]
                else:
                    # Beyond data range: use last available
                    return self.starfit_releases[node].iloc[-1]
            else:
                # Fallback: use catchment inflow if STARFIT releases not available
                if date_lag in self.timeseries_data.index:
                    return self.timeseries_data.loc[date_lag, node]
                else:
                    return self.timeseries_data[node].iloc[-1]

        # Non-reservoir nodes: return catchment inflow with water consumption adjustment
        if date_lag in self.timeseries_data.index:
            Yhat_lag_prediction = self.timeseries_data.loc[date_lag, node]
        else:
            Yhat_lag_prediction = self.timeseries_data[node].iloc[-1]

        # Get lag-1 prediction for water consumption calculation
        date_lag_minus_1 = date_t + pd.Timedelta(days=lag - 1)
        if date_lag_minus_1 in self.timeseries_data.index:
            Yhat_lag_minus1_prediction = self.timeseries_data.loc[date_lag_minus_1, node]
        else:
            Yhat_lag_minus1_prediction = self.timeseries_data[node].iloc[-1]

        # Apply water consumption adjustment (same as base class, lines 402-416)
        if node in reservoir_list + majorflow_list:
            pywr_node = (
                f"reservoir_{node}" if node in reservoir_list else f"link_{node}"
            )
            wd = self.catchment_wc.loc[pywr_node, "Total_WD_MGD"]
            cu = self.catchment_wc.loc[pywr_node, "Total_CU_WD_Ratio"]

            consumption_prediction = min(
                Yhat_lag_prediction, cu * min(Yhat_lag_minus1_prediction, wd)
            )
            return Yhat_lag_prediction - consumption_prediction

        return Yhat_lag_prediction


class PredictedInflowEnsemblePreprocessor(PredictedInflowPreprocessor):
    """
    Generates ensemble predictions for inflows at Montague and Trenton using MPI parallelization.

    Processes multiple realization members from an ensemble HDF5 file and saves predictions
    in HDF5 format compatible with PredictionEnsemble parameter.
    """

    # Routes make_predictions() through the vectorized perfect-foresight kernel
    # (_predict_perfect_foresight_series). False restores the scalar reference
    # path (PredictedInflowPreprocessor._predict_value per day) — flipped per
    # instance by the equivalence tests.
    _vectorize_perfect_foresight = True

    def __init__(
        self,
        flow_type,
        ensemble_hdf5_file,
        realization_ids=None,
        start_date=None,
        end_date=None,
        modes=("regression_disagg",),
        use_log=True,
        remove_zeros=False,
        use_const=False,
        use_mpi=False,
        comm=None,
    ):
        """
        Initialize the PredictedInflowEnsemblePreprocessor.

        Args:
            flow_type: Label for the dataset.
            ensemble_hdf5_file: Path to HDF5 file containing ensemble inflow data.
            realization_ids: List of realization IDs to process. If None, uses all available.
            start_date: Start date for predictions. If None, match input data.
            end_date: End date for predictions. If None, match input data.
            modes: Prediction modes to use.
            use_log: Whether to use log transformation.
            remove_zeros: Whether to remove zero values.
            use_const: Whether to use constant in regression.
            comm: MPI communicator. If None and use_mpi=True, uses MPI.COMM_WORLD.
        """
        super().__init__(
            flow_type, start_date, end_date, modes, use_log, remove_zeros, use_const
        )

        self.ensemble_hdf5_file = ensemble_hdf5_file
        self.realization_ids = realization_ids

        self.use_mpi = use_mpi
        if self.use_mpi:
            if comm is not None:
                self.comm = comm
            else:
                from mpi4py import MPI
                self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            self.comm = None
            self.rank = 0
            self.size = 1

        # Update output path for ensemble predictions
        self.output_dirs = {
            "predicted_inflows_mgd.hdf5": self.pn.sc.get(f"flows/{self.flow_type}")
            / "predicted_inflows_mgd.hdf5",
        }

        # Storage for ensemble results
        self.ensemble_predictions = {}

        # Per-realization STARFIT releases preloaded from
        # presimulated_releases_mgd.hdf5 in load() when perfect_foresight is in modes.
        # Maps str(realization_id) -> DataFrame[datetime x reservoir_name].
        self._starfit_release_data = {}

        # Only the travel-time nodes feed predictions (both regression combos
        # and the perfect-foresight kernel), so per-rank ensemble reads pull
        # just these — the same read narrowing 99fd7d6 applied to the STARFIT
        # presim preprocessor. Order follows pywrdrb_all_nodes.
        self._prediction_nodes = [
            n for n in pywrdrb_all_nodes
            if n in self.node_to_trenton_travel_time
            or n in self.node_to_montague_travel_time
        ]

    def _extract_realization_from_open_file(self, hdf5_file, realization_id):
        """
        Extract a single realization from an already-open HDF5 file.

        This is an optimized version of extract_realization_from_hdf5() that works
        with an already-open file handle to avoid repeated file open/close operations.

        Parameters
        ----------
        hdf5_file : h5py.File
            Open HDF5 file handle
        realization_id : str
            The realization ID to extract

        Returns
        -------
        pd.DataFrame
            DataFrame containing the extracted realization data
        """
        # Extract timeseries data from realization for each prediction node
        # (the travel-time nodes are the only columns any mode reads).
        data = {}

        for node in self._prediction_nodes:
            node_data = hdf5_file[node]
            column_labels = node_data.attrs["column_labels"]

            err_msg = f"The specified realization {realization_id} is not available in the HDF file."
            # Convert both to strings for consistent comparison
            column_labels_str = [str(label) for label in column_labels]
            assert str(realization_id) in column_labels_str, (
                err_msg + f" Realizations available: {column_labels}"
            )

            data[node] = node_data[str(realization_id)][:]

        dates = node_data["date"][:].tolist()
        data["datetime"] = dates

        # Combine into dataframe
        df = pd.DataFrame(data, index=dates)
        df.index = pd.to_datetime(df.index.astype(str))
        return df

    def _shifted_values(self, frame, node, lag, index, shift_cache):
        """Column ``node`` of ``frame`` at ``index + lag`` days, vectorized.

        The elementwise twin of the scalar per-day lookup
        ``frame.loc[date_t + Timedelta(days=lag), node]`` with the
        out-of-range fallback ``frame[node].iloc[-1]`` — in BOTH directions
        (negative lags occur: Trenton lag 1 with travel time 4 gives lag -3,
        so the first days of those series take the record's LAST value,
        exactly as the scalar path does). Membership is tested by date via
        ``get_indexer`` (no index-continuity assumption). Values are gathered
        without any dtype cast so float32/float64 promotion happens at the
        same points as in the scalar path.
        """
        key = (id(frame.index), lag)
        pos = shift_cache.get(key)
        if pos is None:
            target = index + pd.Timedelta(days=lag)
            tpos = frame.index.get_indexer(target)
            # -1 = date not present -> the scalar path's .iloc[-1] fallback.
            pos = np.where(tpos >= 0, tpos, len(frame.index) - 1)
            shift_cache[key] = pos
        return frame[node].to_numpy()[pos]

    def _predict_perfect_foresight_series(self, index, node, lag, shift_cache):
        """Vectorized elementwise twin of the scalar perfect-foresight branch.

        Mirrors ``PredictedInflowPreprocessor._predict_value`` (the
        ``mode == "perfect_foresight"`` body) over the whole prediction index
        at once: every operation is the same IEEE-754 op in the same order,
        ``min()`` becomes ``np.minimum`` with identical argument order, and
        the two catchment_wc constants are hoisted out of the (former) day
        loop. Keep in sync with the scalar path — the scalar path is the
        reference (same contract as
        ``STARFITOfflineSimulator.simulate_reservoir_ensemble``).
        """
        # NYC reservoirs: 0.0 (releases are the control variable). No NYC
        # node appears in the travel-time dicts, so this guard is only kept
        # to preserve the scalar semantics verbatim.
        if node in reservoir_list_nyc:
            return np.zeros(len(index))

        # STARFIT reservoirs: pre-simulated release (no WC adjustment),
        # falling back to raw catchment inflow when releases are unavailable.
        if node in starfit_reservoir_list:
            if (
                self.starfit_releases is not None
                and node in self.starfit_releases.columns
            ):
                return self._shifted_values(
                    self.starfit_releases, node, lag, index, shift_cache
                )
            return self._shifted_values(
                self.timeseries_data, node, lag, index, shift_cache
            )

        # Non-reservoir nodes: catchment inflow with water consumption
        # adjustment (scalar twin: Yhat_lag / Yhat_lag_minus1 lookups, then
        # min(Yhat_lag, cu * min(Yhat_lag_minus1, wd))).
        y = self._shifted_values(self.timeseries_data, node, lag, index, shift_cache)
        if node in reservoir_list + majorflow_list:
            ym1 = self._shifted_values(
                self.timeseries_data, node, lag - 1, index, shift_cache
            )
            pywr_node = (
                f"reservoir_{node}" if node in reservoir_list else f"link_{node}"
            )
            wd = self.catchment_wc.loc[pywr_node, "Total_WD_MGD"]
            cu = self.catchment_wc.loc[pywr_node, "Total_CU_WD_Ratio"]
            consumption = np.minimum(y, cu * np.minimum(ym1, wd))
            return y - consumption

        return y

    def make_predictions(self, regressions):
        """Vectorized twin of ``PredictedTimeseriesPreprocessor.make_predictions``.

        Structurally identical to the base implementation — same index
        subsetting, same ``pred_df`` construction, same per-column zero init,
        and the same ``pred_df[col] += ...`` accumulation in the same
        travel-time-dict (node, lag) order, so the summation order is
        preserved bit-for-bit. The only change: perfect_foresight series come
        from the vectorized kernel instead of a per-day scalar loop. Any
        non-perfect-foresight mode still routes through the scalar
        ``_predict_value``. Keep the skeleton in sync with the base method.
        """
        if not self._vectorize_perfect_foresight:
            return super().make_predictions(regressions)

        if self.timeseries_data is not None:
            index = self.timeseries_data.index
        else:
            raise ValueError("No data loaded for making predictions")
        if self.start_date is not None:
            index = index[index >= self.start_date]
        if self.end_date is not None:
            index = index[index <= self.end_date]

        # Date-membership shifting requires unique, sorted axes (the scalar
        # path would fail differently on duplicates; fail loudly here).
        axes = [("prediction index", index),
                ("timeseries_data index", self.timeseries_data.index)]
        if self.starfit_releases is not None:
            axes.append(("starfit_releases index", self.starfit_releases.index))
        for label, ax in axes:
            if not (ax.is_unique and ax.is_monotonic_increasing):
                raise ValueError(
                    f"{label} must be unique and sorted for the vectorized "
                    f"perfect-foresight path."
                )

        pred_df = pd.DataFrame({"datetime": index})
        node_lags = self.get_prediction_node_lag_combinations()
        shift_cache = {}

        for col, node_lag_mode_list in node_lags.items():
            pred_df[col] = np.zeros(len(index))
            for (node, lag), mode in node_lag_mode_list:
                if mode == "perfect_foresight":
                    vals = self._predict_perfect_foresight_series(
                        index, node, lag, shift_cache
                    )
                else:
                    vals = np.array([
                        self._predict_value(idx, index[idx], node, lag, mode, regressions)
                        for idx in range(len(index))
                    ])
                pred_df[col] += vals

        return pred_df

    def load(self):
        """Load catchment water consumption and each rank's realization slice.

        Water consumption CSV is broadcast from rank 0 as a raw string to avoid
        pickle-based DataFrame serialization. Each rank then opens the ensemble
        HDF5 independently to read only its own realization slice, replacing the
        rank-0-reads-all + pickle-scatter pattern that fails with MPI_ERR_ARG
        when the aggregate scatter payload exceeds INT_MAX at high rank counts.
        """
        # Load water consumption CSV on rank 0; broadcast as raw string.
        fname = self.input_dirs["sw_avg_wateruse_pywrdrb_catchments_mgd.csv"]

        if self.rank == 0:
            print(f"Rank 0: Loading catchment water consumption data...")
            with open(fname, "r") as f:
                wc_str = f.read()
        else:
            wc_str = None

        if self.use_mpi:
            wc_str = self.comm.bcast(wc_str, root=0)

        wc = pd.read_csv(io.StringIO(wc_str))
        wc.index = wc["node"]
        self.catchment_wc = wc

        # Resolve realization ids on rank 0, broadcast to all ranks via a sentinel
        # envelope so a rank-0 failure raises everywhere instead of hanging in bcast.
        # Input HDF5 uses node-first layout: realization ids are stored in
        # /<node>/.attrs["column_labels"], not as top-level keys.
        def _get_ids():
            if self.realization_ids is not None:
                return [str(r) for r in self.realization_ids]
            with h5py.File(self.ensemble_hdf5_file, "r") as f:
                labels = f[pywrdrb_all_nodes[0]].attrs["column_labels"]
                return [str(l) for l in labels]

        if self.use_mpi:
            self.realization_ids = bcast_with_error(self.comm, self.rank, _get_ids)
        else:
            self.realization_ids = _get_ids()

        # Each rank reads only its own realization slice. Concurrent read-only opens
        # on the same HDF5 are safe on Lustre/GPFS with the serial h5py driver
        # (HDF5_USE_FILE_LOCKING=FALSE is set on _mpi_utils import).
        if self.use_mpi:
            my_ids = list(np.array_split(self.realization_ids, self.size)[self.rank])
            self.comm.Barrier()  # align all ranks before concurrent opens
        else:
            my_ids = list(self.realization_ids)

        t0 = time.time()
        with h5py.File(self.ensemble_hdf5_file, "r") as f:
            self.realization_data = {
                str(rid): self._extract_realization_from_open_file(f, rid)
                for rid in my_ids
            }
        print(
            f"[rank {self.rank}/{self.size}] load: "
            f"read {len(my_ids)} realizations in {time.time() - t0:.1f}s"
        )

        if self.use_mpi:
            self.comm.Barrier()  # all ranks synced before process() proceeds

        # Read precomputed STARFIT releases for perfect_foresight mode.
        # Each rank loads only its own realization slice from
        # presimulated_releases_mgd.hdf5 (written by STARFITReleaseEnsemblePreprocessor).
        # Concurrent read-only HDF5 opens across ranks are safe because
        # _mpi_utils sets HDF5_USE_FILE_LOCKING=FALSE at import time.
        if "perfect_foresight" in self.modes:
            presim_hdf5 = os.path.join(
                str(self.pn.sc.get(f"flows/{self.flow_type}")),
                "presimulated_releases_mgd.hdf5",
            )
            if not os.path.exists(presim_hdf5):
                raise FileNotFoundError(
                    f"Pre-simulated STARFIT releases ensemble HDF5 not found: {presim_hdf5}\n"
                    f"perfect_foresight mode now requires this artifact. Generate it via:\n"
                    f"  from pywrdrb.pre import STARFITReleaseEnsemblePreprocessor\n"
                    f"  STARFITReleaseEnsemblePreprocessor(inflow_type='{self.flow_type}', "
                    f"realization_ids={list(self.realization_ids)}).run()"
                )
            with h5py.File(presim_hdf5, "r") as hf:
                # Validate every reservoir we need is present.
                missing_res = [
                    r for r in starfit_reservoir_list if r not in hf.keys()
                ]
                if missing_res:
                    raise ValueError(
                        f"Missing reservoirs in {presim_hdf5}: {missing_res}. "
                        f"Regenerate the file with the full starfit_reservoir_list."
                    )

                # Read the canonical date axis once.
                first_node = starfit_reservoir_list[0]
                date_node = hf[first_node]
                date_key = "date" if "date" in date_node else "datetime"
                raw_dates = date_node[date_key][:]
                date_index = pd.to_datetime(
                    pd.Index(
                        [
                            d.decode() if isinstance(d, bytes) else str(d)
                            for d in raw_dates
                        ]
                    )
                )

                # Validate every realization we need is present (column_labels of any node).
                available = {
                    str(l) for l in hf[first_node].attrs["column_labels"]
                }
                for rid in my_ids:
                    rid_s = str(rid)
                    if rid_s not in available:
                        raise ValueError(
                            f"Realization {rid_s} not present in {presim_hdf5}. "
                            f"Available: {sorted(available, key=lambda x: (len(x), x))[:10]}..."
                        )
                    rel_data = {
                        node: hf[node][rid_s][:] for node in starfit_reservoir_list
                    }
                    self._starfit_release_data[rid_s] = pd.DataFrame(
                        rel_data, index=date_index
                    )

        if self.rank == 0:
            print(
                f"Processing {len(self.realization_ids)} realizations across {self.size} processes"
            )

    def process(self):
        """Process ensemble predictions using MPI parallelization."""
        if not hasattr(self, "realization_data"):
            self.load()

        # Distribute realizations across MPI processes
        realizations_per_rank = np.array_split(self.realization_ids, self.size)
        my_realizations = realizations_per_rank[self.rank]

        local_predictions = {}

        if self.rank == 0:
            print(
                f"Rank {self.rank}: Processing {len(my_realizations)} realizations..."
            )

        for i, realization_id in enumerate(my_realizations):
            if self.rank == 0 and (i + 1) % max(1, len(my_realizations) // 5) == 0:
                print(
                    f"Rank 0: Processing realization {i+1}/{len(my_realizations)}: {realization_id}"
                )

            # Use pre-loaded realization data (each rank reads its own slice in load())
            self.timeseries_data = self.realization_data[str(realization_id)]

            # Use pre-loaded STARFIT releases for this realization (perfect_foresight mode).
            # The releases were read from presimulated_releases_mgd.hdf5 in load().
            if self._starfit_release_data:
                self.starfit_releases = self._starfit_release_data[
                    str(realization_id)
                ]

            # Train regressions and make predictions for this realization
            regressions = self.train_regressions()
            realization_predictions = self.make_predictions(regressions)

            local_predictions[str(realization_id)] = realization_predictions

        if self.rank == 0:
            print(f"Rank 0: Completed processing all assigned realizations")

        # Collect all local predictions on rank 0 via point-to-point send/recv,
        # replacing comm.gather which hits INT_MAX pickle limits at high rank counts.
        if self.use_mpi:
            merged = point_to_point_gather(
                self.comm, self.rank, self.size, local_predictions
            )
            if self.rank == 0:
                self.ensemble_predictions = merged
        else:
            self.ensemble_predictions = dict(local_predictions)

    def save(self):
        """Save ensemble predictions to HDF5 format."""
        if self.rank == 0:
            if not self.ensemble_predictions:
                raise ValueError(
                    "No ensemble predictions to save. Run process() first."
                )

            fname = self.output_dirs["predicted_inflows_mgd.hdf5"]

            # All realizations share one date axis, so the (expensive) string
            # formatting runs once and is reused whenever the axis matches;
            # a per-group ``datetime`` dataset is still written for every
            # realization (PredictionEnsemble reads it from the group).
            ref_datetime = None
            ref_strings = None

            with h5py.File(fname, "w") as hf:
                for realization_id, predictions_df in self.ensemble_predictions.items():
                    # Create group for this realization
                    realization_group = hf.create_group(str(realization_id))

                    # Store datetime. h5py 3.x requires an explicit string
                    # dtype for Python object arrays of strings; without it
                    # h5py raises ``TypeError: Object dtype dtype('O') has
                    # no native HDF5 equivalent``.
                    dt_series = predictions_df["datetime"]
                    if ref_datetime is None or not dt_series.equals(ref_datetime):
                        ref_datetime = dt_series
                        ref_strings = dt_series.astype(str).values
                    realization_group.create_dataset(
                        "datetime",
                        data=ref_strings,
                        dtype=h5py.string_dtype(encoding="utf-8"),
                    )

                    # Store prediction columns
                    for col in predictions_df.columns:
                        if col != "datetime":
                            realization_group.create_dataset(
                                col, data=predictions_df[col].values
                            )

            print(f"Saved ensemble predictions to {fname}")

        # Ensure all processes wait for save to complete
        if self.use_mpi:
            self.comm.barrier()
