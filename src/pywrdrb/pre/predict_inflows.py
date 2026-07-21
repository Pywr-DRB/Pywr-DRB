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
- "gage_flow": Raw natural gage flow lookup (legacy, does not account for STARFIT operations)

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
"""
import io
import h5py
import numpy as np
import pandas as pd
from pywrdrb.pre.predict_timeseries import PredictedTimeseriesPreprocessor
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
    (e.g., regression, perfect foresight, gage flow).

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
                Options: "regression_disagg", "perfect_foresight", "gage_flow"
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
            "gage_flow",
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
        Loads catchment inflows, gage flows, STARFIT releases, and water consumption data.

        For regression modes: Uses catchment_inflow_mgd.csv (marginal/incremental flows)
        For perfect_foresight mode: Uses catchment_inflow_mgd.csv + pre-simulated STARFIT releases
        For gage_flow mode: Uses gage_flow_mgd.csv (total natural flow at gages)
        """
        # Determine which data sources to load based on modes
        has_regression = any(mode.startswith("regression") for mode in self.modes)
        has_perfect_foresight = "perfect_foresight" in self.modes
        has_gage_flow = "gage_flow" in self.modes

        # Load catchment inflow data for regression and perfect_foresight modes
        # used to predict inflows at Montague and Trenton via aggregation
        if has_regression or has_perfect_foresight:
            fname = self.input_dirs["catchment_inflow_mgd.csv"]
            self.timeseries_data = pd.read_csv(fname, index_col=0, parse_dates=True)
            self.timeseries_data.index = pd.DatetimeIndex(self.timeseries_data.index)
        else:
            self.timeseries_data = None

        # Load gage flow data for gage_flow mode
        # provides total natural flow directly at gage locations (delMontague, delTrenton)
        if has_gage_flow:
            gage_fname = self.pn.sc.get(f"flows/{self.flow_type}") / "gage_flow_mgd.csv"
            self.gage_data = pd.read_csv(gage_fname, index_col=0, parse_dates=True)
            self.gage_data.index = pd.DatetimeIndex(self.gage_data.index)
        else:
            self.gage_data = None

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
        if self.timeseries_data is None and self.gage_data is None:
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
        For gage_flow: Uses gage data directly at target locations (no aggregation)
        """

        # Dictionary to hold regression combination
        # keys are strings of the form "target_lag_mode"
        # values are lists of tuples (node, lag)
        combos = {}

        # Montague predictions
        for lag in [1, 2]:
            for mode in self.modes:
                col = f"delMontague_lag{lag}_{mode}"

                if mode == "gage_flow":
                    # Use gage data directly - no aggregation needed
                    combos[col] = [(("delMontague", lag), mode)]
                elif mode == "perfect_foresight" or mode.startswith("regression"):
                    # Aggregate upstream catchments with travel times
                    combos[col] = []
                    for node, travel_time in self.node_to_montague_travel_time.items():
                        combos[col].append(((node, lag - travel_time), mode))

        # Trenton predictions
        for lag in [1, 2, 3, 4]:
            for mode in self.modes:
                col = f"delTrenton_lag{lag}_{mode}"

                if mode == "gage_flow":
                    # Use gage data directly - no aggregation needed
                    combos[col] = [(("delTrenton", lag), mode)]
                elif mode == "perfect_foresight" or mode.startswith("regression"):
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

        # STARFIT simulator instance (created once, reused per realization)
        self._starfit_simulator = None

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
        # Extract timeseries data from realization for each node
        data = {}

        for node in pywrdrb_all_nodes:
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

    def load(self):
        """Load available realization IDs, catchment water consumption, and all realization data.

        Rank 0 performs all HDF5 reads. Data is distributed to other ranks using
        MPI primitives that avoid pickle-based large-object broadcasts:
        - wc CSV is broadcast as a raw UTF-8 string (avoids DataFrame pickle)
        - realization DataFrames are scattered so each rank receives only its slice
        """

        ### Load water consumption CSV on rank 0; broadcast as raw string to avoid
        # pickle-based DataFrame bcast which fails on some HPC MPI stacks.
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

        ### Rank 0 reads all realization data from HDF5, then scatters each rank's
        # assigned slice. This avoids (a) concurrent file opens and (b) broadcasting
        # a large dict of all DataFrames to every rank.
        if self.rank == 0:
            print(f"Rank 0: Reading all realization data from HDF5...")
            with h5py.File(self.ensemble_hdf5_file, "r") as f:
                if self.realization_ids is None:
                    self.realization_ids = [key for key in f.keys()]
                all_data = {
                    rid: self._extract_realization_from_open_file(f, rid)
                    for rid in self.realization_ids
                }
        else:
            all_data = None

        if self.use_mpi:
            # Broadcast the realization ID list (small) so all ranks know the full set
            self.realization_ids = self.comm.bcast(self.realization_ids, root=0)

            # Build per-rank slices on rank 0, then scatter one slice per rank
            if self.rank == 0:
                slices = [
                    {rid: all_data[rid] for rid in chunk}
                    for chunk in np.array_split(self.realization_ids, self.size)
                ]
                print(f"Rank 0: Scattering realization data to {self.size} ranks...")
            else:
                slices = None
            self.realization_data = self.comm.scatter(slices, root=0)
        else:
            self.realization_data = all_data

        # Initialize STARFIT simulator once if perfect_foresight mode is used
        if "perfect_foresight" in self.modes:
            from pywrdrb.pre.generate_presimulated_releases import (
                STARFITOfflineSimulator,
            )

            self._starfit_simulator = STARFITOfflineSimulator(initial_volume_frac=0.8)
            self._starfit_simulator.load_parameters()

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

            # Use pre-loaded realization data (read by rank 0 and broadcast in load())
            self.timeseries_data = self.realization_data[str(realization_id)]

            # Run STARFIT simulation for this realization if perfect_foresight
            if self._starfit_simulator is not None:
                self.starfit_releases = self._starfit_simulator.simulate_all(
                    self.timeseries_data
                )

            # Train regressions and make predictions for this realization
            regressions = self.train_regressions()
            realization_predictions = self.make_predictions(regressions)

            local_predictions[str(realization_id)] = realization_predictions

        if self.rank == 0:
            print(f"Rank 0: Completed processing all assigned realizations")

        # Gather all predictions to rank 0
        if self.use_mpi:
            all_predictions = self.comm.gather(local_predictions, root=0)
        else:
            all_predictions = [local_predictions]

        if self.rank == 0:
            # Combine predictions from all processes
            for predictions_dict in all_predictions:
                self.ensemble_predictions.update(predictions_dict)

    def save(self):
        """Save ensemble predictions to HDF5 format."""
        if self.rank == 0:
            if not self.ensemble_predictions:
                raise ValueError(
                    "No ensemble predictions to save. Run process() first."
                )

            fname = self.output_dirs["predicted_inflows_mgd.hdf5"]

            with h5py.File(fname, "w") as hf:
                for realization_id, predictions_df in self.ensemble_predictions.items():
                    # Create group for this realization
                    realization_group = hf.create_group(str(realization_id))

                    # Store datetime
                    datetime_strings = predictions_df["datetime"].astype(str).values
                    realization_group.create_dataset("datetime", data=datetime_strings)

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
