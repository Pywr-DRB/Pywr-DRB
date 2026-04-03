"""
Preprocessor for generating NJ diversion predictions for Delaware-Raritan Canal operations.

Overview:
This module creates lag-based predictions for NJ diversions from the
Delaware-Raritan Canal, which are needed to properly implement the NYC and lower basin
reservoir releases, when predicted flow is less than target flow.
It uses regression models trained on historical diversion data to forecast demand
1-4 days ahead, which helps determine appropriate reservoir releases that account
for downstream travel time and anticipated withdrawals.

Technical Notes:
- Extends PredictedTimeseriesPreprocessor specifically for diversion predictions
- The historical NJ diversions are based on DR Canal gauge flow
- Uses historical NJ diversion data from the extrapolated dataset
- Creates prediction files for "demand_nj" used by FFMP parameters
- Supports multiple prediction modes (regression_disagg, perfect_foresight, etc.)
- Predictions follow the pattern "demand_nj_lag{1-4}_{mode}"
- Output is saved as a CSV file in the data/diversions directory for use by the model

Example usage:
from pywrdrb.pre import PredictedDiversionPreprocessor
diversion_predictor = PredictedDiversionPreprocessor(
    start_date="1945-01-01",
    end_date="2023-12-31",
    modes=("regression_disagg",),
)
diversion_predictor.process()
diversion_predictor.save()


Links:
- See SI for Hamilton et al. (2024) for more details on the method formulation.

Change Log:
TJA, 2025-05-07, review+docstrings
"""

import io
import h5py
import numpy as np

import pandas as pd
from pywrdrb.pre.predict_timeseries import PredictedTimeseriesPreprocessor
from pywrdrb.utils.hdf5 import extract_realization_from_hdf5


__all__ = ["PredictedDiversionPreprocessor", "PredictedDiversionEnsemblePreprocessor"]


class PredictedDiversionPreprocessor(PredictedTimeseriesPreprocessor):
    """
    Predicts NJ diversions from the Delaware-Raritan Canal using regression models.
    (e.g., regression, perfect foresight, moving average).

    Example usage:
    ```python
    from pywrdrb.pre import PredictedDiversionPreprocessor
    diversion_predictor = PredictedDiversionPreprocessor(
        start_date="1983-10-01",
        end_date="2016-12-31",
        modes=("regression_disagg",),
    )
    diversion_predictor.process()
    diversion_predictor.save()
    ```
    """

    def __init__(
        self,
        flow_type=None,
        start_date=None,
        end_date=None,
        modes=("regression_disagg",),
        use_log=True,
        remove_zeros=False,
        use_const=False,
    ):
        """Initialize the PredictedDiversionPreprocessor.

        Parameters
        ----------
        start_date : str, None
            Start date for the time series. If None, match the input data.
        end_date : str, None
            End date for the time series. If None, match the input data.
        modes : tuple
            Modes to use for prediction. Default is ('regression_disagg',). Options include:
            "regression_disagg", "perfect_foresight", "moving_average", "same_day".
        use_log : bool
            Whether to use log transformation for model vars. Default is True.
        remove_zeros : bool
            Whether to remove zero values. Default is False.
        use_const : bool
            Whether to use a constant/intercept in regression. Default is False.

        Returns
        -------
        None
        """

        # Initialize the PredictedTimeseriesPreprocessor
        super().__init__(
            flow_type, start_date, end_date, use_log, remove_zeros, use_const
        )

        # Valid prediction modes (fully tested and supported)
        self.regression_mode_options = [
            "regression_disagg",
            "perfect_foresight",
        ]

        # NOTE: The following modes have partial implementations but are not fully
        # tested or validated. They are left in the codebase for potential future
        # development but removed from valid options:
        # - "moving_average": Uses 7-day rolling average for predictions
        # - "same_day": Uses current day's observation as prediction
        # To enable these modes, complete their implementation and add thorough testing.

        # Modes being used; check validity
        self.modes = modes
        for mode in self.modes:
            assert (
                mode in self.regression_mode_options
            ), f"Invalid regression mode: {mode}. Must be one of {self.regression_mode_options}."

        # Input files used for prediction
        if flow_type is None:
            # use default historical diversions
            diversion_fname = self.pn.diversions.get(
                "diversion_nj_extrapolated_mgd.csv"
            )
            output_fname = self.pn.get("diversions") / "predicted_diversions_mgd.csv"
        else:
            diversion_fname = (
                self.pn.sc.get(f"flows/{self.flow_type}")
                / "diversion_nj_extrapolated_mgd.csv"
            )
            output_fname = (
                self.pn.sc.get(f"flows/{self.flow_type}")
                / "predicted_diversions_mgd.csv"
            )

        self.input_dirs = {
            "diversion_nj_extrapolated_mgd.csv": diversion_fname,
        }

        # Output locations for predicted timeseries
        self.output_dirs = {
            "predicted_diversions_mgd.csv": output_fname,
        }

    def load(self):
        """Load NJ diversions and catchment WC data (used for structural compatibility).

        Parameters
        ----------
        None

        Returns
        -------
        None
            timeseries_data is stored as a class attribute.
        """

        ### Load NJ diversions data
        fname = self.input_dirs["diversion_nj_extrapolated_mgd.csv"]
        df = pd.read_csv(fname, parse_dates=["datetime"])
        df.index = pd.DatetimeIndex(df["datetime"])
        df["demand_nj"] = df["D_R_Canal"]

        # # subset to the input start_date if provided
        # self.start_date = self.start_date if self.start_date is not None else df.index[0]
        # self.end_date = self.end_date if self.end_date is not None else df.index[-1]
        # self.timeseries_data = subset_timeseries(df, self.start_date, self.end_date)
        self.timeseries_data = df.copy()

    def process(self):
        """Run full prediction workflow.

        Steps:
        1. Load timeseries data (if not already loaded).
        2. Train regressions on the data.
        3. Make predictions using the trained regressions.

        Parameters
        ----------
        None

        Returns
        -------
        None
            The predicted_timeseries is stored as a class attribute.
        """
        if self.timeseries_data is None:
            self.load()
        regressions = self.train_regressions()
        self.predicted_timeseries = self.make_predictions(regressions)

    def save(self):
        """Save predicted diversion time series to CSV.

        Parameters
        ----------
        None

        Returns
        -------
        None
            The predicted timeseries is saved to the specified output directory.
        """
        # Make sure the predictions are done successfully
        assert (
            self.predicted_timeseries is not None
        ), "Predicted timeseries is None. Cannot save."

        # Save
        fname = self.output_dirs["predicted_diversions_mgd.csv"]
        self.predicted_timeseries.to_csv(fname, index=False)

    def get_prediction_node_lag_combinations(self):
        """Return dict of predicted diversion column names formatted as (node, lag, mode) tuples.

        Parameters
        ----------
        None

        Returns
        -------
        combos : dict
            Dictionary where keys are column names (e.g., "demand_nj_lag1_regression_disagg")
            and values are lists of tuples defining the ((node, lag), mode) for each regression.
        """
        combos = {}
        node = "demand_nj"
        for lag in [1, 2, 3, 4]:
            for mode in self.modes:
                col = f"{node}_lag{lag}_{mode}"
                combos[col] = [((node, lag), mode)]
        return combos


class PredictedDiversionEnsemblePreprocessor(PredictedDiversionPreprocessor):
    """
    Generates ensemble predictions for NJ diversions using MPI parallelization.

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
        Initialize the PredictedDiversionEnsemblePreprocessor.

        Args:
            flow_type: Label for the dataset.
            ensemble_hdf5_file: Path to HDF5 file containing ensemble diversion data.
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
            "predicted_diversions_mgd.hdf5": self.pn.sc.get(f"flows/{self.flow_type}")
            / "predicted_diversions_mgd.hdf5",
        }

        # Storage for ensemble results
        self.ensemble_predictions = {}

    def load(self):
        """Load available realization IDs and all realization data.

        Rank 0 performs all HDF5 reads, then scatters each rank's assigned slice.
        This avoids (a) concurrent file opens and (b) broadcasting a large dict of
        all DataFrames to every rank, both of which cause MPI_ERR_OTHER on HPC.
        """
        ### Rank 0 reads all realization data from HDF5, then scatters per-rank slices.
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

        if self.rank == 0:
            print(
                f"Processing {len(self.realization_ids)} realizations across {self.size} processes"
            )

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
        realization_group = hdf5_file[str(realization_id)]

        # Extract column labels
        column_labels = realization_group.attrs["column_labels"]

        # Extract timeseries data for each location
        data = {}
        for label in column_labels:
            dataset = realization_group[label]
            data[label] = dataset[:]

        # Get date indices - handle both 'date' and 'datetime' keys
        if "datetime" in realization_group.keys():
            dates = realization_group["datetime"][:].tolist()
        elif "date" in realization_group.keys():
            dates = realization_group["date"][:].tolist()
        else:
            raise KeyError("Neither 'date' nor 'datetime' found in realization group")

        data["datetime"] = dates

        # Combine into dataframe
        df = pd.DataFrame(data, index=dates)
        df.index = pd.to_datetime(df.index.astype(str))
        return df

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

            # Create 'demand_nj' column from 'D_R_Canal' (matching base class load() behavior)
            if "D_R_Canal" in self.timeseries_data.columns:
                self.timeseries_data["demand_nj"] = self.timeseries_data[
                    "D_R_Canal"
                ]

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

            fname = self.output_dirs["predicted_diversions_mgd.hdf5"]

            with h5py.File(fname, "w") as hf:
                for realization_id, predictions_df in self.ensemble_predictions.items():
                    # Create group for this realization
                    realization_group = hf.create_group(str(realization_id))

                    # Store column labels as attribute for compatibility with extract_realization_from_hdf5
                    column_labels = list(predictions_df.columns)
                    realization_group.attrs["column_labels"] = column_labels

                    # Store datetime
                    datetime_strings = predictions_df["datetime"].astype(str).values
                    realization_group.create_dataset("datetime", data=datetime_strings)

                    # Store prediction columns
                    for col in predictions_df.columns:
                        if col != "datetime":
                            realization_group.create_dataset(
                                col, data=predictions_df[col].values
                            )

            print(f"Saved ensemble diversion predictions to {fname}")

        # Ensure all processes wait for save to complete
        if self.use_mpi:
            self.comm.barrier()
