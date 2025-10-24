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

import h5py
import numpy as np

import pandas as pd
from pywrdrb.pre.predict_timeseries import PredictedTimeseriesPreprocessor
from pywrdrb.utils.hdf5 import extract_realization_from_hdf5


__all__ = ["PredictedDiversionPreprocessor", 
           "PredictedDiversionEnsemblePreprocessor"]

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
    def __init__(self,
                 flow_type=None,
                 start_date=None,
                 end_date=None,
                 modes=('regression_disagg',),
                 use_log=True,
                 remove_zeros=False,
                 use_const=False):
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
        super().__init__(flow_type, 
                         start_date, 
                         end_date, 
                         use_log, 
                         remove_zeros, 
                         use_const)
        
        # List of regression modes options
        self.regression_mode_options = [
            "regression_disagg",
            "perfect_foresight",
            "moving_average",
            "same_day",
        ]

        # Modes being used; check validity        
        self.modes = modes
        for mode in self.modes:
            assert mode in self.regression_mode_options, f"Invalid regression mode: {mode}. Must be one of {self.regression_mode_options}."
    
        # Input files used for prediction
        if flow_type is None:
            # use default historical diversions
            diversion_fname = self.pn.diversions.get("diversion_nj_extrapolated_mgd.csv")
            output_fname = self.pn.get("diversions") / "predicted_diversions_mgd.csv"
        else:
            diversion_fname = self.pn.sc.get(f"flows/{self.flow_type}") / "diversion_nj_extrapolated_mgd.csv"
            output_fname = self.pn.sc.get(f"flows/{self.flow_type}") / "predicted_diversions_mgd.csv"
            
        self.input_dirs = {
            "diversion_nj_extrapolated_mgd.csv" : diversion_fname,
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
        assert self.predicted_timeseries is not None, "Predicted timeseries is None. Cannot save."
        
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
    
    def __init__(self,
                 flow_type,
                 ensemble_hdf5_file,
                 realization_ids=None,
                 start_date=None,
                 end_date=None,
                 modes=('regression_disagg',),
                 use_log=True,
                 remove_zeros=False,
                 use_const=False,
                 use_mpi=False):
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
        """
        super().__init__(flow_type, start_date, end_date, modes, use_log, remove_zeros, use_const)
        
        self.ensemble_hdf5_file = ensemble_hdf5_file
        self.realization_ids = realization_ids
        
        self.use_mpi = use_mpi
        if self.use_mpi:
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
            "predicted_diversions_mgd.hdf5": self.pn.sc.get(f"flows/{self.flow_type}") / "predicted_diversions_mgd.hdf5",
        }
        
        # Storage for ensemble results
        self.ensemble_predictions = {}

    def load(self):
        """Load available realization IDs to be distributed."""        
        # Get available realization IDs if not specified
        if self.realization_ids is None:
            with h5py.File(self.ensemble_hdf5_file, 'r') as f:
                self.realization_ids = [key for key in f.keys()]
        
        if self.rank == 0:
            print(f"Processing {len(self.realization_ids)} realizations across {self.size} processes")

    def process(self):
        """Process ensemble predictions using MPI parallelization."""
        if not hasattr(self, 'realization_ids'):
            self.load()
        
        # Distribute realizations across MPI processes
        
        realizations_per_rank = np.array_split(self.realization_ids, self.size)
        my_realizations = realizations_per_rank[self.rank]
        
        local_predictions = {}
        
        # Process assigned realizations
        for realization_id in my_realizations:
            if self.rank == 0:
                print(f"Processing realization {realization_id}")
            
            # Extract realization data
            # Note: diversion HDF5 files are stored by realization, not by node
            self.timeseries_data = extract_realization_from_hdf5(
                self.ensemble_hdf5_file,
                realization_id,
                stored_by_node=False
            )

            # Create 'demand_nj' column from 'D_R_Canal' (matching base class load() behavior)
            if 'D_R_Canal' in self.timeseries_data.columns:
                self.timeseries_data["demand_nj"] = self.timeseries_data["D_R_Canal"]

            # Train regressions and make predictions for this realization
            regressions = self.train_regressions()
            realization_predictions = self.make_predictions(regressions)
            
            local_predictions[str(realization_id)] = realization_predictions
        
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
                raise ValueError("No ensemble predictions to save. Run process() first.")
            
            fname = self.output_dirs["predicted_diversions_mgd.hdf5"]
            
            with h5py.File(fname, 'w') as hf:
                for realization_id, predictions_df in self.ensemble_predictions.items():
                    # Create group for this realization
                    realization_group = hf.create_group(realization_id)

                    # Store column labels as attribute for compatibility with extract_realization_from_hdf5
                    column_labels = list(predictions_df.columns)
                    realization_group.attrs['column_labels'] = column_labels

                    # Store datetime
                    datetime_strings = predictions_df['datetime'].astype(str).values
                    realization_group.create_dataset('datetime', data=datetime_strings)

                    # Store prediction columns
                    for col in predictions_df.columns:
                        if col != 'datetime':
                            realization_group.create_dataset(col, data=predictions_df[col].values)
            
            print(f"Saved ensemble diversion predictions to {fname}")
        
        # Ensure all processes wait for save to complete
        if self.use_mpi:
            self.comm.barrier()