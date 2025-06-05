"""
Preprocessor for generating inflow predictions at nodes in the pywrdrb model.

Overview: 
This class generates lag-based inflow predictions/forecasts at Montague and Trenton,
which are used to determine NYC and lower basin reservoir operations while accounting for travel time. 
It uses regression models, trained on historical data, to predict flows 1-4 days ahead
based on catchment-level data and travel times between nodes. The output data has 
multiple different columns corresponding to different prediction nodes, lead time lags, and 
regression modes. 

Technical Notes: 
- Extends PredictedTimeseriesPreprocessor with specific inflow prediction logic
- Incorporates travel times to properly account for flow routing
- Adjusts predictions for water consumption in each catchment
- Outputs predictions for 1-4 days ahead at Montague and Trenton

Example Usage:
from pywrdrb.pre import PredictedInflowPreprocessor
inflow_predictor = PredictedInflowPreprocessor(flow_type="pub_nhmv10_BC_withObsScaled", start_date="1945-01-01", end_date="2023-12-31", modes=("regression_disagg",),)
inflow_predictor.process()
inflow_predictor.save()

Links:
- See SI for Hamilton et al. (2024) for more details on the method formulation.

Change Log:
TJA, 2025-05-07, Minor fixes + docstrings
"""
import h5py
import pandas as pd
from pywrdrb.pre.predict_timeseries import PredictedTimeseriesPreprocessor
from pywrdrb.utils.hdf5 import extract_realization_from_hdf5

__all__ = ["PredictedInflowPreprocessor",
           "PredictedInflowEnsemblePreprocessor"]

class PredictedInflowPreprocessor(PredictedTimeseriesPreprocessor):
    """
    Predicts catchment inflows at Montague and Trenton using specified modes
    (e.g., regression, perfect foresight, moving average).
    
    Example usage:
    ```python
    from pywrdrb.pre import PredictedInflowPreprocessor

    inflow_predictor = PredictedInflowPreprocessor(flow_type="nhmv10", start_date="1983-10-01", end_date="2016-12-31", modes=("regression_disagg",),)
    
    inflow_predictor.process()
    inflow_predictor.save()
    ```
    """
    def __init__(self,
                 flow_type,
                 start_date=None,
                 end_date=None,
                 modes=('regression_disagg',),
                 use_log=True,
                 remove_zeros=False,
                 use_const=False):
        """
        Initialize the PredictedInflowPreprocessor.
        
        Args:
            flow_type (str): Label for the dataset.
            start_date (bool, None): Start date for the time series. If None, match the input data.
            end_date (bool, None): End date for the time series. If None, match the input data.
            modes (tuple): Modes to use for prediction. Default is ('regression_disagg',).
            use_log (bool): Whether to use log transformation. Default is True.
            remove_zeros (bool): Whether to remove zero values. Default is False.
            use_const (bool): Whether to use a constant in regression. Default is False.
        
        Returns:
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
        self.input_dirs = {
            "sw_avg_wateruse_pywrdrb_catchments_mgd.csv": self.pn.catchment_withdrawals.get("sw_avg_wateruse_pywrdrb_catchments_mgd.csv"),
            "catchment_inflow_mgd.csv": self.pn.sc.get(f"flows/{self.flow_type}") / "catchment_inflow_mgd.csv",
        }
        
        # Output locations for predicted timeseries
        self.output_dirs = {
            "predicted_inflows_mgd.csv": self.pn.sc.get(f"flows/{self.flow_type}") / "predicted_inflows_mgd.csv",
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
        

    def load(self):
        """
        Loads catchment inflows and water consumption data.
        """
        # Load catchment inflow data
        # used to predict inflows at Montague and Trenton
        fname = self.input_dirs["catchment_inflow_mgd.csv"]
        self.timeseries_data = pd.read_csv(fname, index_col=0, parse_dates=True)
        self.timeseries_data.index = pd.DatetimeIndex(self.timeseries_data.index)

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
        assert self.predicted_timeseries is not None, "Predicted timeseries is None. Cannot save."
        
        # Save
        fname = self.output_dirs["predicted_inflows_mgd.csv"]
        self.predicted_timeseries.to_csv(fname, index=False)



    def process(self):
        """Run full prediction workflow."""
        # FIXED: Ensure data is loaded first
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
        """
        
        # Dictionary to hold regression combination
        # keys are strings of the form "target_lag_mode"
        # values are lists of tuples (node, lag)
        combos = {}

        # Montague
        # (node, (lag - travel_time)) pairs
        for lag in [1, 2]:
            for node, travel_time in self.node_to_montague_travel_time.items():
                
                if lag - travel_time < 0:
                    continue
                
                
                node_lag = [
                    (node, lag - travel_time),
                ]
                
                # Create node-lag pairs for each prediction mode
                node_lag = [(node, lag - travel_time)]
                
                # Add combinations for all requested modes
                for mode in self.modes:
                    col = f"delMontague_lag{lag}_{mode}"
                    if col not in combos:
                        combos[col] = []
                    # Add this node's contribution to the prediction
                    combos[col].extend([((n, l), mode) for (n, l) in node_lag])

        # For Trenton
        # (node, (lag - travel_time)) pairs
        for lag in [1, 2, 3, 4]:
            for node, travel_time in self.node_to_trenton_travel_time.items():
                
                if lag - travel_time < 0:
                    continue
                
                node_lag = [
                    (node, lag - travel_time),
                ]

                # Add combinations for all requested modes
                for mode in self.modes:
                    col = f"delTrenton_lag{lag}_{mode}"
                    if col not in combos:
                        combos[col] = []
                    # Add this node's contribution to the prediction
                    combos[col].extend([((n, l), mode) for (n, l) in node_lag])

        return combos


class PredictedInflowEnsemblePreprocessor(PredictedInflowPreprocessor):
    """
    Generates ensemble predictions for inflows at Montague and Trenton using MPI parallelization.
    
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
        Initialize the PredictedInflowEnsemblePreprocessor.
        
        Args:
            flow_type: Label for the dataset.
            ensemble_hdf5_file: Path to HDF5 file containing ensemble inflow data.
            realization_ids: List of realization IDs to process. If None, uses all available.
            start_date: Start date for predictions.
            end_date: End date for predictions.
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
            "predicted_inflows_mgd.hdf5": self.pn.sc.get(f"flows/{self.flow_type}") / "predicted_inflows_mgd.hdf5",
        }
        
        # Storage for ensemble results
        self.ensemble_predictions = {}

    def load(self):
        """Load available realization IDs and catchment water consumption data."""
        # Load water consumption data (same for all realizations)
        fname = self.input_dirs["sw_avg_wateruse_pywrdrb_catchments_mgd.csv"]
        wc = pd.read_csv(fname)
        wc.index = wc["node"]
        self.catchment_wc = wc
        
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
            self.timeseries_data = extract_realization_from_hdf5(
                self.ensemble_hdf5_file, 
                realization_id, 
                stored_by_node=True
            )
            
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
            
            fname = self.output_dirs["predicted_inflows_mgd.hdf5"]
            
            with h5py.File(fname, 'w') as hf:
                for realization_id, predictions_df in self.ensemble_predictions.items():
                    # Create group for this realization
                    realization_group = hf.create_group(realization_id)
                    
                    # Store datetime
                    datetime_strings = predictions_df['datetime'].astype(str).values
                    realization_group.create_dataset('datetime', data=datetime_strings)
                    
                    # Store prediction columns
                    for col in predictions_df.columns:
                        if col != 'datetime':
                            realization_group.create_dataset(col, data=predictions_df[col].values)
            
            print(f"Saved ensemble predictions to {fname}")
        
        # Ensure all processes wait for save to complete
        if self.use_mpi:
            self.comm.barrier()