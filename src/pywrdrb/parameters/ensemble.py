"""
Custom parameter used to handle simulation ensemble input data.

Overview:
Pywr is designed to handle running simulations in parallel, but it is helpful to have some 
custom parameters to help facilitate this. These parameters are used to load ensemble, then
store the relevant realizations in a pandas DataFrame which is accessible during simulation. 

Technical Notes:
- The FlowEnsemble is used to access to inflow ensemble timeseries during the simulation period.
- The PredictionEnsemble is used to access an ensemble of flow prediction timeseries, which is used to inform NYC releases.
- #TODO: 
    - Should add some documentation or other standardization for the inflow ensemble data formating 

Links:
NA

Change Log:
TJA, 2025-05-07, Add docs.
"""
import os
import numpy as np
import pandas as pd
import h5py

from pywr.parameters import Parameter

from pywrdrb.path_manager import get_pn_object

pn = get_pn_object()


class FlowEnsemble(Parameter):
    """This parameter provides access to inflow ensemble timeseries.

    For a given inflow ensemble file, we want to load and access specific realizations
    for a given model run. These realizations are loaded from an HDF5 file, then 
    stored in a pandas DataFrame for easy access during simulation.

    Methods
    -------
    setup()
        Perform setup operations for the parameter. Automated pywr operation.
    value(timestep, scenario_index)
        Return the current flow for the specified timestep and scenario index. 
    load(model, data)
        Load the parameter from the model dictionary.
        
    Attributes
    ----------
    inflow_ensemble_indices : list
        The realization indices of the inflow ensemble to be used for this simulation.
    inflow_column_indices : list
        The column indices of the inflow ensemble DataFrame corresponding to the realization indices.
    inflow_ensemble : DataFrame
        The DataFrame containing the inflow ensemble data, indexed by datetime.
    """

    def __init__(self, model, name, inflow_type, inflow_ensemble_indices, **kwargs):
        """Initialize the FlowEnsemble parameter.
        
        Parameters
        ----------
        model : Model
            The pywrdrb.Model object.
        name : str
            The name of the node in the model.
        inflow_type : str
            The dataset label. Expects to find an HDF5 file with inflow ensemble data in the pn.flows.input_dir directory.
        inflow_ensemble_indices : list
            The realization indices of the inflow ensemble to be used for this simulation.
        **kwargs : dict
            Additional keyword arguments to be passed to the pywr.Parameter class. None used. 
        
        Returns
        -------
        None
        """
        super().__init__(model, **kwargs)

        # ensemble input file
        input_dir = pn.sc.get(f"flows/{inflow_type}")
        filename = os.path.join(input_dir, f"catchment_inflow_mgd.hdf5")

        # Load from hfd5 specific realizations
        with h5py.File(filename, "r") as file:
            node_inflow_ensemble = file[name]
            column_labels = node_inflow_ensemble.attrs["column_labels"]

            # Get timeseries
            data = {}
            for label in column_labels:
                data[label] = node_inflow_ensemble[label][:]

            date_column = "datetime" if "datetime" in column_labels else "date"
            datetime = node_inflow_ensemble[date_column][:].tolist()

        # Store in DF
        inflow_df = pd.DataFrame(data, index=datetime)
        inflow_df.index = pd.to_datetime(inflow_df.index.astype(str))

        ## Match ensemble indices to columns
        # inflow_ensemble_indices is a list of integers;
        # We need to: 
        # 1) verify that the indices are included in the df
        # 2) find the columns corresponding to these realization IDs
        inflow_ensemble_columns = []
        for real_id in inflow_ensemble_indices:
            assert (
                f"{real_id}" in inflow_df.columns
            ), f"The specified inflow_ensemble_index {real_id} is not available in the HDF file."
            inflow_ensemble_columns.append(
                np.argwhere(inflow_df.columns == f"{real_id}")[0][0]
            )

        self.inflow_ensemble_indices = inflow_ensemble_indices
        self.inflow_column_indices = inflow_ensemble_columns
        self.inflow_ensemble = inflow_df.iloc[:, inflow_ensemble_columns]

    def setup(self):
        """Perform setup operations for the parameter."""
        super().setup()

    def value(self, timestep, scenario_index):
        """Return the current flow across scenarios for the specified timestep and scenario index.

        This is automaticalled called by pywr during each timestep of the simulation.
        The timestep and scenario_index are passed in by pywr automatically.
        The scenario_index is used to determine which realization to use.

        Parameters
        ----------
        timestep : Timestep
            The timestep being evaluated.
        scenario_index : ScenarioIndex
            The index of the simulation scenario.
        
        Returns
        -------
        float
            The inflow value for the specified timestep and scenario.
        """
        s_id = self.inflow_ensemble_indices[scenario_index.global_id]
        return self.inflow_ensemble.loc[timestep.datetime, f"{s_id}"]

    @classmethod
    def load(cls, model, data):
        """Load the parameter using the pywrdrb.Model dictionary.
        
        Parameters
        ----------
        model : Model
            The pywrdrb.Model object.
        data : dict
            The dictionary containing the parameter data. Must include inflow_ensemble_indices and inflow_type.
        
        Returns
        -------
        FlowEnsemble
            An instance of the FlowEnsemble class, for the given model specifications.
        """
        name = data.pop("node")
        inflow_ensemble_indices = data.pop("inflow_ensemble_indices")
        inflow_type = data.pop("inflow_type")
        return cls(model, name, inflow_type, inflow_ensemble_indices, **data)


FlowEnsemble.register()


class PredictionEnsemble(Parameter):
    """Loads and stored ensemble of prediction timeseries used to inform NYC releases during simulation.

    When calculating NYC release, we need use forecast/predicted downstream flows to calculate the
    releases neede to maintain the Montague and Trenton flow targets in 1-4 days ahead.
    These predictions are generated prior to the simulation (e.g., pywrdrb.pre.PredictedInflowPreprocessor) 
    and stored in an HDF5 file, with unique predictions for each realization member. 

    Methods
    -------
    setup()
        Perform setup operations for the parameter. Automated pywr operation.
    value(timestep, scenario_index)
        Return the current flow for the specified timestep and scenario index.
    load(model, data)
        Load the parameter from the model dictionary.

    Attributes
    ----------
    ensemble_indices : list
        The realization indices of the inflow ensemble to be used for this simulation.
    pred_column_indices : list
        The column indices of the inflow ensemble DataFrame corresponding to the realization indices.
    pred_ensemble : DataFrame
        The DataFrame containing the inflow ensemble data, indexed by datetime.
    """
    def __init__(self, model, column, 
                 inflow_type, ensemble_indices, 
                 **kwargs):
        """Initialize the PredictionEnsemble parameter.
        
        Parameters
        ----------
        model : Model
            The pywrdrb.Model object.
        column : str
            The name of the column in the HDF5 file to be used for the ensemble.
        inflow_type : str
            The dataset label. Expects to find an HDF5 file with inflow ensemble data in the pn.flows.input_dir directory.
        ensemble_indices : list
            The realization indices of the inflow ensemble to be used for this simulation.
        **kwargs : dict
            Additional keyword arguments to be passed to the pywr.Parameter class. None used.
        
        Returns
        -------
        None        
        """
        
        super().__init__(model, **kwargs)

        # input file corresponding to the inflow_type
        input_dir = pn.sc.get(f"flows/{inflow_type}")
        filename = os.path.join(input_dir, f"predicted_inflows_mgd.hdf5")
        prediction_ensemble = {}

        # Load from hfd5 specific realizations
        with h5py.File(filename, "r") as file:
            for i in ensemble_indices:
                prediction_realization = file[f"{i}"]

                column_labels = list(prediction_realization.keys())
                assert (
                    column in column_labels
                ), f"The specified column {column} is not available in the HDF file."

                # Get timeseries values
                prediction_ensemble[f"{i}"] = prediction_realization[column][:]

            # Pull datetime from one of the realizations
            date_column = "datetime" if "datetime" in column_labels else "date"
            datetime = prediction_realization[date_column][:].tolist()

        # Store in DF
        prediction_ensemble_df = pd.DataFrame(prediction_ensemble, index=datetime)
        prediction_ensemble_df.index = pd.to_datetime(
            prediction_ensemble_df.index.astype(str)
        )

        ## Match ensemble indices to columns
        # inflow_ensemble_indices is a list of integers;
        # We need to: 
        # 1) verify that the indices are included in the df
        # 2) find the columns corresponding to these realization IDs
        ensemble_columns = []
        for real_id in ensemble_indices:
            assert (
                f"{real_id}" in prediction_ensemble_df.columns
            ), f"The specified inflow_ensemble_index {real_id} is not available in the HDF file."
            ensemble_columns.append(
                np.argwhere(prediction_ensemble_df.columns == f"{real_id}")[0][0]
            )

        self.pred_ensemble_indices = ensemble_indices
        self.pred_column_indices = ensemble_columns
        self.pred_ensemble = prediction_ensemble_df.iloc[:, ensemble_columns]

    def setup(self):
        """Perform setup operations for the parameter."""
        super().setup()

    def value(self, timestep, scenario_index):
        """Return the current flow across scenarios for the specified timestep and scenario index.

        This is automaticalled called by pywr during each timestep of the simulation.
        The timestep and scenario_index are passed in by pywr automatically.

        Parameters
        ----------
        timestep : Timestep
            The timestep being evaluated.
        scenario_index : ScenarioIndex
            The index of the simulation scenario.
        
        Returns
        -------
        float
            The prediction value for the specified timestep and scenario.
        """
        s_id = self.pred_ensemble_indices[scenario_index.global_id]
        return self.pred_ensemble.loc[timestep.datetime, f"{s_id}"]

    @classmethod
    def load(cls, model, data):
        """Load the parameter using the pywrdrb.Model dictionary.
        
        Parameters
        ----------
        model : Model
            The pywrdrb.Model object.
        data : dict
                The dictionary containing the parameter data. Must include column, ensemble_indices and inflow_type.
        
        Returns
        -------
        PredictionEnsemble
            An instance of the PredictionEnsemble class, for the given model specifications.
        """
        column = data.pop("column")
        ensemble_indices = data.pop("ensemble_indices")
        inflow_type = data.pop("inflow_type")
        return cls(model, column, inflow_type, ensemble_indices, **data)


PredictionEnsemble.register()
