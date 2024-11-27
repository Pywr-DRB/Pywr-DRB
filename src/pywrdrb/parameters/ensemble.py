"""
Defines custom Pywr Parameters used to manage streamflow ensembles.

FlowEnsemble:
    Provides access to inflow ensemble timeseries during the simulation period.

PredictionEnsemble:
    Provides access to an ensemble of flow prediction timeseries used to inform FFMP releases.
"""
import os
import numpy as np
import pandas as pd
import h5py

from pywr.parameters import Parameter

from pywrdrb.utils.hdf5 import (
    get_hdf5_realization_numbers,
    extract_realization_from_hdf5,
)
from pywrdrb import get_directory

input_dir = get_directory().input_dir

class FlowEnsemble(Parameter):
    """This parameter provides access to inflow ensemble timeseries during the simulation period.

    Args:
        model (Model): The Pywr model.
        name (str): The node name.
        inflow_type (str): The dataset label; Options: 'obs_pub', 'nhmv10', 'nwmv21'
        inflow_ensemble_indices (list): Indices of the inflow ensemble to be used.
        **kwargs: Additional keyword arguments.

    Returns:
        None
    """

    def __init__(self, model, name, inflow_type, inflow_ensemble_indices, **kwargs):
        super().__init__(model, **kwargs)

        from pywrdrb import get_directory
        input_dir = get_directory().input_dir

        if "syn" in inflow_type:
            input_dir_ = os.path.join(input_dir, "synthetic_ensembles")
        elif ("pub" in inflow_type) and ("ensemble" in inflow_type):
            input_dir_ = os.path.join(input_dir, "historic_ensembles")

        filename = os.path.join(input_dir_, f"catchment_inflow_{inflow_type}.hdf5")

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
        # We need to 1) verify that the indices are included in the df
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

        Args:
            timestep (Timestep): The timestep being evaluated.
            scenario_index (ScenarioIndex): The index of the scenario.

        Returns:
            float: The flow value for the specified timestep and scenario.
        """
        s_id = self.inflow_ensemble_indices[scenario_index.global_id]
        return self.inflow_ensemble.loc[timestep.datetime, f"{s_id}"]

    @classmethod
    def load(cls, model, data):
        name = data.pop("node")
        inflow_ensemble_indices = data.pop("inflow_ensemble_indices")
        inflow_type = data.pop("inflow_type")
        return cls(model, name, inflow_type, inflow_ensemble_indices, **data)


FlowEnsemble.register()


class PredictionEnsemble(Parameter):
    """This parameter provides access to
    an ensemble of flow prediction timeseries used to inform FFMP releases.

    Args:
        model (Model): The Pywr model.
        column (str): The label of the prediction column.
        inflow_type (str): The dataset label; Options: 'obs_pub_nhmv10_ObsScaled_ensemble', 'obs_pub_nwmv21_ObsScaled_ensemble'
        ensemble_indices (list): Indices of the inflow ensemble to be used.
        **kwargs: Additional keyword arguments.

    Returns:
        None
    """

    def __init__(self, model, column, inflow_type, ensemble_indices, **kwargs):
        super().__init__(model, **kwargs)

        filename = os.path.join(input_dir, f"predicted_inflows_diversions_{inflow_type}.hdf5")
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
        # We need to 1) verify that the indices are included in the df
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

        Args:
            timestep (Timestep): The timestep being evaluated.
            scenario_index (ScenarioIndex): The index of the scenario.

        Returns:
            float: The flow value for the specified timestep and scenario.
        """
        s_id = self.pred_ensemble_indices[scenario_index.global_id]
        return self.pred_ensemble.loc[timestep.datetime, f"{s_id}"]

    @classmethod
    def load(cls, model, data):
        column = data.pop("column")
        ensemble_indices = data.pop("ensemble_indices")
        inflow_type = data.pop("inflow_type")
        return cls(model, column, inflow_type, ensemble_indices, **data)


PredictionEnsemble.register()
