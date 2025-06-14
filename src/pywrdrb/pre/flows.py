"""
Datapreprocessor for flow data preparation in the Delaware River Basin (DRB) using Pywr-DRB.

Overview: 
Each class in this module is responsible for loading, processing, and saving for one type of flow data.

Key Steps: 
Each class have a common implementation of the following steps:
1. load() - Load raw data from the input directories or retrieve from API.
2. process() - Process the loaded raw data to prepare it for use in Pywr-DRB.
3. save() - Save the processed data to the output directories.

Change Log:
Chung-Yi Lin, 2025-05-02, None
"""

import pandas as pd
from pywrdrb.pre.datapreprocessor_ABC import DataPreprocessor
from pywrdrb.pywr_drb_node_data import (
    nhm_site_matches, nwm_site_matches, wrf_hydro_site_matches, upstream_nodes_dict, downstream_node_lags
)
from pywrdrb.utils.lists import reservoir_list_nyc

__all__ = [
    "NHMFlowDataPreprocessor",
    "NWMFlowDataPreprocessor",
    "NHMWithObsScaledFlowDataPreprocessor",
    "NWMWithObsScaledFlowDataPreprocessor",
    "WRFAORCWithObsScaledFlowDataPreprocessor",
]
    
def _subtract_upstream_catchment_inflows(inflows):
    """
    Subtracts upstream catchment inflows from the input inflows timeseries.

    Inflow timeseries are cumulative. For each downstream node, this function subtracts 
    the flow into all upstream nodes so that it represents only the direct catchment 
    inflows into this node. It also accounts for time lags between distant nodes.

    Parameters
    ----------
    inflows : pandas.DataFrame
        The inflows timeseries dataframe.

    Returns
    -------
    pandas.DataFrame
        The modified inflows timeseries dataframe with upstream catchment inflows subtracted.
    """
    inflows = inflows.copy()
    for node, upstreams in upstream_nodes_dict.items():
        for upstream in upstreams:
            lag = downstream_node_lags[upstream]
            if lag > 0:
                inflows.loc[inflows.index[lag:], node] -= inflows.loc[
                    inflows.index[:-lag], upstream
                ].values
                ### subtract same-day flow without lagging for first lag days, since we don't have data before 0 for lagging
                inflows.loc[inflows.index[:lag], node] -= inflows.loc[
                    inflows.index[:lag], upstream
                ].values
            else:
                inflows[node] -= inflows[upstream]

        ### if catchment inflow is negative after subtracting upstream, set to 0
        inflows.loc[inflows[node] < 0, node] = 0

        ### delTrenton node should have zero catchment inflow because coincident with DRCanal
        ### -> make sure that is still so after subtraction process
        inflows["delTrenton"] *= 0.0
    return inflows

def _match_gagues(df, site_matches_id):
    """
    Matches USGS gage sites to Pywr-DRB nodes and returns a DataFrame of inflows.
    
    parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing flow data indexed by datetime.
    site_matches_id : dict
        A dictionary mapping Pywr-DRB node names to USGS gage site IDs.
    
    returns
    -------
    pandas.DataFrame
        A DataFrame with Pywr-DRB node names as columns and datetime as index, containing the inflows.
    """
    for node, site in site_matches_id.items():
        if node == "cannonsville":
            inflows = pd.DataFrame(df.loc[:, site].sum(axis=1))
            inflows.columns = [node]
            inflows["datetime"] = inflows.index
            inflows.index = inflows["datetime"]
            inflows = inflows.iloc[:, :-1]
        else:
            inflows[node] = df[site].sum(axis=1)
    return inflows
        
def _add_upstream_catchment_inflows(inflows, exclude_NYC=False):
    """
    Adds upstream catchment inflows to get cumulative flow at downstream nodes. This is inverse of _subtract_upstream_catchment_inflows()

    Inflow timeseries are cumulative. For each downstream node, this function adds the flow into all upstream nodes so
    that it represents cumulative inflows into the downstream node. It also accounts for time lags between distant nodes.

    parameters
    ---------- 
    inflows : pandas.DataFrame
        The inflows timeseries dataframe.
    exclude_NYC : bool, optional
        If True, excludes NYC reservoirs from the upstream inflows (default is False).
    
    returns
    -------
    pandas.DataFrame
        The modified inflows timeseries dataframe with upstream catchment inflows added.
    """
    ### loop over upstream_nodes_dict in reverse direction to avoid double counting
    inflows = inflows.copy()
    for node in list(upstream_nodes_dict.keys())[::-1]:
        for upstream in upstream_nodes_dict[node]:
            if exclude_NYC == False or upstream not in reservoir_list_nyc:
                lag = downstream_node_lags[upstream]
                if lag > 0:
                    inflows.loc[inflows.index[lag:], node] += inflows.loc[
                        inflows.index[:-lag], upstream
                    ].values
                    ### add same-day flow without lagging for first lag days, since we don't have data before 0 for lagging
                    inflows.loc[inflows.index[:lag], node] += inflows.loc[
                        inflows.index[:lag], upstream
                    ].values
                else:
                    inflows[node] += inflows[upstream]

        ### if catchment inflow is negative after adding upstream, set to 0 (note: this shouldnt happen)
        inflows.loc[inflows[node] < 0, node] = 0
    return inflows
        
class NHMFlowDataPreprocessor(DataPreprocessor):
    def __init__(self):
        """
        Create NHM flow inputs for both catchment inflows and gauge flows by matching USGS 
        gage sites to nodes in Pywr-DRB.

        For reservoirs, the matched gages are actually downstream, but assume this flows 
        into the reservoir from the upstream catchment.
        For river nodes, upstream reservoir inflows are subtracted from the flow at the 
        river node USGS gage.
        For nodes related to USGS gages downstream of reservoirs, the currently 
        redundant flow with assumed inflow is subtracted, resulting in an additional 
        catchment flow of 0 until this is updated.
        Saves csv file, & returns dataframe whose columns are names of Pywr-DRB nodes.
        """
        super().__init__()
        self.flow_type = "nhmv10"
        self.pn.flows.mkdir(self.flow_type) # Create the directory if it does not exist
        self.input_dirs = {
            "streamflow_nhmv10_mgd.csv": self.pn.flows._hydro_model_flow_output.get() / "streamflow_nhmv10_mgd.csv",
        }
        self.output_dirs = {
            "gage_flow_mgd.csv": self.pn.sc.get(f"flows/{self.flow_type}") / "gage_flow_mgd.csv",
            "catchment_inflow_mgd.csv": self.pn.sc.get(f"flows/{self.flow_type}") / "catchment_inflow_mgd.csv",
        }
        # Need to make sure that pn is reloaded such that file directories are added 
        # to shortcuts. (see __init__.py)
        
    def load(self):
        """
        Load the raw data from the input directories defined in `self.input_dirs`.
        Reads the streamflow data from a CSV file and stores it in `self.raw_data`.
        """
        filename = self.input_dirs["streamflow_nhmv10_mgd.csv"]
        df = pd.read_csv(filename, sep=",", index_col=0)
        df.index = pd.to_datetime(df.index)
        self.raw_data["streamflow_nhmv10_mgd.csv"] = df
        
    def process(self):
        """
        Process the loaded raw data to prepare it for use in Pywr-DRB.
        This includes matching USGS gage sites to Pywr-DRB nodes and adjusting inflows
        for upstream catchment inflows.
        
        The processed data is stored in `self.processed_data` with keys for gage flows
        and catchment inflows.
        """
        df = self.raw_data["streamflow_nhmv10_mgd.csv"]
        # 1. Match inflows for each Pywr-DRB node
        # 1.1 Reservoir inflows
        site_matches_id = nhm_site_matches
        inflows = _match_gagues(df, site_matches_id)
        self.processed_data["gage_flow_mgd.csv"] = inflows.copy()
        
        # 2. Inflow timeseries are cumulative. So for each downstream node, subtract 
        # the flow into all upstream nodes so this represents only direct catchment 
        # inflows into this node. Account for time lags between distant nodes.
        inflows = _subtract_upstream_catchment_inflows(inflows)
        # For downstream nodes, this represents the catchment inflow with upstream node 
        # inflows subtracted
        self.processed_data["catchment_inflow_mgd.csv"] = inflows.copy()
    
    def save(self, file_format='csv'):
        """
        Save the processed data to the output directories defined in `self.output_dirs`.
        """
        for filename, df in self.processed_data.items():
            if file_format == 'csv':
                df.to_csv(self.output_dirs[filename])
                print(f"Data saved to {self.output_dirs[filename]}")

class NWMFlowDataPreprocessor(DataPreprocessor):
    def __init__(self):
        """
        Create NWM flow inputs for both catchment inflows and gauge flows by matching USGS 
        gage sites to nodes in Pywr-DRB.

        For reservoirs, the matched gages are actually downstream, but assume this flows 
        into the reservoir from the upstream catchment.
        For river nodes, upstream reservoir inflows are subtracted from the flow at the 
        river node USGS gage.
        For nodes related to USGS gages downstream of reservoirs, the currently 
        redundant flow with assumed inflow is subtracted, resulting in an additional 
        catchment flow of 0 until this is updated.
        Saves csv file, & returns dataframe whose columns are names of Pywr-DRB nodes.
        """
        super().__init__()
        self.flow_type = "nwmv21"
        self.pn.flows.mkdir(self.flow_type) # Create the directory if it does not exist
        self.input_dirs = {
            "streamflow_nwmv21_mgd.csv": self.pn.flows._hydro_model_flow_output.get() / "streamflow_nwmv21_mgd.csv",
        }
        self.output_dirs = {
            "gage_flow_mgd.csv": self.pn.sc.get(f"flows/{self.flow_type}") / "gage_flow_mgd.csv",
            "catchment_inflow_mgd.csv": self.pn.sc.get(f"flows/{self.flow_type}") / "catchment_inflow_mgd.csv",
        }
        # Need to make sure that pn is reloaded such that file directories are added 
        # to shortcuts. (see __init__.py)
        
    def load(self):
        """
        Load the raw data from the input directories defined in `self.input_dirs`.
        Reads the streamflow data from a CSV file and stores it in `self.raw_data`.
        """
        filename = self.input_dirs["streamflow_nwmv21_mgd.csv"]
        df = pd.read_csv(filename, sep=",", index_col=0)
        df.index = pd.to_datetime(df.index)
        self.raw_data["streamflow_nwmv21_mgd.csv"] = df
        
    def process(self):
        """
        Process the loaded raw data to prepare it for use in Pywr-DRB.
        This includes matching USGS gage sites to Pywr-DRB nodes and adjusting inflows
        for upstream catchment inflows.
        
        The processed data is stored in `self.processed_data` with keys for gage flows
        and catchment inflows.
        """
        df = self.raw_data["streamflow_nwmv21_mgd.csv"]
        # 1. Match inflows for each Pywr-DRB node
        # 1.1 Reservoir inflows
        site_matches_id = nwm_site_matches
        inflows = _match_gagues(df, site_matches_id)
        self.processed_data["gage_flow_mgd.csv"] = inflows.copy()
        
        # 2. Inflow timeseries are cumulative. So for each downstream node, subtract 
        # the flow into all upstream nodes so this represents only direct catchment 
        # inflows into this node. Account for time lags between distant nodes.
        inflows = _subtract_upstream_catchment_inflows(inflows)
        # For downstream nodes, this represents the catchment inflow with upstream node 
        # inflows subtracted
        self.processed_data["catchment_inflow_mgd.csv"] = inflows.copy()
    
    def save(self, file_format='csv'):
        """
        Save the processed data to the output directories defined in `self.output_dirs`.
        """
        for filename, df in self.processed_data.items():
            if file_format == 'csv':
                df.to_csv(self.output_dirs[filename])
                print(f"Data saved to {self.output_dirs[filename]}")

class NHMWithObsScaledFlowDataPreprocessor(DataPreprocessor):
    def __init__(self):
        """
        Create NHM hybrid flow inputs for both catchment inflows and gauge flows by matching USGS 
        gage sites to nodes in Pywr-DRB.

        For reservoirs, the matched gages are actually downstream, but assume this flows 
        into the reservoir from the upstream catchment.
        For river nodes, upstream reservoir inflows are subtracted from the flow at the 
        river node USGS gage.
        For nodes related to USGS gages downstream of reservoirs, the currently 
        redundant flow with assumed inflow is subtracted, resulting in an additional 
        catchment flow of 0 until this is updated.
        Saves csv file, & returns dataframe whose columns are names of Pywr-DRB nodes.
        """
        super().__init__()
        self.flow_type = "nhmv10_withObsScaled"
        self.pn.flows.mkdir(self.flow_type) # Create the directory if it does not exist
        self.input_dirs = {
            "streamflow_nhmv10_mgd.csv": self.pn.flows._hydro_model_flow_output.get() / "streamflow_nhmv10_mgd.csv",
            "scaled_inflows_nhmv10.csv": self.pn.flows._scaled_inflows.get() / "scaled_inflows_nhmv10.csv",
        }
        # github.com/Pywr-DRB/Input-Data-Retrieval/blob/main/inflow_scaling_regression.py
        self.output_dirs = {
            "gage_flow_mgd.csv": self.pn.sc.get(f"flows/{self.flow_type}") / "gage_flow_mgd.csv",
            "catchment_inflow_mgd.csv": self.pn.sc.get(f"flows/{self.flow_type}") / "catchment_inflow_mgd.csv",
        }
        # Need to make sure that pn is reloaded such that file directories are added 
        # to shortcuts. (see __init__.py)
        
    def load(self):
        """
        Load the raw data from the input directories defined in `self.input_dirs`.
        Reads the streamflow data from a CSV file and stores it in `self.raw_data`.
        """
        filename = self.input_dirs["streamflow_nhmv10_mgd.csv"]
        df = pd.read_csv(filename, sep=",", index_col=0)
        df.index = pd.to_datetime(df.index)
        self.raw_data["streamflow_nhmv10_mgd.csv"] = df
        
        filename = self.input_dirs["scaled_inflows_nhmv10.csv"]
        df = pd.read_csv(filename, sep=",", index_col=0)
        df.index = pd.to_datetime(df.index)
        self.raw_data["scaled_inflows_nhmv10.csv"] = df
        
    def process(self):
        """
        Process the loaded raw data to prepare it for use in Pywr-DRB.
        This includes matching USGS gage sites to Pywr-DRB nodes and adjusting inflows
        for upstream catchment inflows, then hybrid with the scaled obseved inflows.
        
        The processed data is stored in `self.processed_data` with keys for gage flows
        and catchment inflows.
        """
        # First process the nhmv10 data
        df = self.raw_data["streamflow_nhmv10_mgd.csv"]
        # 1. Match inflows for each Pywr-DRB node
        # 1.1 Reservoir inflows
        site_matches_id = nhm_site_matches
        inflows = _match_gagues(df, site_matches_id)
        inflows = _subtract_upstream_catchment_inflows(inflows)
        
        # Second process the scaled inflows
        scaled_obs = self.raw_data["scaled_inflows_nhmv10.csv"]
        overlap_index = inflows.index.intersection(scaled_obs.index)
        
        inflows = inflows.loc[overlap_index]
        scaled_obs = scaled_obs.loc[overlap_index]
        
        for reservoir in reservoir_list_nyc + ["fewalter", "beltzvilleCombined"]:
            inflows[reservoir] = scaled_obs[reservoir]
        self.processed_data["catchment_inflow_mgd.csv"] = inflows.copy()
        
        inflows = _add_upstream_catchment_inflows(inflows)
        self.processed_data["gage_flow_mgd.csv"] = inflows.copy()
        
    def save(self, file_format='csv'):
        """
        Save the processed data to the output directories defined in `self.output_dirs`.
        """
        for filename, df in self.processed_data.items():
            if file_format == 'csv':
                df.to_csv(self.output_dirs[filename])
                print(f"Data saved to {self.output_dirs[filename]}")

class NWMWithObsScaledFlowDataPreprocessor(DataPreprocessor):
    def __init__(self):
        """
        Create NWM hybrid flow inputs for both catchment inflows and gauge flows by matching USGS 
        gage sites to nodes in Pywr-DRB.

        For reservoirs, the matched gages are actually downstream, but assume this flows 
        into the reservoir from the upstream catchment.
        For river nodes, upstream reservoir inflows are subtracted from the flow at the 
        river node USGS gage.
        For nodes related to USGS gages downstream of reservoirs, the currently 
        redundant flow with assumed inflow is subtracted, resulting in an additional 
        catchment flow of 0 until this is updated.
        Saves csv file, & returns dataframe whose columns are names of Pywr-DRB nodes.
        """
        super().__init__()
        self.flow_type = "nwmv21_withObsScaled"
        self.pn.flows.mkdir(self.flow_type) # Create the directory if it does not exist
        self.input_dirs = {
            "streamflow_nwmv21_mgd.csv": self.pn.flows._hydro_model_flow_output.get() / "streamflow_nwmv21_mgd.csv",
            "scaled_inflows_nwmv21.csv": self.pn.flows._scaled_inflows.get() / "scaled_inflows_nwmv21.csv",
        }
        # github.com/Pywr-DRB/Input-Data-Retrieval/blob/main/inflow_scaling_regression.py
        self.output_dirs = {
            "gage_flow_mgd.csv": self.pn.sc.get(f"flows/{self.flow_type}") / "gage_flow_mgd.csv",
            "catchment_inflow_mgd.csv": self.pn.sc.get(f"flows/{self.flow_type}") / "catchment_inflow_mgd.csv",
        }
        # Need to make sure that pn is reloaded such that file directories are added 
        # to shortcuts. (see __init__.py)
        
    def load(self):
        """
        Load the raw data from the input directories defined in `self.input_dirs`.
        Reads the streamflow data from a CSV file and stores it in `self.raw_data`.
        """
        filename = self.input_dirs["streamflow_nwmv21_mgd.csv"]
        df = pd.read_csv(filename, sep=",", index_col=0)
        df.index = pd.to_datetime(df.index)
        self.raw_data["streamflow_nwmv21_mgd.csv"] = df
        
        filename = self.input_dirs["scaled_inflows_nwmv21.csv"]
        df = pd.read_csv(filename, sep=",", index_col=0)
        df.index = pd.to_datetime(df.index)
        self.raw_data["scaled_inflows_nwmv21.csv"] = df
        
    def process(self):
        """
        Process the loaded raw data to prepare it for use in Pywr-DRB.
        This includes matching USGS gage sites to Pywr-DRB nodes and adjusting inflows
        for upstream catchment inflows, then hybrid with the scaled obseved inflows.
        
        The processed data is stored in `self.processed_data` with keys for gage flows
        and catchment inflows.
        """
        # First process the nwmv21 data
        df = self.raw_data["streamflow_nwmv21_mgd.csv"]
        # 1. Match inflows for each Pywr-DRB node
        # 1.1 Reservoir inflows
        site_matches_id = nwm_site_matches
        inflows = _match_gagues(df, site_matches_id)
        inflows = _subtract_upstream_catchment_inflows(inflows)
        
        # Second process the scaled inflows
        scaled_obs = self.raw_data["scaled_inflows_nwmv21.csv"]
        overlap_index = inflows.index.intersection(scaled_obs.index)
        
        inflows = inflows.loc[overlap_index]
        scaled_obs = scaled_obs.loc[overlap_index]
        
        for reservoir in reservoir_list_nyc + ["fewalter", "beltzvilleCombined"]:
            inflows[reservoir] = scaled_obs[reservoir]
        self.processed_data["catchment_inflow_mgd.csv"] = inflows.copy()
        
        inflows = _add_upstream_catchment_inflows(inflows)
        self.processed_data["gage_flow_mgd.csv"] = inflows.copy() 
    
    def save(self, file_format='csv'):
        """
        Save the processed data to the output directories defined in `self.output_dirs`.
        """
        for filename, df in self.processed_data.items():
            if file_format == 'csv':
                df.to_csv(self.output_dirs[filename])
                print(f"Data saved to {self.output_dirs[filename]}")
                             
class WRFAORCWithObsScaledFlowDataPreprocessor(DataPreprocessor):
    def __init__(self):
        """
        Create wrf hydro (calib) hybrid flow inputs for both catchment inflows and gauge 
        flows by matching USGS gage sites to nodes in Pywr-DRB.

        For reservoirs, the matched gages are actually downstream, but assume this flows 
        into the reservoir from the upstream catchment.
        For river nodes, upstream reservoir inflows are subtracted from the flow at the 
        river node USGS gage.
        For nodes related to USGS gages downstream of reservoirs, the currently 
        redundant flow with assumed inflow is subtracted, resulting in an additional 
        catchment flow of 0 until this is updated.
        Saves csv file, & returns dataframe whose columns are names of Pywr-DRB nodes.
        """
        super().__init__()
        self.flow_type = "wrfaorc_withObsScaled"
        self.pn.flows.mkdir(self.flow_type) # Create the directory if it does not exist
        self.input_dirs = {
            "streamflow_wrfaorc_calib_nlcd2016.csv": self.pn.flows._hydro_model_flow_output.get() / "streamflow_wrfaorc_calib_nlcd2016.csv",
            "scaled_inflows_wrfaorc.csv": self.pn.flows._scaled_inflows.get() / "scaled_inflows_wrfaorc.csv",
        }
        # github.com/Pywr-DRB/Input-Data-Retrieval/blob/main/inflow_scaling_regression.py
        self.output_dirs = {
            "gage_flow_mgd.csv": self.pn.sc.get(f"flows/{self.flow_type}") / "gage_flow_mgd.csv",
            "catchment_inflow_mgd.csv": self.pn.sc.get(f"flows/{self.flow_type}") / "catchment_inflow_mgd.csv",
        }
        # Need to make sure that pn is reloaded such that file directories are added 
        # to shortcuts. (see __init__.py)
        
    def load(self):
        """
        Load the raw data from the input directories defined in `self.input_dirs`.
        Reads the streamflow data from a CSV file and stores it in `self.raw_data`.
        """
        filename = self.input_dirs["streamflow_wrfaorc_calib_nlcd2016.csv"]
        df = pd.read_csv(filename, sep=",", index_col=0)
        df.index = pd.to_datetime(df.index)
        self.raw_data["streamflow_wrfaorc_mgd.csv"] = df
        
        filename = self.input_dirs["scaled_inflows_wrfaorc.csv"]
        df = pd.read_csv(filename, sep=",", index_col=0)
        df.index = pd.to_datetime(df.index)
        self.raw_data["scaled_inflows_wrfaorc.csv"] = df
        
    def process(self):
        """
        Process the loaded raw data to prepare it for use in Pywr-DRB.
        This includes matching USGS gage sites to Pywr-DRB nodes and adjusting inflows
        for upstream catchment inflows, then hybrid with the scaled obseved inflows.
        
        The processed data is stored in `self.processed_data` with keys for gage flows
        and catchment inflows.
        """
        # First process the wrfaorc data
        df = self.raw_data["streamflow_wrfaorc_mgd.csv"]
        # 1. Match inflows for each Pywr-DRB node
        # 1.1 Reservoir inflows
        site_matches_id = wrf_hydro_site_matches
        inflows = _match_gagues(df, site_matches_id)
        inflows = _subtract_upstream_catchment_inflows(inflows)
        
        # Second process the scaled inflows
        scaled_obs = self.raw_data["scaled_inflows_wrfaorc.csv"]
        overlap_index = inflows.index.intersection(scaled_obs.index)
        
        inflows = inflows.loc[overlap_index]
        scaled_obs = scaled_obs.loc[overlap_index]
        
        for reservoir in reservoir_list_nyc + ["fewalter", "beltzvilleCombined"]:
            inflows[reservoir] = scaled_obs[reservoir]
        inflows.index.name = "datetime"
        self.processed_data["catchment_inflow_mgd.csv"] = inflows.copy()
        
        inflows = _add_upstream_catchment_inflows(inflows)
        self.processed_data["gage_flow_mgd.csv"] = inflows.copy()
        
    def save(self, file_format='csv'):
        """
        Save the processed data to the output directories defined in `self.output_dirs`.
        """
        for filename, df in self.processed_data.items():
            if file_format == 'csv':
                df.to_csv(self.output_dirs[filename])
                print(f"Data saved to {self.output_dirs[filename]}")