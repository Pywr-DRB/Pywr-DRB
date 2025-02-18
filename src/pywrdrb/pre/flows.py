
import pandas as pd
from . import DataPreprocessor
from ..pywr_drb_node_data import (
    nhm_site_matches, nwm_site_matches, upstream_nodes_dict, downstream_node_lags
)


    
def subtract_upstream_catchment_inflows(inflows):
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

class NHMFlowDataPreprocessor(DataPreprocessor):
    def __init__(self):
        """
        Create flow inputs for both catchment inflows and gauge flows by matching USGS 
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
        self._dirs.data.flows.mkdir(self.flow_type) # Create the directory if it does not exist
        self.input_dirs = {
            "streamflow_daily_nhmv10_mgd.csv": self._dirs.data.flows._modeled_gages.get() / "streamflow_daily_nhmv10_mgd.csv",
        }
        self.output_dirs = {
            "gage_flow_mgd.csv": self._dirs.data.flows.get(self.flow_type) / "gage_flow_mgd.csv",
            "catchment_inflow_mgd.csv": self._dirs.data.flows.get(self.flow_type) / "catchment_inflow_mgd.csv",
        }
        # Need to make sure that _dirs is reloaded such that file directories are added 
        # to shortcuts. (see __init__.py)
        
    def load(self):
        filename = self.input_dirs["streamflow_daily_nhmv10_mgd.csv"]
        df = pd.read_csv(filename, sep=",", index_col=0)
        df.index = pd.to_datetime(df.index)
        self.raw_data["streamflow_daily_nhmv10_mgd.csv"] = df
        
    def process(self):
        df = self.raw_data["streamflow_daily_nhmv10_mgd.csv"]
        # 1. Match inflows for each Pywr-DRB node
        # 1.1 Reservoir inflows
        site_matches_id = nhm_site_matches
        for node, site in site_matches_id.items():
            if node == "cannonsville":
                inflows = pd.DataFrame(df.loc[:, site].sum(axis=1))
                inflows.columns = [node]
                inflows["datetime"] = inflows.index
                inflows.index = inflows["datetime"]
                inflows = inflows.iloc[:, :-1]
            else:
                inflows[node] = df[site].sum(axis=1)
        self.processed_data["gage_flow_mgd.csv"] = inflows.copy()
        
        # 2. Inflow timeseries are cumulative. So for each downstream node, subtract 
        # the flow into all upstream nodes so this represents only direct catchment 
        # inflows into this node. Account for time lags between distant nodes.
        inflows = subtract_upstream_catchment_inflows(inflows)
        # For downstream nodes, this represents the catchment inflow with upstream node 
        # inflows subtracted
        self.processed_data["catchment_inflow_mgd.csv"] = inflows.copy()
    
    def save(self, file_format='csv'):
        for filename, df in self.processed_data.items():
            if file_format == 'csv':
                df.to_csv(self.output_dirs[filename])
                print(f"Data saved to {self.output_dirs[filename]}")

class NWMFlowDataPreprocessor(DataPreprocessor):
    def __init__(self):
        """
        Create flow inputs for both catchment inflows and gauge flows by matching USGS 
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
        self._dirs.data.flows.mkdir(self.flow_type) # Create the directory if it does not exist
        self.input_dirs = {
            "streamflow_daily_nwmv21_mgd.csv": self._dirs.data.flows._modeled_gages.get() / "streamflow_daily_nwmv21_mgd.csv",
        }
        self.output_dirs = {
            "gage_flow_mgd.csv": self._dirs.data.flows.get(self.flow_type) / "gage_flow_mgd.csv",
            "catchment_inflow_mgd.csv": self._dirs.data.flows.get(self.flow_type) / "catchment_inflow_mgd.csv",
        }
        # Need to make sure that _dirs is reloaded such that file directories are added 
        # to shortcuts. (see __init__.py)
        
    def load(self):
        filename = self.input_dirs["streamflow_daily_nwmv21_mgd.csv"]
        df = pd.read_csv(filename, sep=",", index_col=0)
        df.index = pd.to_datetime(df.index)
        self.raw_data["streamflow_daily_nwmv21_mgd.csv"] = df
        
    def process(self):
        df = self.raw_data["streamflow_daily_nwmv21_mgd.csv"]
        # 1. Match inflows for each Pywr-DRB node
        # 1.1 Reservoir inflows
        site_matches_id = nwm_site_matches
        for node, site in site_matches_id.items():
            if node == "cannonsville":
                inflows = pd.DataFrame(df.loc[:, site].sum(axis=1))
                inflows.columns = [node]
                inflows["datetime"] = inflows.index
                inflows.index = inflows["datetime"]
                inflows = inflows.iloc[:, :-1]
            else:
                inflows[node] = df[site].sum(axis=1)
        self.processed_data["gage_flow_mgd.csv"] = inflows.copy()
        
        # 2. Inflow timeseries are cumulative. So for each downstream node, subtract 
        # the flow into all upstream nodes so this represents only direct catchment 
        # inflows into this node. Account for time lags between distant nodes.
        inflows = subtract_upstream_catchment_inflows(inflows)
        # For downstream nodes, this represents the catchment inflow with upstream node 
        # inflows subtracted
        self.processed_data["catchment_inflow_mgd.csv"] = inflows.copy()
    
    def save(self, file_format='csv'):
        for filename, df in self.processed_data.items():
            if file_format == 'csv':
                df.to_csv(self.output_dirs[filename])
                print(f"Data saved to {self.output_dirs[filename]}")







def match_gages(df, dataset_label, site_matches_id):
    """
    Matches USGS gage sites to nodes in Pywr-DRB.

    For reservoirs, the matched gages are actually downstream, but assume this flows into the reservoir from the upstream catchment.
    For river nodes, upstream reservoir inflows are subtracted from the flow at the river node USGS gage.
    For nodes related to USGS gages downstream of reservoirs, the currently redundant flow with assumed inflow is subtracted, resulting in an additional catchment flow of 0 until this is updated.
    Saves csv file, & returns dataframe whose columns are names of Pywr-DRB nodes.

    Args:
        df (pandas.DataFrame): The input dataframe.
        dataset_label (str): The label for the dataset.
        site_matches_id (dict): A dictionary containing the site matches for Pywr-DRB nodes.
        upstream_nodes_dict (dict): A dictionary containing the upstream nodes for each node.

    Returns:
        pandas.DataFrame: The resulting dataframe whose columns are names of Pywr-DRB nodes.
    """

    ### 1. Match inflows for each Pywr-DRB node
    ## 1.1 Reservoir inflows
    for node, site in site_matches_id.items():
        if node == "cannonsville":
            if ("obs_pub" in dataset_label) and (site == None):
                inflows = pd.DataFrame(df.loc[:, node])
            else:
                inflows = pd.DataFrame(df.loc[:, site].sum(axis=1))
            inflows.columns = [node]
            inflows["datetime"] = inflows.index
            inflows.index = inflows["datetime"]
            inflows = inflows.iloc[:, :-1]
        else:
            if ("obs_pub" in dataset_label) and (site == None):
                inflows[node] = df[node]
            else:
                inflows[node] = df[site].sum(axis=1)

    if "obs_pub" not in dataset_label:
        ## Save full flows to csv
        # For downstream nodes, this represents the full flow for results comparison
        inflows.to_csv(f"{input_dir}gage_flow_{dataset_label}.csv")

    ### 2. Inflow timeseries are cumulative. So for each downstream node, subtract the flow into all upstream nodes so
    ###    this represents only direct catchment inflows into this node. Account for time lags between distant nodes.
    inflows = subtract_upstream_catchment_inflows(inflows)

    ## Save catchment inflows to csv
    # For downstream nodes, this represents the catchment inflow with upstream node inflows subtracted
    inflows.to_csv(f"{input_dir}catchment_inflow_{dataset_label}.csv")

    if "obs_pub" in dataset_label:
        ## For PUB, to get full gage flow we want to re-add up cumulative flows after doing previous catchment subtraction.
        # For downstream nodes, this represents the full flow for results comparison
        inflows = add_upstream_catchment_inflows(inflows)
        inflows.to_csv(f"{input_dir}gage_flow_{dataset_label}.csv")