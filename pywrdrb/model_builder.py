"""
Contains functions used to construct a pywrdrb model in JSON format.
"""

import json
import pandas as pd

from pywrdrb.utils.directories import input_dir, model_data_dir
from pywrdrb.utils.lists import (
    majorflow_list,
    reservoir_list,
    reservoir_list_nyc,
    modified_starfit_reservoir_list,
)
from pywrdrb.utils.lists import drbc_lower_basin_reservoirs
from pywrdrb.utils.constants import cfs_to_mgd
from pywrdrb.pywr_drb_node_data import (
    immediate_downstream_nodes_dict,
    downstream_node_lags,
)


### model options/parameters
flow_prediction_mode = "regression_disagg"  ### 'regression_agg', 'regression_disagg', 'perfect_foresight', 'same_day', 'moving_average'
use_lower_basin_mrf_contributions = True


def get_reservoir_capacity(reservoir):
    istarf = pd.read_csv(f"{model_data_dir}drb_model_istarf_conus.csv")
    return float(
        istarf["Adjusted_CAP_MG"].loc[istarf["reservoir"] == reservoir].iloc[0]
    )


### get max reservoir releases for NYC from historical data
def get_reservoir_max_release(reservoir, release_type):
    assert reservoir in reservoir_list_nyc, f"No max release data for {reservoir}"
    assert release_type in ["controlled", "flood"]

    ### constrain most releases based on historical controlled release max
    if release_type == "controlled":
        hist_releases = pd.read_excel(
            input_dir + "/historic_NYC/Pep_Can_Nev_releases_daily_2000-2021.xlsx"
        )
        if reservoir == "pepacton":
            max_hist_release = hist_releases["Pepacton Controlled Release"].max()
        elif reservoir == "cannonsville":
            max_hist_release = hist_releases["Cannonsville Controlled Release"].max()
        elif reservoir == "neversink":
            max_hist_release = hist_releases["Neversink Controlled Release"].max()
        return max_hist_release * cfs_to_mgd

    ### constrain flood releases based on FFMP Table 5 instead. Note there is still a separate high cost spill path that can exceed this value.
    elif release_type == "flood":
        if reservoir == "pepacton":
            max_hist_release = 2400
        elif reservoir == "cannonsville":
            max_hist_release = 4200
        elif reservoir == "neversink":
            max_hist_release = 3400
        return max_hist_release * cfs_to_mgd


### get max reservoir releases for NYC from historical data
def get_reservoir_max_diversion_NYC(reservoir):
    assert reservoir in reservoir_list_nyc, f"No max diversion data for {reservoir}"
    hist_diversions = pd.read_excel(
        input_dir + "/historic_NYC/Pep_Can_Nev_diversions_daily_2000-2021.xlsx"
    )
    if reservoir == "pepacton":
        max_hist_diversion = hist_diversions["East Delaware Tunnel"].max()
    elif reservoir == "cannonsville":
        max_hist_diversion = hist_diversions["West Delaware Tunnel"].max()
    elif reservoir == "neversink":
        max_hist_diversion = hist_diversions["Neversink Tunnel"].max()
    return max_hist_diversion * cfs_to_mgd


class PywrdrbModelBuilder():
    def __init__(self, inflow_type, start_date, end_date, options={
        "inflow_ensemble_indices":None,
        "use_hist_NycNjDeliveries":True, # If True, we use historical NYC/NJ deliveries as demand, else we use predicted demand. Otherwise, assume demand is equal to max allotment under FFMP
        "predict_temperature":False, # If True, we use LSTM model to predict temperature at Lordville
        "predict_salinity":False, # If True, we use LSTM model to predict salinity at Trenton
        }):
        # Create model dict to hold all model nodes, edges, params, etc, following Pywr protocol. 
        # This will be saved to JSON at end.
        self.NSCENARIOS = 1
        self.initial_volume_frac = 0.8 # get initial reservoir storages as 80% of capacity
        self.inflow_type = inflow_type
        self.start_date = start_date
        self.end_date = end_date
        self.timestep = 1
        self.options = options

        # Tracking purposes
        self.reservoirs = []
        self.summary_report = {}

        self.edges = []
        self.parameters = []

        # Variables
        self.levels = ["1a", "1b", "1c", "2", "3", "4", "5"] # Reservoir operational regimes
        self.EPS = 1e-8

        self.model_dict = {
            "metadata": {"title": "DRB", "description": "Pywr DRB representation", "minimum_version": "0.4"},
            "timestepper": {"start": start_date, "end": end_date, "timestep": self.timestep},
            "scenarios": [{"name": "inflow", "size": self.NSCENARIOS}], # default to 1 scenario
            "nodes": [],
            "edges": [],
            "parameters": {}
        }                

    def reset_model(self):
        self.NSCENARIOS = 1
        self.model_dict = {
            "metadata": {"title": "DRB", "description": "Pywr DRB representation", "minimum_version": "0.4"},
            "timestepper": {"start": self.start_date, "end": self.end_date, "timestep": self.timestep},
            "scenarios": [{"name": "inflow", "size": self.NSCENARIOS}], # default to 1 scenario
            "nodes": [],
            "edges": [],
            "parameters": {}
        }
    #!! revisit this to incorporate other scenario strategies
                   
    def make_model(self):

        ####################################################################
        ### Add pywr scenarios
        ####################################################################

        ### Add inflow scenarios
        # We might want to extend this to parameter as well.
        inflow_ensemble_indices = self.options.get("inflow_ensemble_indices")
        if inflow_ensemble_indices is not None:
            self.add_ensemble_inflow_scenarios(inflow_ensemble_indices)

        # Define "scenarios" based on flow multiplier -> only run one with 1.0 for now
        else: # N_SCENARIOS = 1:     
            self.model_dict["parameters"]["flow_factor"] = {
                "type": "constantscenario",
                "scenario": "inflow",
                "values": [1.0],
            }

        #######################################################################
        ### Add major nodes (e.g., reservoirs) to model, along with corresponding minor 
        ### nodes (e.g., withdrawals), edges, & parameters
        #######################################################################

        ### Get downstream node to link to for the current node
        for node, downstream_node in immediate_downstream_nodes_dict.items():
            # Get flow lag (days) between current node and its downstream connection
            downstream_lag = downstream_node_lags[node]
            
            # Reservoir node
            if node in reservoir_list:
                self.add_node_major_reservoir(node, downstream_lag, downstream_node)
            # River node
            else:
                has_catchment = False if node == "delTrenton" else True
                self.add_node_major_river(node, downstream_lag, downstream_node, has_catchment)

        #######################################################################
        ### Add additional nodes & edges beyond those associated with major nodes above
        #######################################################################
        self.add_node_nyc_aggregated_storage_and_link()
        self.add_node_nyc_and_nj_deliveries()
        self.add_node_final_basin_outlet()

        #######################################################################
        ### Add additional parameters beyond those associated with major nodes above
        #######################################################################
        self.add_parameter_nyc_and_nj_demands()
        self.add_parameter_nyc_reservoirs_operational_regimes()
        self.add_parameter_nyc_and_nj_delivery_constraints()
        self.add_parameter_nyc_reservoirs_min_require_flow()
        self.add_parameter_nyc_reservoirs_flood_control()
        # Can be removed after volume balancing rules are implemented
        self.add_parameter_nyc_reservoirs_variable_cost_based_on_fractional_storage()
        self.add_parameter_nyc_reservoirs_current_volume()
        self.add_parameter_nyc_reservoirs_aggregated_info()
        self.add_parameter_montague_trenton_flow_targets()
        self.add_parameter_predicted_lagged_non_nyc_inflows_to_Montague_and_Trenton_and_lagged_nj_demands()
        self.add_parameter_nyc_reservoirs_balancing_methods()

        #######################################################################
        ### Couple temperature & salinaty LSTM model
        #######################################################################
        if self.options.get("predict_temperature"):
            self.add_parameter_couple_temp_lstm()
        if self.options.get("predict_salinity"):
            self.add_parameter_couple_salinity_lstm()

    def write_model(self, model_filename):
        """Write the model to a JSON file."""
        with open(f"{model_filename}", "w") as o:
            json.dump(self.model_dict, o, indent=4)

    def add_ensemble_inflow_scenarios(self, inflow_ensemble_indices):
        # Parallel strategy used in pywr
        self.NSCENARIOS = len(inflow_ensemble_indices)
        self.model_dict["scenarios"] = [{"name": "inflow", "size": self.NSCENARIOS}]

    def add_node_major_reservoir(self, reservoir_name, downstream_lag, downstream_node):
        """
        Add a major reservoir node to the model. This step will also add a cluster of 
        nodes that are connected to the reservoir, including catchment, withdrawal, 
        consumption, release, overflow, and delay. 
        """
        node_name = f"reservoir_{reservoir_name}"

        model_dict = self.model_dict

        # Initial settings
        initial_volume_frac = self.initial_volume_frac
        regulatory_release = (
                True
                if reservoir_name in (reservoir_list_nyc + drbc_lower_basin_reservoirs)
                else False
            )
        starfit_release = True if reservoir_name not in reservoir_list_nyc else False
        variable_cost = True if (regulatory_release and not starfit_release) else False

        capacity = (
            get_reservoir_capacity(f"modified_{reservoir_name}")
            if reservoir_name in modified_starfit_reservoir_list
            else get_reservoir_capacity(reservoir_name)
            )

        #############################################
        ################# Add nodes #################
        #############################################

        ##### Add reservoir node
        initial_volume = capacity * initial_volume_frac
        reservoir = {
            "name": node_name,
            "type": "storage",
            "max_volume": f"max_volume_{reservoir_name}",
            "initial_volume": initial_volume,
            "initial_volume_pc": initial_volume_frac,
            "cost": -10.0 if not variable_cost else f"storage_cost_{reservoir_name}",
        }
        model_dict["nodes"].append(reservoir)
        self.reservoirs.append(reservoir_name)

        
        ##### Add catchment node that sends inflows to reservoir
        catchment = {
            "name": f"catchment_{reservoir_name}",
            "type": "catchment",
            "flow": f"flow_{reservoir_name}",
        }
        model_dict["nodes"].append(catchment)

        ##### Add withdrawal node that withdraws flow from catchment for human use
        withdrawal = {
            "name": f"catchmentWithdrawal_{reservoir_name}",
            "type": "link",
            "cost": -15.0,
            "max_flow": f"max_flow_catchmentWithdrawal_{reservoir_name}",
        }
        model_dict["nodes"].append(withdrawal)

        ##### Add consumption node that removes flow from model - the rest of withdrawals return back to reservoir
        consumption = {
            "name": f"catchmentConsumption_{reservoir_name}",
            "type": "output",
            "cost": -2000.0,
            "max_flow": f"max_flow_catchmentConsumption_{reservoir_name}",
        }
        model_dict["nodes"].append(consumption)

        ##### Add outflow node that reservoir release flow to downstream node
        # Get the reservoir release rules        
        # Lower basin reservoirs with no FFMP connection
        if starfit_release and not regulatory_release:
            outflow = {
                "name": f"outflow_{reservoir_name}",
                "type": "link",
                "cost": -500.0,
                "max_flow": f"starfit_release_{reservoir_name}",
            }
        # Lower basin reservoirs which contribute to Montague/Trenton
        elif starfit_release and regulatory_release:
            outflow = {
                "name": f"outflow_{reservoir_name}",
                "type": "link",
                "cost": -500.0,
                "max_flow": f"downstream_release_target_{reservoir_name}",
            }
        # NYC Reservoirs
        elif regulatory_release and not starfit_release:
            outflow = {
                "name": f"outflow_{reservoir_name}",
                "type": "link",
                "cost": -1000.0,
                "max_flow": f"downstream_release_target_{reservoir_name}",
            }
        model_dict["nodes"].append(outflow)

        ##### Add secondary high-cost flow path for spill above max_flow
        outflow = {
            "name": f"spill_{reservoir_name}", 
            "type": "link", 
            "cost": 5000.0}
        model_dict["nodes"].append(outflow)

        ##### Add delay node to account for flow travel time between nodes. Lag unit is days.
        if downstream_lag > 0:
            delay = {
                "name": f"delay_{reservoir_name}", 
                "type": "DelayNode", 
                "days": downstream_lag
                }
            model_dict["nodes"].append(delay)

        #############################################
        ########## Add edges between nodes ##########
        #############################################

        model_dict["edges"] += [
            # catchment to reservoir
            [f"catchment_{reservoir_name}", node_name],
            # catchment to withdrawal
            [f"catchment_{reservoir_name}", f"catchmentWithdrawal_{reservoir_name}"],
            # withdrawal to consumption
            [f"catchmentWithdrawal_{reservoir_name}", f"catchmentConsumption_{reservoir_name}"],
            # withdrawal to reservoir
            [f"catchmentWithdrawal_{reservoir_name}", node_name],
            # reservoir to outflow
            [node_name, f"outflow_{reservoir_name}"],
            # reservoir to high-cost spill (secondary path)
            [node_name, f"spill_{reservoir_name}"]
        ]

         # reservoir downstream node (via outflow node if one exists)
        if downstream_node in majorflow_list:
            downstream_name = f"link_{downstream_node}"
        elif downstream_node == "output_del":
            downstream_name = downstream_node
        else:
            downstream_name = f"reservoir_{downstream_node}"

        # outflow to (delay to) downstream node
        if downstream_lag > 0:
            model_dict["edges"] += [
                [f"outflow_{reservoir_name}", f"delay_{reservoir_name}"],
                [f"spill_{reservoir_name}", f"delay_{reservoir_name}"],
                [f"delay_{reservoir_name}", downstream_name],
            ]
        else:
            model_dict["edges"] += [
                [f"outflow_{reservoir_name}", downstream_name],
                [f"spill_{reservoir_name}", downstream_name]
            ]
        
        #############################################
        ########## Add standard parameters ##########
        #############################################

        # max volume of reservoir, from GRanD database except where adjusted from other sources (eg NYC)
        model_dict["parameters"][f"max_volume_{reservoir_name}"] = {
            "type": "constant",
            "url": "drb_model_istarf_conus.csv",
            "column": "Adjusted_CAP_MG",
            "index_col": "reservoir",
            "index": f"modified_{reservoir_name}"
            if reservoir_name in modified_starfit_reservoir_list
            else reservoir_name,
        }

        # add max release constraints for NYC reservoirs
        if regulatory_release and not starfit_release:
            model_dict["parameters"][f"controlled_max_release_{reservoir_name}"] = {
                "type": "constant",
                "value": get_reservoir_max_release(reservoir_name, "controlled"),
            }
            model_dict["parameters"][f"flood_max_release_{reservoir_name}"] = {
                "type": "constant",
                "value": get_reservoir_max_release(reservoir_name, "flood"),
            }

        # for STARFIT reservoirs, use custom parameter
        if starfit_release:
            model_dict["parameters"][f"starfit_release_{reservoir_name}"] = {
                "type": "STARFITReservoirRelease",
                "node": reservoir_name,
            }

        ### assign inflows to nodes
        inflow_ensemble_indices = self.options.get("inflow_ensemble_indices", None)
        inflow_type = self.inflow_type
        if inflow_ensemble_indices:
            ### Use custom FlowEnsemble parameter to handle scenario indexing
            model_dict["parameters"][f"flow_{reservoir_name}"] = {
                "type": "FlowEnsemble",
                "node": reservoir_name,
                "inflow_type": inflow_type,
                "inflow_ensemble_indices": inflow_ensemble_indices,
            }

        else:
            inflow_source = f"{input_dir}catchment_inflow_{inflow_type}.csv"
            ### Use single-scenario historic data
            model_dict["parameters"][f"flow_{reservoir_name}"] = {
                "type": "dataframe",
                "url": inflow_source,
                "column": reservoir_name,
                "index_col": "datetime",
                "parse_dates": True,
            }

        # get max flow for catchment withdrawal nodes based on DRBC data
        model_dict["parameters"][f"max_flow_catchmentWithdrawal_{reservoir_name}"] = {
            "type": "constant",
            "url": f"{input_dir}sw_avg_wateruse_Pywr-DRB_Catchments.csv",
            "column": "Total_WD_MGD",
            "index_col": "node",
            "index": node_name,
        }

        # get max flow for catchment consumption nodes based on DRBC data
        # assume the consumption_t = R * withdrawal_{t-1}, where R is the ratio of avg 
        # consumption to withdrawal from DRBC data
        model_dict["parameters"][f"catchmentConsumptionRatio_{reservoir_name}"] = {
            "type": "constant",
            "url": f"{input_dir}sw_avg_wateruse_Pywr-DRB_Catchments.csv",
            "column": "Total_CU_WD_Ratio",
            "index_col": "node",
            "index": node_name,
        }
        model_dict["parameters"][f"prev_flow_catchmentWithdrawal_{reservoir_name}"] = {
            "type": "flow",
            "node": f"catchmentWithdrawal_{reservoir_name}",
        }
        model_dict["parameters"][f"max_flow_catchmentConsumption_{reservoir_name}"] = {
            "type": "aggregated",
            "agg_func": "product",
            "parameters": [
                f"catchmentConsumptionRatio_{reservoir_name}",
                f"prev_flow_catchmentWithdrawal_{reservoir_name}",
            ],
        }

    def add_node_major_river(self, name, downstream_lag, downstream_node, has_catchment=True):

        model_dict = self.model_dict
        node_name = f"link_{name}"

        #############################################
        ################# Add nodes #################
        #############################################
        river = {"name": node_name, "type": "link"}
        model_dict["nodes"].append(river)

        if has_catchment: 
            # Most of the time river nodes will have catchment nodes ecxept for delTrenton
            # Add catchment node that sends inflows to reservoir
            catchment = {
                "name": f"catchment_{name}",
                "type": "catchment",
                "flow": f"flow_{name}",
            }
            model_dict["nodes"].append(catchment)

            # Add withdrawal node that withdraws flow from catchment for human use
            withdrawal = {
                "name": f"catchmentWithdrawal_{name}",
                "type": "link",
                "cost": -15.0,
                "max_flow": f"max_flow_catchmentWithdrawal_{name}",
            }
            model_dict["nodes"].append(withdrawal)

            # Add consumption node that removes flow from model - the rest of 
            # withdrawals return back to reservoir
            consumption = {
                "name": f"catchmentConsumption_{name}",
                "type": "output",
                "cost": -2000.0,
                "max_flow": f"max_flow_catchmentConsumption_{name}",
            }
            model_dict["nodes"].append(consumption)

        if downstream_lag > 0:
            delay = {
                "name": f"delay_{name}", 
                "type": "DelayNode", 
                "days": downstream_lag
                }
            model_dict["nodes"].append(delay)

        #############################################
        ########## Add edges between nodes ##########
        #############################################
        if has_catchment:
            model_dict["edges"] += [
                # catchment to river
                [f"catchment_{name}", node_name],
                # catchment to withdrawal
                [f"catchment_{name}", f"catchmentWithdrawal_{name}"],
                # withdrawal to consumption
                [f"catchmentWithdrawal_{name}", f"catchmentConsumption_{name}"],
                # withdrawal to river
                [f"catchmentWithdrawal_{name}", node_name]
            ]

        # reservoir downstream node (via outflow node if one exists)
        #!! consider to move into make_model
        if downstream_node in majorflow_list:
            downstream_name = f"link_{downstream_node}"
        elif downstream_node == "output_del":
            downstream_name = downstream_node
        else:
            downstream_name = f"reservoir_{downstream_node}"

        # outflow to (delay to) downstream node
        if downstream_lag > 0:
            model_dict["edges"] += [
                [node_name, f"delay_{name}"],
                [f"delay_{name}", downstream_name],
            ]
        else:
            model_dict["edges"].append([node_name, downstream_name])
        
        #############################################
        ########## Add standard parameters ##########
        #############################################
        if has_catchment:
            inflow_ensemble_indices = self.options.get("inflow_ensemble_indices", None)
            inflow_type = self.inflow_type
            ### Assign inflows to nodes
            if inflow_ensemble_indices:
                ### Use custom FlowEnsemble parameter to handle scenario indexing
                model_dict["parameters"][f"flow_{name}"] = {
                    "type": "FlowEnsemble",
                    "node": name,
                    "inflow_type": inflow_type,
                    "inflow_ensemble_indices": inflow_ensemble_indices,
                }

            else:
                inflow_source = f"{input_dir}catchment_inflow_{inflow_type}.csv"

                ### Use single-scenario historic data
                model_dict["parameters"][f"flow_{name}"] = {
                    "type": "dataframe",
                    "url": inflow_source,
                    "column": name,
                    "index_col": "datetime",
                    "parse_dates": True,
                }

            ### get max flow for catchment withdrawal nodes based on DRBC data
            model_dict["parameters"][f"max_flow_catchmentWithdrawal_{name}"] = {
                "type": "constant",
                "url": f"{input_dir}sw_avg_wateruse_Pywr-DRB_Catchments.csv",
                "column": "Total_WD_MGD",
                "index_col": "node",
                "index": node_name,
            }

            ### get max flow for catchment consumption nodes based on DRBC data
            ### assume the consumption_t = R * withdrawal_{t-1}, where R is the ratio of avg consumption to withdrawal from DRBC data
            model_dict["parameters"][f"catchmentConsumptionRatio_{name}"] = {
                "type": "constant",
                "url": f"{input_dir}sw_avg_wateruse_Pywr-DRB_Catchments.csv",
                "column": "Total_CU_WD_Ratio",
                "index_col": "node",
                "index": node_name,
            }
            model_dict["parameters"][f"prev_flow_catchmentWithdrawal_{name}"] = {
                "type": "flow",
                "node": f"catchmentWithdrawal_{name}",
            }
            model_dict["parameters"][f"max_flow_catchmentConsumption_{name}"] = {
                "type": "aggregated",
                "agg_func": "product",
                "parameters": [
                    f"catchmentConsumptionRatio_{name}",
                    f"prev_flow_catchmentWithdrawal_{name}",
                ],
            }
        pass          
    
    def add_node_final_basin_outlet(self):
        model_dict = self.model_dict
        ### Add final basin outlet node
        model_dict["nodes"].append({
                "name": "output_del",
                "type": "output"
            })
        
    def add_node_nyc_aggregated_storage_and_link(self):
        model_dict = self.model_dict
        ### Node for NYC aggregated storage across 3 reservoirs
        model_dict["nodes"].append({
                "name": "reservoir_agg_nyc",
                "type": "aggregatedstorage",
                "storage_nodes": [f"reservoir_{r}" for r in reservoir_list_nyc],
            })

        ### Nodes linking each NYC reservoir to NYC deliveries
        # reservoir_list_nyc = ["cannonsville", "pepacton", "neversink"]
        for r in reservoir_list_nyc:
            model_dict["nodes"].append({
                    "name": f"link_{r}_nyc",
                    "type": "link",
                    "cost": -500.0,
                    "max_flow": f"max_flow_delivery_nyc_{r}",
                })
    
    def add_node_nyc_and_nj_deliveries(self):
        model_dict = self.model_dict
        ### Nodes for NYC & NJ deliveries
        for d in ["nyc", "nj"]:
            model_dict["nodes"].append({
                    "name": f"delivery_{d}",
                    "type": "output",
                    "cost": -500.0,
                    "max_flow": f"max_flow_delivery_{d}",
                })
        
        ### Edges linking each NYC reservoir to NYC deliveries
        for r in reservoir_list_nyc:
            model_dict["edges"].append([f"reservoir_{r}", f"link_{r}_nyc"])
            model_dict["edges"].append([f"link_{r}_nyc", "delivery_nyc"])
        ### Edge linking Delaware River at DRCanal to NJ deliveries
        model_dict["edges"].append(["link_delDRCanal", "delivery_nj"])

    def add_parameter_nyc_and_nj_demands(self):
        """Add nyc and nj demands to model parameters."""
        model_dict = self.model_dict
        use_hist_NycNjDeliveries = self.options.get("use_hist_NycNjDeliveries", True)

        ### Demands for NYC & NJ deliveries
        # If this flag is True, we assume demand is equal to historical deliveries timeseries
        if use_hist_NycNjDeliveries:
            # NYC
            model_dict["parameters"][f"demand_nyc"] = {
                "type": "dataframe",
                "url": f"{input_dir}deliveryNYC_ODRM_extrapolated.csv",
                "column": "aggregate",
                "index_col": "datetime",
                "parse_dates": True,
            }
            # NJ
            model_dict["parameters"][f"demand_nj"] = {
                "type": "dataframe",
                "url": f"{input_dir}deliveryNJ_DRCanal_extrapolated.csv",
                "index_col": "datetime",
                "parse_dates": True,
            }
        # Otherwise, assume demand is equal to max allotment under FFMP
        else:
            # NYC
            model_dict["parameters"][f"demand_nyc"] = {
                "type": "constant",
                "url": "drb_model_constants.csv",
                "column": "value",
                "index_col": "parameter",
                "index": "max_flow_baseline_delivery_nyc",
            }
            # NJ
            model_dict["parameters"][f"demand_nj"] = {
                "type": "constant",
                "url": "drb_model_constants.csv",
                "column": "value",
                "index_col": "parameter",
                "index": "max_flow_baseline_monthlyAvg_delivery_nj",
            }

    def add_parameter_nyc_reservoirs_operational_regimes(self):
        model_dict = self.model_dict
        # Levels defining operational regimes for NYC reservoirs base on combined storage: 1a, 1b, 1c, 2, 3, 4, 5.
        # Note 1a assumed to fill remaining space, doesnt need to be defined here.
        levels = self.levels
        for level in levels[1:]:
            model_dict["parameters"][f"level{level}"] = {
                "type": "dailyprofile",
                "url": "drb_model_dailyProfiles.csv",
                "index_col": "profile",
                "index": f"level{level}",
            }

        ### Control curve index that tells us which level the aggregated NYC storage is currently in
        model_dict["parameters"]["drought_level_agg_nyc"] = {
            "type": "controlcurveindex",
            "storage_node": "reservoir_agg_nyc",
            "control_curves": [f"level{level}" for level in levels[1:]],
        }

        ### Factors defining delivery profiles for NYC and NJ, for each storage level: 1a, 1b, 1c, 2, 3, 4, 5.
        demands = ["nyc", "nj"]
        for demand in demands:
            for level in levels:
                model_dict["parameters"][f"level{level}_factor_delivery_{demand}"] = {
                    "type": "constant",
                    "url": "drb_model_constants.csv",
                    "column": "value",
                    "index_col": "parameter",
                    "index": f"level{level}_factor_delivery_{demand}",
                }

        ### Indexed arrays that dictate cutbacks to NYC & NJ deliveries, based on current storage level and DOY
        for demand in demands:
            model_dict["parameters"][f"drought_factor_delivery_{demand}"] = {
                "type": "indexedarray",
                "index_parameter": "drought_level_agg_nyc",
                "params": [f"level{level}_factor_delivery_{demand}" for level in levels],
            }

        ### Control curve index that tells us which level each individual NYC reservoir's storage is currently in
        for reservoir in reservoir_list_nyc:
            model_dict["parameters"][f"drought_level_{reservoir}"] = {
                "type": "controlcurveindex",
                "storage_node": f"reservoir_{reservoir}",
                "control_curves": [f"level{level}" for level in levels[1:]],
            }

    def add_parameter_nyc_and_nj_delivery_constraints(self):
        model_dict = self.model_dict

        #######################################################################
        ### NYC & NJ delivery baseline under normal conditions
        #######################################################################
        # Max allowable delivery to NYC (on moving avg)
        model_dict["parameters"]["max_flow_baseline_delivery_nyc"] = {
            "type": "constant",
            "url": "drb_model_constants.csv",
            "column": "value",
            "index_col": "parameter",
            "index": "max_flow_baseline_delivery_nyc",
        }

        # NJ has both a daily limit and monthly average limit
        model_dict["parameters"]["max_flow_baseline_daily_delivery_nj"] = {
            "type": "constant",
            "url": "drb_model_constants.csv",
            "column": "value",
            "index_col": "parameter",
            "index": "max_flow_baseline_daily_delivery_nj",
        }
        model_dict["parameters"]["max_flow_baseline_monthlyAvg_delivery_nj"] = {
            "type": "constant",
            "url": "drb_model_constants.csv",
            "column": "value",
            "index_col": "parameter",
            "index": "max_flow_baseline_monthlyAvg_delivery_nj",
        }

        #######################################################################
        ### FFMP-based NYC & NJ delivery constraints under drought conditions
        #######################################################################
        # Max allowable daily delivery to NYC & NJ- this will be very large for levels 
        # 1&2 (so that moving average limit in the next parameter is active), but apply 
        # daily flow limits for more drought stages.
        model_dict["parameters"]["max_flow_drought_delivery_nyc"] = {
            "type": "aggregated",
            "agg_func": "product",
            "parameters": ["max_flow_baseline_delivery_nyc", "drought_factor_delivery_nyc"],
        }
        # drought_factor_delivery_nyc is defined in add_parameter_nyc_reservoirs_operational_regimes()

        # Max allowable delivery to NYC in current time step to maintain moving avg
        # limit. Based on custom Pywr parameter.
        model_dict["parameters"]["max_flow_ffmp_delivery_nyc"] = {
            "type": "FfmpNycRunningAvg",
            "node": "delivery_nyc",
            "max_avg_delivery": "max_flow_baseline_delivery_nyc",
        }

        # Now actual max flow to NYC delivery node is the min of demand, daily FFMP 
        # limit, and daily limit to meet FFMP moving avg limit
        model_dict["parameters"]["max_flow_delivery_nyc"] = {
            "type": "aggregated",
            "agg_func": "min",
            "parameters": [
                "demand_nyc", # defined in add_parameter_nyc_and_nj_demands()
                "max_flow_drought_delivery_nyc",
                "max_flow_ffmp_delivery_nyc",
            ],
        }

        # Max allowable delivery to NJ in current time step to maintain moving avg limit. 
        # Based on custom Pywr parameter.
        model_dict["parameters"]["max_flow_ffmp_delivery_nj"] = {
            "type": "FfmpNjRunningAvg",
            "node": "delivery_nj",
            "max_avg_delivery": "max_flow_baseline_monthlyAvg_delivery_nj",
            "max_daily_delivery": "max_flow_baseline_daily_delivery_nj",
            "drought_factor": "drought_factor_delivery_nj",
        }
        # FfmpNjRunningAvg is registered in ffmp.py

        # Now actual max flow to NJ delivery node is the min of demand and daily limit
        # to meet FFMP moving avg limit
        model_dict["parameters"]["max_flow_delivery_nj"] = {
            "type": "aggregated",
            "agg_func": "min",
            "parameters": ["demand_nj", "max_flow_ffmp_delivery_nj"],
        }

    # Revisit this after reading the reports
    def add_parameter_nyc_reservoirs_min_require_flow(self):
        model_dict = self.model_dict
        levels = self.levels
        # Baseline release flow rate for each NYC reservoir, dictated by FFMP
        # reservoir_list_nyc = ["cannonsville", "pepacton", "neversink"]
        for reservoir in reservoir_list_nyc:
            model_dict["parameters"][f"mrf_baseline_{reservoir}"] = {
                "type": "constant",
                "url": "drb_model_constants.csv",
                "column": "value",
                "index_col": "parameter",
                "index": f"mrf_baseline_{reservoir}",
            }
        
        ### Factor governing changing release reqs from NYC reservoirs, based on aggregated storage across 3 reservoirs
        for reservoir in reservoir_list_nyc:
            model_dict["parameters"][f"mrf_drought_factor_agg_{reservoir}"] = {
                "type": "indexedarray",
                "index_parameter": "drought_level_agg_nyc",
                "params": [f"level{level}_factor_mrf_{reservoir}" for level in levels],
            }
        
        ### Levels defining operational regimes for individual NYC reservoirs, as opposed to aggregated level across 3 reservoirs
        for reservoir in reservoir_list_nyc:
            for level in levels:
                model_dict["parameters"][f"level{level}_factor_mrf_{reservoir}"] = {
                    "type": "dailyprofile",
                    "url": "drb_model_dailyProfiles.csv",
                    "index_col": "profile",
                    "index": f"level{level}_factor_mrf_{reservoir}",
                }

        ### Factor governing changing release reqs from NYC reservoirs, based on individual storage for particular reservoir
        for reservoir in reservoir_list_nyc:
            model_dict["parameters"][f"mrf_drought_factor_individual_{reservoir}"] = {
                "type": "indexedarray",
                "index_parameter": f"drought_level_{reservoir}",
                "params": [f"level{level}_factor_mrf_{reservoir}" for level in levels],
            }

        ### Factor governing changing release reqs from NYC reservoirs, depending on whether aggregated or individual storage level is activated
        ### Based on custom Pywr parameter.
        for reservoir in reservoir_list_nyc:
            model_dict["parameters"][f"mrf_drought_factor_combined_final_{reservoir}"] = {
                "type": "NYCCombinedReleaseFactor",
                "node": f"reservoir_{reservoir}",
            }

        ### FFMP mandated releases from NYC reservoirs
        for reservoir in reservoir_list_nyc:
            model_dict["parameters"][f"mrf_target_individual_{reservoir}"] = {
                "type": "aggregated",
                "agg_func": "product",
                "parameters": [
                    f"mrf_baseline_{reservoir}",
                    f"mrf_drought_factor_combined_final_{reservoir}",
                ],
            }

        ### sum of FFMP mandated releases from NYC reservoirs
        model_dict["parameters"]["mrf_target_individual_agg_nyc"] = {
            "type": "aggregated",
            "agg_func": "sum",
            "parameters": [
                f"mrf_target_individual_{reservoir}" for reservoir in reservoir_list_nyc
            ],
        }

    def add_parameter_nyc_reservoirs_flood_control(self):
        model_dict = self.model_dict
        ### extra flood releases for NYC reservoirs specified in FFMP
        for reservoir in reservoir_list_nyc:
            model_dict["parameters"][f"weekly_rolling_mean_flow_{reservoir}"] = {
                "type": "RollingMeanFlowNode",
                "node": f"reservoir_{reservoir}",
                "timesteps": 7,
                "name": f"flow_{reservoir}",
                "initial_flow": 0,
            }
            model_dict["parameters"][f"flood_release_{reservoir}"] = {
                "type": "NYCFloodRelease",
                "node": f"reservoir_{reservoir}",
            }

        ### sum of flood control releases from NYC reservoirs
        model_dict["parameters"]["flood_release_agg_nyc"] = {
            "type": "aggregated",
            "agg_func": "sum",
            "parameters": [
                f"flood_release_{reservoir}" for reservoir in reservoir_list_nyc
            ],
        }

    # Can be removed after volume balancing rules are implemented
    def add_parameter_nyc_reservoirs_variable_cost_based_on_fractional_storage(self):
        ### variable storage cost for each reservoir, based on its fractional storage
        ### Note: may not need this anymore now that we have volume balancing rules. but maybe makes sense to leave in for extra protection.
        model_dict = self.model_dict
        EPS = self.EPS
        volumes = {
            "cannonsville": get_reservoir_capacity("cannonsville"),
            "pepacton": get_reservoir_capacity("pepacton"),
            "neversink": get_reservoir_capacity("neversink"),
        }
        for reservoir in reservoir_list_nyc:
            model_dict["parameters"][f"storage_cost_{reservoir}"] = {
                "type": "interpolatedvolume",
                "values": [-100, -1],
                "node": f"reservoir_{reservoir}",
                "volumes": [-EPS, volumes[reservoir] + EPS],
            }
    
    def add_parameter_nyc_reservoirs_current_volume(self):
        model_dict = self.model_dict
        EPS = self.EPS
        ### current volume stored in each reservoir
        for reservoir in reservoir_list_nyc + ["agg_nyc"]:
            model_dict["parameters"][f"volume_{reservoir}"] = {
                "type": "interpolatedvolume",
                "values": [-EPS, 1000000],
                "node": f"reservoir_{reservoir}",
                "volumes": [-EPS, 1000000],
            }

    def add_parameter_nyc_reservoirs_aggregated_info(self):
        model_dict = self.model_dict
        EPS = self.EPS
        ### current volume stored in the aggregated storage node 
        model_dict["parameters"]["volume_agg_nyc"] = {
            "type": "interpolatedvolume",
            "values": [-EPS, 1000000],
            "node": "reservoir_agg_nyc",
            "volumes": [-EPS, 1000000],
        }

        ### aggregated inflows to NYC reservoirs
        model_dict["parameters"]["flow_agg_nyc"] = {
            "type": "aggregated",
            "agg_func": "sum",
            "parameters": [f"flow_{reservoir}" for reservoir in reservoir_list_nyc],
        }

        ### aggregated max volume across NYC reservoirs
        model_dict["parameters"]["max_volume_agg_nyc"] = {
            "type": "aggregated",
            "agg_func": "sum",
            "parameters": [f"max_volume_{reservoir}" for reservoir in reservoir_list_nyc],
        }

    def add_parameter_montague_trenton_flow_targets(self):
        model_dict = self.model_dict
        levels = self.levels
        ### Baseline flow target at Montague & Trenton
        mrfs = ["delMontague", "delTrenton"]
        for mrf in mrfs:
            model_dict["parameters"][f"mrf_baseline_{mrf}"] = {
                "type": "constant",
                "url": "drb_model_constants.csv",
                "column": "value",
                "index_col": "parameter",
                "index": f"mrf_baseline_{mrf}",
            }

        ### Seasonal multiplier factors for Montague & Trenton flow targets based on drought level of NYC aggregated storage
        for mrf in mrfs:
            for level in levels:
                model_dict["parameters"][f"level{level}_factor_mrf_{mrf}"] = {
                    "type": "monthlyprofile",
                    "url": "drb_model_monthlyProfiles.csv",
                    "index_col": "profile",
                    "index": f"level{level}_factor_mrf_{mrf}",
                }

        ### Current value of seasonal multiplier factor for Montague & Trenton flow targets based on drought level of NYC aggregated storage
        for mrf in mrfs:
            model_dict["parameters"][f"mrf_drought_factor_{mrf}"] = {
                "type": "indexedarray",
                "index_parameter": "drought_level_agg_nyc",
                "params": [f"level{level}_factor_mrf_{mrf}" for level in levels],
            }

        ### Total Montague & Trenton flow targets based on drought level of NYC aggregated storage
        for mrf in mrfs:
            model_dict["parameters"][f"mrf_target_{mrf}"] = {
                "type": "aggregated",
                "agg_func": "product",
                "parameters": [f"mrf_baseline_{mrf}", f"mrf_drought_factor_{mrf}"],
            }
    
    #?? Not yet understand how this works
    def add_parameter_predicted_lagged_non_nyc_inflows_to_Montague_and_Trenton_and_lagged_nj_demands(self):
        model_dict = self.model_dict
        inflow_ensemble_indices = self.options.get("inflow_ensemble_indices")
        inflow_type = self.inflow_type
        ### total predicted lagged non-NYC inflows to Montague & Trenton, and predicted lagged NJ demands
        if inflow_ensemble_indices is None:
            for mrf, lag in zip(
                (
                    "delMontague",
                    "delMontague",
                    "delTrenton",
                    "delTrenton",
                    "delTrenton",
                    "delTrenton",
                ),
                (1, 2, 1, 2, 3, 4),
            ):
                label = f"{mrf}_lag{lag}_{flow_prediction_mode}"
                model_dict["parameters"][f"predicted_nonnyc_gage_flow_{mrf}_lag{lag}"] = {
                    "type": "dataframe",
                    "url": f"{input_dir}predicted_inflows_diversions_{inflow_type}.csv",
                    "column": label,
                    "index_col": "datetime",
                    "parse_dates": True,
                }
            ### now get predicted nj demand
            for lag in range(1, 5):
                label = f"demand_nj_lag{lag}_{flow_prediction_mode}"
                model_dict["parameters"][f"predicted_demand_nj_lag{lag}"] = {
                    "type": "dataframe",
                    "url": f"{input_dir}predicted_inflows_diversions_{inflow_type}.csv",
                    "column": label,
                    "index_col": "datetime",
                    "parse_dates": True,
                }
        else:
            ### Use custom PredictionEnsemble parameter to handle scenario indexing
            for mrf, lag in zip(
                (
                    "delMontague",
                    "delMontague",
                    "delTrenton",
                    "delTrenton",
                    "delTrenton",
                    "delTrenton",
                ),
                (1, 2, 1, 2, 3, 4),
            ):
                label = f"{mrf}_lag{lag}_{flow_prediction_mode}"

                model_dict["parameters"][f"predicted_nonnyc_gage_flow_{mrf}_lag{lag}"] = {
                    "type": "PredictionEnsemble",
                    "column": label,
                    "inflow_type": inflow_type,
                    "ensemble_indices": inflow_ensemble_indices,
                }

            ### now get predicted nj demand (this is the same across ensemble)
            # Drop the '_ensemble' suffix from the inflow_type
            predicted_inflow_type = inflow_type.replace("_ensemble", "")
            predicted_inflow_type = (
                predicted_inflow_type.replace("syn_", "")
                if "syn_" in predicted_inflow_type
                else predicted_inflow_type
            )

            for lag in range(1, 5):
                label = f"demand_nj_lag{lag}_{flow_prediction_mode}"
                model_dict["parameters"][f"predicted_demand_nj_lag{lag}"] = {
                    "type": "dataframe",
                    "url": f"{input_dir}predicted_inflows_diversions_{predicted_inflow_type}.csv",
                    "column": label,
                    "index_col": "datetime",
                    "parse_dates": True,
                }

    #?? Not yet understand how this works
    def add_parameter_nyc_reservoirs_balancing_methods(self):
        model_dict = self.model_dict
        ### Get total release needed from NYC reservoirs to satisfy Montague & Trenton flow targets,
        ### above and beyond their individually mandated releases, & after accounting for non-NYC inflows and NJ diversions.

        # This first step is based on predicted inflows to Montague in 2 days and Trenton in 4 days, and is used
        # to calculate balanced releases from all 3 reservoirs. But only Cannonsville & Pepacton actually use these
        # releases, because Neversink is adjusted later because it is 1 day closer travel time & has more info.
        # Uses custom Pywr parameter.

        step = 1
        ### first get release needed to meet Montague target in 2 days
        model_dict["parameters"][f"release_needed_mrf_montague_step{step}"] = {
            "type": "TotalReleaseNeededForDownstreamMRF",
            "mrf": "delMontague",
            "step": step,
        }
        ### now get addl release (above Montague release) needed to meet Trenton target in 4 days
        model_dict["parameters"][f"release_needed_mrf_trenton_step{step}"] = {
            "type": "TotalReleaseNeededForDownstreamMRF",
            "mrf": "delTrenton",
            "step": step,
        }
        ### total mrf release needed is sum of Montague & Trenton. This Step 1 is for Cannonsville/Pepacton releases.
        model_dict["parameters"][f"total_agg_mrf_montagueTrenton_step{step}"] = {
            "type": "aggregated",
            "agg_func": "sum",
            "parameters": [
                f"release_needed_mrf_montague_step{step}",
                f"release_needed_mrf_trenton_step{step}",
            ],
        }

        ## Max available Trenton contribution from each available lower basin reservoir. Step 1 is for Cannonsville/Pepacton release, so 4 days ahead for Trenton.
        for reservoir in drbc_lower_basin_reservoirs:
            model_dict["parameters"][f"max_mrf_trenton_step{step}_{reservoir}"] = {
                "type": "LowerBasinMaxMRFContribution",
                "node": f"reservoir_{reservoir}",
                "step": step,
            }

        ## Aggregate total expected lower basin contribution to Trenton
        model_dict["parameters"][f"lower_basin_agg_mrf_trenton_step{step}"] = {
            "type": "VolBalanceLowerBasinMRFAggregate",
            "step": step,
        }

        ### now calculate actual Cannonsville & Pepacton releases to meet Montague&Trenton, with assumed releases for Neversink & lower basin
        for reservoir in ["cannonsville", "pepacton"]:
            model_dict["parameters"][f"mrf_montagueTrenton_{reservoir}"] = {
                "type": f"VolBalanceNYCDownstreamMRF_step{step}",
                "node": f"reservoir_{reservoir}",
            }

        ### step 2:
        ### now update Neversink release requirement based on yesterday's Can&Pep releases & extra day of flow observations,
        ### and similarly update lower basin releases with extra day of information
        step = 2
        ### get previous day's releases from Can & Pep
        for reservoir in ["cannonsville", "pepacton"]:
            lag = 1
            model_dict["parameters"][f"outflow_{reservoir}_rollmean{lag}"] = {
                "type": "rollingmeanflownode",
                "node": f"outflow_{reservoir}",
                "days": lag,
            }
            model_dict["parameters"][f"spill_{reservoir}_rollmean{lag}"] = {
                "type": "rollingmeanflownode",
                "node": f"spill_{reservoir}",
                "days": lag,
            }
            model_dict["parameters"][f"release_{reservoir}_lag{lag}"] = {
                "type": "LaggedReservoirRelease",
                "node": reservoir,
                "lag": lag,
            }

        ### get release needed to meet Montague target in 1 days
        model_dict["parameters"][f"release_needed_mrf_montague_step{step}"] = {
            "type": "TotalReleaseNeededForDownstreamMRF",
            "mrf": "delMontague",
            "step": step,
        }
        ### now get addl release (above Montague release) needed to meet Trenton target in 3 days
        model_dict["parameters"][f"release_needed_mrf_trenton_step{step}"] = {
            "type": "TotalReleaseNeededForDownstreamMRF",
            "mrf": "delTrenton",
            "step": step,
        }
        ### total mrf release needed is sum of Montague & Trenton. This Step 2 is for Neversink releases.
        model_dict["parameters"][f"total_agg_mrf_montagueTrenton_step{step}"] = {
            "type": "aggregated",
            "agg_func": "sum",
            "parameters": [
                f"release_needed_mrf_montague_step{step}",
                f"release_needed_mrf_trenton_step{step}",
            ],
        }

        ## Max available Trenton contribution from each available lower basin reservoir. Step 2 is for Neversink release, so 3 days ahead for Trenton.
        for reservoir in drbc_lower_basin_reservoirs:
            model_dict["parameters"][f"max_mrf_trenton_step{step}_{reservoir}"] = {
                "type": "LowerBasinMaxMRFContribution",
                "node": f"reservoir_{reservoir}",
                "step": step,
            }

        ## Aggregate total expected lower basin contribution to Trenton
        model_dict["parameters"][f"lower_basin_agg_mrf_trenton_step{step}"] = {
            "type": "VolBalanceLowerBasinMRFAggregate",
            "step": step,
        }

        ### Now assign Neversink releases to meet Montague/Trenton mrf, after accting for previous Can/Pep releases & expected lower basin contribution
        model_dict["parameters"][f"mrf_montagueTrenton_neversink"] = {
            "type": f"VolBalanceNYCDownstreamMRF_step{step}",
        }

        ### step 3:
        ### Update lower basin releases that are 2 days from Trenton equivalent (Beltzville & Blue Marsh)
        step = 3

        ### get previous day's releases from Neversink
        for reservoir in ["neversink"]:
            lag = 1
            model_dict["parameters"][f"outflow_{reservoir}_rollmean{lag}"] = {
                "type": "rollingmeanflownode",
                "node": f"outflow_{reservoir}",
                "days": lag,
            }
            model_dict["parameters"][f"spill_{reservoir}_rollmean{lag}"] = {
                "type": "rollingmeanflownode",
                "node": f"spill_{reservoir}",
                "days": lag,
            }
            model_dict["parameters"][f"release_{reservoir}_lag{lag}"] = {
                "type": "LaggedReservoirRelease",
                "node": reservoir,
                "lag": lag,
            }
        ### and get 2-days ago release from Can/Pep
        for reservoir in ["cannonsville", "pepacton"]:
            lag = 2
            model_dict["parameters"][f"outflow_{reservoir}_rollmean{lag}"] = {
                "type": "rollingmeanflownode",
                "node": f"outflow_{reservoir}",
                "days": lag,
            }
            model_dict["parameters"][f"spill_{reservoir}_rollmean{lag}"] = {
                "type": "rollingmeanflownode",
                "node": f"spill_{reservoir}",
                "days": lag,
            }
            model_dict["parameters"][f"release_{reservoir}_lag{lag}"] = {
                "type": "LaggedReservoirRelease",
                "node": reservoir,
                "lag": lag,
            }

        ### now get addl release (above NYC releases from last 2 days) needed to meet Trenton target in 2 days
        model_dict["parameters"][f"release_needed_mrf_trenton_step{step}"] = {
            "type": "TotalReleaseNeededForDownstreamMRF",
            "mrf": "delTrenton",
            "step": step,
        }

        ## Max available Trenton contribution from each available lower basin reservoir. Step 3 is for Beltzville/BlueMarsh release, so 2 days ahead for Trenton.
        for reservoir in drbc_lower_basin_reservoirs:
            model_dict["parameters"][f"max_mrf_trenton_step{step}_{reservoir}"] = {
                "type": "LowerBasinMaxMRFContribution",
                "node": f"reservoir_{reservoir}",
                "step": step,
            }

        ## Aggregate total expected lower basin contribution to Trenton
        model_dict["parameters"][f"lower_basin_agg_mrf_trenton_step{step}"] = {
            "type": "VolBalanceLowerBasinMRFAggregate",
            "step": step,
        }

        ## Now dispatch actual individual releases from lower basin reservoirs for meeting Trenton - step 3 is Blue marsh/beltzville
        for reservoir in ["beltzvilleCombined", "blueMarsh"]:
            model_dict["parameters"][f"mrf_trenton_{reservoir}"] = {
                "type": "VolBalanceLowerBasinMRFIndividual",
                "node": f"reservoir_{reservoir}",
                "step": step,
            }

        ### step 4:
        ### Update lower basin releases that are 1 days from Trenton equivalent (Nockamixon)
        step = 4

        ### get previous day's releases from Beltzville & Blue Marsh
        for reservoir in ["beltzvilleCombined", "blueMarsh"]:
            lag = 1
            model_dict["parameters"][f"outflow_{reservoir}_rollmean{lag}"] = {
                "type": "rollingmeanflownode",
                "node": f"outflow_{reservoir}",
                "days": lag,
            }
            model_dict["parameters"][f"spill_{reservoir}_rollmean{lag}"] = {
                "type": "rollingmeanflownode",
                "node": f"spill_{reservoir}",
                "days": lag,
            }
            model_dict["parameters"][f"release_{reservoir}_lag{lag}"] = {
                "type": "LaggedReservoirRelease",
                "node": reservoir,
                "lag": lag,
            }

        ### and get 2-days ago releases from Neversink
        for reservoir in ["neversink"]:
            lag = 2
            model_dict["parameters"][f"outflow_{reservoir}_rollmean{lag}"] = {
                "type": "rollingmeanflownode",
                "node": f"outflow_{reservoir}",
                "days": lag,
            }
            model_dict["parameters"][f"spill_{reservoir}_rollmean{lag}"] = {
                "type": "rollingmeanflownode",
                "node": f"spill_{reservoir}",
                "days": lag,
            }
            model_dict["parameters"][f"release_{reservoir}_lag{lag}"] = {
                "type": "LaggedReservoirRelease",
                "node": reservoir,
                "lag": lag,
            }
        ### and get 3-days ago release from Can/Pep
        for reservoir in ["cannonsville", "pepacton"]:
            lag = 3
            model_dict["parameters"][f"outflow_{reservoir}_rollmean{lag}"] = {
                "type": "rollingmeanflownode",
                "node": f"outflow_{reservoir}",
                "days": lag,
            }
            model_dict["parameters"][f"spill_{reservoir}_rollmean{lag}"] = {
                "type": "rollingmeanflownode",
                "node": f"spill_{reservoir}",
                "days": lag,
            }
            model_dict["parameters"][f"release_{reservoir}_lag{lag}"] = {
                "type": "LaggedReservoirRelease",
                "node": reservoir,
                "lag": lag,
            }

        ### now get addl release (above NYC releases from steps 1-2 & lower basin releases from step 3) needed to meet Trenton target in 1 days
        model_dict["parameters"][f"release_needed_mrf_trenton_step{step}"] = {
            "type": "TotalReleaseNeededForDownstreamMRF",
            "mrf": "delTrenton",
            "step": step,
        }

        ## Max available Trenton contribution from each available lower basin reservoir. Step 4 is just for Nockamixon, 1 day above Trenton.
        for reservoir in drbc_lower_basin_reservoirs:
            model_dict["parameters"][f"max_mrf_trenton_step{step}_{reservoir}"] = {
                "type": "LowerBasinMaxMRFContribution",
                "node": f"reservoir_{reservoir}",
                "step": step,
            }

        ## Aggregate total expected lower basin contribution to Trenton
        model_dict["parameters"][f"lower_basin_agg_mrf_trenton_step{step}"] = {
            "type": "VolBalanceLowerBasinMRFAggregate",
            "step": step,
        }

        ## Now dispatch actual individual releases from lower basin reservoirs for meeting Trenton - step 4 is just Nockamixon
        for reservoir in ["nockamixon"]:
            model_dict["parameters"][f"mrf_trenton_{reservoir}"] = {
                "type": "VolBalanceLowerBasinMRFIndividual",
                "node": f"reservoir_{reservoir}",
                "step": step,
            }

        ### get final downstream release from each NYC reservoir, which is the sum of its
        ###    individually-mandated release from FFMP, flood control release, and its contribution to the
        ### Montague/Trenton targets
        for reservoir in reservoir_list_nyc:
            model_dict["parameters"][f"downstream_release_target_{reservoir}"] = {
                "type": "aggregated",
                "agg_func": "sum",
                "parameters": [
                    f"mrf_target_individual_{reservoir}",
                    f"flood_release_{reservoir}",
                    f"mrf_montagueTrenton_{reservoir}",
                ],
            }
        ## From Lower Basin reservoirs: sum of STARFIT and mrf contribution
        for reservoir in drbc_lower_basin_reservoirs:
            model_dict["parameters"][f"downstream_release_target_{reservoir}"] = {
                "type": "aggregated",
                "agg_func": "sum",
                "parameters": [f"mrf_trenton_{reservoir}", f"starfit_release_{reservoir}"],
            }

        ### now distribute NYC deliveries across 3 reservoirs with volume balancing after accounting for downstream releases
        for reservoir in reservoir_list_nyc:
            ### max diversion to NYC from each reservoir based on historical data
            model_dict["parameters"][f"hist_max_flow_delivery_nyc_{reservoir}"] = {
                "type": "constant",
                "value": get_reservoir_max_diversion_NYC(reservoir),
            }
            ### Target diversion from each NYC reservoir to satisfy NYC demand,
            ### accounting for historical max diversion constraints
            ### and attempting to balance storages across 3 NYC reservoirs
            ### Uses custom Pywr parameter.
            model_dict["parameters"][f"max_flow_delivery_nyc_{reservoir}"] = {
                "type": "VolBalanceNYCDemand",
                "node": f"reservoir_{reservoir}",
            }

    #!! Not yet complete
    def add_parameter_couple_temp_lstm(self):
        model_dict = self.model_dict
        try:
            from pywrdrb.parameters.temperature import TemperaturePrediction

            model_dict["parameters"]["predicted_mu_max_of_temperature"] = {
                "type": "TemperaturePrediction"
            }
        except Exception as e:
            print(f"Temperature prediction model not available. Error: {e}")

    #!! Not yet complete
    def add_parameter_couple_salinity_lstm(self):
        model_dict = self.model_dict
        try:
            from pywrdrb.parameters.salinity import SalinityPrediction

            model_dict["parameters"]["predicted_salinity"] = {"type": "SalinityPrediction"}
        except Exception as e:
            print(f"Salinity prediction model not available. Error: {e}")
         

        