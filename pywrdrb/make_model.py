"""
Contains functions used to construct a pywrdrb model in JSON format.
"""

import json
import pandas as pd
from collections import defaultdict

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
    upstream_nodes_dict,
    immediate_downstream_nodes_dict,
    downstream_node_lags,
)
from pywrdrb.parameters.lower_basin_ffmp import lag_days_from_Trenton

### model options/parameters
flow_prediction_mode = "regression_disagg"  ### 'regression_agg', 'regression_disagg', 'perfect_foresight', 'same_day', 'moving_average'
use_lower_basin_mrf_contributions = True


EPS = 1e-8


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


##########################################################################################
### add_major_node()
##########################################################################################


def add_major_node(
    model,
    name,
    node_type,
    inflow_type,
    starfit_release,
    regulatory_release,
    downstream_node=None,
    downstream_lag=0,
    capacity=None,
    initial_volume_frac=None,
    variable_cost=None,
    has_catchment=True,
    inflow_ensemble_indices=None,
    run_starfit_sensitivity_analysis=False,
    sensitivity_analysis_scenarios=list(range(0, 2001)),
):
    """
    Add a major node to the model.

    Major node types include reservoirs and rivers. This function adds the major node
    and all standard minor nodes that belong to each major node (i.e., catchment,
    withdrawal, consumption, outflow), along with their standard parameters and edges.
    All nodes, edges, and parameters are added to the model dictionary, which is then returned.

    Args:
        model (dict): The dictionary holding all model elements, which will be written to a JSON file upon completion.
        name (str): The name of the major node.
        node_type (str): The type of major node, either 'reservoir' or 'river'.
        inflow_type (str): The inflow type, e.g., 'nhmv10'.
        outflow_type (str): Define what type of outflow node to use (if any), either 'starfit' or 'regulatory'.
        downstream_node (str): The name of the node directly downstream for writing the edge network.
        downstream_lag (int): The travel time (in days) between flow leaving a node and reaching its downstream node.
        capacity (float): The capacity of the reservoir in million gallons (MG) (reservoirs only).
        initial_volume_frac (float): The initial fraction full for the reservoir (reservoirs only).
                                     Note: This is a fraction, not a percentage, despite being named "initial_volume_pc" in the Pywr JSON by convention.
        variable_cost (bool): If False, the cost is fixed throughout the simulation.
                              If True, it varies according to a state-dependent parameter (reservoirs only).
        has_catchment (bool): True if the node has a catchment with inflows and withdrawal/consumption.
                              False for artificial nodes that coincide with another (e.g., delTrenton, which shares a catchment with delDRCanal).
        inflow_ensemble_indices (None or list): List of ensemble indices for inflows (optional).
        run_starfit_sensitivity_analysis (bool): If True, run STARFIT sensitivity analysis. Default is False.
        sensitivity_analysis_scenarios (list): List of scenario ID numbers for STARFIT sensitivity analysis.

    Returns:
        dict: The updated model dictionary.
    """

    ### NYC reservoirs are a bit more complex, leave some of model creation in csv files for now
    is_NYC_reservoir = name in ["cannonsville", "pepacton", "neversink"]
    ### does it have explicit outflow node for starfit or regulatory behavior?
    has_outflow_node = True if (starfit_release or regulatory_release) else False
    ### first add major node to dict
    if node_type == "reservoir":
        node_name = f"reservoir_{name}"
        initial_volume = capacity * initial_volume_frac
        reservoir = {
            "name": node_name,
            "type": "storage",
            "max_volume": f"max_volume_{name}",
            "initial_volume": initial_volume,
            "initial_volume_pc": initial_volume_frac,
            "cost": -10.0 if not variable_cost else f"storage_cost_{name}",
        }
        model["nodes"].append(reservoir)
    elif node_type == "river":
        node_name = f"link_{name}"
        river = {"name": node_name, "type": "link"}
        model["nodes"].append(river)

    if has_catchment:
        ### add catchment node that sends inflows to reservoir
        catchment = {
            "name": f"catchment_{name}",
            "type": "catchment",
            "flow": f"flow_{name}",
        }
        model["nodes"].append(catchment)

        ### add withdrawal node that withdraws flow from catchment for human use
        withdrawal = {
            "name": f"catchmentWithdrawal_{name}",
            "type": "link",
            "cost": -15.0,
            "max_flow": f"max_flow_catchmentWithdrawal_{name}",
        }
        model["nodes"].append(withdrawal)

        ### add consumption node that removes flow from model - the rest of withdrawals return back to reservoir
        consumption = {
            "name": f"catchmentConsumption_{name}",
            "type": "output",
            "cost": -2000.0,
            "max_flow": f"max_flow_catchmentConsumption_{name}",
        }
        model["nodes"].append(consumption)

    ### add outflow node (if any), either using STARFIT rules or regulatory targets.
    ### Note reservoirs must have either starfit or regulatory, river nodes have either regulatory or None
    if has_outflow_node:
        # Lower basin reservoirs with no FFMP connection
        if starfit_release and not regulatory_release:
            outflow = {
                "name": f"outflow_{name}",
                "type": "link",
                "cost": -500.0,
                "max_flow": f"starfit_release_{name}",
            }
        # Lower basin reservoirs which contribute to Montague/Trenton
        elif starfit_release and regulatory_release:
            outflow = {
                "name": f"outflow_{name}",
                "type": "link",
                "cost": -500.0,
                "max_flow": f"downstream_release_target_{name}",
            }
        # NYC Reservoirs
        elif regulatory_release and not starfit_release:
            outflow = {
                "name": f"outflow_{name}",
                "type": "link",
                "cost": -1000.0,
                "max_flow": f"downstream_release_target_{name}",
            }
        model["nodes"].append(outflow)

        ### add secondary high-cost flow path for spill above max_flow
        outflow = {"name": f"spill_{name}", "type": "link", "cost": 5000.0}
        model["nodes"].append(outflow)

    ### add Delay node to account for flow travel time between nodes. Lag unit is days.
    if downstream_lag > 0:
        delay = {"name": f"delay_{name}", "type": "DelayNode", "days": downstream_lag}
        model["nodes"].append(delay)

    ### now add edges of model flow network
    if has_catchment:
        ### catchment to reservoir
        model["edges"].append([f"catchment_{name}", node_name])
        ### catchment to withdrawal
        model["edges"].append([f"catchment_{name}", f"catchmentWithdrawal_{name}"])
        ### withdrawal to consumption
        model["edges"].append(
            [f"catchmentWithdrawal_{name}", f"catchmentConsumption_{name}"]
        )
        ### withdrawal to reservoir
        model["edges"].append([f"catchmentWithdrawal_{name}", node_name])
    ### reservoir downstream node (via outflow node if one exists)
    if downstream_node in majorflow_list:
        downstream_name = f"link_{downstream_node}"
    elif downstream_node == "output_del":
        downstream_name = downstream_node
    else:
        downstream_name = f"reservoir_{downstream_node}"
    if has_outflow_node:
        model["edges"].append([node_name, f"outflow_{name}"])
        if downstream_lag > 0:
            model["edges"].append([f"outflow_{name}", f"delay_{name}"])
            model["edges"].append([f"delay_{name}", downstream_name])
        else:
            model["edges"].append([f"outflow_{name}", downstream_name])
        ### add secondary high-cost spill path
        model["edges"].append([node_name, f"spill_{name}"])
        if downstream_lag > 0:
            model["edges"].append([f"spill_{name}", f"delay_{name}"])
        else:
            model["edges"].append([f"spill_{name}", downstream_name])

    else:
        if downstream_lag > 0:
            model["edges"].append([node_name, f"delay_{name}"])
            model["edges"].append([f"delay_{name}", downstream_name])
        else:
            model["edges"].append([node_name, downstream_name])

    ################################################################
    ### now add standard parameters
    ################################################################
    ### max volume of reservoir, from GRanD database except where adjusted from other sources (eg NYC)
    if node_type == "reservoir":
        model["parameters"][f"max_volume_{name}"] = {
            "type": "constant",
            "url": "drb_model_istarf_conus.csv",
            "column": "Adjusted_CAP_MG",
            "index_col": "reservoir",
            "index": f"modified_{name}"
            if name in modified_starfit_reservoir_list
            else name,
        }

    ### add max release constraints for NYC reservoirs
    if regulatory_release and not starfit_release:
        model["parameters"][f"controlled_max_release_{name}"] = {
            "type": "constant",
            "value": get_reservoir_max_release(name, "controlled"),
        }
        model["parameters"][f"flood_max_release_{name}"] = {
            "type": "constant",
            "value": get_reservoir_max_release(name, "flood"),
        }

    ### For STARFIT reservoirs, use custom parameter
    if starfit_release:
        model["parameters"][f"starfit_release_{name}"] = {
            "type": "STARFITReservoirRelease",
            "node": name,
            "run_starfit_sensitivity_analysis": run_starfit_sensitivity_analysis,
            "sensitivity_analysis_scenarios": sensitivity_analysis_scenarios,
        }

    if has_catchment:
        ### Assign inflows to nodes
        if inflow_ensemble_indices:
            ### Use custom FlowEnsemble parameter to handle scenario indexing
            model["parameters"][f"flow_{name}"] = {
                "type": "FlowEnsemble",
                "node": name,
                "inflow_type": inflow_type,
                "inflow_ensemble_indices": inflow_ensemble_indices,
            }

        else:
            inflow_source = f"{input_dir}catchment_inflow_{inflow_type}.csv"

            ### Use single-scenario historic data
            model["parameters"][f"flow_{name}"] = {
                "type": "dataframe",
                "url": inflow_source,
                "column": name,
                "index_col": "datetime",
                "parse_dates": True,
            }

        ### get max flow for catchment withdrawal nodes based on DRBC data
        model["parameters"][f"max_flow_catchmentWithdrawal_{name}"] = {
            "type": "constant",
            "url": f"{input_dir}sw_avg_wateruse_Pywr-DRB_Catchments.csv",
            "column": "Total_WD_MGD",
            "index_col": "node",
            "index": node_name,
        }

        ### get max flow for catchment consumption nodes based on DRBC data
        ### assume the consumption_t = R * withdrawal_{t-1}, where R is the ratio of avg consumption to withdrawal from DRBC data
        model["parameters"][f"catchmentConsumptionRatio_{name}"] = {
            "type": "constant",
            "url": f"{input_dir}sw_avg_wateruse_Pywr-DRB_Catchments.csv",
            "column": "Total_CU_WD_Ratio",
            "index_col": "node",
            "index": node_name,
        }
        model["parameters"][f"prev_flow_catchmentWithdrawal_{name}"] = {
            "type": "flow",
            "node": f"catchmentWithdrawal_{name}",
        }
        model["parameters"][f"max_flow_catchmentConsumption_{name}"] = {
            "type": "aggregated",
            "agg_func": "product",
            "parameters": [
                f"catchmentConsumptionRatio_{name}",
                f"prev_flow_catchmentWithdrawal_{name}",
            ],
        }

    return model


##########################################################################################
### drb_make_model()
##########################################################################################


def make_model(
    inflow_type,
    model_filename,
    start_date,
    end_date,
    use_hist_NycNjDeliveries=True,
    inflow_ensemble_indices=None,
    predict_temperature=False,
    predict_salinity=False,
    run_starfit_sensitivity_analysis=False,
    sensitivity_analysis_scenarios=list(range(0, 2001)),
):
    """
    Creates the JSON file used by Pywr to define the model, including all nodes, edges, and parameters.

    Args:
        inflow_type (str): Type of inflow.
        start_date (str): Start date of the model.
        end_date (str): End date of the model.
        use_hist_NycNjDeliveries (bool): Flag indicating whether to use historical NYC and NJ deliveries.
        inflow_ensemble_indices (list): List of inflow ensemble indices.
        predict_temperature (bool): If True, use LSTM model to predict temperature. Must have BMI repository set up. Default is False.
        predict_salinity (bool): If True, use LSTM model to predict salinity. NOT YET IMPLEMENTED.
        run_starfit_sensitivity_analysis (bool): If True, run STARFIT sensitivity analysis. Default is False.
        sensitivity_analysis_scenarios (list): List of scenario ID numbers for STARFIT sensitivity analysis.

    Returns:
        dict: Model JSON representation.
    """

    #######################################################################
    ### Basic model info
    #######################################################################

    if inflow_ensemble_indices:
        N_SCENARIOS = len(inflow_ensemble_indices)

    elif sensitivity_analysis_scenarios and not inflow_ensemble_indices:
        N_SCENARIOS = len(sensitivity_analysis_scenarios)

    else:
        N_SCENARIOS = 1

    ### create dict to hold all model nodes, edges, params, etc, following Pywr protocol. This will be saved to JSON at end.
    model = {
        "metadata": {
            "title": "DRB",
            "description": "Pywr DRB representation",
            "minimum_version": "0.4",
        },
        "timestepper": {"start": start_date, "end": end_date, "timestep": 1},
        "scenarios": [{"name": "inflow", "size": N_SCENARIOS}],
    }

    model["nodes"] = []
    model["edges"] = []
    model["parameters"] = {}

    #######################################################################
    ### add major nodes (e.g., reservoirs) to model, along with corresponding minor nodes (e.g., withdrawals), edges, & parameters
    #######################################################################

    ### get initial reservoir storages as 80% of capacity
    initial_volume_frac = 0.8

    ### get downstream node to link to for the current node
    for node, downstream_node in immediate_downstream_nodes_dict.items():
        node_type = "reservoir" if node in reservoir_list else "river"

        ### get outflow regulatory constraint type for reservoirs
        if node_type == "reservoir":
            regulatory_release = (
                True
                if node in (reservoir_list_nyc + drbc_lower_basin_reservoirs)
                else False
            )
            starfit_release = True if node not in reservoir_list_nyc else False
        else:
            regulatory_release = False
            starfit_release = False

        ### get flow lag (days) between current node and its downstream connection
        downstream_lag = downstream_node_lags[node]

        variable_cost = True if (regulatory_release and not starfit_release) else False
        if node_type == "reservoir":
            capacity = (
                get_reservoir_capacity(f"modified_{node}")
                if node in modified_starfit_reservoir_list
                else get_reservoir_capacity(node)
            )

        has_catchment = False if node == "delTrenton" else True

        ### set up major node
        model = add_major_node(
            model,
            node,
            node_type,
            inflow_type,
            starfit_release,
            regulatory_release,
            downstream_node,
            downstream_lag,
            capacity,
            initial_volume_frac,
            variable_cost,
            has_catchment,
            inflow_ensemble_indices,
            run_starfit_sensitivity_analysis=run_starfit_sensitivity_analysis,
            sensitivity_analysis_scenarios=sensitivity_analysis_scenarios,
        )
    #######################################################################
    ### Add additional nodes beyond those associated with major nodes above
    #######################################################################

    ### Node for NYC aggregated storage across 3 reservoirs
    model["nodes"].append(
        {
            "name": "reservoir_agg_nyc",
            "type": "aggregatedstorage",
            "storage_nodes": [f"reservoir_{r}" for r in reservoir_list_nyc],
        }
    )

    ### Nodes linking each NYC reservoir to NYC deliveries
    for r in reservoir_list_nyc:
        model["nodes"].append(
            {
                "name": f"link_{r}_nyc",
                "type": "link",
                "cost": -500.0,
                "max_flow": f"max_flow_delivery_nyc_{r}",
            }
        )

    ### Nodes for NYC & NJ deliveries
    for d in ["nyc", "nj"]:
        model["nodes"].append(
            {
                "name": f"delivery_{d}",
                "type": "output",
                "cost": -500.0,
                "max_flow": f"max_flow_delivery_{d}",
            }
        )

    ### Node for final model sink in Delaware Bay
    model["nodes"].append({"name": "output_del", "type": "output"})

    #######################################################################
    ### Add additional edges beyond those associated with major nodes above
    #######################################################################

    ### Edges linking each NYC reservoir to NYC deliveries
    for r in reservoir_list_nyc:
        model["edges"].append([f"reservoir_{r}", f"link_{r}_nyc"])
        model["edges"].append([f"link_{r}_nyc", "delivery_nyc"])

    ### Edge linking Delaware River at DRCanal to NJ deliveries
    model["edges"].append(["link_delDRCanal", "delivery_nj"])

    #######################################################################
    ### Add additional parameters beyond those associated with major nodes above
    #######################################################################

    ### Define "scenarios" based on flow multiplier -> only run one with 1.0 for now
    if N_SCENARIOS == 1:
        model["parameters"]["flow_factor"] = {
            "type": "constantscenario",
            "scenario": "inflow",
            "values": [1.0],
        }

    ### demand for NYC
    if use_hist_NycNjDeliveries:
        ### if this flag is True, we assume demand is equal to historical deliveries timeseries
        model["parameters"][f"demand_nyc"] = {
            "type": "dataframe",
            "url": f"{input_dir}deliveryNYC_ODRM_extrapolated.csv",
            "column": "aggregate",
            "index_col": "datetime",
            "parse_dates": True,
        }
    else:
        ### otherwise, assume demand is equal to max allotment under FFMP
        model["parameters"][f"demand_nyc"] = {
            "type": "constant",
            "url": "drb_model_constants.csv",
            "column": "value",
            "index_col": "parameter",
            "index": "max_flow_baseline_delivery_nyc",
        }

    ### repeat for NJ deliveries
    if use_hist_NycNjDeliveries:
        ### if this flag is True, we assume demand is equal to historical deliveries timeseries
        model["parameters"][f"demand_nj"] = {
            "type": "dataframe",
            "url": f"{input_dir}deliveryNJ_DRCanal_extrapolated.csv",
            "index_col": "datetime",
            "parse_dates": True,
        }
    else:
        ### otherwise, assume demand is equal to max allotment under FFMP
        model["parameters"][f"demand_nj"] = {
            "type": "constant",
            "url": "drb_model_constants.csv",
            "column": "value",
            "index_col": "parameter",
            "index": "max_flow_baseline_monthlyAvg_delivery_nj",
        }

    ### max allowable delivery to NYC (on moving avg)
    model["parameters"]["max_flow_baseline_delivery_nyc"] = {
        "type": "constant",
        "url": "drb_model_constants.csv",
        "column": "value",
        "index_col": "parameter",
        "index": "max_flow_baseline_delivery_nyc",
    }

    ### NJ has both a daily limit and monthly average limit
    model["parameters"]["max_flow_baseline_daily_delivery_nj"] = {
        "type": "constant",
        "url": "drb_model_constants.csv",
        "column": "value",
        "index_col": "parameter",
        "index": "max_flow_baseline_daily_delivery_nj",
    }
    model["parameters"]["max_flow_baseline_monthlyAvg_delivery_nj"] = {
        "type": "constant",
        "url": "drb_model_constants.csv",
        "column": "value",
        "index_col": "parameter",
        "index": "max_flow_baseline_monthlyAvg_delivery_nj",
    }

    ### levels defining operational regimes for NYC reservoirs base on combined storage: 1a, 1b, 1c, 2, 3, 4, 5.
    ### Note 1a assumed to fill remaining space, doesnt need to be defined here.
    levels = ["1a", "1b", "1c", "2", "3", "4", "5"]
    for level in levels[1:]:
        model["parameters"][f"level{level}"] = {
            "type": "dailyprofile",
            "url": "drb_model_dailyProfiles.csv",
            "index_col": "profile",
            "index": f"level{level}",
        }

    ### Control curve index that tells us which level the aggregated NYC storage is currently in
    model["parameters"]["drought_level_agg_nyc"] = {
        "type": "controlcurveindex",
        "storage_node": "reservoir_agg_nyc",
        "control_curves": [f"level{level}" for level in levels[1:]],
    }

    ### factors defining delivery profiles for NYC and NJ, for each storage level: 1a, 1b, 1c, 2, 3, 4, 5.
    demands = ["nyc", "nj"]
    for demand in demands:
        for level in levels:
            model["parameters"][f"level{level}_factor_delivery_{demand}"] = {
                "type": "constant",
                "url": "drb_model_constants.csv",
                "column": "value",
                "index_col": "parameter",
                "index": f"level{level}_factor_delivery_{demand}",
            }

    ### Indexed arrays that dictate cutbacks to NYC & NJ deliveries, based on current storage level and DOY
    for demand in demands:
        model["parameters"][f"drought_factor_delivery_{demand}"] = {
            "type": "indexedarray",
            "index_parameter": "drought_level_agg_nyc",
            "params": [f"level{level}_factor_delivery_{demand}" for level in levels],
        }

    ### Max allowable daily delivery to NYC & NJ- this will be very large for levels 1&2 (so that moving average limit
    ### in the next parameter is active), but apply daily flow limits for more drought stages.
    model["parameters"]["max_flow_drought_delivery_nyc"] = {
        "type": "aggregated",
        "agg_func": "product",
        "parameters": ["max_flow_baseline_delivery_nyc", "drought_factor_delivery_nyc"],
    }

    ### Max allowable delivery to NYC in current time step to maintain moving avg limit. Based on custom Pywr parameter.
    model["parameters"]["max_flow_ffmp_delivery_nyc"] = {
        "type": "FfmpNycRunningAvg",
        "node": "delivery_nyc",
        "max_avg_delivery": "max_flow_baseline_delivery_nyc",
    }

    ### Max allowable delivery to NJ in current time step to maintain moving avg limit. Based on custom Pywr parameter.
    model["parameters"]["max_flow_ffmp_delivery_nj"] = {
        "type": "FfmpNjRunningAvg",
        "node": "delivery_nj",
        "max_avg_delivery": "max_flow_baseline_monthlyAvg_delivery_nj",
        "max_daily_delivery": "max_flow_baseline_daily_delivery_nj",
        "drought_factor": "drought_factor_delivery_nj",
    }

    ### Now actual max flow to NYC delivery node is the min of demand, daily FFMP limit, and daily limit to meet FFMP moving avg limit
    model["parameters"]["max_flow_delivery_nyc"] = {
        "type": "aggregated",
        "agg_func": "min",
        "parameters": [
            "demand_nyc",
            "max_flow_drought_delivery_nyc",
            "max_flow_ffmp_delivery_nyc",
        ],
    }

    ### Now actual max flow to NJ delivery node is the min of demand and daily limit to meet FFMP moving avg limit
    model["parameters"]["max_flow_delivery_nj"] = {
        "type": "aggregated",
        "agg_func": "min",
        "parameters": ["demand_nj", "max_flow_ffmp_delivery_nj"],
    }

    ### baseline release flow rate for each NYC reservoir, dictated by FFMP
    for reservoir in reservoir_list_nyc:
        model["parameters"][f"mrf_baseline_{reservoir}"] = {
            "type": "constant",
            "url": "drb_model_constants.csv",
            "column": "value",
            "index_col": "parameter",
            "index": f"mrf_baseline_{reservoir}",
        }

    ### Control curve index that tells us which level each individual NYC reservoir's storage is currently in
    for reservoir in reservoir_list_nyc:
        model["parameters"][f"drought_level_{reservoir}"] = {
            "type": "controlcurveindex",
            "storage_node": f"reservoir_{reservoir}",
            "control_curves": [f"level{level}" for level in levels[1:]],
        }

    ### Factor governing changing release reqs from NYC reservoirs, based on aggregated storage across 3 reservoirs
    for reservoir in reservoir_list_nyc:
        model["parameters"][f"mrf_drought_factor_agg_{reservoir}"] = {
            "type": "indexedarray",
            "index_parameter": "drought_level_agg_nyc",
            "params": [f"level{level}_factor_mrf_{reservoir}" for level in levels],
        }

    ### Levels defining operational regimes for individual NYC reservoirs, as opposed to aggregated level across 3 reservoirs
    for reservoir in reservoir_list_nyc:
        for level in levels:
            model["parameters"][f"level{level}_factor_mrf_{reservoir}"] = {
                "type": "dailyprofile",
                "url": "drb_model_dailyProfiles.csv",
                "index_col": "profile",
                "index": f"level{level}_factor_mrf_{reservoir}",
            }

    ### Factor governing changing release reqs from NYC reservoirs, based on individual storage for particular reservoir
    for reservoir in reservoir_list_nyc:
        model["parameters"][f"mrf_drought_factor_individual_{reservoir}"] = {
            "type": "indexedarray",
            "index_parameter": f"drought_level_{reservoir}",
            "params": [f"level{level}_factor_mrf_{reservoir}" for level in levels],
        }

    ### Factor governing changing release reqs from NYC reservoirs, depending on whether aggregated or individual storage level is activated
    ### Based on custom Pywr parameter.
    for reservoir in reservoir_list_nyc:
        model["parameters"][f"mrf_drought_factor_combined_final_{reservoir}"] = {
            "type": "NYCCombinedReleaseFactor",
            "node": f"reservoir_{reservoir}",
        }

    ### FFMP mandated releases from NYC reservoirs
    for reservoir in reservoir_list_nyc:
        model["parameters"][f"mrf_target_individual_{reservoir}"] = {
            "type": "aggregated",
            "agg_func": "product",
            "parameters": [
                f"mrf_baseline_{reservoir}",
                f"mrf_drought_factor_combined_final_{reservoir}",
            ],
        }

    ### sum of FFMP mandated releases from NYC reservoirs
    model["parameters"]["mrf_target_individual_agg_nyc"] = {
        "type": "aggregated",
        "agg_func": "sum",
        "parameters": [
            f"mrf_target_individual_{reservoir}" for reservoir in reservoir_list_nyc
        ],
    }

    ### extra flood releases for NYC reservoirs specified in FFMP
    for reservoir in reservoir_list_nyc:
        model["parameters"][f"weekly_rolling_mean_flow_{reservoir}"] = {
            "type": "RollingMeanFlowNode",
            "node": f"reservoir_{reservoir}",
            "timesteps": 7,
            "name": f"flow_{reservoir}",
            "initial_flow": 0,
        }
        model["parameters"][f"flood_release_{reservoir}"] = {
            "type": "NYCFloodRelease",
            "node": f"reservoir_{reservoir}",
        }

    ### sum of flood control releases from NYC reservoirs
    model["parameters"]["flood_release_agg_nyc"] = {
        "type": "aggregated",
        "agg_func": "sum",
        "parameters": [
            f"flood_release_{reservoir}" for reservoir in reservoir_list_nyc
        ],
    }

    ### variable storage cost for each reservoir, based on its fractional storage
    ### Note: may not need this anymore now that we have volume balancing rules. but maybe makes sense to leave in for extra protection.
    volumes = {
        "cannonsville": get_reservoir_capacity("cannonsville"),
        "pepacton": get_reservoir_capacity("pepacton"),
        "neversink": get_reservoir_capacity("neversink"),
    }
    for reservoir in reservoir_list_nyc:
        model["parameters"][f"storage_cost_{reservoir}"] = {
            "type": "interpolatedvolume",
            "values": [-100, -1],
            "node": f"reservoir_{reservoir}",
            "volumes": [-EPS, volumes[reservoir] + EPS],
        }

    ### current volume stored in each reservoir, plus the aggregated storage node
    for reservoir in reservoir_list_nyc + ["agg_nyc"]:
        model["parameters"][f"volume_{reservoir}"] = {
            "type": "interpolatedvolume",
            "values": [-EPS, 1000000],
            "node": f"reservoir_{reservoir}",
            "volumes": [-EPS, 1000000],
        }

    ### aggregated inflows to NYC reservoirs
    model["parameters"]["flow_agg_nyc"] = {
        "type": "aggregated",
        "agg_func": "sum",
        "parameters": [f"flow_{reservoir}" for reservoir in reservoir_list_nyc],
    }

    ### aggregated max volume across NYC reservoirs
    model["parameters"]["max_volume_agg_nyc"] = {
        "type": "aggregated",
        "agg_func": "sum",
        "parameters": [f"max_volume_{reservoir}" for reservoir in reservoir_list_nyc],
    }

    ### Baseline flow target at Montague & Trenton
    mrfs = ["delMontague", "delTrenton"]
    for mrf in mrfs:
        model["parameters"][f"mrf_baseline_{mrf}"] = {
            "type": "constant",
            "url": "drb_model_constants.csv",
            "column": "value",
            "index_col": "parameter",
            "index": f"mrf_baseline_{mrf}",
        }

    ### Seasonal multiplier factors for Montague & Trenton flow targets based on drought level of NYC aggregated storage
    for mrf in mrfs:
        for level in levels:
            model["parameters"][f"level{level}_factor_mrf_{mrf}"] = {
                "type": "monthlyprofile",
                "url": "drb_model_monthlyProfiles.csv",
                "index_col": "profile",
                "index": f"level{level}_factor_mrf_{mrf}",
            }

    ### Current value of seasonal multiplier factor for Montague & Trenton flow targets based on drought level of NYC aggregated storage
    for mrf in mrfs:
        model["parameters"][f"mrf_drought_factor_{mrf}"] = {
            "type": "indexedarray",
            "index_parameter": "drought_level_agg_nyc",
            "params": [f"level{level}_factor_mrf_{mrf}" for level in levels],
        }

    ### Total Montague & Trenton flow targets based on drought level of NYC aggregated storage
    for mrf in mrfs:
        model["parameters"][f"mrf_target_{mrf}"] = {
            "type": "aggregated",
            "agg_func": "product",
            "parameters": [f"mrf_baseline_{mrf}", f"mrf_drought_factor_{mrf}"],
        }

    # ### total predicted lagged non-NYC inflows to Montague & Trenton, and predicted lagged NJ demands
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
            model["parameters"][f"predicted_nonnyc_gage_flow_{mrf}_lag{lag}"] = {
                "type": "dataframe",
                "url": f"{input_dir}predicted_inflows_diversions_{inflow_type}.csv",
                "column": label,
                "index_col": "datetime",
                "parse_dates": True,
            }
        ### now get predicted nj demand
        for lag in range(1, 5):
            label = f"demand_nj_lag{lag}_{flow_prediction_mode}"
            model["parameters"][f"predicted_demand_nj_lag{lag}"] = {
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

            model["parameters"][f"predicted_nonnyc_gage_flow_{mrf}_lag{lag}"] = {
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
            model["parameters"][f"predicted_demand_nj_lag{lag}"] = {
                "type": "dataframe",
                "url": f"{input_dir}predicted_inflows_diversions_{predicted_inflow_type}.csv",
                "column": label,
                "index_col": "datetime",
                "parse_dates": True,
            }
        print(
            "WARNING: Ensemble mode not tested/verified for Montague & Trenton flow forecasts"
        )

    ### Get total release needed from NYC reservoirs to satisfy Montague & Trenton flow targets,
    ### above and beyond their individually mandated releases, & after accting for non-NYC inflows and NJ diversions.
    ### THis first step is based on predicted inflows to Montague in 2 days and Trenton in 4 days, and is used
    ### to calculate balanced releases from all 3 reservoirs. But only Cannonsville & Pepacton actually use these
    ### releases, because Neversink is adjusted later because it is 1 day closer travel time & has more info.
    ### Uses custom Pywr parameter.

    step = 1
    ### first get release needed to meet Montague target in 2 days
    model["parameters"][f"release_needed_mrf_montague_step{step}"] = {
        "type": "TotalReleaseNeededForDownstreamMRF",
        "mrf": "delMontague",
        "step": step,
    }
    ### now get addl release (above Montague release) needed to meet Trenton target in 4 days
    model["parameters"][f"release_needed_mrf_trenton_step{step}"] = {
        "type": "TotalReleaseNeededForDownstreamMRF",
        "mrf": "delTrenton",
        "step": step,
    }
    ### total mrf release needed is sum of Montague & Trenton. This Step 1 is for Cannonsville/Pepacton releases.
    model["parameters"][f"total_agg_mrf_montagueTrenton_step{step}"] = {
        "type": "aggregated",
        "agg_func": "sum",
        "parameters": [
            f"release_needed_mrf_montague_step{step}",
            f"release_needed_mrf_trenton_step{step}",
        ],
    }

    ## Max available Trenton contribution from each available lower basin reservoir. Step 1 is for Cannonsville/Pepacton release, so 4 days ahead for Trenton.
    for reservoir in drbc_lower_basin_reservoirs:
        model["parameters"][f"max_mrf_trenton_step{step}_{reservoir}"] = {
            "type": "LowerBasinMaxMRFContribution",
            "node": f"reservoir_{reservoir}",
            "step": step,
        }

    ## Aggregate total expected lower basin contribution to Trenton
    model["parameters"][f"lower_basin_agg_mrf_trenton_step{step}"] = {
        "type": "VolBalanceLowerBasinMRFAggregate",
        "step": step,
    }

    ### now calculate actual Cannonsville & Pepacton releases to meet Montague&Trenton, with assumed releases for Neversink & lower basin
    for reservoir in ["cannonsville", "pepacton"]:
        model["parameters"][f"mrf_montagueTrenton_{reservoir}"] = {
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
        model["parameters"][f"outflow_{reservoir}_rollmean{lag}"] = {
            "type": "rollingmeanflownode",
            "node": f"outflow_{reservoir}",
            "days": lag,
        }
        model["parameters"][f"spill_{reservoir}_rollmean{lag}"] = {
            "type": "rollingmeanflownode",
            "node": f"spill_{reservoir}",
            "days": lag,
        }
        model["parameters"][f"release_{reservoir}_lag{lag}"] = {
            "type": "LaggedReservoirRelease",
            "node": reservoir,
            "lag": lag,
        }

    ### get release needed to meet Montague target in 1 days
    model["parameters"][f"release_needed_mrf_montague_step{step}"] = {
        "type": "TotalReleaseNeededForDownstreamMRF",
        "mrf": "delMontague",
        "step": step,
    }
    ### now get addl release (above Montague release) needed to meet Trenton target in 3 days
    model["parameters"][f"release_needed_mrf_trenton_step{step}"] = {
        "type": "TotalReleaseNeededForDownstreamMRF",
        "mrf": "delTrenton",
        "step": step,
    }
    ### total mrf release needed is sum of Montague & Trenton. This Step 2 is for Neversink releases.
    model["parameters"][f"total_agg_mrf_montagueTrenton_step{step}"] = {
        "type": "aggregated",
        "agg_func": "sum",
        "parameters": [
            f"release_needed_mrf_montague_step{step}",
            f"release_needed_mrf_trenton_step{step}",
        ],
    }

    ## Max available Trenton contribution from each available lower basin reservoir. Step 2 is for Neversink release, so 3 days ahead for Trenton.
    for reservoir in drbc_lower_basin_reservoirs:
        model["parameters"][f"max_mrf_trenton_step{step}_{reservoir}"] = {
            "type": "LowerBasinMaxMRFContribution",
            "node": f"reservoir_{reservoir}",
            "step": step,
        }

    ## Aggregate total expected lower basin contribution to Trenton
    model["parameters"][f"lower_basin_agg_mrf_trenton_step{step}"] = {
        "type": "VolBalanceLowerBasinMRFAggregate",
        "step": step,
    }

    ### Now assign Neversink releases to meet Montague/Trenton mrf, after accting for previous Can/Pep releases & expected lower basin contribution
    model["parameters"][f"mrf_montagueTrenton_neversink"] = {
        "type": f"VolBalanceNYCDownstreamMRF_step{step}",
    }

    ### step 3:
    ### Update lower basin releases that are 2 days from Trenton equivalent (Beltzville & Blue Marsh)
    step = 3

    ### get previous day's releases from Neversink
    for reservoir in ["neversink"]:
        lag = 1
        model["parameters"][f"outflow_{reservoir}_rollmean{lag}"] = {
            "type": "rollingmeanflownode",
            "node": f"outflow_{reservoir}",
            "days": lag,
        }
        model["parameters"][f"spill_{reservoir}_rollmean{lag}"] = {
            "type": "rollingmeanflownode",
            "node": f"spill_{reservoir}",
            "days": lag,
        }
        model["parameters"][f"release_{reservoir}_lag{lag}"] = {
            "type": "LaggedReservoirRelease",
            "node": reservoir,
            "lag": lag,
        }
    ### and get 2-days ago release from Can/Pep
    for reservoir in ["cannonsville", "pepacton"]:
        lag = 2
        model["parameters"][f"outflow_{reservoir}_rollmean{lag}"] = {
            "type": "rollingmeanflownode",
            "node": f"outflow_{reservoir}",
            "days": lag,
        }
        model["parameters"][f"spill_{reservoir}_rollmean{lag}"] = {
            "type": "rollingmeanflownode",
            "node": f"spill_{reservoir}",
            "days": lag,
        }
        model["parameters"][f"release_{reservoir}_lag{lag}"] = {
            "type": "LaggedReservoirRelease",
            "node": reservoir,
            "lag": lag,
        }

    ### now get addl release (above NYC releases from last 2 days) needed to meet Trenton target in 2 days
    model["parameters"][f"release_needed_mrf_trenton_step{step}"] = {
        "type": "TotalReleaseNeededForDownstreamMRF",
        "mrf": "delTrenton",
        "step": step,
    }

    ## Max available Trenton contribution from each available lower basin reservoir. Step 3 is for Beltzville/BlueMarsh release, so 2 days ahead for Trenton.
    for reservoir in drbc_lower_basin_reservoirs:
        model["parameters"][f"max_mrf_trenton_step{step}_{reservoir}"] = {
            "type": "LowerBasinMaxMRFContribution",
            "node": f"reservoir_{reservoir}",
            "step": step,
        }

    ## Aggregate total expected lower basin contribution to Trenton
    model["parameters"][f"lower_basin_agg_mrf_trenton_step{step}"] = {
        "type": "VolBalanceLowerBasinMRFAggregate",
        "step": step,
    }

    ## Now dispatch actual individual releases from lower basin reservoirs for meeting Trenton - step 3 is Blue marsh/beltzville
    for reservoir in ["beltzvilleCombined", "blueMarsh"]:
        model["parameters"][f"mrf_trenton_{reservoir}"] = {
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
        model["parameters"][f"outflow_{reservoir}_rollmean{lag}"] = {
            "type": "rollingmeanflownode",
            "node": f"outflow_{reservoir}",
            "days": lag,
        }
        model["parameters"][f"spill_{reservoir}_rollmean{lag}"] = {
            "type": "rollingmeanflownode",
            "node": f"spill_{reservoir}",
            "days": lag,
        }
        model["parameters"][f"release_{reservoir}_lag{lag}"] = {
            "type": "LaggedReservoirRelease",
            "node": reservoir,
            "lag": lag,
        }

    ### and get 2-days ago releases from Neversink
    for reservoir in ["neversink"]:
        lag = 2
        model["parameters"][f"outflow_{reservoir}_rollmean{lag}"] = {
            "type": "rollingmeanflownode",
            "node": f"outflow_{reservoir}",
            "days": lag,
        }
        model["parameters"][f"spill_{reservoir}_rollmean{lag}"] = {
            "type": "rollingmeanflownode",
            "node": f"spill_{reservoir}",
            "days": lag,
        }
        model["parameters"][f"release_{reservoir}_lag{lag}"] = {
            "type": "LaggedReservoirRelease",
            "node": reservoir,
            "lag": lag,
        }
    ### and get 3-days ago release from Can/Pep
    for reservoir in ["cannonsville", "pepacton"]:
        lag = 3
        model["parameters"][f"outflow_{reservoir}_rollmean{lag}"] = {
            "type": "rollingmeanflownode",
            "node": f"outflow_{reservoir}",
            "days": lag,
        }
        model["parameters"][f"spill_{reservoir}_rollmean{lag}"] = {
            "type": "rollingmeanflownode",
            "node": f"spill_{reservoir}",
            "days": lag,
        }
        model["parameters"][f"release_{reservoir}_lag{lag}"] = {
            "type": "LaggedReservoirRelease",
            "node": reservoir,
            "lag": lag,
        }

    ### now get addl release (above NYC releases from steps 1-2 & lower basin releases from step 3) needed to meet Trenton target in 1 days
    model["parameters"][f"release_needed_mrf_trenton_step{step}"] = {
        "type": "TotalReleaseNeededForDownstreamMRF",
        "mrf": "delTrenton",
        "step": step,
    }

    ## Max available Trenton contribution from each available lower basin reservoir. Step 4 is just for Nockamixon, 1 day above Trenton.
    for reservoir in drbc_lower_basin_reservoirs:
        model["parameters"][f"max_mrf_trenton_step{step}_{reservoir}"] = {
            "type": "LowerBasinMaxMRFContribution",
            "node": f"reservoir_{reservoir}",
            "step": step,
        }

    ## Aggregate total expected lower basin contribution to Trenton
    model["parameters"][f"lower_basin_agg_mrf_trenton_step{step}"] = {
        "type": "VolBalanceLowerBasinMRFAggregate",
        "step": step,
    }

    ## Now dispatch actual individual releases from lower basin reservoirs for meeting Trenton - step 4 is just Nockamixon
    for reservoir in ["nockamixon"]:
        model["parameters"][f"mrf_trenton_{reservoir}"] = {
            "type": "VolBalanceLowerBasinMRFIndividual",
            "node": f"reservoir_{reservoir}",
            "step": step,
        }

    ### get final downstream release from each NYC reservoir, which is the sum of its
    ###    individually-mandated release from FFMP, flood control release, and its contribution to the
    ### Montague/Trenton targets
    for reservoir in reservoir_list_nyc:
        model["parameters"][f"downstream_release_target_{reservoir}"] = {
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
        model["parameters"][f"downstream_release_target_{reservoir}"] = {
            "type": "aggregated",
            "agg_func": "sum",
            "parameters": [f"mrf_trenton_{reservoir}", f"starfit_release_{reservoir}"],
        }

    ### now distribute NYC deliveries across 3 reservoirs with volume balancing after accounting for downstream releases
    for reservoir in reservoir_list_nyc:
        ### max diversion to NYC from each reservoir based on historical data
        model["parameters"][f"hist_max_flow_delivery_nyc_{reservoir}"] = {
            "type": "constant",
            "value": get_reservoir_max_diversion_NYC(reservoir),
        }
        ### Target diversion from each NYC reservoir to satisfy NYC demand,
        ### accounting for historical max diversion constraints
        ### and attempting to balance storages across 3 NYC reservoirs
        ### Uses custom Pywr parameter.
        model["parameters"][f"max_flow_delivery_nyc_{reservoir}"] = {
            "type": "VolBalanceNYCDemand",
            "node": f"reservoir_{reservoir}",
        }

    ### Temperature prediction at Lordville using LSTM model
    if predict_temperature:
        try:
            from pywrdrb.parameters.temperature import TemperaturePrediction

            model["parameters"]["predicted_mu_max_of_temperature"] = {
                "type": "TemperaturePrediction"
            }
        except Exception as e:
            print(f"Temperature prediction model not available. Error: {e}")

    ### Salinity prediction at Trenton
    if predict_salinity:
        try:
            from pywrdrb.parameters.salinity import SalinityPrediction

            model["parameters"]["predicted_salinity"] = {"type": "SalinityPrediction"}
        except Exception as e:
            print(f"Salinity prediction model not available. Error: {e}")

    #######################################################################
    ### save full model as json
    #######################################################################

    with open(f"{model_filename}", "w") as o:
        json.dump(model, o, indent=4)
