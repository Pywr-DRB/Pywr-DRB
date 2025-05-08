"""
Flexible Flow Management Program (FFMP) Pywr Parameters

Overview:
This module defines custom Pywr parameter classes to implement the Flexible Flow Management Program (FFMP)
for the New York City (NYC) reservoir system, including Cannonsville, Pepacton, and Neversink.

The FFMP governs how reservoirs release water for downstream targets (Montague and Trenton), individual
mandated releases, drought management, and flood operations. Each class in this module implements a specific
component of FFMP logic and constraints, including NYC and NJ running average releases, release factors,
flood control, demand balancing, and multi-step routing to downstream targets.

Key Functionalities:
- Enforce FFMP rules via running average delivery limits
- Compute release factors using drought levels and storage indicators
- Determine flood releases to return to NOR within 7 days
- Calculate NYC contributions to Montague/Trenton targets across time-lagged steps
- Balance deliveries and releases across reservoirs based on current storage and release constraints

Technical Notes:
- These custom parameters must be registered using `.register()` to be used within a Pywr model
- Many parameters use scenario-specific state arrays to track values dynamically
- Dates are tracked with `pandas.Timedelta` for temporal logic
- Depends on `pywrdrb.utils.lists.reservoir_list_nyc`, and `pywrdrb.utils.constants.epsilon`

Links:
- FFMP 2018 Document (Appendix A): https://webapps.usgs.gov/odrm/ffmp/Appendix_A_FFMP%2020180716%20Final.pdf

Change Log:
Marilyn Smith, 2025-05-07, Added complete metadata and NumPy-style docstrings
"""

import numpy as np
import pandas as pd
from pywr.parameters import Parameter, load_parameter

from pywrdrb.utils.lists import reservoir_list_nyc
from pywrdrb.utils.constants import epsilon

class FfmpNycRunningAvgParameter(Parameter):
    """
    Enforces the NYC FFMP delivery constraint using a running average over time.

    This custom Pywr parameter limits the amount of water that can be delivered from an NYC reservoir
    based on a long-term daily average (`max_avg_delivery`). It ensures that daily releases stay within
    the specified delivery budget and adjusts dynamically based on past deliveries.

    Attributes
    ----------
    node : pywr.Node
        The node from which water is being delivered (usually an NYC reservoir).
    max_avg_delivery : float
        Maximum allowable average delivery volume per day.
    max_delivery : numpy.ndarray
        Array holding the daily remaining delivery allowance per scenario.
    timestep : int
        The number of days per timestep (usually 1 for daily models).
    datetime : pandas.Timestamp
        The current model date used for enforcing reset logic.

    Methods
    -------
    setup()
        Allocates an array to store per-scenario delivery limits.
    reset()
        Initializes delivery limits and datetime to model start.
    value(timestep, scenario_index)
        Returns the remaining delivery volume for the given scenario.
    after()
        Updates the running delivery limit based on the prior day’s release.
    load(model, data)
        Class method to load the parameter from JSON/YAML model data.
    """

    def __init__(self, model, node, max_avg_delivery, **kwargs):
        """
        Initialize the FfmpNycRunningAvgParameter.

        Parameters
        ----------
        model : pywr.Model
            The Pywr model instance.
        node : pywr.Node
            The node associated with this delivery parameter.
        max_avg_delivery : pywr.parameters.ConstantParameter
            The maximum daily average delivery limit for the FFMP constraint.
        **kwargs
            Additional keyword arguments passed to the base `Parameter` class.

        Notes
        -----
        This parameter is typically used on NYC reservoir delivery nodes to ensure
        long-term delivery limits are not exceeded in the FFMP simulation framework.
        """
        super().__init__(model, **kwargs)
        self.node = node
        self.max_avg_delivery = max_avg_delivery.get_constant_value()
        self.children.add(max_avg_delivery)

    def setup(self):
        """
        Allocate internal arrays used to track max delivery per scenario.
        """
        super().setup()
        num_scenarios = len(self.model.scenarios.combinations)
        self.max_delivery = np.empty([num_scenarios], np.float64)

    def reset(self):
        """
        Reset the delivery budget and internal clock for all scenarios.

        This sets the remaining daily delivery to the maximum allowed average,
        scaled by the timestep length (usually 1 day).
        """
        self.timestep = self.model.timestepper.delta
        self.datetime = pd.to_datetime(self.model.timestepper.start)
        self.max_delivery[...] = self.max_avg_delivery * self.timestep

    def value(self, timestep, scenario_index):
        """
        Get the remaining delivery volume for the current timestep and scenario.

        Parameters
        ----------
        timestep : pywr.core.Timestep
            The current timestep.
        scenario_index : pywr.core.ScenarioIndex
            The scenario index indicating which ensemble member is running.

        Returns
        -------
        float
            The remaining delivery volume allowed for this timestep and scenario.
        """
        return self.max_delivery[scenario_index.global_id]

    def after(self):
        """
        Update the delivery budget based on the previous timestep’s release.

        - On May 31st, the delivery budget is reset to the maximum average delivery.
        - On other days, the budget is adjusted by subtracting the actual delivery
          and adding back the daily average delivery allowance.

        Notes
        -----
        - Delivery budgets cannot become negative.
        - Internal date is incremented by 1 day per timestep.
        """
        ### if it is may 31, reset max delivery to original value (800)
        if self.datetime.month == 5 and self.datetime.day == 31:
            self.max_delivery[...] = self.max_avg_delivery * self.timestep
        ### else update the requirement based on running average: maxdel_t = maxdel_{t-1} - flow_{t-1} + max_avg_del
        else:
            self.max_delivery += (
                self.max_avg_delivery - self.node.flow
            ) * self.timestep
            self.max_delivery[
                self.max_delivery < 0
            ] = 0  # max delivery cannot be less than zero
        ### update date for tomorrow
        self.datetime += pd.Timedelta(1, "d")

    @classmethod
    def load(cls, model, data):
        """
        Load the parameter from model and configuration data.

        Parameters
        ----------
        model : pywr.Model
            The Pywr model instance.
        data : dict
            A dictionary of parameter configuration from the model input file.
            Must include:
            - "node": Name of the node using this parameter.
            - "max_avg_delivery": Parameter ID of the constant average delivery limit.

        Returns
        -------
        FfmpNycRunningAvgParameter
            The fully initialized parameter object.
        """
        node = model.nodes[data.pop("node")]
        max_avg_delivery = load_parameter(model, data.pop("max_avg_delivery"))
        return cls(model, node, max_avg_delivery, **data)


### have to register the custom parameter so Pywr recognizes it
FfmpNycRunningAvgParameter.register()


class FfmpNjRunningAvgParameter(Parameter):
    """
    Enforces NJ FFMP delivery limits using a drought-adjusted running average.

    This parameter tracks delivery volumes from a NJ node (typically a reservoir or intake)
    and dynamically adjusts allowable delivery based on drought factor conditions. It ensures
    compliance with both long-term average and daily maximum delivery constraints, resetting
    when drought levels change or on the first of each month under normal conditions.

    Attributes
    ----------
    node : pywr.Node
        Node from which water is being delivered.
    max_avg_delivery : float
        Long-term average daily delivery volume (base level, before drought adjustment).
    max_daily_delivery : float
        Maximum allowable delivery volume for a single day.
    drought_factor : pywr.Parameter
        Multiplier reflecting drought status. Varies over time and by scenario.
    max_delivery : numpy.ndarray
        Current remaining delivery volume allowed for each scenario.
    current_drought_factor : numpy.ndarray
        Latest drought factor values, one per scenario.
    previous_drought_factor : numpy.ndarray
        Drought factor values from the previous timestep, one per scenario.
    timestep : int
        Length of each timestep (e.g., 1 day).
    datetime : pandas.Timestamp
        Internal tracking of current model date.

    Methods
    -------
    setup()
        Allocate arrays to store per-scenario delivery limits and drought factors.
    reset()
        Initialize delivery limits and drought factors at model start.
    value(timestep, scenario_index)
        Return the remaining delivery volume for a given scenario.
    after()
        Update delivery limits after each timestep, accounting for flow and drought factor.
    load(model, data)
        Load and instantiate this parameter using model configuration data.
    """

    def __init__(
        self,
        model,
        node,
        max_avg_delivery,
        max_daily_delivery,
        drought_factor,
        **kwargs,
    ):
        """
        Initialize FfmpNjRunningAvgParameter.

        Parameters
        ----------
        model : pywr.Model
            The Pywr model instance.
        node : pywr.Node
            Node from which water is being delivered.
        max_avg_delivery : pywr.parameters.ConstantParameter
            Long-term average daily delivery (unadjusted by drought).
        max_daily_delivery : pywr.parameters.ConstantParameter
            Maximum allowable daily delivery.
        drought_factor : pywr.Parameter
            Drought adjustment multiplier that varies over time.
        **kwargs
            Additional keyword arguments passed to the base Parameter class.

        Notes
        -----
        The delivery logic accounts for drought level transitions and applies caps on
        both minimum (0) and maximum (daily limit) allowed deliveries.
        """
        super().__init__(model, **kwargs)
        self.node = node
        self.max_avg_delivery = max_avg_delivery.get_constant_value()
        self.max_daily_delivery = max_daily_delivery.get_constant_value()
        self.drought_factor = drought_factor
        self.children.add(max_avg_delivery)
        self.children.add(max_daily_delivery)
        self.children.add(drought_factor)

    def setup(self):
        """
        Allocate arrays to store delivery volume and drought factors per scenario.
        """
        super().setup()
        num_scenarios = len(self.model.scenarios.combinations)
        self.max_delivery = np.empty([num_scenarios], np.float64)
        self.current_drought_factor = np.empty([num_scenarios], np.float64)
        self.previous_drought_factor = np.empty([num_scenarios], np.float64)

    def reset(self):
        """
        Reset delivery limits and drought factors at the start of the simulation.

        This method sets up initial delivery budgets based on the average
        delivery value and initializes drought factors to 1.0 for all scenarios.
        """
        self.timestep = self.model.timestepper.delta
        self.datetime = pd.to_datetime(self.model.timestepper.start)
        self.max_delivery[...] = self.max_avg_delivery * self.timestep
        self.current_drought_factor[...] = 1.0
        self.previous_drought_factor[...] = 1.0

    def value(self, timestep, scenario_index):
        """
        Return the delivery budget remaining for the specified scenario.

        Parameters
        ----------
        timestep : pywr.core.Timestep
            The current timestep in the simulation.
        scenario_index : pywr.core.ScenarioIndex
            Index of the scenario for which to retrieve the value.

        Returns
        -------
        float
            Remaining allowable delivery volume for the given scenario.
        """
        return self.max_delivery[scenario_index.global_id]

    def after(self):
        """
        Update delivery budgets based on flow and drought factor changes.

        - If the drought factor has not changed, the budget is adjusted using a running average.
        - If the drought factor changes, the delivery budget is reset.
        - On the first day of the month, if under normal conditions (factor = 1.0), the budget is reset.
        - Values are capped at zero (minimum) and the max daily limit (maximum).

        Notes
        -----
        This logic implements FFMP requirements for delivery constraints across drought phases.
        """
        self.current_drought_factor[...] = self.drought_factor.get_all_values()
        ### loop over scenarios
        for s, factor in enumerate(self.current_drought_factor):
            ### first check if today's drought_factor is same as yesterday's
            if factor == self.previous_drought_factor[s]:
                ### if today is same drought factor as yesterday, and factor is 1.0, we reset running avg on first day of each month
                if (self.datetime.day == 1) and (factor == 1.0):
                    self.max_delivery[s] = (
                        self.max_avg_delivery * factor * self.timestep
                    )
                ### else, update requirement based on running average: maxdel_t = maxdel_{t-1} - flow_{t-1} + max_avg_del
                else:
                    self.max_delivery[s] += (
                        self.max_avg_delivery * factor - self.node.flow[s]
                    ) * self.timestep
            ### if today's drought factor is different from yesterday, we always reset running avg
            else:
                self.max_delivery[s] = self.max_avg_delivery * factor * self.timestep

        ### max delivery cannot be less than zero
        self.max_delivery[self.max_delivery < 0] = 0
        ### max delivery cannot be larger than daily limit
        self.max_delivery[
            self.max_delivery > self.max_daily_delivery
        ] = self.max_daily_delivery
        ### update date & previous factor for tomorrow
        self.datetime += pd.Timedelta(1, "d")
        self.previous_drought_factor[...] = self.current_drought_factor[...]

    @classmethod
    def load(cls, model, data):
        """
        Load the FfmpNjRunningAvgParameter from a model config dictionary.

        Parameters
        ----------
        model : pywr.Model
            The Pywr model instance.
        data : dict
            Dictionary from JSON/YAML input containing keys:
            - "node": Node name.
            - "max_avg_delivery": ID of average delivery parameter.
            - "max_daily_delivery": ID of daily cap parameter.
            - "drought_factor": ID of drought factor parameter.

        Returns
        -------
        FfmpNjRunningAvgParameter
            Initialized parameter instance ready for use in the model.
        """
        node = model.nodes[data.pop("node")]
        max_avg_delivery = load_parameter(model, data.pop("max_avg_delivery"))
        max_daily_delivery = load_parameter(model, data.pop("max_daily_delivery"))
        drought_factor = load_parameter(model, data.pop("drought_factor"))
        return cls(
            model, node, max_avg_delivery, max_daily_delivery, drought_factor, **data
        )


### have to register the custom parameter so Pywr recognizes it
FfmpNjRunningAvgParameter.register()



class NYCCombinedReleaseFactor(Parameter):
    """
    Calculates the release factor for an NYC reservoir based on drought level and storage rule logic.

    This parameter determines whether an NYC reservoir’s release should be based on its
    individual storage (flood operations) or on the aggregate storage across all three
    NYC reservoirs (normal or drought operations). It returns a weighted multiplier to 
    adjust baseline reservoir releases depending on the prevailing drought level.

    Attributes
    ----------
    node : pywr.Node
        The Pywr node associated with this parameter.
    drought_level_agg_nyc : pywr.Parameter
        Parameter representing the aggregated NYC drought level index.
    mrf_drought_factor_agg_reservoir : pywr.Parameter
        Drought factor parameter to use when operating based on aggregated NYC storage.
    mrf_drought_factor_individual_reservoir : pywr.Parameter
        Drought factor parameter to use when operating based on individual reservoir storage.

    Methods
    -------
    value(timestep, scenario_index)
        Calculate and return the reservoir’s release factor based on current drought logic.
    load(model, data)
        Load the parameter from a model configuration dictionary.

    Notes
    -----
    The logic used to compute the release factor follows the formulation 
    described in the GitHub issue dated 8/30/2022:
    https://github.com/users/ahamilton144/projects/1/views/1?pane=issue&itemId=7839486
    """

    def __init__(
        self,
        model,
        node,
        drought_level_agg_nyc,
        mrf_drought_factor_agg_reservoir,
        mrf_drought_factor_individual_reservoir,
        **kwargs,
    ):
        """
        Initialize NYCCombinedReleaseFactor.

        Parameters
        ----------
        model : pywr.Model
            The Pywr model instance.
        node : pywr.Node
            The reservoir node for which the release factor is applied.
        drought_level_agg_nyc : pywr.Parameter
            Parameter indicating NYC system-wide drought index.
        mrf_drought_factor_agg_reservoir : pywr.Parameter
            Drought multiplier for aggregated reservoir operations.
        mrf_drought_factor_individual_reservoir : pywr.Parameter
            Drought multiplier for individual reservoir operations.
        **kwargs
            Additional keyword arguments passed to the base Parameter class.
        """
        super().__init__(model, **kwargs)
        self.node = node
        self.drought_level_agg_nyc = drought_level_agg_nyc
        self.mrf_drought_factor_agg_reservoir = mrf_drought_factor_agg_reservoir
        self.mrf_drought_factor_individual_reservoir = (
            mrf_drought_factor_individual_reservoir
        )
        self.children.add(drought_level_agg_nyc)
        self.children.add(mrf_drought_factor_agg_reservoir)
        self.children.add(mrf_drought_factor_individual_reservoir)

    def value(self, timestep, scenario_index):
        """
        Compute the current release factor for an NYC reservoir.

        Uses a weighted logic depending on whether the current drought index 
        indicates normal/drought (use aggregate) or flood (use individual reservoir).

        Parameters
        ----------
        timestep : pywr.Timestep
            The current timestep.
        scenario_index : pywr.ScenarioIndex
            The scenario index used to retrieve parameter values.

        Returns
        -------
        float
            The final release multiplier for the reservoir.

        Notes
        -----
        The formula used is:
        ```
        factor = min(max(D_agg - 2, 0), 1) * factor_agg
               + min(max(3 - D_agg, 0), 1) * factor_indiv
        ```
        where D_agg is the drought level index for the NYC system.
        """
        ### $$ factor_{combined-cannonsville} = \min(\max(levelindex_{aggregated} - 2, 0), 1) * factor_{cannonsville}[levelindex_{aggregated}] +
        ###                                     \min(\max(3 - levelindex_{aggregated}, 0), 1) * factor_{cannonsville}[levelindex_{cannonsville}] $$

        return min(
            max(self.drought_level_agg_nyc.get_value(scenario_index) - 2, 0), 1
        ) * self.mrf_drought_factor_agg_reservoir.get_value(scenario_index) + min(
            max(3 - self.drought_level_agg_nyc.get_value(scenario_index), 0), 1
        ) * self.mrf_drought_factor_individual_reservoir.get_value(
            scenario_index
        )

    @classmethod
    def load(cls, model, data):
        """
        Load NYCCombinedReleaseFactor from a model configuration dictionary.

        Parameters
        ----------
        model : pywr.Model
            The Pywr model instance.
        data : dict
            Dictionary containing model configuration. Expected keys:
            - "node": Name of the reservoir node.
            - "mrf_drought_factor_agg_<reservoir>"
            - "mrf_drought_factor_individual_<reservoir>"

        Returns
        -------
        NYCCombinedReleaseFactor
            The loaded and initialized parameter instance.
        """
        reservoir = data.pop("node")
        node = model.nodes[reservoir]
        reservoir = reservoir.split("_")[1]
        drought_level_agg_nyc = load_parameter(model, f"drought_level_agg_nyc")
        mrf_drought_factor_agg_reservoir = load_parameter(
            model, f"mrf_drought_factor_agg_{reservoir}"
        )
        mrf_drought_factor_individual_reservoir = load_parameter(
            model, f"mrf_drought_factor_individual_{reservoir}"
        )
        return cls(
            model,
            node,
            drought_level_agg_nyc,
            mrf_drought_factor_agg_reservoir,
            mrf_drought_factor_individual_reservoir,
            **data,
        )


### have to register the custom parameter so Pywr recognizes it
NYCCombinedReleaseFactor.register()


class NYCFloodRelease(Parameter):
    """
    Computes excess flood control release for NYC reservoirs based on storage thresholds.

    If the drought level is 1a or 1b, this parameter calculates the additional volume that must be 
    released to bring storage back to the 1b/1c boundary over a 7-day period, following the rules 
    outlined in the FFMP (Flood and Drought Operating Plan) guidance.

    Attributes
    ----------
    node : pywr.Node
        Node representing the reservoir outlet.
    drought_level_reservoir : pywr.Parameter
        Parameter indicating the current drought level of the reservoir.
    level1c : pywr.Parameter
        Threshold parameter for the 1b/1c storage boundary.
    volume_reservoir : pywr.Parameter
        Current volume of water stored in the reservoir.
    max_volume_reservoir : pywr.Parameter
        Maximum physical storage capacity of the reservoir.
    weekly_rolling_mean_flow_reservoir : pywr.Parameter
        Weekly average of inflows to the reservoir.
    max_release_reservoir : pywr.Parameter
        Maximum allowable flood release for the reservoir.
    mrf_target_individual_reservoir : pywr.Parameter
        Baseline release target for the reservoir.

    Methods
    -------
    value(timestep, scenario_index)
        Returns the flood release volume for a given scenario and timestep.
    load(model, data)
        Class method for constructing the parameter from model config data.

    Notes
    -----
    For drought levels below 2 (i.e., 1a or 1b), excess storage is released over
    7 days, accounting for recent inflows and baseline release requirements.
    """

    def __init__(
        self,
        model,
        node,
        drought_level_reservoir,
        level1c,
        volume_reservoir,
        max_volume_reservoir,
        weekly_rolling_mean_flow_reservoir,
        max_release_reservoir,
        mrf_target_individual_reservoir,
        **kwargs,
    ):
        """
        Initialize NYCFloodRelease parameter.

        Parameters
        ----------
        model : pywr.Model
            The Pywr model instance.
        node : pywr.Node
            Node associated with the parameter.
        drought_level_reservoir : pywr.Parameter
            Parameter indicating the drought level at the reservoir.
        level1c : pywr.Parameter
            Storage boundary separating level 1b and 1c.
        volume_reservoir : pywr.Parameter
            Current volume in the reservoir.
        max_volume_reservoir : pywr.Parameter
            Full capacity of the reservoir.
        weekly_rolling_mean_flow_reservoir : pywr.Parameter
            Recent 7-day average of reservoir inflow.
        max_release_reservoir : pywr.Parameter
            Maximum release capacity for flood control.
        mrf_target_individual_reservoir : pywr.Parameter
            Minimum release target (baseline MRF) for the reservoir.
        **kwargs
            Additional keyword arguments passed to the base Parameter class.
        """
        super().__init__(model, **kwargs)
        self.node = node
        self.drought_level_reservoir = drought_level_reservoir
        self.level1c = level1c
        self.volume_reservoir = volume_reservoir
        self.max_volume_reservoir = max_volume_reservoir
        self.weekly_rolling_mean_flow_reservoir = weekly_rolling_mean_flow_reservoir
        self.max_release_reservoir = max_release_reservoir
        self.mrf_target_individual_reservoir = mrf_target_individual_reservoir

        self.children.add(drought_level_reservoir)
        self.children.add(level1c)
        self.children.add(volume_reservoir)
        self.children.add(max_volume_reservoir)
        self.children.add(weekly_rolling_mean_flow_reservoir)
        self.children.add(max_release_reservoir)
        self.children.add(mrf_target_individual_reservoir)

    def value(self, timestep, scenario_index):
        """
        Calculate the flood control release required to reduce reservoir storage.

        Parameters
        ----------
        timestep : pywr.Timestep
            Current timestep in the model run.
        scenario_index : pywr.ScenarioIndex
            Index specifying which scenario is being evaluated.

        Returns
        -------
        float
            Required flood control release (in volume units per day).

        Notes
        -----
        Logic only applies when drought level is < 2 (i.e., level 1a or 1b).
        Release ensures reservoir storage returns to level 1c threshold in 7 days.
        """
        ### extra flood releases needed if we are in level 1a or 1b
        if self.drought_level_reservoir.get_value(scenario_index) < 2:
            ## calculate the total excess volume needed to be release in next 7 days:
            ## assume for now this is just the current storage minus the level 1b/1c boundary, plus 7 * 7-day rolling avg inflow.
            excess_volume = (
                self.volume_reservoir.get_value(scenario_index)
                - (
                    self.level1c.get_value(scenario_index)
                    * self.max_volume_reservoir.get_value(scenario_index)
                )
                + self.weekly_rolling_mean_flow_reservoir.get_value(scenario_index) * 7
            )
            flood_release = max(
                min(
                    excess_volume / 7
                    - self.mrf_target_individual_reservoir.get_value(scenario_index),
                    self.max_release_reservoir.get_value(scenario_index)
                    - self.mrf_target_individual_reservoir.get_value(scenario_index),
                ),
                0,
            )

            return flood_release

        else:
            return 0.0

    @classmethod
    def load(cls, model, data):
        """
        Load NYCFloodRelease from model configuration dictionary.

        Parameters
        ----------
        model : pywr.Model
            The Pywr model instance.
        data : dict
            Dictionary specifying parameter configuration. Must include:
                - node
                - volume_<reservoir>
                - max_volume_<reservoir>
                - weekly_rolling_mean_flow_<reservoir>
                - flood_max_release_<reservoir>
                - mrf_target_individual_<reservoir>

        Returns
        -------
        NYCFloodRelease
            The constructed parameter instance.
        """
        reservoir = data.pop("node")
        node = model.nodes[reservoir]
        reservoir = reservoir.split("_")[1]
        drought_level_reservoir = load_parameter(model, f"drought_level_agg_nyc")
        level1c = load_parameter(model, "level1c")
        volume_reservoir = load_parameter(model, f"volume_{reservoir}")
        max_volume_reservoir = load_parameter(model, f"max_volume_{reservoir}")
        weekly_rolling_mean_flow_reservoir = load_parameter(
            model, f"weekly_rolling_mean_flow_{reservoir}"
        )
        max_release_reservoir = load_parameter(model, f"flood_max_release_{reservoir}")
        mrf_target_individual_reservoir = load_parameter(
            model, f"mrf_target_individual_{reservoir}"
        )

        return cls(
            model,
            node,
            drought_level_reservoir,
            level1c,
            volume_reservoir,
            max_volume_reservoir,
            weekly_rolling_mean_flow_reservoir,
            max_release_reservoir,
            mrf_target_individual_reservoir,
            **data,
        )


### have to register the custom parameter so Pywr recognizes it
NYCFloodRelease.register()


### Updated generic version of VolBalanceNYCDownstreamMRFTargetAgg_step1 that gets total release needed
###    to meet downstream flow target for any mrf (delMontague or delTrenton) and any step in multi-day staggered process
class TotalReleaseNeededForDownstreamMRF(Parameter):
    """
    Calculates the total NYC FFMP releases required to meet downstream Montague or Trenton flow targets.

    This parameter computes the additional water release needed from upstream reservoirs to satisfy
    regulatory flow requirements at downstream control points (Montague or Trenton), accounting for
    predicted natural flows, previous releases, and any mandatory individual or flood releases.

    Attributes
    ----------
    model : pywr.Model
        The Pywr model instance.
    mrf : str
        Flow target location, either 'delMontague' or 'delTrenton'.
    step : int
        Position in the multi-day staggered release sequence (1 through 4).
    predicted_nonnyc_gage_flow_mrf : pywr.Parameter
        Predicted natural flow at the control point, excluding NYC releases.
    predicted_demand_nj : pywr.Parameter or None
        Forecasted New Jersey demand (used for Trenton only).
    mrf_target_flow : pywr.Parameter
        Required flow target at the control point (Montague or Trenton).
    release_needed_mrf_montague : pywr.Parameter or None
        Additional upstream release needed at Montague (used when mrf = 'delTrenton').
    mrf_target_individual_nyc : pywr.Parameter or None
        Combined mandatory FFMP releases from NYC reservoirs (step 1–2 only).
    flood_release_nyc : pywr.Parameter or None
        Combined flood control releases from NYC reservoirs (step 1–2 only).
    previous_release_reservoirs : list of pywr.Parameter
        Reservoir releases from prior timesteps, used to estimate future arrivals.

    Methods
    -------
    value(timestep, scenario_index)
        Compute the NYC release required to satisfy the MRF target at the specified step.
    load(model, data)
        Class method to construct the parameter instance from a model configuration dictionary.

    Notes
    -----
    Steps 1–4 correspond to staggered lag travel times to meet Trenton and Montague MRFs.
    Required release is clipped at zero (i.e., no negative releases).
    """

    def __init__(
        self,
        model,
        step,
        mrf,
        predicted_nonnyc_gage_flow_mrf,
        predicted_demand_nj,
        mrf_target_flow,
        release_needed_mrf_montague,
        mrf_target_individual_nyc,
        flood_release_nyc,
        previous_release_reservoirs,
        **kwargs,
    ):
        """
        Initialize a TotalReleaseNeededForDownstreamMRF parameter instance.

        Parameters
        ----------
        model : pywr.Model
            Pywr model object.
        step : int
            Step in the lagged release sequence (1 to 4).
        mrf : str
            Flow control location: 'delMontague' or 'delTrenton'.
        predicted_nonnyc_gage_flow_mrf : pywr.Parameter
            Predicted natural flow at MRF gage.
        predicted_demand_nj : pywr.Parameter or None
            NJ demand forecast (used if mrf='delTrenton').
        mrf_target_flow : pywr.Parameter
            Regulatory flow requirement at control point.
        release_needed_mrf_montague : pywr.Parameter or None
            Upstream MRF contribution (used if mrf='delTrenton').
        mrf_target_individual_nyc : pywr.Parameter or None
            NYC individual FFMP release target (step 1–2 only).
        flood_release_nyc : pywr.Parameter or None
            NYC flood release volume (step 1–2 only).
        previous_release_reservoirs : list of pywr.Parameter
            Prior releases used for lagged flow accounting.
        **kwargs
            Additional arguments passed to the base class.
        """
        super().__init__(model, **kwargs)
        self.mrf = mrf
        self.step = step
        self.predicted_nonnyc_gage_flow_mrf = predicted_nonnyc_gage_flow_mrf
        self.mrf_target_flow = mrf_target_flow

        self.children.add(predicted_nonnyc_gage_flow_mrf)
        self.children.add(mrf_target_flow)

        ### we only have to acct for previous NYC releases after step 1
        if step > 1:
            self.previous_release_reservoirs = previous_release_reservoirs
            for p in previous_release_reservoirs:
                self.children.add(p)
                
        ### we only have to acct for NYC FFMP individual & flood releases in steps 1-2
        if step < 3:
            self.mrf_target_individual_nyc = mrf_target_individual_nyc
            self.flood_release_nyc = flood_release_nyc
            self.children.add(mrf_target_individual_nyc)
            self.children.add(flood_release_nyc)

        ### nj demand prediction only needed for Trenton.
        if mrf == "delTrenton":
            self.predicted_demand_nj = predicted_demand_nj
            self.release_needed_mrf_montague = release_needed_mrf_montague
            self.children.add(predicted_demand_nj)
            self.children.add(release_needed_mrf_montague)

    def value(self, timestep, scenario_index):
        """
        Compute additional NYC releases needed to meet the downstream flow target.

        Parameters
        ----------
        timestep : pywr.Timestep
            Current model timestep.
        scenario_index : pywr.ScenarioIndex
            Index identifying the scenario being evaluated.

        Returns
        -------
        float
            Required NYC FFMP release volume to meet MRF (non-negative).
        """
        ### we only have to acct for previous NYC releases after step 1
        if self.step > 1:
            previous_release_reservoirs_total = sum(
                [p.get_value(scenario_index) for p in self.previous_release_reservoirs]
            )
        else:
            previous_release_reservoirs_total = 0.0

        ### we only have to acct for NYC FFMP individual & flood releases in steps 1-2
        if self.step < 3:
            mrf_target_individual_nyc = self.mrf_target_individual_nyc.get_value(
                scenario_index
            )
            flood_release_nyc = self.flood_release_nyc.get_value(scenario_index)
            if self.mrf == "delTrenton":
                release_needed_mrf_montague = (
                    self.release_needed_mrf_montague.get_value(scenario_index)
                )
        else:
            mrf_target_individual_nyc = 0.0
            flood_release_nyc = 0.0
            release_needed_mrf_montague = 0.0

        ### we only need to account for upstream Montague mrf releases & NJ diversions if currently mrf=Trenton
        if self.mrf == "delTrenton":
            release_needed = max(
                self.mrf_target_flow.get_value(scenario_index)
                - self.predicted_nonnyc_gage_flow_mrf.get_value(scenario_index)
                - mrf_target_individual_nyc
                - flood_release_nyc
                - previous_release_reservoirs_total
                - release_needed_mrf_montague
                + self.predicted_demand_nj.get_value(scenario_index),
                0.0,
            )
        else:
            release_needed = max(
                self.mrf_target_flow.get_value(scenario_index)
                - self.predicted_nonnyc_gage_flow_mrf.get_value(scenario_index)
                - mrf_target_individual_nyc
                - flood_release_nyc
                - previous_release_reservoirs_total,
                0.0,
            )

        return release_needed

    @classmethod
    def load(cls, model, data):
        """
        Load a TotalReleaseNeededForDownstreamMRF instance from model configuration.

        Parameters
        ----------
        model : pywr.Model
            The Pywr model instance.
        data : dict
            Parameter configuration dictionary. Must include:
                - "mrf": either 'delMontague' or 'delTrenton'
                - "step": int in [1, 2, 3, 4]

        Returns
        -------
        TotalReleaseNeededForDownstreamMRF
            Configured parameter instance.
        """
        assert "mrf" in data.keys() and "step" in data.keys()
        assert data["mrf"] in ["delMontague", "delTrenton"]
        assert data["step"] in [1, 2, 3, 4]

        ### get predicted flow parameters based on mrf target & number of days ahead we are looking
        mrf = data.pop("mrf")
        step = data.pop("step")
        days_ahead = 3 - step if mrf == "delMontague" else 5 - step
        mrf_target_flow = load_parameter(model, f"mrf_target_{mrf}")
        predicted_nonnyc_gage_flow_mrf = load_parameter(
            model, f"predicted_nonnyc_gage_flow_{mrf}_lag{days_ahead}"
        )

        ### now fill in previous releases and current-step flood/ffmp releases based on step
        if step == 1:
            ### for step 1, we aggegate flood/indiv releases across 3 NYC reservoirs, and don't have any previous releases to acct for
            mrf_target_individual_nyc = load_parameter(
                model, "mrf_target_individual_agg_nyc"
            )
            flood_release_nyc = load_parameter(model, "flood_release_agg_nyc")
            previous_release_reservoirs = []
        elif step == 2:
            ### for step 2, we only need to acct for neversink flood/indiv, and we have can/pep previous releases
            mrf_target_individual_nyc = load_parameter(
                model, "mrf_target_individual_neversink"
            )
            flood_release_nyc = load_parameter(model, "flood_release_neversink")
            previous_release_reservoirs = [
                load_parameter(model, f"release_{r}_lag1")
                for r in ["cannonsville", "pepacton"]
            ]
        elif step == 3:
            ### for step 3, we only need to acct for can/pep/nev previous releases, 
            # not any NYC indiv/flood releases this time step
            mrf_target_individual_nyc = None
            flood_release_nyc = None
            previous_release_reservoirs = [
                load_parameter(model, f"release_{r}_lag2")
                for r in ["cannonsville", "pepacton"]
            ]
            previous_release_reservoirs += [
                load_parameter(model, f"release_{r}_lag1") for r in ["neversink"]
            ]
        elif step == 4:
            # for step 3, we only need to acct for can/pep/nev previous releases, 
            # not any NYC indiv/flood releases this time step
            mrf_target_individual_nyc = None
            flood_release_nyc = None
            previous_release_reservoirs = [
                load_parameter(model, f"release_{r}_lag3")
                for r in ["cannonsville", "pepacton"]
            ]
            previous_release_reservoirs += [
                load_parameter(model, f"release_{r}_lag2") for r in ["neversink"]
            ]
            previous_release_reservoirs += [
                load_parameter(model, f"release_{r}_lag1")
                for r in ["beltzvilleCombined", "blueMarsh"]
            ]
        else:
            print(
                "TotalReleaseNeededForDownstreamMRF shouldnt be here - only 4 steps implemented"
            )

        if mrf == "delTrenton":
            predicted_demand_nj = load_parameter(
                model, f"predicted_demand_nj_lag{days_ahead}"
            )  ### only used for delTrenton
            if step < 3:
                release_needed_mrf_montague = load_parameter(
                    model, f"release_needed_mrf_montague_step{step}"
                )  ### only used for delTrenton
            else:
                release_needed_mrf_montague = None

        else:
            predicted_demand_nj = None
            release_needed_mrf_montague = None

        return cls(
            model,
            step,
            mrf,
            predicted_nonnyc_gage_flow_mrf,
            predicted_demand_nj,
            mrf_target_flow,
            release_needed_mrf_montague,
            mrf_target_individual_nyc,
            flood_release_nyc,
            previous_release_reservoirs,
            **data,
        )


### have to register the custom parameter so Pywr recognizes it
TotalReleaseNeededForDownstreamMRF.register()


### updated parameter to consistently assign release targets for all 3 reservoirs, above and beyond individual mandated releases.
###     - total release needed for Montague & Trenton targets from FFMP
###     - accounts for max release constraints at each reservoir


class VolBalanceNYCDownstreamMRF_step1(Parameter):
    """
    Assigns release targets for all 3 NYC reservoirs, above and beyond individual mandated releases,
    to meet Montague & Trenton MRF targets. Accounts for max release constraints at each reservoir.

    Attributes
    ----------
    reservoir : str
        The reservoir associated with this specific parameter instance.
    nodes : dict
        Dictionary of Pywr nodes corresponding to the NYC reservoirs.
    parameters : dict
        Dictionary of associated parameters required for MRF calculations.
    max_vol_reservoirs : list
        List of maximum volume parameters for NYC reservoirs.
    vol_reservoirs : list
        List of volume parameters for NYC reservoirs.
    flow_reservoirs : list
        List of current flow parameters for NYC reservoirs.
    max_release_reservoirs : list
        List of maximum release constraints for each NYC reservoir.
    mrf_target_individual_reservoirs : list
        List of MRF release requirements for each individual NYC reservoir.
    flood_release_reservoirs : list
        List of flood release volumes for each NYC reservoir.
    num_reservoirs : int
        Number of coordinating NYC reservoirs.
    
    Methods
    -------
    split_required_mrf_across_nyc_reservoirs(requirement_total, scenario_index)
        Splits total NYC release requirement among reservoirs using volume-balancing and constraints.
    
    value(timestep, scenario_index)
        Returns the MRF release target for the current reservoir and timestep.
    
    load(model, data)
        Loads the parameter from model and dictionary input.
    """

    def __init__(self, model, reservoir, nodes, parameters, **kwargs):
        """
        Initialize the VolBalanceNYCDownstreamMRF_step1 parameter.

        This parameter distributes the required additional NYC releases (beyond
        individual mandated releases) across the three NYC reservoirs using a 
        volume-balancing method. It also accounts for maximum release constraints 
        at each reservoir.

        Parameters
        ----------
        model : Model
            The Pywr model instance.
        reservoir : str
            The reservoir associated with this specific parameter instance.
        nodes : dict
            Dictionary of Pywr Node objects for NYC reservoirs.
        parameters : dict
            Dictionary of Parameter objects required for MRF calculation,
            including storage, flow, flood release, and max release values
            for all NYC reservoirs, as well as aggregate system parameters.
        **kwargs
            Additional keyword arguments passed to the Pywr Parameter base class.

        Notes
        -----
        This parameter is evaluated individually for each reservoir but uses shared
        data across the three NYC reservoirs (Cannonsville, Pepacton, Neversink)
        to compute coordinated releases that satisfy downstream flow targets.
        """
        super().__init__(model, **kwargs)
        self.reservoir = reservoir
        self.nodes = nodes
        self.parameters = parameters

        self.num_reservoirs = len(nodes)

        # Individual parameters
        for param_name, param in parameters.items():
            setattr(self, param_name, param)
            self.children.add(param)

        # Grouped NYC parameters
        self.max_vol_reservoirs = [
            parameters[f"max_volume_{res}"] for res in reservoir_list_nyc
        ]
        self.vol_reservoirs = [
            parameters[f"volume_{res}"] for res in reservoir_list_nyc
        ]
        self.flow_reservoirs = [parameters[f"flow_{res}"] for res in reservoir_list_nyc]
        self.max_release_reservoirs = [
            parameters[f"controlled_max_release_{res}"] for res in reservoir_list_nyc
        ]
        self.mrf_target_individual_reservoirs = [
            parameters[f"mrf_target_individual_{res}"] for res in reservoir_list_nyc
        ]
        self.flood_release_reservoirs = [
            parameters[f"flood_release_{res}"] for res in reservoir_list_nyc
        ]

    def split_required_mrf_across_nyc_reservoirs(
        self, requirement_total, scenario_index
    ):
        """
        Splits the total required downstream flow release across NYC reservoirs,
        adjusting for max release constraints and relative reservoir contributions.

        Parameters
        ----------
        requirement_total : float
            Total additional release required to meet Montague/Trenton MRF flow targets.
        scenario_index : ScenarioIndex
            Index indicating which scenario the value applies to.

        Returns
        -------
        targets : list of float
            Target release amounts for each NYC reservoir.
        
        Notes
        -----
        Uses a volume-based distribution method that scales by reservoir contributions to the 
        total system volume. Enforces individual release constraints and adjusts iteratively.
        """
        # Get max release constraints
        max_releases_reservoirs = [
            max(
                self.max_release_reservoirs[i].get_value(scenario_index)
                - self.mrf_target_individual_reservoirs[i].get_value(scenario_index)
                - self.flood_release_reservoirs[i].get_value(scenario_index),
                0,
            )
            for i in range(self.num_reservoirs)
        ]

        ## Use only NYC releases to meet Montague & Trenton targets
        targets = [-1] * self.num_reservoirs
        for i in range(self.num_reservoirs):
            targets[i] = (
                self.vol_reservoirs[i].get_value(scenario_index)
                + self.flow_reservoirs[i].get_value(scenario_index)
                - self.mrf_target_individual_reservoirs[i].get_value(scenario_index)
                - self.flood_release_reservoirs[i].get_value(scenario_index)
                - (
                    self.max_vol_reservoirs[i].get_value(scenario_index)
                    / self.max_volume_agg_nyc.get_value(scenario_index)
                )
                * (
                    self.volume_agg_nyc.get_value(scenario_index)
                    + self.flow_agg_nyc.get_value(scenario_index)
                    - self.mrf_target_individual_agg_nyc.get_value(scenario_index)
                    - self.flood_release_agg_nyc.get_value(scenario_index)
                    - requirement_total
                )
            )
            ### enforce nonnegativity and reservoir max release constraint. Set min to 0.01 instead of 0 so that it can be activated if extra flow needed below.
            targets[i] = min(max(targets[i], 0.01), max_releases_reservoirs[i])

        if np.isnan(targets).any():
            print(
                f"Warning: NaNs present in NYC release target. Possibly due to zero storage."
            )
            targets = [target if not np.isnan(target) else 0.0 for target in targets]

        ### sum total release across 3 reservoirs.
        target_sum = sum(targets)

        ### if target_sum > requirement_total (which happens if one of targets was initially negative before
        ###   taking max(0)) -> fix this by reducing nonzero releases proportionally
        if target_sum > requirement_total + epsilon:
            for i in range(self.num_reservoirs):
                targets[i] *= requirement_total / target_sum
        target_sum = sum(targets)

        ###if sum this is less than total_agg_mrf_montagueTrenton,
        ### that means one of the reservoirs had exceeded max release before min() above
        ### -> rescale unconstrained reservoirs to counteract
        fully_constrained = False if (target_sum > epsilon) else True
        count = 0
        while requirement_total - epsilon > target_sum and not fully_constrained:
            increasable_flow = 0
            unincreasable_flow = 0
            ### find the total "increasable" flow that is not coming from reservoirs with 0 release or max_release
            for i in range(self.num_reservoirs):
                if targets[i] < max_releases_reservoirs[i]:
                    increasable_flow += targets[i]
                else:
                    unincreasable_flow += targets[i]
            if increasable_flow > epsilon:
                for i in range(self.num_reservoirs):
                    targets[i] = min(
                        targets[i]
                        * (requirement_total - unincreasable_flow)
                        / increasable_flow,
                        max_releases_reservoirs[i],
                    )
            else:
                fully_constrained = True
                # Check for nans which occur when fully empty
                if np.isnan(targets).any():
                    print(
                        f"Warning: NaNs present in NYC release target. Possibly due to zero storage."
                    )
                targets = [
                    target if not np.isnan(target) else 0.0 for target in targets
                ]
            target_sum = sum(targets)
            count += 1
            if count > 5:
                print(
                    "shouldnt be here 1 VolBalanceNYCDownstreamMRF_step1",
                    count,
                    requirement_total,
                    target_sum,
                    increasable_flow,
                    targets,
                    max_releases_reservoirs,
                )

        if target_sum > requirement_total + epsilon:
            print(
                "shouldnt be here 2 VolBalanceNYCDownstreamMRF_step1",
                count,
                target_sum,
                requirement_total,
            )
            print([self.vol_reservoirs[i].get_value(scenario_index) for i in range(3)])
            print([self.flow_reservoirs[i].get_value(scenario_index) for i in range(3)])
            print(
                [
                    self.mrf_target_individual_reservoirs[i].get_value(scenario_index)
                    for i in range(3)
                ]
            )
            print(
                [
                    self.flood_release_reservoirs[i].get_value(scenario_index)
                    for i in range(3)
                ]
            )
            print(
                [self.max_vol_reservoirs[i].get_value(scenario_index) for i in range(3)]
            )
            print(self.max_volume_agg_nyc.get_value(scenario_index))
            print(self.volume_agg_nyc.get_value(scenario_index))
            print(self.flow_agg_nyc.get_value(scenario_index))
            print(self.mrf_target_individual_agg_nyc.get_value(scenario_index))
            print(self.flood_release_agg_nyc.get_value(scenario_index))
            print(requirement_total)
            print(max_releases_reservoirs)
            print(targets)
        return targets

    def value(self, timestep, scenario_index):
        """
        Returns the target release value for the specific NYC reservoir instance
        based on the total MRF release requirement and reservoir balancing logic.

        Parameters
        ----------
        timestep : Timestep
            The current model timestep.
        scenario_index : ScenarioIndex
            Index of the simulation scenario.

        Returns
        -------
        float
            The release target for the given reservoir to help meet the MRF flow target.
        """
        sid = scenario_index.global_id
        ### calculate contributions for all 3 NYC reservoirs in consistent way.
        ### Note: ideally we would only do this once. But may not be possible to have parameter with array output,
        ###       so for now we just repeat this procedure for each reservoir.

        ### first calculate contributions to Trenton&Montague flow targets based on volume balancing formula.
        ### These are above and beyond what is needed for individual FFMP mandated releases
        nyc_requirement_total = self.total_agg_mrf_montagueTrenton_step1.get_value(
            scenario_index
        ) 
        
        # lower basin release is already accounted for in total_agg_mrf_montagueTrenton
        # - self.lower_basin_agg_mrf_trenton_step1.get_value(scenario_index)

        # If no downstream flow required, return 0.0
        if nyc_requirement_total < epsilon:
            return 0.0
        else:
            # Split required MRF releases across NYC reservoirs
            nyc_targets = self.split_required_mrf_across_nyc_reservoirs(
                nyc_requirement_total, scenario_index
            )

            ## Return target for the reservoir of interest
            for i in range(self.num_reservoirs):
                if self.reservoir == reservoir_list_nyc[i]:
                    return nyc_targets[i]
            print(
                "shouldnt be here 3 VolBalanceNYCDownstreamMRF_step1 ",
                self.reservoir,
                nyc_targets,
            )

    @classmethod
    def load(cls, model, data):
        """
        Loads the class from a model and a data dictionary.

        Parameters
        ----------
        model : Model
            The Pywr model instance.
        data : dict
            Dictionary containing configuration including the reservoir node and parameters.

        Returns
        -------
        VolBalanceNYCDownstreamMRF_step1
            Instance of the class with associated nodes and parameters loaded.
        """
        reservoir = data.pop("node")
        reservoir = reservoir.split("_")[1]

        node_list = [f"reservoir_{i}" for i in reservoir_list_nyc]

        parameter_list = [
            "max_volume_agg_nyc",
            "volume_agg_nyc",
            "drought_level_agg_nyc",
            "total_agg_mrf_montagueTrenton_step1",
            "lower_basin_agg_mrf_trenton_step1",
            "flow_agg_nyc",
            "mrf_target_individual_agg_nyc",
            "flood_release_agg_nyc",
        ]

        for res in reservoir_list_nyc:
            parameter_list += [
                f"max_volume_{res}",
                f"volume_{res}",
                f"flow_{res}",
                f"controlled_max_release_{res}",
                f"mrf_target_individual_{res}",
                f"flood_release_{res}",
            ]

        # Load nodes
        nodes = {}
        for node in node_list:
            nodes[node] = model.nodes[node]

        # Load parameters
        parameters = {}
        for param in parameter_list:
            parameters[param] = load_parameter(model, param)

        return cls(model, reservoir, nodes, parameters, **data)


### have to register the custom parameter so Pywr recognizes it
VolBalanceNYCDownstreamMRF_step1.register()


class VolBalanceNYCDownstreamMRF_step2(Parameter):
    """
    Calculates the volume of additional flow that Neversink Reservoir must release during Step 2
    of the staggered MRF coordination process, accounting for prior contributions and constraints.

    This parameter computes how much Neversink can contribute toward Montague and/or Trenton
    flow targets, beyond its mandated minimum and flood releases, constrained by its
    controllable release capacity.

    Attributes
    ----------
    step : int
        The coordination step (fixed as 2 for this class).
    total_agg_mrf_montagueTrenton : Parameter
        Total release required from NYC reservoirs to meet MRF targets at Montague or Trenton.
    mrf_target_individual_neversink : Parameter
        Minimum mandated release from Neversink reservoir under FFMP.
    flood_release_neversink : Parameter
        Scheduled flood control release from Neversink.
    max_release_neversink : Parameter
        Maximum controlled release capacity from Neversink.
    lower_basin_agg_mrf_trenton : Parameter
        Lower basin contribution to Trenton flow target (informational only).

    Methods
    -------
    value(timestep, scenario_index)
        Computes the feasible additional release from Neversink to help meet flow targets.
    load(model, data)
        Loads all required model parameters and returns the parameter instance.
    """

    def __init__(
        self,
        model,
        step,
        total_agg_mrf_montagueTrenton,
        mrf_target_individual_neversink,
        flood_release_neversink,
        max_release_neversink,
        lower_basin_agg_mrf_trenton,
        **kwargs,
    ):
        """
        Initialize the VolBalanceNYCDownstreamMRF_step2 parameter.

        This class handles Step 2 of the NYC coordinated release process,
        determining how much additional water Neversink can release given
        its constraints and obligations.

        Parameters
        ----------
        model : Model
            The Pywr model instance.
        step : int
            Coordination step (should be 2).
        total_agg_mrf_montagueTrenton : Parameter
            Total volume of release required across all NYC reservoirs to meet Montague/Trenton target.
        mrf_target_individual_neversink : Parameter
            Required minimum FFMP release from Neversink.
        flood_release_neversink : Parameter
            Additional flood-mitigation release from Neversink.
        max_release_neversink : Parameter
            Maximum allowed controlled release from Neversink.
        lower_basin_agg_mrf_trenton : Parameter
            Flow from lower basin reservoirs toward Trenton (used for context, not directly in computation).
        **kwargs
            Additional keyword arguments passed to the base Pywr Parameter class.

        Notes
        -----
        This step assumes Cannonsville and Pepacton have already contributed in Step 1.
        Neversink’s share is constrained by its physical release limits and prior mandated releases.
        """
        super().__init__(model, **kwargs)
        self.step = step
        self.total_agg_mrf_montagueTrenton = total_agg_mrf_montagueTrenton
        self.mrf_target_individual_neversink = mrf_target_individual_neversink
        self.flood_release_neversink = flood_release_neversink
        self.max_release_neversink = max_release_neversink
        self.lower_basin_agg_mrf_trenton = lower_basin_agg_mrf_trenton

        self.children.add(step)
        self.children.add(total_agg_mrf_montagueTrenton)
        self.children.add(mrf_target_individual_neversink)
        self.children.add(flood_release_neversink)
        self.children.add(max_release_neversink)
        self.children.add(lower_basin_agg_mrf_trenton)

    def value(self, timestep, scenario_index):
        """
        Compute the additional volume that Neversink can release to meet downstream flow targets.

        This method evaluates how much Neversink Reservoir can contribute during Step 2,
        after accounting for its FFMP-required release, flood release, and maximum release capacity.

        Parameters
        ----------
        timestep : Timestep
            The current model timestep.
        scenario_index : ScenarioIndex
            Index specifying the scenario for which the value is evaluated.

        Returns
        -------
        float
            The additional feasible release from Neversink Reservoir (above minimums) to support the flow target.
        """
        max_release_neversink = max(
            self.max_release_neversink.get_value(scenario_index)
            - self.mrf_target_individual_neversink.get_value(scenario_index)
            - self.flood_release_neversink.get_value(scenario_index),
            0,
        )

        release_neversink = max(
            min(
                max_release_neversink,
                self.total_agg_mrf_montagueTrenton.get_value(scenario_index) # - self.lower_basin_agg_mrf_trenton.get_value(scenario_index),
            ),
            0,
        )
        return release_neversink

    @classmethod
    def load(cls, model, data):
        """
        Load model parameters and create a VolBalanceNYCDownstreamMRF_step2 instance.

        This class method retrieves all necessary model parameters from the data dictionary
        and initializes the parameter object for Step 2 release balancing.

        Parameters
        ---------
        model : Model
            The Pywr model instance.
        data : dict
            Dictionary containing metadata and configuration inputs.

        Returns
        -------
        VolBalanceNYCDownstreamMRF_step2
            A configured parameter instance for Step 2 NYC release balancing.

        Notes
        -----
        This method is automatically called when the Pywr model is built using a data dictionary
        that references this custom parameter class.
        """
        step = 2
        total_agg_mrf_montagueTrenton = load_parameter(
            model, f"total_agg_mrf_montagueTrenton_step{step}"
        )
        mrf_target_individual_neversink = load_parameter(
            model, "mrf_target_individual_neversink"
        )
        flood_release_neversink = load_parameter(model, "flood_release_neversink")
        max_release_neversink = load_parameter(
            model, "controlled_max_release_neversink"
        )
        lower_basin_agg_mrf_trenton = load_parameter(
            model, f"lower_basin_agg_mrf_trenton_step{step}"
        )
        return cls(
            model,
            step,
            total_agg_mrf_montagueTrenton,
            mrf_target_individual_neversink,
            flood_release_neversink,
            max_release_neversink,
            lower_basin_agg_mrf_trenton,
            **data,
        )


### have to register the custom parameter so Pywr recognizes it
VolBalanceNYCDownstreamMRF_step2.register()


class VolBalanceNYCDemand(Parameter):
    """
    Computes NYC reservoir delivery targets to meet downstream demands while balancing relative storage across the system.

    This parameter allocates NYC delivery volume across the three upstream reservoirs (Cannonsville, Pepacton, Neversink)
    based on storage and inflow conditions, maximum diversion capacity, and existing release obligations. The goal is to
    keep reservoir storage levels balanced while fulfilling the NYC system delivery target.

    See comments on this GitHub issue for the equations and logic:
    https://github.com/users/ahamilton144/projects/1/views/1?pane=issue&itemId=7840442

    Attributes
    ----------
    reservoir : str
        The name of the NYC reservoir associated with this instance.
    nodes : list
        List of model nodes corresponding to NYC reservoirs.
    num_reservoirs : int
        Number of NYC reservoirs (typically 3).
    max_volume_agg_nyc : Parameter
        Total maximum volume capacity of the combined NYC reservoir system.
    volume_agg_nyc : Parameter
        Total current volume of the combined NYC reservoir system.
    max_flow_delivery_nyc : Parameter
        Maximum allowed total delivery to NYC at the current time.
    flow_agg_nyc : Parameter
        Current total flow delivered to NYC from all reservoirs.
    max_vol_reservoirs : list of Parameter
        Maximum volume for each individual NYC reservoir.
    vol_reservoirs : list of Parameter
        Current volume for each NYC reservoir.
    flow_reservoirs : list of Parameter
        Current delivery flow for each NYC reservoir.
    hist_max_flow_delivery_nycs : list of Parameter
        Historical maximum diversion rate for each NYC reservoir.
    mrf_target_individual_reservoirs : list of Parameter
        Minimum required release (mandated) for each NYC reservoir under FFMP.
    downstream_release_target_reservoirs : list of Parameter
        Additional downstream release obligations for each NYC reservoir.
    flood_release_reservoirs : list of Parameter
        Flood mitigation releases scheduled for each NYC reservoir.

    Methods
    -------
    value(timestep, scenario_index)
        Calculates the delivery target for the specific NYC reservoir to balance storage and meet total delivery needs.
    load(model, data)
        Class method to initialize the parameter from a Pywr model and input data dictionary.

    Notes
    -----
    This parameter helps balance the NYC system's deliveries while preserving storage equity.
    The delivery calculation is performed separately for each reservoir instance, due to
    Pywr’s scalar parameter constraints.
    """

    def __init__(
        self,
        model,
        reservoir,
        nodes,
        max_volume_agg_nyc,
        volume_agg_nyc,
        max_flow_delivery_nyc,
        flow_agg_nyc,
        max_vol_reservoirs,
        vol_reservoirs,
        flow_reservoirs,
        hist_max_flow_delivery_nycs,
        mrf_target_individual_reservoirs,
        downstream_release_target_reservoirs,
        flood_release_reservoirs,
        **kwargs,
    ):
        """
        Initialize an instance of the VolBalanceNYCDemand parameter.

        Parameters
        ----------
        model : Model
            Pywr model object.
        reservoir : str
            Name of the NYC reservoir for which this parameter instance applies.
        nodes : list
            List of Pywr reservoir nodes (Cannonsville, Pepacton, Neversink).
        max_volume_agg_nyc : Parameter
            Total maximum volume of the NYC reservoir system.
        volume_agg_nyc : Parameter
            Combined current volume of the NYC reservoir system.
        max_flow_delivery_nyc : Parameter
            Target total NYC delivery rate.
        flow_agg_nyc : Parameter
            Current combined delivery from all reservoirs to NYC.
        max_vol_reservoirs : list of Parameter
            Maximum storage capacity for each reservoir.
        vol_reservoirs : list of Parameter
            Current volume for each reservoir.
        flow_reservoirs : list of Parameter
            Current delivery flow for each reservoir.
        hist_max_flow_delivery_nycs : list of Parameter
            Historical maximum delivery constraints for each reservoir.
        mrf_target_individual_reservoirs : list of Parameter
            Minimum flow release requirements for each reservoir (e.g., FFMP).
        downstream_release_target_reservoirs : list of Parameter
            Required additional downstream releases per reservoir.
        flood_release_reservoirs : list of Parameter
            Flood control releases per reservoir.
        **kwargs
            Additional keyword arguments for base Parameter class.

        Notes
        -----
        The `value` method internally repeats balancing calculations for each reservoir
        individually because array-valued parameters are not natively supported in Pywr.
        """
        super().__init__(model, **kwargs)
        self.reservoir = reservoir
        self.nodes = nodes
        self.num_reservoirs = len(nodes)
        self.max_volume_agg_nyc = max_volume_agg_nyc
        self.volume_agg_nyc = volume_agg_nyc
        self.max_flow_delivery_nyc = max_flow_delivery_nyc
        self.flow_agg_nyc = flow_agg_nyc
        self.max_vol_reservoirs = max_vol_reservoirs
        self.vol_reservoirs = vol_reservoirs
        self.flow_reservoirs = flow_reservoirs
        self.hist_max_flow_delivery_nycs = hist_max_flow_delivery_nycs
        self.mrf_target_individual_reservoirs = mrf_target_individual_reservoirs
        self.downstream_release_target_reservoirs = downstream_release_target_reservoirs
        self.flood_release_reservoirs = flood_release_reservoirs

        self.children.add(max_volume_agg_nyc)
        self.children.add(volume_agg_nyc)
        self.children.add(max_flow_delivery_nyc)
        self.children.add(flow_agg_nyc)
        for i in range(len(reservoir_list_nyc)):
            self.children.add(max_vol_reservoirs[i])
            self.children.add(vol_reservoirs[i])
            self.children.add(flow_reservoirs[i])
            self.children.add(hist_max_flow_delivery_nycs[i])
            self.children.add(mrf_target_individual_reservoirs[i])
            self.children.add(downstream_release_target_reservoirs[i])
            self.children.add(flood_release_reservoirs[i])

    def value(self, timestep, scenario_index):
        """
        Calculate the NYC delivery target for this reservoir that balances storage and meets system-wide demand.

        Parameters
        ----------
        timestep : Timestep
            Current model timestep.
        scenario_index : ScenarioIndex
            Current scenario index in the simulation.

        Returns
        -------
        float
            Target diversion to NYC from this reservoir on the current timestep.

        Notes
        -----
        This function allocates deliveries proportionally using a volume-balancing approach while
        enforcing individual reservoir diversion constraints. If one or more reservoirs are
        constrained, deliveries from others may be scaled up within feasible limits.
        """
        sid = scenario_index.global_id
        ### calculate diversions from all 3 NYC reservoirs in consistent way.
        ### Note: ideally we would only do this once. But may not be possible to have parameter with array output,
        ###       so for now we just repeat this procedure for each reservoir.

        ### first calculate the contributions to NYC delivery for this reservoir to balance storages across reservoirs
        requirement_total = self.max_flow_delivery_nyc.get_value(scenario_index)
        max_diversions = [
            self.hist_max_flow_delivery_nycs[i].get_value(scenario_index)
            for i in range(self.num_reservoirs)
        ]
        mrf_individual_total = sum(
            [
                self.mrf_target_individual_reservoirs[i].get_value(scenario_index)
                for i in range(self.num_reservoirs)
            ]
        )
        mrf_downstream_total = sum(
            [
                self.downstream_release_target_reservoirs[i].get_value(scenario_index)
                for i in range(self.num_reservoirs)
            ]
        )
        flood_release_total = sum(
            [
                self.flood_release_reservoirs[i].get_value(scenario_index)
                for i in range(self.num_reservoirs)
            ]
        )

        if requirement_total < epsilon:
            return 0.0
        else:
            targets = [-1] * self.num_reservoirs
            for i in range(self.num_reservoirs):
                targets[i] = (
                    self.vol_reservoirs[i].get_value(scenario_index)
                    + self.flow_reservoirs[i].get_value(scenario_index)
                    - self.mrf_target_individual_reservoirs[i].get_value(scenario_index)
                    - self.downstream_release_target_reservoirs[i].get_value(
                        scenario_index
                    )
                    - self.flood_release_reservoirs[i].get_value(scenario_index)
                    - (
                        self.max_vol_reservoirs[i].get_value(scenario_index)
                        / self.max_volume_agg_nyc.get_value(scenario_index)
                    )
                    * (
                        self.volume_agg_nyc.get_value(scenario_index)
                        + self.flow_agg_nyc.get_value(scenario_index)
                        - mrf_individual_total
                        - mrf_downstream_total
                        - flood_release_total
                        - requirement_total
                    )
                )

                ### enforce nonnegativity and reservoir max release constraint. Set min to 0.01 instead of 0 so that it can be activated if extra flow needed below.
                targets[i] = min(max(targets[i], 0.01), max_diversions[i])

            ### sum total release across 3 reservoirs.
            target_sum = sum(targets)

            ### if target_sum > requirement_total (which happens if one of targets was initially negative before
            ###   taking max(0)) -> fix this by reducing nonzero releases proportionally
            if target_sum > requirement_total + epsilon:
                for i in range(self.num_reservoirs):
                    targets[i] *= requirement_total / target_sum
            target_sum = sum(targets)

            ###if sum this is less than requirement_total,
            ### that means one of the reservoirs had exceeded max release before min() above
            ### -> rescale unconstrained reservoirs to counteract
            fully_constrained = False
            count = 0
            while requirement_total - epsilon > target_sum and not fully_constrained:
                increasable_flow = 0
                unincreasable_flow = 0
                ### find the total "increasable" flow that is not coming from reservoirs with 0 release or max_release
                for i in range(self.num_reservoirs):
                    if targets[i] < max_diversions[i]:
                        increasable_flow += targets[i]
                    else:
                        unincreasable_flow += targets[i]
                if increasable_flow > epsilon:
                    for i in range(self.num_reservoirs):
                        targets[i] = min(
                            targets[i]
                            * (requirement_total - unincreasable_flow)
                            / increasable_flow,
                            max_diversions[i],
                        )
                else:
                    fully_constrained = True
                    print("fully constrained", targets)
                target_sum = sum(targets)
                count += 1
                if count > 5:
                    print(
                        "shouldnt be here 1 VolBalanceNYCDemand, ",
                        count,
                        requirement_total,
                        target_sum,
                        increasable_flow,
                        targets,
                        max_diversions,
                    )
            if target_sum > requirement_total + epsilon:
                print(
                    "shouldnt be here 2 VolBalanceNYCDemand, ",
                    count,
                    target_sum,
                    requirement_total,
                )

            ### now return target for the reservoir of interest
            for i in range(self.num_reservoirs):
                if self.reservoir == reservoir_list_nyc[i]:
                    return targets[i]

    @classmethod
    def load(cls, model, data):
        """
        Load and initialize the VolBalanceNYCDemand parameter from model and config data.

        Parameters
        ----------
        model : Model
            The Pywr model object.
        data : dict
            Dictionary with parameter configuration values (including the 'node' field).

        Returns
        -------
        VolBalanceNYCDemand
            Instantiated parameter object, ready for use in simulation.

        Notes
        -----
        The `node` name is used to infer which NYC reservoir (Cannonsville, Pepacton, or Neversink)
        this parameter is associated with. All required supporting parameters are also loaded here.
        """
        reservoir = data.pop("node")
        reservoir = reservoir.split("_")[1]
        nodes = [
            model.nodes[f"reservoir_{reservoir}"] for reservoir in reservoir_list_nyc
        ]
        max_volume_agg_nyc = load_parameter(model, "max_volume_agg_nyc")
        volume_agg_nyc = load_parameter(model, "volume_agg_nyc")
        max_flow_delivery_nyc = load_parameter(model, "max_flow_delivery_nyc")
        flow_agg_nyc = load_parameter(model, "flow_agg_nyc")
        max_vol_reservoirs = [
            load_parameter(model, f"max_volume_{reservoir}")
            for reservoir in reservoir_list_nyc
        ]
        vol_reservoirs = [
            load_parameter(model, f"volume_{reservoir}")
            for reservoir in reservoir_list_nyc
        ]
        flow_reservoirs = [
            load_parameter(model, f"flow_{reservoir}")
            for reservoir in reservoir_list_nyc
        ]
        hist_max_flow_delivery_nycs = [
            load_parameter(model, f"hist_max_flow_delivery_nyc_{reservoir}")
            for reservoir in reservoir_list_nyc
        ]
        mrf_target_individual_reservoirs = [
            load_parameter(model, f"mrf_target_individual_{reservoir}")
            for reservoir in reservoir_list_nyc
        ]
        downstream_release_target_reservoirs = [
            load_parameter(model, f"downstream_release_target_{reservoir}")
            for reservoir in reservoir_list_nyc
        ]
        flood_release_reservoirs = [
            load_parameter(model, f"flood_release_{reservoir}")
            for reservoir in reservoir_list_nyc
        ]
        return cls(
            model,
            reservoir,
            nodes,
            max_volume_agg_nyc,
            volume_agg_nyc,
            max_flow_delivery_nyc,
            flow_agg_nyc,
            max_vol_reservoirs,
            vol_reservoirs,
            flow_reservoirs,
            hist_max_flow_delivery_nycs,
            mrf_target_individual_reservoirs,
            downstream_release_target_reservoirs,
            flood_release_reservoirs,
            **data,
        )
### have to register the custom parameter so Pywr recognizes it
VolBalanceNYCDemand.register()
