"""
This file contains different class objects which are used to construct custom Pywr parameters.

The parameters created here are used to implement the flexible flow management program (FFMP)
for the three NYC reservoirs.
"""

import numpy as np
import pandas as pd
from pywr.parameters import Parameter, load_parameter

from pywrdrb.utils.lists import reservoir_list_nyc
from pywrdrb.utils.constants import epsilon

class FfmpNycRunningAvgParameter(Parameter):
    """
    Enforces the constraint on NYC deliveries from the FFMP, based on a running average.

    Args:
        model (Model): The Pywr model instance.
        node (Node): The node associated with the parameter.
        max_avg_delivery (ConstantParameter): The maximum average delivery constant parameter.

    Attributes:
        node (Node): The node associated with the parameter.
        max_avg_delivery (float): The maximum average delivery value.
        max_delivery (ndarray): An array to hold the parameter state.

    Methods:
        setup(): Allocates an array to hold the parameter state.
        reset(): Resets the amount remaining in all states to the initial value.
        value(timestep, scenario_index): Returns the current volume remaining for the scenario.
        after(): Updates the parameter requirement based on running average and updates the date for tomorrow.

    Class Methods:
        load(model, data): Loads the parameter from model and data dictionary.
    """

    def __init__(self, model, node, max_avg_delivery, **kwargs):
        super().__init__(model, **kwargs)
        self.node = node
        self.max_avg_delivery = max_avg_delivery.get_constant_value()
        self.children.add(max_avg_delivery)

    def setup(self):
        """Allocates an array to hold the parameter state."""
        super().setup()
        num_scenarios = len(self.model.scenarios.combinations)
        self.max_delivery = np.empty([num_scenarios], np.float64)

    def reset(self):
        """Resets the amount remaining in all states to the initial value."""
        self.timestep = self.model.timestepper.delta
        self.datetime = pd.to_datetime(self.model.timestepper.start)
        self.max_delivery[...] = self.max_avg_delivery * self.timestep

    def value(self, timestep, scenario_index):
        """
        Returns the current volume remaining for the scenario.

        Args:
            timestep (Timestep): The current timestep.
            scenario_index (ScenarioIndex): The scenario index.

        Returns:
            float: The current volume remaining for the scenario.
        """
        return self.max_delivery[scenario_index.global_id]

    def after(self):
        """Updates the parameter requirement based on running average and updates the date for tomorrow."""
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
        Loads the parameter from model and data dictionary.

        Args:
            model (Model): The Pywr model instance.
            data (dict): The data dictionary containing the parameter information.

        Returns:
            FfmpNycRunningAvgParameter: The loaded parameter instance.
        """
        node = model.nodes[data.pop("node")]
        max_avg_delivery = load_parameter(model, data.pop("max_avg_delivery"))
        return cls(model, node, max_avg_delivery, **data)


### have to register the custom parameter so Pywr recognizes it
FfmpNycRunningAvgParameter.register()


class FfmpNjRunningAvgParameter(Parameter):
    """
    Enforces the constraint on NJ deliveries from the FFMP, based on a running average.

    Args:
        model (Model): The Pywr model instance.
        node (Node): The node associated with the parameter.

    Attributes:
        node (Node): The node associated with the parameter.
        max_avg_delivery (float): The maximum average delivery value.
        max_daily_delivery (float): The maximum daily delivery value.
        drought_factor (Parameter): The drought factor parameter.
        max_delivery (ndarray): An array to hold the parameter state.
        current_drought_factor (ndarray): An array to hold the current drought factor.
        previous_drought_factor (ndarray): An array to hold the previous drought factor.

    Methods:
        setup(): Allocates arrays to hold the parameter state and drought factors.
        reset(): Resets the amount remaining in all states to the initial value.
        value(timestep, scenario_index): Returns the current volume remaining for the scenario.
        after(): Updates the parameter requirement based on running average and updates the date for tomorrow.

    Class Methods:
        load(model, data): Loads the parameter from model and data dictionary.

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
        super().__init__(model, **kwargs)
        self.node = node
        self.max_avg_delivery = max_avg_delivery.get_constant_value()
        self.max_daily_delivery = max_daily_delivery.get_constant_value()
        self.drought_factor = drought_factor
        self.children.add(max_avg_delivery)
        self.children.add(max_daily_delivery)
        self.children.add(drought_factor)

    def setup(self):
        """Allocate an array to hold the parameter state."""
        super().setup()
        num_scenarios = len(self.model.scenarios.combinations)
        self.max_delivery = np.empty([num_scenarios], np.float64)
        self.current_drought_factor = np.empty([num_scenarios], np.float64)
        self.previous_drought_factor = np.empty([num_scenarios], np.float64)

    def reset(self):
        """Resets the amount remaining in all states to the initial value."""
        self.timestep = self.model.timestepper.delta
        self.datetime = pd.to_datetime(self.model.timestepper.start)
        self.max_delivery[...] = self.max_avg_delivery * self.timestep
        self.current_drought_factor[...] = 1.0
        self.previous_drought_factor[...] = 1.0

    def value(self, timestep, scenario_index):
        """
        Returns the current volume remaining for the scenario.

        Args:
            timestep (Timestep): The current timestep.
            scenario_index (ScenarioIndex): The scenario index.

        Returns:
            float: The current volume remaining for the scenario.

        """
        return self.max_delivery[scenario_index.global_id]

    def after(self):
        """Updates the parameter requirement based on running average and updates the date for tomorrow."""
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
    Decides whether an NYC reservoir's release is dictated by its own
    storage (in the case of flood operations) or the aggregate storage across the three NYC reservoirs
    (in the case of normal or drought operations). It returns the "factor" which is a multiplier to baseline release
    value for the reservoir.
    See 8/30/2022 comment on this GitHub issue for the equation & logic:
    https://github.com/users/ahamilton144/projects/1/views/1?pane=issue&itemId=7839486

    Args:
        model (Model): The Pywr model instance.
        node (Node): The node associated with the parameter.

    Attributes:
        node (Node): The node associated with the parameter.
        drought_level_agg_nyc (Parameter): The drought level aggregate NYC parameter.
        mrf_drought_factor_agg_reservoir (Parameter): The MRF drought factor aggregate reservoir parameter.
        mrf_drought_factor_individual_reservoir (Parameter): The MRF drought factor individual reservoir parameter.

    Methods:
        value(timestep, scenario_index): Returns the overall release factor for the NYC reservoir.

    Class Methods:
        load(model, data): Loads the parameter from model and data dictionary.

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
        Returns the overall release factor for the NYC reservoir, depending on whether it is flood stage
        (in which case we use the reservoirs individual storage) or normal/drought stage
        (in which case we use aggregate storage across the NYC reservoirs).

        Args:
            timestep (Timestep): The current timestep.
            scenario_index (ScenarioIndex): The scenario index.

        Returns:
            float: The overall release factor for the NYC reservoir.
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
        Loads the parameter from model and data dictionary.

        Args:
            model (Model): The Pywr model instance.
            data (dict): The data dictionary containing the parameter information.

        Returns:
            NYCCombinedReleaseFactor: The loaded parameter instance.
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
    Calculates any excess flood control releases needed to reduce NYC reservoir's storage back down to
    level 1b/1c boundary within 7 days. See Page 21 FFMP for details.

    Attributes:
        node (Node): The node associated with the parameter.
        drought_level_reservoir (Parameter): The drought level reservoir parameter.
        level1c (Parameter): The level 1c parameter.
        volume_reservoir (Parameter): The volume reservoir parameter.
        max_volume_reservoir (Parameter): The max volume reservoir parameter.
        weekly_rolling_mean_flow_reservoir (Parameter): The weekly rolling mean flow reservoir parameter.
        max_release_reservoir (Parameter): The max release reservoir parameter.
        mrf_target_individual_reservoir (Parameter): The MRF target individual reservoir parameter.

    Methods:
        value(timestep, scenario_index): Returns the excess flood control releases needed to reduce NYC reservoir's storage.

    Class Methods:
        load(model, data): Loads the parameter from model and data dictionary.
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
        Returns the excess flood control releases needed to reduce NYC reservoir's storage back down to
        level 1b/1c boundary within 7 days.

        Args:
            timestep (Timestep): The current timestep.
            scenario_index (ScenarioIndex): The scenario index.

        Returns:
            float: Flood release for given reservoir, timestep, and scenario
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
        Loads the parameter from model and data dictionary.

        Args:
            model (Model): The Pywr model instance.
            data (dict): The data dictionary containing the parameter information.

        Returns:
            NYCCombinedReleaseFactor: The loaded parameter instance.
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
    Calculates the total releases needed from FFMP reservoirs to meet Montague or Trenton target,
    above and beyond their individual direct mandated releases and flood control releases.

    Attributes:
        model (Model): The Pywr model instance.
        mrf (str): The MRF target for which we are calculating the total release needed.
        step (int): The step in the calculation process, to account for lag travel to downstream sites.
        predicted_nonnyc_gage_flow_mrf (Parameter): The predicted non-NYC gage flow MRF parameter.
        predicted_demand_nj (Parameter): The predicted demand NJ parameter.
        mrf_target_flow (Parameter): The MRF target flow parameter.
        release_needed_mrf_montague (Parameter): The release needed MRF Montague parameter.
        mrf_target_individual_nyc (Parameter): The MRF target individual NYC parameter.
        flood_release_nyc (Parameter): The flood release NYC parameter.
        previous_release_reservoirs (list): The list of previous release reservoirs parameters.

    Methods:
        value(timestep, scenario_index): Returns the total releases needed from FFMP reservoirs to meet Montague or Trenton target.
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
        """Returns the total releases needed from FFMP reservoirs to meet Montague or Trenton target,
        above and beyond their individual direct mandated releases and flood control releases.

        Args:
            timestep (Timestep): The current timestep.
            scenario_index (ScenarioIndex): The scenario index.

        Returns:
            float: NYC releases needed to help meet flow target
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
        Loads the parameter from model and data dictionary.

        Args:
            model (Model): The Pywr model instance.
            data (dict): The data dictionary containing the parameter information.
                --- mrf: either 'delMontague' or 'delTrenton'
                --- days_ahead: int between 1 & 4 representing the number of days ahead we are trying to meet flow req
        Returns:

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
            ### for step 3, we only need to acct for can/pep/nev previous releases, not any NYC indiv/flood releases this time step
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
            ### for step 3, we only need to acct for can/pep/nev previous releases, not any NYC indiv/flood releases this time step
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
    to meet Montague & Trenton targets. Accounts for max release constraints at each reservoir.

    Attributes:
        model (Model): The Pywr model instance.
        reservoir (str): The reservoir associated with the parameter.
        nodes (list): The list of nodes associated with the parameter.
        parameters (dict): The dictionary of parameters associated with the parameter.
        max_vol_reservoirs (list): The list of max volume reservoir parameters.
        vol_reservoirs (list): The list of volume reservoir parameters.
        flow_reservoirs (list): The list of flow reservoir parameters.
        max_release_reservoirs (list): The list of max release reservoir parameters.
        mrf_target_individual_reservoirs (list): The list of MRF target individual reservoir parameters.
        flood_release_reservoirs (list): The list of flood release reservoir parameters.
        num_reservoirs (int): The number of coordinating reservoirs.

    Methods:
        split_required_mrf_across_nyc_reservoirs(requirement_total, scenario_index):
        value(timestep, scenario_index):
    """

    def __init__(self, model, reservoir, nodes, parameters, **kwargs):
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
        """ """
        sid = scenario_index.global_id
        ### calculate contributions for all 3 NYC reservoirs in consistent way.
        ### Note: ideally we would only do this once. But may not be possible to have parameter with array output,
        ###       so for now we just repeat this procedure for each reservoir.

        ### first calculate contributions to Trenton&Montague flow targets based on volume balancing formula.
        ### These are above and beyond what is needed for individual FFMP mandated releases
        nyc_requirement_total = self.total_agg_mrf_montagueTrenton_step1.get_value(
            scenario_index
        ) - self.lower_basin_agg_mrf_trenton_step1.get_value(scenario_index)

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
        """Setup the parameter."""
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
    """ """

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
        """Returns the total flow needed from Neversink to meet Montague and Trenton targets,
        above and beyond their individual direct mandated releases.

        Args:
            timestep (Timestep): The current timestep.
            scenario_index (ScenarioIndex): The scenario index.

        Returns:
            float: The total flow needed from Neversink to meet Montague and Trenton targets.
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
                self.total_agg_mrf_montagueTrenton.get_value(scenario_index)
                - self.lower_basin_agg_mrf_trenton.get_value(scenario_index),
            ),
            0,
        )
        return release_neversink

    @classmethod
    def load(cls, model, data):
        """
        Loads the parameter from model and data dictionary.

        Args:
            model (Model): The Pywr model instance.
            data (dict): The data dictionary containing the parameter information.

        Returns:
            VolBalanceNYCDemandTarget: The loaded parameter instance.
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
    Updates the contribution to NYC deliveries made by each of the NYC
    reservoirs, in such a way as to balance the relative storages across the three reservoirs.
    See comments on this GitHub issue for the equations & logic:
    https://github.com/users/ahamilton144/projects/1/views/1?pane=issue&itemId=7840442

    Args:
        model (Model): The Pywr model instance.
        node (Node): The node associated with the parameter.

    Attributes:
        reservoir (str): The reservoir associated with the parameter.
        node (Node): The node associated with the parameter.
        max_volume_agg_nyc (Parameter): The maximum volume aggregate NYC parameter.
        volume_agg_nyc (Parameter): The volume aggregate NYC parameter.
        max_flow_delivery_nyc (Parameter): The maximum flow delivery NYC parameter.
        flow_agg_nyc (Parameter): The flow aggregate NYC parameter.
        max_vol_reservoir (Parameter): The maximum volume reservoir parameter.
        vol_reservoir (Parameter): The volume reservoir parameter.
        flow_reservoir (Parameter): The flow reservoir parameter.

    Methods:
        value(timestep, scenario_index): Returns the target NYC delivery for this reservoir to balance storages across reservoirs.

    Class Methods:
        load(model, data): Loads the parameter from model and data dictionary.

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
        """Setup the parameter."""
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
