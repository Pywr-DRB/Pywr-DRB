"""
This file contains different class objects which are used to construct custom Pywr parameters.

The parameters created here are used to implement the flexible flow management program (FFMP)
for the three NYC reservoirs.
"""

import numpy as np
import pandas as pd
from pywr.parameters import Parameter, load_parameter
from utils.lists import reservoir_list_nyc
from utils.constants import epsilon

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
        """Updates the parameter requirement based on running average and updates the date for tomorrow.
        """
        ### if it is may 31, reset max delivery to original value (800)
        if self.datetime.month == 5 and self.datetime.day == 31:
            self.max_delivery[...] = self.max_avg_delivery * self.timestep
        ### else update the requirement based on running average: maxdel_t = maxdel_{t-1} - flow_{t-1} + max_avg_del
        else:
            self.max_delivery += (self.max_avg_delivery - self.node.flow) * self.timestep
            self.max_delivery[self.max_delivery < 0] = 0  # max delivery cannot be less than zero
        ### update date for tomorrow
        self.datetime += pd.Timedelta(1, 'd')

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
    def __init__(self, model, node, max_avg_delivery, max_daily_delivery, drought_factor, **kwargs):
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
                    self.max_delivery[s] = self.max_avg_delivery * factor * self.timestep
                ### else, update requirement based on running average: maxdel_t = maxdel_{t-1} - flow_{t-1} + max_avg_del
                else:
                    self.max_delivery[s] += (self.max_avg_delivery * factor - self.node.flow[s]) * self.timestep
            ### if today's drought factor is different from yesterday, we always reset running avg
            else:
                self.max_delivery[s] = self.max_avg_delivery * factor * self.timestep

        ### max delivery cannot be less than zero
        self.max_delivery[self.max_delivery < 0] = 0
        ### max delivery cannot be larger than daily limit
        self.max_delivery[self.max_delivery > self.max_daily_delivery] = self.max_daily_delivery
        ### update date & previous factor for tomorrow
        self.datetime += pd.Timedelta(1, 'd')
        self.previous_drought_factor[...] = self.current_drought_factor[...]

    @classmethod
    def load(cls, model, data):
        node = model.nodes[data.pop("node")]
        max_avg_delivery = load_parameter(model, data.pop("max_avg_delivery"))
        max_daily_delivery = load_parameter(model, data.pop("max_daily_delivery"))
        drought_factor = load_parameter(model, data.pop('drought_factor'))
        return cls(model, node, max_avg_delivery, max_daily_delivery, drought_factor, **data)

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
    def __init__(self, model, node, drought_level_agg_nyc, mrf_drought_factor_agg_reservoir,
                 mrf_drought_factor_individual_reservoir, **kwargs):
        super().__init__(model, **kwargs)
        self.node = node
        self.drought_level_agg_nyc = drought_level_agg_nyc
        self.mrf_drought_factor_agg_reservoir = mrf_drought_factor_agg_reservoir
        self.mrf_drought_factor_individual_reservoir = mrf_drought_factor_individual_reservoir
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

        return min(max(self.drought_level_agg_nyc.get_value(scenario_index) - 2, 0), 1) * \
                    self.mrf_drought_factor_agg_reservoir.get_value(scenario_index) + \
               min(max(3 - self.drought_level_agg_nyc.get_value(scenario_index), 0), 1) * \
                    self.mrf_drought_factor_individual_reservoir.get_value(scenario_index)

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
        reservoir = reservoir.split('_')[1]
        drought_level_agg_nyc = load_parameter(model, f'drought_level_agg_nyc')
        mrf_drought_factor_agg_reservoir = load_parameter(model, f'mrf_drought_factor_agg_{reservoir}')
        mrf_drought_factor_individual_reservoir = load_parameter(model, f'mrf_drought_factor_individual_{reservoir}')
        return cls(model, node, drought_level_agg_nyc, mrf_drought_factor_agg_reservoir,
                   mrf_drought_factor_individual_reservoir, **data)

### have to register the custom parameter so Pywr recognizes it
NYCCombinedReleaseFactor.register()





class NYCFloodRelease(Parameter):
    """
    Calculates any excess flood control releases needed to reduce NYC reservoir's storage back down to
    level 1b/1c boundary within 7 days. See Page 21 FFMP for details.
    """
    def __init__(self, model, node, drought_level_reservoir, level1c, volume_reservoir, max_volume_reservoir,
                    weekly_rolling_mean_flow_reservoir, max_release_reservoir, mrf_target_individual_reservoir, **kwargs):
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

        """
        ### extra flood releases needed if we are in level 1a or 1b
        if self.drought_level_reservoir.get_value(scenario_index) < 2:
            ## calculate the total excess volume needed to be release in next 7 days:
            ## assume for now this is just the current storage minus the level 1b/1c boundary, plus 7 * 7-day rolling avg inflow.
            excess_volume = (self.volume_reservoir.get_value(scenario_index) - (self.level1c.get_value(scenario_index) * \
                                                       self.max_volume_reservoir.get_value(scenario_index) ) + \
                             self.weekly_rolling_mean_flow_reservoir.get_value(scenario_index) * 7)
            flood_release = max(min(excess_volume / 7 - self.mrf_target_individual_reservoir.get_value(scenario_index),
                                    self.max_release_reservoir.get_value(scenario_index) - \
                                    self.mrf_target_individual_reservoir.get_value(scenario_index)),
                                0)

            # ### calculate the total excess volume needed to be release today:
            # ### assume for now this is just the current storage minus the level 1b/1c boundary, plus today's inflow
            # ### Could probably improve this by using forecast into future
            # excess_volume = (self.volume_reservoir.get_value(scenario_index) - (self.level1c.get_value(scenario_index) * \
            #                                            self.max_volume_reservoir.get_value(scenario_index) ) + \
            #                  self.flow_reservoir.get_value(scenario_index))
            # flood_release = min(max(excess_volume - self.mrf_target_individual_reservoir.get_value(scenario_index),
            #                         0), self.max_release_reservoir.get_value(scenario_index) -
            #                     self.mrf_target_individual_reservoir.get_value(scenario_index))

            return flood_release

        else:
            return 0.



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
        reservoir = reservoir.split('_')[1]
        drought_level_reservoir = load_parameter(model, f'drought_level_agg_nyc')
        level1c = load_parameter(model, 'level1c')
        volume_reservoir = load_parameter(model, f'volume_{reservoir}')
        max_volume_reservoir = load_parameter(model, f'max_volume_{reservoir}')
        weekly_rolling_mean_flow_reservoir = load_parameter(model, f'weekly_rolling_mean_flow_{reservoir}')
        max_release_reservoir = load_parameter(model, f'flood_max_release_{reservoir}')
        mrf_target_individual_reservoir = load_parameter(model, f'mrf_target_individual_{reservoir}')

        return cls(model, node, drought_level_reservoir, level1c, volume_reservoir, max_volume_reservoir,
                    weekly_rolling_mean_flow_reservoir, max_release_reservoir, mrf_target_individual_reservoir, **data)

### have to register the custom parameter so Pywr recognizes it
NYCFloodRelease.register()






class VolBalanceNYCDownstreamMRFTargetAgg_step1CanPep(Parameter):
    """
    Calculates the total releases from NYC reservoirs needed to meet the Montague and Trenton flow targets,
    after subtracting out flows from the rest of the basin, and adding max deliveries to NJ, and
    subtracting mandated individual FFMP releases & flood releases for NYC reservoirs.

    Args:
        model (Model): The Pywr model instance.

    Attributes:
        predicted_nonnyc_gage_flow_delMontague_lag2 (Parameter): The volume balance flow aggregate non-NYC Montague delivery parameter.
        predicted_nonnyc_gage_flow_delTrenton_lag4 (Parameter): The volume balance flow aggregate non-NYC Trenton delivery parameter.
        mrf_target_delMontague (Parameter): The MRF target Montague delivery parameter.
        predicted_demand_nj_lag4 (Parameter): The predicted lagged demand to NJ parameter.
        mrf_target_delTrenton (Parameter): The MRF target Trenton delivery parameter.

    Methods:
        value(timestep, scenario_index): Returns the total flow needed from NYC reservoirs to meet Montague and Trenton targets.

    Class Methods:
        load(model, data): Loads the parameter from model and data dictionary.

    """
    def __init__(self, model, predicted_nonnyc_gage_flow_delMontague_lag2, mrf_target_delMontague,\
                 predicted_nonnyc_gage_flow_delTrenton_lag4, predicted_demand_nj_lag4, mrf_target_delTrenton,
                 mrf_target_individual_agg_nyc, flood_release_agg_nyc, **kwargs):
        super().__init__(model, **kwargs)
        self.predicted_nonnyc_gage_flow_delMontague_lag2 = predicted_nonnyc_gage_flow_delMontague_lag2
        self.mrf_target_delMontague = mrf_target_delMontague
        self.predicted_nonnyc_gage_flow_delTrenton_lag4 = predicted_nonnyc_gage_flow_delTrenton_lag4
        self.predicted_demand_nj_lag4 = predicted_demand_nj_lag4
        self.mrf_target_delTrenton = mrf_target_delTrenton
        self.mrf_target_individual_agg_nyc = mrf_target_individual_agg_nyc
        self.flood_release_agg_nyc = flood_release_agg_nyc

        self.children.add(predicted_nonnyc_gage_flow_delMontague_lag2)
        self.children.add(mrf_target_delMontague)
        self.children.add(predicted_nonnyc_gage_flow_delTrenton_lag4)
        self.children.add(predicted_demand_nj_lag4)
        self.children.add(mrf_target_delTrenton)
        self.children.add(mrf_target_individual_agg_nyc)
        self.children.add(flood_release_agg_nyc)

    def value(self, timestep, scenario_index):
        """Returns the total flow needed from NYC reservoirs to meet Montague and Trenton targets,
        above and beyond their individual direct mandated releases and flood control releases.

        Args:
            timestep (Timestep): The current timestep.
            scenario_index (ScenarioIndex): The scenario index.

        Returns:
            float: The total flow needed from NYC reservoirs to meet Montague and Trenton targets.
        """
        req_delMontague = max(self.mrf_target_delMontague.get_value(scenario_index) -
                              self.predicted_nonnyc_gage_flow_delMontague_lag2.get_value(scenario_index) -
                              self.mrf_target_individual_agg_nyc.get_value(scenario_index) -
                              self.flood_release_agg_nyc.get_value(scenario_index),
                              0.)
        req_delTrenton = max(self.mrf_target_delTrenton.get_value(scenario_index) -
                             self.predicted_nonnyc_gage_flow_delTrenton_lag4.get_value(scenario_index) -
                             self.mrf_target_individual_agg_nyc.get_value(scenario_index) -
                             self.flood_release_agg_nyc.get_value(scenario_index) +
                             self.predicted_demand_nj_lag4.get_value(scenario_index),
                             0.)
        return max(req_delMontague, req_delTrenton)

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
        predicted_nonnyc_gage_flow_delMontague_lag2 = load_parameter(model, 'predicted_nonnyc_gage_flow_delMontague_lag2')
        mrf_target_delMontague = load_parameter(model, 'mrf_target_delMontague')
        predicted_nonnyc_gage_flow_delTrenton_lag4 = load_parameter(model, 'predicted_nonnyc_gage_flow_delTrenton_lag4')
        predicted_demand_nj_lag4 = load_parameter(model, 'predicted_demand_nj_lag4')
        mrf_target_delTrenton = load_parameter(model, 'mrf_target_delTrenton')
        mrf_target_individual_agg_nyc = load_parameter(model, 'mrf_target_individual_agg_nyc')
        flood_release_agg_nyc = load_parameter(model, 'flood_release_agg_nyc')

        return cls(model, predicted_nonnyc_gage_flow_delMontague_lag2, mrf_target_delMontague,\
                   predicted_nonnyc_gage_flow_delTrenton_lag4, predicted_demand_nj_lag4, mrf_target_delTrenton,
                   mrf_target_individual_agg_nyc, flood_release_agg_nyc, **data)

### have to register the custom parameter so Pywr recognizes it
VolBalanceNYCDownstreamMRFTargetAgg_step1CanPep.register()




### updated parameter to consistently assign release targets for all 3 reservoirs, above and beyond individual mandated releases.
###     - total release needed for Montague & Trenton targets from FFMP
###     - accounts for max release constraints at each reservoir

class VolBalanceNYCDownstreamMRF_step1CanPep(Parameter):

    def __init__(self, model, reservoir, nodes, max_volume_agg_nyc, volume_agg_nyc, volbalance_relative_mrf_montagueTrenton_step1CanPep,
                 flow_agg_nyc, max_vol_reservoirs, vol_reservoirs, flow_reservoirs, max_release_reservoirs,
                 mrf_target_individual_reservoirs, mrf_target_individual_agg_nyc,
                 flood_release_reservoirs, flood_release_agg_nyc, **kwargs):
        super().__init__(model, **kwargs)
        self.reservoir = reservoir
        self.nodes = nodes
        self.num_reservoirs = len(nodes)
        self.max_volume_agg_nyc = max_volume_agg_nyc
        self.volume_agg_nyc = volume_agg_nyc
        self.volbalance_relative_mrf_montagueTrenton_step1CanPep = volbalance_relative_mrf_montagueTrenton_step1CanPep
        self.flow_agg_nyc = flow_agg_nyc
        self.max_vol_reservoirs = max_vol_reservoirs
        self.vol_reservoirs = vol_reservoirs
        self.flow_reservoirs = flow_reservoirs
        self.max_release_reservoirs = max_release_reservoirs
        self.mrf_target_individual_reservoirs = mrf_target_individual_reservoirs
        self.mrf_target_individual_agg_nyc  = mrf_target_individual_agg_nyc
        self.flood_release_reservoirs = flood_release_reservoirs
        self.flood_release_agg_nyc = flood_release_agg_nyc

        self.children.add(max_volume_agg_nyc)
        self.children.add(volume_agg_nyc)
        self.children.add(volbalance_relative_mrf_montagueTrenton_step1CanPep)
        self.children.add(flow_agg_nyc)
        self.children.add(mrf_target_individual_agg_nyc)
        self.children.add(flood_release_agg_nyc)
        for i in range(len(reservoir_list_nyc)):
            self.children.add(max_vol_reservoirs[i])
            self.children.add(vol_reservoirs[i])
            self.children.add(flow_reservoirs[i])
            self.children.add(max_release_reservoirs[i])
            self.children.add(mrf_target_individual_reservoirs[i])
            self.children.add(flood_release_reservoirs[i])


    def value(self, timestep, scenario_index):
        """
        """
        sid = scenario_index.global_id
        ### calculate contributions for all 3 NYC reservoirs in consistent way.
        ### Note: ideally we would only do this once. But may not be possible to have parameter with array output,
        ###       so for now we just repeat this procedure for each reservoir.

        ### first calculate contributions to Trenton&Montague flow targets based on volume balancing formula.
        ### These are above and beyond what is needed for individual FFMP mandated releases
        requirement_total = self.volbalance_relative_mrf_montagueTrenton_step1CanPep.get_value(scenario_index)
        max_releases_reservoirs = [max(self.max_release_reservoirs[i].get_value(scenario_index) - \
                                       self.mrf_target_individual_reservoirs[i].get_value(scenario_index) -
                                       self.flood_release_reservoirs[i].get_value(scenario_index),
                                       0) for i in range(self.num_reservoirs)]
        targets = [-1] * self.num_reservoirs
        for i in range(self.num_reservoirs):
            targets[i] = self.vol_reservoirs[i].get_value(scenario_index) + \
                         self.flow_reservoirs[i].get_value(scenario_index) - \
                         self.mrf_target_individual_reservoirs[i].get_value(scenario_index) - \
                         self.flood_release_reservoirs[i].get_value(scenario_index) - \
                         (self.max_vol_reservoirs[i].get_value(scenario_index) / \
                          self.max_volume_agg_nyc.get_value(scenario_index)) * \
                         (self.volume_agg_nyc.get_value(scenario_index) + \
                          self.flow_agg_nyc.get_value(scenario_index) - \
                          self.mrf_target_individual_agg_nyc.get_value(scenario_index) - \
                          self.flood_release_agg_nyc.get_value(scenario_index) - \
                          requirement_total)
            ### enforce nonnegativity and reservoir max release constraint
            targets[i] = min(max(targets[i], 0), max_releases_reservoirs[i])

        ### sum total release across 3 reservoirs. if this is less than volbalance_relative_mrf_montagueTrenton_step1CanPep,
        ### that means one of the reservoirs had negative value or exceeded max release above
        ### -> rescale unconstrained reservoirs to counteract
        target_sum = sum(targets)
        fully_constrained = False
        count = 0
        while requirement_total - epsilon > target_sum and not fully_constrained:
            increasable_flow = 0
            ### find the total "increasable" flow that is not coming from reservoirs with 0 release or max_release
            for i in range(self.num_reservoirs):
                if targets[i] < max_releases_reservoirs[i]:
                    increasable_flow += targets[i]
            if increasable_flow > epsilon:
                for i in range(self.num_reservoirs):
                    targets[i] = min(targets[i] * requirement_total / increasable_flow, max_releases_reservoirs[i])
            else:
                fully_constrained = True
            target_sum = sum(targets)
            count += 1
            if count > 5:
                print(count, requirement_total, target_sum, increasable_flow, targets, max_releases_reservoirs)

        ### now return target for the reservoir of interest
        for i in range(self.num_reservoirs):
            if self.reservoir == reservoir_list_nyc[i]:
                return targets[i]


    @classmethod
    def load(cls, model, data):
        """
        """
        reservoir = data.pop("node")
        reservoir = reservoir.split('_')[1]
        nodes = [model.nodes[f'reservoir_{reservoir}'] for reservoir in reservoir_list_nyc]
        max_volume_agg_nyc = load_parameter(model, 'max_volume_agg_nyc')
        volume_agg_nyc = load_parameter(model, 'volume_agg_nyc')
        volbalance_relative_mrf_montagueTrenton_step1CanPep = load_parameter(model, 'volbalance_relative_mrf_montagueTrenton_step1CanPep')
        flow_agg_nyc = load_parameter(model, 'flow_agg_nyc')
        max_vol_reservoirs = [load_parameter(model, f'max_volume_{reservoir}') for reservoir in reservoir_list_nyc]
        vol_reservoirs = [load_parameter(model, f'volume_{reservoir}') for reservoir in reservoir_list_nyc]
        flow_reservoirs = [load_parameter(model, f'flow_{reservoir}') for reservoir in reservoir_list_nyc]
        max_release_reservoirs = [load_parameter(model, f'controlled_max_release_{reservoir}') for reservoir in reservoir_list_nyc]
        mrf_target_individual_reservoirs = [load_parameter(model, f'mrf_target_individual_{reservoir}') for reservoir in reservoir_list_nyc]
        mrf_target_individual_agg_nyc = load_parameter(model, 'mrf_target_individual_agg_nyc')
        flood_release_reservoirs = [load_parameter(model, f'flood_release_{reservoir}') for reservoir in reservoir_list_nyc]
        flood_release_agg_nyc = load_parameter(model, 'flood_release_agg_nyc')

        return cls(model, reservoir, nodes, max_volume_agg_nyc, volume_agg_nyc, volbalance_relative_mrf_montagueTrenton_step1CanPep,
                   flow_agg_nyc, max_vol_reservoirs, vol_reservoirs, flow_reservoirs, max_release_reservoirs,
                   mrf_target_individual_reservoirs, mrf_target_individual_agg_nyc,
                   flood_release_reservoirs, flood_release_agg_nyc, **data)

### have to register the custom parameter so Pywr recognizes it
VolBalanceNYCDownstreamMRF_step1CanPep.register()





class VolBalanceNYCDownstreamMRF_step2Nev(Parameter):
    """
    Calculates the total releases from Neversink needed to meet the Montague and Trenton flow targets,
    after subtracting out predicted flows from the rest of the basin plus Cannonsville & Pepacton releases from day before
    and adding max deliveries to NJ, and subtracting mandated individual FFMP releases for Neversink

    Args:
        model (Model): The Pywr model instance.

    Attributes:
        predicted_nonnyc_gage_flow_delMontague_lag2 (Parameter): The volume balance flow aggregate non-NYC Montague delivery parameter.
        mrf_target_delMontague (Parameter): The MRF target Montague delivery parameter.
        volbalance_flow_agg_nonnyc_delTrenton (Parameter): The volume balance flow aggregate non-NYC Trenton delivery parameter.
        predicted_demand_nj_lag3 (Parameter): The lagged predicted NJ demand in 3 days
        mrf_target_delTrenton (Parameter): The MRF target Trenton delivery parameter.

    Methods:
        value(timestep, scenario_index): Returns the total flow needed from NYC reservoirs to meet Montague and Trenton targets.

    Class Methods:
        load(model, data): Loads the parameter from model and data dictionary.

    """
    def __init__(self, model, predicted_nonnyc_gage_flow_delMontague_lag1, mrf_target_delMontague,\
                 predicted_nonnyc_gage_flow_delTrenton_lag3, predicted_demand_nj_lag3, mrf_target_delTrenton,
                 mrf_target_individual_neversink, flood_release_neversink, max_release_neversink,
                 prev_release_reservoirs, **kwargs):
        super().__init__(model, **kwargs)
        self.predicted_nonnyc_gage_flow_delMontague_lag1 = predicted_nonnyc_gage_flow_delMontague_lag1
        self.mrf_target_delMontague = mrf_target_delMontague
        self.predicted_nonnyc_gage_flow_delTrenton_lag3 = predicted_nonnyc_gage_flow_delTrenton_lag3
        self.predicted_demand_nj_lag3 = predicted_demand_nj_lag3
        self.mrf_target_delTrenton = mrf_target_delTrenton
        self.mrf_target_individual_neversink = mrf_target_individual_neversink
        self.flood_release_neversink = flood_release_neversink
        self.max_release_neversink = max_release_neversink
        self.prev_release_reservoirs = prev_release_reservoirs

        self.children.add(predicted_nonnyc_gage_flow_delMontague_lag1)
        self.children.add(mrf_target_delMontague)
        self.children.add(predicted_nonnyc_gage_flow_delTrenton_lag3)
        self.children.add(predicted_demand_nj_lag3)
        self.children.add(mrf_target_delTrenton)
        self.children.add(mrf_target_individual_neversink)
        self.children.add(flood_release_neversink)
        self.children.add(max_release_neversink)
        for i in range(2):
            self.children.add(prev_release_reservoirs[i])

    def value(self, timestep, scenario_index):
        """Returns the total flow needed from NYC reservoirs to meet Montague and Trenton targets,
        above and beyond their individual direct mandated releases.

        Args:
            timestep (Timestep): The current timestep.
            scenario_index (ScenarioIndex): The scenario index.

        Returns:
            float: The total flow needed from NYC reservoirs to meet Montague and Trenton targets.
        """
        max_release_neversink = max(self.max_release_neversink.get_value(scenario_index) - \
                                    self.mrf_target_individual_neversink.get_value(scenario_index) - \
                                    self.flood_release_neversink.get_value(scenario_index), 0)
        prev_release_CanPep_total = sum([self.prev_release_reservoirs[i].get_value(scenario_index) for i in range(2)])
        req_delMontague = max(min(self.mrf_target_delMontague.get_value(scenario_index) - \
                                  self.predicted_nonnyc_gage_flow_delMontague_lag1.get_value(scenario_index) - \
                                  self.mrf_target_individual_neversink.get_value(scenario_index) - \
                                  self.flood_release_neversink.get_value(scenario_index) - \
                                  prev_release_CanPep_total,
                                  max_release_neversink),
                              0)
        req_delTrenton = max(min(self.mrf_target_delTrenton.get_value(scenario_index) - \
                                 self.predicted_nonnyc_gage_flow_delTrenton_lag3.get_value(scenario_index) - \
                                 self.mrf_target_individual_neversink.get_value(scenario_index) - \
                                 self.flood_release_neversink.get_value(scenario_index) - \
                                 prev_release_CanPep_total + \
                                 self.predicted_demand_nj_lag3.get_value(scenario_index),
                                 max_release_neversink),
                             0)
        return max(req_delMontague, req_delTrenton)

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
        predicted_nonnyc_gage_flow_delMontague_lag1 = load_parameter(model, 'predicted_nonnyc_gage_flow_delMontague_lag1')
        mrf_target_delMontague = load_parameter(model, 'mrf_target_delMontague')
        predicted_nonnyc_gage_flow_delTrenton_lag3 = load_parameter(model, 'predicted_nonnyc_gage_flow_delTrenton_lag3')
        predicted_demand_nj_lag3 = load_parameter(model, 'predicted_demand_nj_lag3')
        mrf_target_delTrenton = load_parameter(model, 'mrf_target_delTrenton')
        mrf_target_individual_neversink = load_parameter(model, 'mrf_target_individual_neversink')
        flood_release_neversink = load_parameter(model, 'flood_release_neversink')
        max_release_neversink = load_parameter(model, 'controlled_max_release_neversink')
        prev_release_reservoirs = [load_parameter(model, f'prev_release_{reservoir}') for reservoir in ['cannonsville','pepacton']]

        return cls(model, predicted_nonnyc_gage_flow_delMontague_lag1, mrf_target_delMontague,\
                   predicted_nonnyc_gage_flow_delTrenton_lag3, predicted_demand_nj_lag3, mrf_target_delTrenton,
                   mrf_target_individual_neversink, flood_release_neversink, max_release_neversink,
                   prev_release_reservoirs, **data)

### have to register the custom parameter so Pywr recognizes it
VolBalanceNYCDownstreamMRF_step2Nev.register()






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
    def __init__(self, model, reservoir, nodes, max_volume_agg_nyc, volume_agg_nyc, max_flow_delivery_nyc,
                 flow_agg_nyc, max_vol_reservoirs, vol_reservoirs, flow_reservoirs, hist_max_flow_delivery_nycs,
                 downstream_release_target_reservoirs, flood_release_reservoirs, **kwargs):
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
            self.children.add(downstream_release_target_reservoirs[i])
            self.children.add(flood_release_reservoirs[i])



    def value(self, timestep, scenario_index):
        sid = scenario_index.global_id
        ### calculate diversions from all 3 NYC reservoirs in consistent way.
        ### Note: ideally we would only do this once. But may not be possible to have parameter with array output,
        ###       so for now we just repeat this procedure for each reservoir.

        ### first calculate the contributions to NYC delivery for this reservoir to balance storages across reservoirs
        requirement_total = self.max_flow_delivery_nyc.get_value(scenario_index)
        max_diversions = [self.hist_max_flow_delivery_nycs[i].get_value(scenario_index) for i in range(self.num_reservoirs)]
        mrf_target_total = sum([self.downstream_release_target_reservoirs[i].get_value(scenario_index) for i in range(self.num_reservoirs)])
        flood_release_total = sum([self.flood_release_reservoirs[i].get_value(scenario_index) for i in range(self.num_reservoirs)])

        targets = [-1] * self.num_reservoirs
        for i in range(self.num_reservoirs):
            targets[i] = self.vol_reservoirs[i].get_value(scenario_index) + \
                         self.flow_reservoirs[i].get_value(scenario_index) - \
                         self.downstream_release_target_reservoirs[i].get_value(scenario_index) - \
                         self.flood_release_reservoirs[i].get_value(scenario_index) - \
                         (self.max_vol_reservoirs[i].get_value(scenario_index) / \
                          self.max_volume_agg_nyc.get_value(scenario_index)) * \
                         (self.volume_agg_nyc.get_value(scenario_index) + \
                          self.flow_agg_nyc.get_value(scenario_index) - \
                          mrf_target_total - flood_release_total - requirement_total)
            ### enforce nonnegativity and reservoir max release constraint
            targets[i] = min(max(targets[i], 0), max_diversions[i])



        ### sum total diversions across 3 reservoirs. if this is less than requirement_total,
        ### that means one of the reservoirs had negative value or exceeded max diversion above
        ### -> rescale unconstrained reservoirs to counteract
        target_sum = sum(targets)
        fully_constrained = False
        count = 0
        while requirement_total - epsilon > target_sum and not fully_constrained:
            increasable_flow = 0
            ### find the total "increasable" flow that is not coming from reservoirs with 0 release or max_release
            for i in range(self.num_reservoirs):
                if targets[i] < max_diversions[i]:
                    increasable_flow += targets[i]
            if increasable_flow > epsilon:
                for i in range(self.num_reservoirs):
                    targets[i] = min(targets[i] * requirement_total / increasable_flow, max_diversions[i])
            else:
                fully_constrained = True
            target_sum = sum(targets)
            count += 1
            if count > 5:
                print(count, requirement_total, target_sum, increasable_flow, targets, max_diversions)

        ### now return target for the reservoir of interest
        for i in range(self.num_reservoirs):
            if self.reservoir == reservoir_list_nyc[i]:
                return targets[i]


    @classmethod
    def load(cls, model, data):
        """
        """
        reservoir = data.pop("node")
        reservoir = reservoir.split('_')[1]
        nodes = [model.nodes[f'reservoir_{reservoir}'] for reservoir in reservoir_list_nyc]
        max_volume_agg_nyc = load_parameter(model, 'max_volume_agg_nyc')
        volume_agg_nyc = load_parameter(model, 'volume_agg_nyc')
        max_flow_delivery_nyc = load_parameter(model, 'max_flow_delivery_nyc')
        flow_agg_nyc = load_parameter(model, 'flow_agg_nyc')
        max_vol_reservoirs = [load_parameter(model, f'max_volume_{reservoir}') for reservoir in reservoir_list_nyc]
        vol_reservoirs = [load_parameter(model, f'volume_{reservoir}') for reservoir in reservoir_list_nyc]
        flow_reservoirs = [load_parameter(model, f'flow_{reservoir}') for reservoir in reservoir_list_nyc]
        hist_max_flow_delivery_nycs = [load_parameter(model, f'hist_max_flow_delivery_nyc_{reservoir}') for reservoir in reservoir_list_nyc]
        downstream_release_target_reservoirs = [load_parameter(model, f'downstream_release_target_{reservoir}') for reservoir in reservoir_list_nyc]
        flood_release_reservoirs = [load_parameter(model, f'flood_release_{reservoir}') for reservoir in reservoir_list_nyc]
        return cls(model, reservoir, nodes, max_volume_agg_nyc, volume_agg_nyc, max_flow_delivery_nyc,
                   flow_agg_nyc, max_vol_reservoirs, vol_reservoirs, flow_reservoirs, hist_max_flow_delivery_nycs,
                   downstream_release_target_reservoirs, flood_release_reservoirs, **data)

### have to register the custom parameter so Pywr recognizes it
VolBalanceNYCDemand.register()
