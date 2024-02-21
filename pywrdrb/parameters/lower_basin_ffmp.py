import numpy as np
import pandas as pd
import math

from pywr.parameters import Parameter, load_parameter

from pywrdrb.utils.directories import model_data_dir
from pywrdrb.utils.constants import cfs_to_mgd, epsilon
from pywrdrb.utils.lists import modified_starfit_reservoir_list, drbc_lower_basin_reservoirs

# Drought emergency lower basin staging levels
# Taken from section 2.5.5 of the DRB Water Code
# Items are: priority_level, reservoir, storage_percentage_lower_bound 
priority_use_during_drought = [
    [1, 'beltzvilleCombined', 0.737],
    [1, 'blueMarsh', 0.689],
    [2, 'nockamixon', 0.687],
    [3, 'beltzvilleCombined', 0.380],
    [4, 'blueMarsh', 0.368],
    [5, 'beltzvilleCombined', 0.034],
    [5, 'blueMarsh', 0.130],
    [6, 'nockamixon', 0.010]]

## Max storage for lower basin reservoirs
# STARFIT relevant max storage often corresponds to flood storage
# Some reservoirs (notably blueMarsh) are operated at much lower % of max flood storage
# These are chosen to match prescribed priority staging levels in the Water Code relative to "usable storage"
# Taken from https://drbc.maps.arcgis.com/apps/dashboards/690464a9958b49e5b49550964641ffd7
drbc_max_usable_storages = {
    'beltzvilleCombined': 13500,
    'blueMarsh':  7450,
    'nockamixon': 13000
    }

## Max discharges at lower reservoirs
max_discharges = {'blueMarsh': 1500*cfs_to_mgd, 
                 'beltzvilleCombined': 1500*cfs_to_mgd,
                 'nockamixon': 1000*cfs_to_mgd,
                 'fewalter': 2000*cfs_to_mgd}


## Max daily MRF contributions from lower basin reservoirs for trenton target flow
# Rough estimate based on observed release patterns during periods where lower basin was used
max_mrf_daily_contributions = {'blueMarsh': 300, 
                 'beltzvilleCombined': 300,
                 'nockamixon': 300,
                 'fewalter': 2000*cfs_to_mgd}

### lag days from Trenton for each drbc lower basin reservoir
lag_days_from_Trenton = {'blueMarsh': 2,
                         'beltzvilleCombined': 2,
                         'nockamixon': 1,
                         'fewalter': 2}

## Conservation releases at lower reservoirs
# Specified in the DRBC Water Code Table 4
# Lower basin drought condition policies are not currently implemented
conservation_releases= {'normal' : {'blueMarsh': 50*cfs_to_mgd,
                                    'beltzvilleCombined': 35*cfs_to_mgd,
                                    'nockamixon' : 11*cfs_to_mgd,
                                    'fewalter': 50*cfs_to_mgd},
                        'lower_basin_drought' : 
                            {'blueMarsh': 30*cfs_to_mgd,
                             'beltzvilleCombined': 15*cfs_to_mgd,
                             'nockamixon' : 7*cfs_to_mgd,
                             'fewalter': 43*cfs_to_mgd}}
normal_conservation_releases = conservation_releases['normal']


class LowerBasinMaxMRFContribution(Parameter):
    """
    Parameter used to track the magnitude of allowable releases from 
    a specific lower basin reservoir which can be used to help NYC meet MRF targets.
    
    This is provided to VolBalanceLowerBasinMRFAggregate to determine the aggregate
    MRF contribution from the lower basin reservoirs.
    """
    def __init__(self, model, 
                 reservoir,
                 step,
                 nodes,
                 parameters,
                 **kwargs):
        super().__init__(model, **kwargs)
        self.debugging = False
        self.reservoir = reservoir
        self.step = step
        self.days_ahead_prediction = 5 - self.step
        self.nodes = nodes
        self.parameters = parameters

        # Reservoirs considered during this step
        self.consider_reservoirs = drbc_lower_basin_reservoirs if (self.step != 1) else ['nockamixon']

        # Current NYC drought level 
        self.drought_level_agg_nyc = self.parameters['drought_level_agg_nyc']
        self.children.add(self.drought_level_agg_nyc)
        
        # Trenton flow requirement
        self.release_needed_mrf_trenton = self.parameters['release_needed_mrf_trenton']
        self.children.add(self.release_needed_mrf_trenton)

        # Max storage scaling to match priority staging
        self.max_volumes = drbc_max_usable_storages
        
        # Table 4 of DRBC Water Code 
        # but drought seems to reference lower basin drought... use normal for now
        self.conservation_releases = normal_conservation_releases
        self.R_min = self.conservation_releases[self.reservoir]
        
        self.max_mrf_daily_contribution = max_mrf_daily_contributions[self.reservoir]
        
        # Make sure attributes are available from storage node
        assert(self.R_min is not None), f'LowerBasinMaxMRFContribution: R_min is None'
        assert(self.max_mrf_daily_contribution is not None), f'LowerBasinMaxMRFContribution: max_mrf_daily_contribution is None'
    
    def print_allowable_storage_info(self, max_allowable, 
                                     current_reservoir_priority,
                                     percent_storage):
        """
        Print info about the allowable storage for this reservoir of interest for debugging purposes.
        Info includes max allowable contribution, relevant priority level, and current storage.
        """
        if max_allowable > 0.0:
            print(f'{self.reservoir} max allowed contr {max_allowable}, priority: {current_reservoir_priority}, S_hat: {percent_storage}')
        else:
            print(f'{self.reservoir} not allowed for MRF, priority: {current_reservoir_priority}, S_hat: {percent_storage}')
        
    
    
    def value(self, timestep, scenario_index):

        # if reservoir is further lag from Trenton than our current step is working on 
        # (eg Beltzville & BLue Marsh in Step 4), return 0
        if lag_days_from_Trenton[self.reservoir] > 5 - self.step:
            return 0.

        # Get NYC current FFMP level
        current_nyc_drought_level = self.drought_level_agg_nyc.get_value(scenario_index)
        is_nyc_drought_emergency = True if current_nyc_drought_level in [6] else False
        
        trenton_requirement = self.release_needed_mrf_trenton.get_value(scenario_index)
        
        # Unlock lower basin reservoirs for MRF if both conditions:
        # a) NYC is in drought conditions, 
        # b) Trenton MRF contributiuons are needed 
        if is_nyc_drought_emergency and (trenton_requirement > 0.0):
            # We want to return the max allowable contribution from this reservoir
            # But need to consider storages and priority staging of each lower basin reservoir
            percent_storages = {}
            max_allowable_releases = {}
            max_allowable = 0.0
            already_used_reservoirs = []
            
            # Find what priority level each lower basin reservoir is in
            # And corresponding usable storage
            for res in drbc_lower_basin_reservoirs:
                
                # Get current storage percentage
                S_max = self.max_volumes[res]
                S_t = self.nodes[res].volume[scenario_index.indices]
                
                # Add inflow and remove required conservation releases from storage
                inflow = self.parameters[f'flow_{res}'].get_value(scenario_index)
                S_t += inflow
                S_t -= self.R_min
                
                # Storage as fraction of max storage
                S_hat_t = S_t / S_max 
                percent_storages[res] = S_hat_t
                
                # tolerance is the volume proportional to 1 day of 300MGD mrf contribution
                # ~2% for beltzville and nockamixon; ~4% for blue marsh
                tolerance = 300 / S_max
                
                # Loop and find current priority level
                for priority_stage in priority_use_during_drought:
                    priority_level, priority_res, stage_lower = priority_stage

                    # Skip if already used (usable storage already assigned)
                    # Skip if not the current res from the outter loop
                    if (res in already_used_reservoirs) or (res != priority_res):
                        continue

                    # if within 2% of current stage lower bound,
                    # and priority level is less than 4, (not the last priority stage) 
                    # then continue and use next priority stage
                    diff = S_hat_t - stage_lower
                    if (diff < tolerance) and (priority_level <= 4):
                        # print(f'{res} storage {S_hat_t} <={tolerance}% above stage {priority_level} lower bound {stage_lower}. Using next priority stage.')
                        continue

                    # else consider storage above the stage lower bound
                    usable_storage = (S_hat_t - stage_lower) * S_max
                    
                    ### assume we dont want to release more than we can sustainably release each day between today and the day the lower basin reservoir will actually release.
                    ### e.g., for Step 1 Cannonsville/Pepacton release calculation, days_ahead_prediction=4. This means today's releases will be combined with Blue Marsh in 2 days,
                    ### so we dont want to expect more water available than an amount that could be released today, tomorrow, & the third day (2 days from now)                    
                    available_mrf_daily_release = usable_storage / max(self.days_ahead_prediction -
                                                                    lag_days_from_Trenton[res] + 1, 1)
                        
                    if available_mrf_daily_release > 0.0:
                        max_allowable_releases[res] = [priority_level, available_mrf_daily_release]
                        already_used_reservoirs.append(res)
                    else:
                        max_allowable_releases[res] = [999, 0.0]

            for res in drbc_lower_basin_reservoirs:
                assert(res in max_allowable_releases.keys()), f'LowerBasinMaxMRFContribution: {res} not in max_allowable_releases.keys()'    
            
            # Find highest priority level to use
            all_priority_levels = [max_allowable_releases[res][0] for res in max_allowable_releases.keys()]
            active_priority = min(all_priority_levels)
            
            # if step 4, then only nockamixon is able to be used
            if self.step == 4:
                active_priority = max_allowable_releases['nockamixon'][0]
            
            # Now set max allowable for this specific reservoir depending if it is highest priority
            current_reservoir_priority = max_allowable_releases[self.reservoir][0]
            if current_reservoir_priority == active_priority:
                max_allowable = max_allowable_releases[self.reservoir][1]
                
                # Constraints : 
                # Dont release more than historically used for MRF (estimated from obs) 
                if max_allowable >= self.max_mrf_daily_contribution:
                    max_allowable = self.max_mrf_daily_contribution
                
                if self.debugging:
                    self.print_allowable_storage_info(max_allowable, 
                                                      current_reservoir_priority,
                                                      percent_storages[self.reservoir])
                    
                # Return max allowable
                assert(max_allowable is not None), f'LowerBasinMaxMRFContribution: max_allowable is None'
                return float(max_allowable)
            else:
                return 0.0
                
        # If NYC is not drought condition, then no MRF contribution
        else:
            return 0.0
    
    @classmethod
    def load(cls, model, data):
        reservoir = data.pop("node")
        step = data.pop('step')
        reservoir = reservoir.split('_')[1]
        nodes = {}
        parameters = {}
        for res in drbc_lower_basin_reservoirs:
            nodes[res] = model.nodes[f'reservoir_{res}']
            parameters[f'flow_{res}'] = load_parameter(model, f'flow_{res}')
        parameters['drought_level_agg_nyc'] = load_parameter(model, 
                                                             f'drought_level_agg_nyc')
        
        # Trenton MRF contributions required by NYC and Lower Basin 
        # (beyond releases already needed for Montague)
        parameters['release_needed_mrf_trenton'] = load_parameter(model, 
                                                                  f'release_needed_mrf_trenton_step{step}')
        return cls(model,
                   reservoir,
                   step,
                   nodes,
                   parameters,
                   **data)

LowerBasinMaxMRFContribution.register()




class VolBalanceLowerBasinMRFAggregate(Parameter):
    def __init__(self, model,
                 release_needed_mrf_trenton,
                 lower_basin_max_mrf_contributions, 
                 **kwargs):
        super().__init__(model, **kwargs)
        self.debugging = False
        self.release_needed_mrf_trenton = release_needed_mrf_trenton
        self.lower_basin_max_mrf_contributions = lower_basin_max_mrf_contributions
        self.drbc_lower_basin_reservoirs = drbc_lower_basin_reservoirs
        
        # CHILD PARAMETERS
        self.children.add(release_needed_mrf_trenton)
        for reservoir in self.drbc_lower_basin_reservoirs:
            self.children.add(lower_basin_max_mrf_contributions[reservoir])
    
    def value(self, timestep, scenario_index):
        """
        Checks how much lower basin reservoir release are needed & allowed to contribute to MRF.
        Returns aggregate MRF contribution from lower basin reservoirs. 
        """    
        
        # If Montague/Trenton MRF target is 0, then return 0
        if self.release_needed_mrf_trenton.get_value(scenario_index) < epsilon:
            return 0.0
        
        # If Montague/Trenton MRF target is not 0, then handle partitioning
        else:
            # Check if lower basin releases are allowed (must be FFMP drought)
            max_aggregate_allowed = sum([self.lower_basin_max_mrf_contributions[reservoir].get_value(scenario_index)
                                            for reservoir in self.drbc_lower_basin_reservoirs])
            
            assert(max_aggregate_allowed is not None), f'VolBalanceLowerBasinMRFAggregate: max_aggregate_allowed is None'
            
            # If no lower basin releases are allowed, then return 0
            if max_aggregate_allowed < epsilon:
                return 0.0
            # Otherwise determine the total lower basin contribution to MRF
            else:
                return min(self.release_needed_mrf_trenton.get_value(scenario_index), max_aggregate_allowed)

    
    
    @classmethod
    def load(cls, model, data):

        step = data.pop('step')

        # Trenton MRF contributions required by NYC and Lower Basin (beyond releases already needed for Montague)
        release_needed_mrf_trenton = load_parameter(model, f'release_needed_mrf_trenton_step{step}')

        # Lower basin reservoir max Trenton contributions
        lower_basin_max_mrf_contributions = {}
        for r in drbc_lower_basin_reservoirs:
            lower_basin_max_mrf_contributions[r] = load_parameter(model, f'max_mrf_trenton_step{step}_{r}')
        return cls(model,
                   release_needed_mrf_trenton,
                   lower_basin_max_mrf_contributions,
                   **data)
        
VolBalanceLowerBasinMRFAggregate.register()




class VolBalanceLowerBasinMRFIndividual(Parameter):
    def __init__(self, model,
                 reservoir, 
                 lower_basin_agg_mrf_trenton,
                 lower_basin_max_mrf_contributions,
                 **kwargs):
        
        super().__init__(model, **kwargs)
        self.debugging = True
        self.reservoir = reservoir
        self.lower_basin_max_mrf_contributions = lower_basin_max_mrf_contributions
        self.lower_basin_agg_mrf_trenton = lower_basin_agg_mrf_trenton
        self.drbc_lower_basin_reservoirs = drbc_lower_basin_reservoirs
        
        # CHILD PARAMETERS
        self.children.add(lower_basin_agg_mrf_trenton)
        for reservoir in drbc_lower_basin_reservoirs:
            self.children.add(lower_basin_max_mrf_contributions[reservoir])
            
        
    def split_lower_basin_mrf_contributions(self, scenario_index):
        """
        Split the MRF contributions from the lower basin reservoirs into 
        individual reservoir contributions. 

        """
        # Get total allowable MRF contribution from lower basin reservoirs
        requirement_total = self.lower_basin_agg_mrf_trenton.get_value(scenario_index)
       
        # Get individual allowable contributions
        # Currently naive handling of priority ordering of lower basin reservoirs
        individual_contributions = {}
        requirement_remaining = requirement_total
        
        total_max_contribution = sum([self.lower_basin_max_mrf_contributions[reservoir].get_value(scenario_index)
                                        for reservoir in self.drbc_lower_basin_reservoirs])
        assert(total_max_contribution >= requirement_total),\
            f'VolBalanceLowerBasinMRFIndividual: total_max_contribution < requirement_total'
        
        for reservoir in drbc_lower_basin_reservoirs:
            if requirement_remaining > 0.0:
                individual_max = self.lower_basin_max_mrf_contributions[reservoir].get_value(scenario_index)
                individual_contributions[reservoir] = min(individual_max, requirement_remaining)
                requirement_remaining -= individual_contributions[reservoir]
            else:
                individual_contributions[reservoir] = 0.0
                
        assert(requirement_remaining < epsilon), \
            f'VolBalanceLowerBasinMRFIndividual: requirement_remaining is not 0: {requirement_remaining}'
        diff = sum(individual_contributions.values()) - requirement_total
        if abs(diff) > epsilon:
            print(f'diff: {diff}')
            print(f'sum individual_contributions: {sum(individual_contributions.values())}')
            print(f'requirement_total: {requirement_total}')
            print(f'requirement_remaining: {requirement_remaining}')
            print(f'Blue Marsh max: {self.lower_basin_max_mrf_contributions["blueMarsh"].get_value(scenario_index)}')
            print(f'Beltzville max: {self.lower_basin_max_mrf_contributions["beltzvilleCombined"].get_value(scenario_index)}')
            print(f'Nockamixon max: {self.lower_basin_max_mrf_contributions["nockamixon"].get_value(scenario_index)}')
            print(f'BlueMarsh contribution: {individual_contributions["blueMarsh"]}')
            print(f'Beltzville contribution: {individual_contributions["beltzvilleCombined"]}')
            print(f'Nockamixon contribution: {individual_contributions["nockamixon"]}')
            
        assert(abs(diff) < epsilon), f'VolBalanceLowerBasinMRFIndividual: sum(individual_constributions) != requirement_total'
        return individual_contributions
    
    def value(self, timestep, scenario_index):
        """
        Checks the aggregate lower basin Montague/Trenton MRF target and 
        the individual reservoir MRF contribution releases.
        """
        
        # Check if lower basin is used for MRF contribution
        if self.lower_basin_agg_mrf_trenton.get_value(scenario_index) < epsilon:
            return 0.0
        
        # Otherwise, split required contributions
        else:
            # Get individual reservoir contributions
            individual_targets = self.split_lower_basin_mrf_contributions(scenario_index)
            # Return the contribution from the reservoir of interest
            release = individual_targets[self.reservoir]
            if self.debugging:
                print(f'S{scenario_index}: {timestep} Lower Basin MRF releases: {individual_targets}')
            return release
    
    @classmethod
    def load(cls, model, data):
        reservoir = data.pop("node")
        reservoir = reservoir.split('_')[1]
        step = data.pop('step')

        lower_basin_agg_mrf_trenton = load_parameter(model, f'lower_basin_agg_mrf_trenton_step{step}')
        
        # Lower basin reservoir max Trenton contributions
        lower_basin_max_mrf_contributions = {}
        for r in drbc_lower_basin_reservoirs:
            lower_basin_max_mrf_contributions[r] = load_parameter(model, f'max_mrf_trenton_step{step}_{r}')

        return cls(model, 
                   reservoir,
                   lower_basin_agg_mrf_trenton,
                   lower_basin_max_mrf_contributions,
                   **data)
        
VolBalanceLowerBasinMRFIndividual.register()