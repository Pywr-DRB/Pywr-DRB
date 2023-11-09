import numpy as np
import pandas as pd
import math

from pywr.parameters import Parameter, load_parameter

from utils.directories import model_data_dir
from utils.constants import cfs_to_mgd, epsilon
from utils.lists import modified_starfit_reservoir_list, drbc_lower_basin_reservoirs

# Drought emergency lower basin staging levels
# Taken from section 2.5.5 of the DRB Water Code
# Items are: priority_level, reservoir, storage_percentage_lower_bound 
priority_use_during_drought = [
    [1, 'beltzvilleCombined', 0.737],
    [1, 'blueMarsh', 0.689],
    [2, 'nockamixon', 0.687],
    [3, 'beltzvilleCombined', 0.380],
    [4, 'blueMarsh', 0.368],
    [5, 'beltzvilleCombined', 0.340],
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


## Max daily MRF contributions from lower basin reservoirs for montagueTrenton target flow
# Rough estimate based on observed release patterns during periods where lower basin was used
max_mrf_daily_contributions = {'blueMarsh': 300, 
                 'beltzvilleCombined': 300,
                 'nockamixon': 300,
                 'fewalter': 2000*cfs_to_mgd}

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
                 nodes,
                 drought_level_agg_nyc,
                 **kwargs):
        super().__init__(model, **kwargs)
        self.reservoir = reservoir
        
        self.nodes = nodes
        self.drought_level_agg_nyc = drought_level_agg_nyc
        
        # Max storage scaling to match priority staging
        self.max_volumes = drbc_max_usable_storages
        
        # Add inheretence
        for res in drbc_lower_basin_reservoirs: 
            self.parents.add(nodes[res])
        self.children.add(drought_level_agg_nyc)
        
        # Table 4 of DRBC Water Code 
        # but drought seems to reference lower basin drought... use normal for now
        self.conservation_releases = normal_conservation_releases
        self.R_min = self.conservation_releases[self.reservoir]
        
        self.max_mrf_daily_contribution = max_mrf_daily_contributions[self.reservoir]
        
        # Make sure attributes are available from storage node
        assert(self.R_min is not None), f'LowerBasinMaxMRFContribution: R_min is None'
        assert(self.max_mrf_daily_contribution is not None), f'LowerBasinMaxMRFContribution: max_mrf_daily_contribution is None'
    
    def value(self, timestep, scenario_index):
        
        # Get NYC current FFMP level
        current_nyc_drought_level = self.drought_level_agg_nyc.value(timestep, scenario_index)
        
        # If NYC is in drought conditions, unlock lower basin reservoirs for MRF
        if current_nyc_drought_level in [6]:
                    
            # We want to return the max allowable contribution from this reservoir
            max_allowable_releases = {}
            max_allowable = 0.0
            used_reservoirs = []
            
            # Find what priority level each lower basin reservoir is in
            # And corresponding usable storage
            for priority_stage in priority_use_during_drought:
                priority_level, res, stage_lower = priority_stage

                if res not in used_reservoirs:
                    # Get storage percentage
                    S_max = self.max_volumes[res]
                    S_t = self.nodes[res].volume[scenario_index.indices]
                    S_hat_t = S_t / S_max

                    usable_storage = (S_hat_t - stage_lower) * S_max
                
                    if usable_storage > 0.0:                     

                        max_allowable_releases[res] = [priority_level, usable_storage]                     
                        used_reservoirs.append(res)
                    else:
                        max_allowable_releases[res] = [999, 0.0]
                else:
                    continue
            
            all_priority_levels = [max_allowable_releases[res][0] for res in max_allowable_releases.keys()]
            high_priority = min(all_priority_levels)
            
            # Now set max allowable for this specific reservoir depending if it is highest priority
            current_reservoir_priority = max_allowable_releases[self.reservoir][0]
            if current_reservoir_priority == high_priority:
                max_allowable = max_allowable_releases[self.reservoir][1]
                
                # Max release constraint
                if max_allowable >= self.max_mrf_daily_contribution:
                    # Conversvation releases are not included
                    max_allowable = self.max_mrf_daily_contribution - self.R_min
                    max_allowable = max(max_allowable, 0.0)
                # Min release constraint
                else:
                    max_allowable =  max((max_allowable - self.R_min), 0.0)
                
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
        reservoir = reservoir.split('_')[1]
        nodes = {}
        # max_volumes = {}
        for res in drbc_lower_basin_reservoirs:
            nodes[res] = model.nodes[f'reservoir_{res}']
            # max_volumes[res] = load_parameter(model, f'max_volume_{res}')
        drought_level_agg_nyc = load_parameter(model, f'drought_level_agg_nyc')
        return cls(model,
                   reservoir, 
                   nodes,
                   drought_level_agg_nyc,
                   **data)

LowerBasinMaxMRFContribution.register()




class VolBalanceLowerBasinMRFAggregate(Parameter):
    def __init__(self, model, 
                 agg_mrf_montagueTrenton, 
                 lower_basin_max_mrf_contributions, 
                 **kwargs):
        super().__init__(model, **kwargs)
        self.agg_mrf_montagueTrenton = agg_mrf_montagueTrenton
        self.lower_basin_max_mrf_contributions = lower_basin_max_mrf_contributions
        self.drbc_lower_basin_reservoirs = drbc_lower_basin_reservoirs
        
        # CHILD PARAMETERS
        self.children.add(agg_mrf_montagueTrenton)
        for reservoir in self.drbc_lower_basin_reservoirs:
            self.children.add(lower_basin_max_mrf_contributions[reservoir])
    
    def value(self, timestep, scenario_index):
        """
        Checks how much lower basin reservoir release are needed & allowed to contribute to MRF.
        Returns aggregate MRF contribution from lower basin reservoirs. 
        """    
        
        # If Montague/Trenton MRF target is 0, then return 0
        if self.agg_mrf_montagueTrenton.value(timestep, scenario_index) < epsilon:
            return 0.0
        
        # If Montague/Trenton MRF target is not 0, then handle partitioning
        else:
            # Check if lower basin releases are allowed (must be FFMP drought)
            max_aggregate_allowed = sum([
                self.lower_basin_max_mrf_contributions[reservoir].value(timestep, scenario_index) 
                for reservoir in self.drbc_lower_basin_reservoirs])
            
            assert(max_aggregate_allowed is not None), f'VolBalanceLowerBasinMRFAggregate: max_aggregate_allowed is None'
            
            # If no lower basin releases are allowed, then return 0
            if max_aggregate_allowed < epsilon:
                return 0.0
            # Otherwise determine the total lower basin contribution to MRF
            else:
                if self.agg_mrf_montagueTrenton.value(timestep, scenario_index) >= max_aggregate_allowed:
                    return max_aggregate_allowed
                else:
                    return self.agg_mrf_montagueTrenton.value(timestep, scenario_index)        
    
    
    @classmethod
    def load(cls, model, data):
        # Total Montague/Trenton MRF contributions required by NYC and Lower Basin
        total_agg_mrf_montagueTrenton = load_parameter(model, 'total_agg_mrf_montagueTrenton')
        
        # Lower basin reservoir max Montague/Trenton contributions
        lower_basin_max_mrf_contributions = {}
        for r in drbc_lower_basin_reservoirs:
            lower_basin_max_mrf_contributions[r] = load_parameter(model, 
                                                                  f'max_mrf_montagueTrenton_{r}')
        return cls(model, 
                   total_agg_mrf_montagueTrenton, 
                   lower_basin_max_mrf_contributions,
                   **data)
        
VolBalanceLowerBasinMRFAggregate.register()




class VolBalanceLowerBasinMRFIndividual(Parameter):
    def __init__(self, model,
                 reservoir, 
                 lower_basin_agg_mrf_montagueTrenton,
                 lower_basin_max_mrf_contributions,
                 **kwargs):
        
        super().__init__(model, **kwargs)
        self.reservoir = reservoir
        self.lower_basin_max_mrf_contributions = lower_basin_max_mrf_contributions
        self.lower_basin_agg_mrf_montagueTrenton = lower_basin_agg_mrf_montagueTrenton   
        self.drbc_lower_basin_reservoirs = drbc_lower_basin_reservoirs
        
        # CHILD PARAMETERS
        self.children.add(lower_basin_agg_mrf_montagueTrenton)
        for reservoir in drbc_lower_basin_reservoirs:
            self.children.add(lower_basin_max_mrf_contributions[reservoir])
            
        
    def split_lower_basin_mrf_contributions(self, timestep, scenario_index):
        """
        Split the MRF contributions from the lower basin reservoirs into 
        individual reservoir contributions. 

        """
        # Get total allowable MRF contribution from lower basin reservoirs
        requirement_total = self.lower_basin_agg_mrf_montagueTrenton.value(timestep, scenario_index)
       
        # Get individual allowable contributions
        # Currently naive handling of priority ordering of lower basin reservoirs
        # TODO: Account for Montague vs Trenton targets
        individual_contributions = {}
        requirement_remaining = requirement_total
        
        total_max_contribution = sum([
            self.lower_basin_max_mrf_contributions[reservoir].value(timestep, scenario_index) 
            for reservoir in self.drbc_lower_basin_reservoirs])
        assert(total_max_contribution >= requirement_total), f'VolBalanceLowerBasinMRFIndividual: total_max_contribution < requirement_total'
        
        for reservoir in drbc_lower_basin_reservoirs:
            if requirement_remaining > 0.0:
                individual_max = self.lower_basin_max_mrf_contributions[reservoir].value(timestep, scenario_index)
                individual_contributions[reservoir] = min(individual_max, requirement_remaining)
                requirement_remaining -= individual_contributions[reservoir]
            else:
                individual_contributions[reservoir] = 0.0
                
        assert(requirement_remaining < epsilon), f'VolBalanceLowerBasinMRFIndividual: requirement_remaining is not 0: {requirement_remaining}'
        diff = sum(individual_contributions.values()) - requirement_total
        if abs(diff) > epsilon:
            print(f'diff: {diff}')
            print(f'sum individual_contributions: {sum(individual_contributions.values())}')
            print(f'requirement_total: {requirement_total}')
            print(f'requirement_remaining: {requirement_remaining}')
            print(f'Blue Marsh max: {self.lower_basin_max_mrf_contributions["blueMarsh"].value(timestep, scenario_index)}')
            print(f'Beltzville max: {self.lower_basin_max_mrf_contributions["beltzvilleCombined"].value(timestep, scenario_index)}')
            print(f'Nockamixon max: {self.lower_basin_max_mrf_contributions["nockamixon"].value(timestep, scenario_index)}')
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
        if self.lower_basin_agg_mrf_montagueTrenton.value(timestep, scenario_index) < epsilon:
            return 0.0
        
        # Otherwise, split required contributions
        else:
            # Get individual reservoir contributions
            individual_targets = self.split_lower_basin_mrf_contributions(timestep, scenario_index)
            
            # Return the contribution from the reservoir of interest
            release = individual_targets[self.reservoir]
            if release > epsilon:
                print(f'Using {self.reservoir} for Montague/Trenton, total: {individual_targets[self.reservoir]}')
            return release
    
    @classmethod
    def load(cls, model, data):
        reservoir = data.pop("node")
        reservoir = reservoir.split('_')[1]
        lower_basin_agg_mrf_montagueTrenton = load_parameter(model, 'lower_basin_agg_mrf_montagueTrenton')
        
        # Lower basin reservoir max Montague/Trenton contributions
        lower_basin_max_mrf_contributions = {}
        for r in drbc_lower_basin_reservoirs:
            lower_basin_max_mrf_contributions[r] = load_parameter(model, 
                                                                  f'max_mrf_montagueTrenton_{r}')
        return cls(model, 
                   reservoir,
                   lower_basin_agg_mrf_montagueTrenton, 
                   lower_basin_max_mrf_contributions,
                   **data)
        
VolBalanceLowerBasinMRFIndividual.register()