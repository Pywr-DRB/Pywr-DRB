"""
Contains parameters that track reservoir bank storages.

Overview:
This module contains parameters that track reservoir bank storages for the New York City (NYC) reservoirs.
These banks represent annually allocated water volumes that can be used for different purposes, and are
described in the 2017 Flexible Flow Management Plan (FFMP) Section 3.c.
At the moment, only the Trenton Equivalent Flow banks is implemented, which is 6.09 BG annually

Technical Notes:
- The Trenton equivalent flow bank is used to support the Trenton flow objective.
- The bank is reset on June 1 of each year.
- When NYC is in drought, the bank should be set to 0.0 (no excess quantity)
- This interacts with other parameters in the pywrdrb.parameters.ffmp module
- #TODO: Consider implementations for other banks
    - Thermal mitigation (1.62 BG)
    - Rapid flow change mitigation (0.65 BG)
    - NJ diversion amelioration (1.65 BG)
    
Links:
- NA

Change Log:
TJA, 2025-05-02, Add docs.
"""

import numpy as np
import pandas as pd

from pywr.parameters import Parameter, load_parameter

max_bank_volumes = {
    "trenton": 6090,  # 6090 MGD (6.09 BG)
    "thermal": 1620,  # 1620 MGD (1.62 BG)
    "rapid_flow": 650,  # 650 MGD (0.65 BG)
    "nj_diversion": 1650,  # 1650 MGD (1.65 BG)
}

bank_options = list(max_bank_volumes.keys())

class IERQRelease_step1(Parameter):
    """
    Tracks the Interim Excess Release Quantity (IERQ) release for the Trenton bank.
    
    Methods
    -------
    setup()
        Allocates an array to hold the parameter state.
    value(timestep, scenario_index)
        Returns the current volume remaining for the scenario.
    after()
        Automatically called after the value() method to update the bank remaining volume.
    load(model, data)
        Standard method to load the parameter from a data dictionary.
    
    Attributes
    ----------
    model : Model
        The Pywr model dict.
    bank : str
        The IERQ bank to track; options: "trenton", "thermal", "rapid_flow", "nj_diversion".
    release_needed : pywr.Parameter
        The parameter that indicates the release needed for the bank.
    step : int
        The step of the model (1 or 2).
    bank_remaining : np.ndarray
        Array to hold the remaining volume for each scenario.
    bank_release : np.ndarray
        Array to hold the release for each scenario.
    datetime : pd.Timestamp
        Datetime index object.
    """
    def __init__(self,
                 model,
                 bank,
                 release_needed,
                 **kwargs,
                 ):
        """Initalize the IERQRelease_step1 parameter.
        
        Parameters
        ----------
        model : Model
            The Pywr model dict.
        bank : str
            The IERQ bank to track; options: "trenton", "thermal", "rapid_flow", "nj_diversion". Only "trenton" is implemented currently.
        release_needed : pywr.Parameter
            The parameter that indicates the release needed for the bank.
        kwargs : dict
            Other keyword arguments passed to the pwyr.Parameter class. None currently used.
        """
        
        super().__init__(model, **kwargs)
        self.bank = bank
        self.step = 1
                
        # Parameter with current release needed
        self.release_needed = release_needed   
        self.parents.add(self.release_needed)     
    
    def setup(self):
        """Allocate arrays to hold the parameter state."""
        super().setup()
        self.num_scenarios = len(self.model.scenarios.combinations)
        self.bank_remaining = np.ones(shape=(self.num_scenarios)) * max_bank_volumes[self.bank]
        self.bank_release = np.empty([self.num_scenarios], np.float64)
        self.datetime = None

    def value(self, timestep, scenario_index):
        """
        Returns the current IERQ volume remaining for this year and scenario.

        Parameters
        ----------
        timestep : Timestep
            The current timestep. Provided by pywr during simulation.
        scenario_index : ScenarioIndex
            The scenario index. Provided by pywr during simulation.

        Returns
        -------
        float 
            The current volume remaining for the scenario.
        """
        if self.datetime is None:
            self.datetime = self.model.timestepper.current.datetime
            self.datetime = pd.Timestamp(self.datetime)
        
        
        trenton_release_needed = self.release_needed.get_value(scenario_index)
        
        
        allowable_release = min(self.bank_remaining[scenario_index.global_id], 
                                trenton_release_needed)
        
        allowable_release = max(allowable_release, 0.0)
        
        self.bank_release[scenario_index.global_id] = allowable_release        
        return self.bank_release[scenario_index.global_id]


    def after(self):
        """Run automatically after the value() method to update the bank remaining volume.
        
        This is used to remove the current Trenton release from the bank remaining volume.
        """
        # Remove today's release from the bank
        timestep = self.model.timestepper.current
        
        self.bank_remaining -= self.bank_release
        
        self.bank_remaining[self.bank_remaining < 0.0] = 0.0
        
        # Reset if May 31
        if self.datetime.month == 5 and self.datetime.day == 31:
            self.bank_remaining = np.ones(shape=(self.num_scenarios)) * max_bank_volumes[self.bank]
        
        # Advance datetime
        self.datetime += pd.Timedelta(1, "d")


    @classmethod
    def load(cls, model, data):
        """
        Load the IERQRelease_step1 parameter from the model dictionary.
        
        This is the standard pywr.Parameter class method. All data used by the parameter 
        should be contained in the dict, which is generated by pywrdrb.ModelBuilder.        
        """
        bank = data.pop("bank")
        
        if bank == "trenton":
            pass
        else:
            return ValueError(f"IERQ bank {bank} not yet implemented for parameter IERQRelease")

        param = f"release_needed_mrf_trenton_after_lower_basin_contributions_step1"
        release_needed_step1 = load_parameter(model, param)

        return cls(
            model, bank, release_needed_step1, **data
        )

# Register
IERQRelease_step1.register()




# class NJDiversionOffset:
#     """
#     TODO: Implement NJ Diversion Offset bank (FFMP Section 4.d)    
#     """
#     def __init__(
#         self,
#         model,
#         **kwargs):
#         pass

# class IERQRemaining(Parameter):
#     """
#     Keeps track of the Interim Excess Release Quantity (IERQ) 
#     remaining for the current year as described in the 2017FFMP Section 3.c.
    
#     Each IERQ bank resets on June 1 every year. 
#     During Drought conditions, IERQ goes to 0.0.
    
#     IERQ (total 10000 MG) is broken up into volumes for:
#     - Trenton equivalent flow (6090 MG)
#     - Thermal mitigation (1620 MG)
#     - Rapid flow change mitigation (650 MG)
#     - NJ diversion amelioration (1650 MG)
    
#     Args:
#     - model: The Pywr model dict.
#     - bank: The IERQ bank to track; options: "trenton", "thermal", "rapid_flow", "nj_diversion".
    
    
#     Methods:
#     - value: Returns the current volume remaining for the given bank scenario.
    
    
#     """
#     def __init__(
#         self,
#         step,
#         model,
#         bank,
#         bank_releases,
#         drought_level_agg_nyc,
#         **kwargs,
#     ):
        
#         super().__init__(model, **kwargs)
#         self.step = step
        
#         # Current NYC drought level
#         self.drought_level_agg_nyc = drought_level_agg_nyc
#         self.children.add(self.drought_level_agg_nyc)
        
#         # Bank release parameter(s)
#         self.bank_releases = bank_releases
#         for release in self.bank_releases:
#             self.children.add(release)
        
#         # Bank name
#         self.bank = bank
#         assert(bank in bank_options), f"IERQ bank {bank} not in {bank_options} for parameter IERQRemaining"
        
#         # max volume for this bank IERQ
#         self.max_bank_volume = max_bank_volumes[bank]
        
        

#     def setup(self):
#         """Allocate an array to hold the parameter state."""
#         super().setup()
#         num_scenarios = len(self.model.scenarios.combinations)
#         self.bank_remaining = np.empty([num_scenarios], np.float64)

        
#     def value(self, timestep, scenario_index):
#         """
#         Returns the current volume remaining for the scenario.

#         Args:
#             timestep (Timestep): The current timestep.
#             scenario_index (ScenarioIndex): The scenario index.

#         Returns:
#             float: The current volume remaining for the scenario.
#         """
        
#         # If NYC in drought, set remaining = 0.0
#         current_nyc_drought_level = self.drought_level_agg_nyc.get_value(scenario_index)
#         is_nyc_drought_emergency = True if current_nyc_drought_level in [6] else False
#         if is_nyc_drought_emergency:
#             self.bank_remaining[scenario_index] = 0.0

#             return self.bank_remaining[scenario_index]

#         if self.step == 1:
#             return self.bank_remaining[scenario_index]
        
#         # if step 2, subtract the release from step 1
#         elif self.step == 2:
#             return self.bank_remaining[scenario_index] - self.bank_releases[0].get_value(scenario_index)



#     def after(self):
#         """
#         """
#         # Remove today's release from the bank
#         timestep = self.model.timestepper.current
        
#         for release in self.bank_releases:
#             todays_release = release.get_all_values()
#             self.bank_remaining -= todays_release

#         self.bank_remaining = np.maximum(self.bank_remaining, 0.0)
        
#         # Reset if May 31
#         if self.datetime.month == 5 and self.datetime.day == 31:
#             self.bank_remaining = self.max_bank_volume
        
#         # Advance datetime
#         self.datetime += pd.Timedelta(1, "d")


#     @classmethod
#     def load(cls, model, data):
#         bank = data.pop("bank")
#         step = data.pop("step")
        
#         assert(bank in bank_options), f"IERQ bank {bank} not in {bank_options} for parameter IERQRemaining"
        
#         if bank == "trenton":
#             bank_release_param_name = f"nyc_mrf_trenton_step1"
#             bank_release_step1 = load_parameter(model, bank_release_param_name)
            
#             bank_release_param_name = f"nyc_mrf_trenton_step2"
#             bank_release_step2 = load_parameter(model, bank_release_param_name)
#             bank_releases = [bank_release_step1, bank_release_step2]

#         else:
#             return ValueError(f"IERQ bank {bank} not yet implemented for parameter IERQRemaining")
        
#         drought_level_agg_nyc = load_parameter(model, f"drought_level_agg_nyc")
#         return cls(
#             model, step, bank, bank_releases, drought_level_agg_nyc, **data
#         )