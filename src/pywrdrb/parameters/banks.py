"""
Contains parameters that track reservoir bank storages and mandatory excess releases.

Overview:
This module contains two categories of NYC release obligations under the 1954 U.S.
Supreme Court Decree (New Jersey v. New York) and the 2017 Flexible Flow Management
Program (FFMP2017):

1. IERQ (Interim Excess Release Quantity) banks — FFMP2017 §3.c
   Annually allocated water volumes for specific flow objectives. Only the Trenton
   Equivalent Flow bank (6.09 BG) is currently wired; thermal/rapid-flow/NJ-diversion
   banks are defined in max_bank_volumes but not yet implemented.

2. ERQ (Excess Release Quantity) — 1954 Decree Art. III-B-1(c)-(d)
   NYC's annual obligation to release excess water to the Delaware River when its
   diversions are below the safe yield. This is the Decree's cooperative reciprocity
   mechanism: NYC's right to divert 800 MGD comes paired with an obligation to
   return water in years when its system is not fully utilized.
   Implemented in ERQRelease.

Technical Notes:
- IERQ bank resets May 31; ERQ seasonal period runs June 15 – March 15.
- ERQ is a Decree obligation (not modifiable by FFMP); the fraction (0.83) and
  seasonal release schedule (120 days, cap 70 BG) are established by the Court.
- ERQ fraction is treated as a Sobol sensitivity parameter to characterize how
  cooperative benefit-sharing affects each party's RRV metrics.

Change Log:
TJA, 2025-05-02, Add docs (IERQRelease_step1).
MS,  2026-06-02, Add ERQRelease (Decree Art. III-B-1(c)-(d)).
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


# ---------------------------------------------------------------------------
# ERQ — Excess Release Quantity (1954 Decree Art. III-B-1(c)-(d))
# ---------------------------------------------------------------------------

class ERQRelease(Parameter):
    """
    Implements NYC's annual Excess Release Quantity (ERQ) obligation to the Delaware River.

    Legal Basis
    -----------
    1954 U.S. Supreme Court Decree, New Jersey v. New York, Art. III-B-1(c)-(d):

    Art. III-B-1(c):
        "The City shall ... release in the aggregate from all its storage reservoirs
        in the upper Delaware watershed, in addition to the quantity of water required
        to be released for the purpose of maintaining the then applicable minimum basic
        rate of flow ..., a quantity of water equal to 83 per cent of the amount by
        which the estimated consumption during such year is less than the City's
        estimate of the continuous safe yield during such year of all its sources
        obtainable without pumping. ... its safe yield in any such year, obtainable
        without pumping, shall be estimated at not less than 1665 m.g.d. after the
        Cannonsville reservoir is put into operation."

    Art. III-B-1(d):
        "The City of New York shall release the excess quantity provided for in
        subsection (c) at rates designed to release the entire quantity in 120 days.
        Commencing with the fifteenth day of June each year, the excess releases shall
        continue for as long a period, but not later than the following March 15 ...
        The excess quantity required to be released in any seasonal period shall in no
        event exceed 70 billion gallons."

    Design
    ------
    Each June 1 the parameter recomputes the upcoming seasonal ERQ:

        ERQ = erq_fraction × max(0, NYC_SAFE_YIELD_ANNUAL_MG − prev_year_consumption_MG)
        ERQ = min(ERQ, MAX_ERQ_MG)   # 70 BG cap

    From June 15 through March 15 the parameter returns a daily release target:

        daily_rate = ERQ / SEASONAL_DAYS   (= ERQ / 120)

    The daily rate is returned as a total across all NYC reservoirs; model_builder
    distributes it equally (1/3 each) across Cannonsville, Pepacton, and Neversink.
    Reservoir storage constraints in pywr naturally limit actual releases when
    reservoirs are drawn down, consistent with the Decree's clause that NYC is not
    required to "release at rates exceeding the capacity of its release works."

    Parameters
    ----------
    model : pywr.Model
    erq_fraction : float
        Fraction used in Art. III-B-1(c) (Decree value: 0.83).
        Intended as a Sobol sensitivity parameter with range [0.65, 1.00].
    nyc_delivery_node_name : str
        Name of the pywr Output node recording NYC diversions (daily MG).
        Default: "delivery_nyc" (pywrdrb standard).

    Decree Constants (class attributes)
    ------------------------------------
    NYC_SAFE_YIELD_FLOOR_MGD : 1665.0
        Minimum safe-yield floor after Cannonsville completion (Decree Art. III-B-1(c)).
    SEASONAL_DAYS : 120
        Release designed to span 120 days (Decree Art. III-B-1(d)).
    MAX_ERQ_MG : 70_000.0
        Annual ERQ cap of 70 billion gallons (Decree Art. III-B-1(d)).
    SEASON_START : (6, 15)
        June 15 — seasonal period commencement (Decree Art. III-B-1(d)).
    SEASON_END : (3, 15)
        March 15 — seasonal period termination (Decree Art. III-B-1(d)).
    NYC_ASSUMED_INIT_CONSUMPTION_MGD : 800.0
        Initial NYC consumption assumed for first-year ERQ calculation (= Decree
        maximum allotment Art. III-A-3; ensures ERQ cap is active from model start).
    """

    # --- Decree constants — do not change without amending the Decree ---
    NYC_SAFE_YIELD_FLOOR_MGD: float  = 1_665.0    # Art. III-B-1(c)
    SEASONAL_DAYS: int               = 120         # Art. III-B-1(d)
    MAX_ERQ_MG: float                = 70_000.0    # Art. III-B-1(d): 70 billion gallons
    SEASON_START: tuple              = (6, 15)     # Art. III-B-1(d): June 15
    SEASON_END: tuple                = (3, 15)     # Art. III-B-1(d): March 15
    DEFAULT_ERQ_FRACTION: float      = 0.83        # Art. III-B-1(c): "83 per cent"
    # Decree Art. III-A-3: 800 MGD allotment after Cannonsville — used as initial condition
    NYC_ASSUMED_INIT_CONSUMPTION_MGD: float = 800.0

    def __init__(
        self,
        model,
        erq_fraction: float = 0.83,
        erq_cap_mg: float = 70_000.0,
        nyc_delivery_node_name: str = "delivery_nyc",
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.erq_fraction = erq_fraction
        # Allow overriding the 70 BG seasonal cap (Art. III-B-1(d)) independently.
        # NOTE: At typical pywrdrb NYC consumption (~545 MGD), raw ERQ is always
        # ~261,000+ MG — far above the 70 BG cap.  This means erq_fraction variation
        # has near-zero effect on outputs; erq_cap_mg is the operationally meaningful
        # Sobol parameter (range [50,000, 100,000] MG = 50–100 BG).
        self.erq_cap_mg = erq_cap_mg
        self.nyc_delivery_node_name = nyc_delivery_node_name

    def setup(self):
        super().setup()
        self.num_scenarios = len(self.model.scenarios.combinations)

        # Annual NYC consumption accumulator (resets each June 1)
        self.current_year_consumption_mg = np.zeros(self.num_scenarios, dtype=np.float64)

        # Initial ERQ: assume NYC consumed its full Decree allotment (800 MGD) in year 0.
        # safe_yield_annual = 1665 MGD × 365.25 d = 607,891 MG
        # consumption_assumed = 800 MGD × 365.25 d = 292,200 MG
        # ERQ_initial = 0.83 × (607,891 − 292,200) = 261,993 MG → capped at 70,000 MG
        safe_yield_annual_mg = self.NYC_SAFE_YIELD_FLOOR_MGD * 365.25
        assumed_consumption  = self.NYC_ASSUMED_INIT_CONSUMPTION_MGD * 365.25
        cap = self.erq_cap_mg  # use instance cap (may differ from class constant)
        initial_erq = min(
            self.erq_fraction * max(0.0, safe_yield_annual_mg - assumed_consumption),
            cap,
        )
        self.erq_remaining  = np.full(self.num_scenarios, initial_erq, dtype=np.float64)
        self.erq_daily_rate = np.full(
            self.num_scenarios,
            initial_erq / self.SEASONAL_DAYS,
            dtype=np.float64,
        )

        # Per-scenario daily release committed in value(); consumed in after()
        self.today_release = np.zeros(self.num_scenarios, dtype=np.float64)

        self.datetime = None

    @staticmethod
    def _in_seasonal_period(month: int, day: int) -> bool:
        """
        True during the Art. III-B-1(d) seasonal period: June 15 – March 15.
        The period spans the calendar year-end (June 15 → Dec 31 → March 15).
        """
        if month > 6 or (month == 6 and day >= 15):   # June 15 – Dec 31
            return True
        if month < 3 or (month == 3 and day <= 15):   # Jan 1 – March 15
            return True
        return False

    def value(self, timestep, scenario_index):
        """
        Return today's ERQ release target (MGD) summed across all NYC reservoirs.

        Zero outside the Art. III-B-1(d) seasonal period (June 15 – March 15) or
        when the annual ERQ has been fully released.
        """
        if self.datetime is None:
            self.datetime = pd.Timestamp(self.model.timestepper.current.datetime)

        sid = scenario_index.global_id

        if not self._in_seasonal_period(self.datetime.month, self.datetime.day):
            self.today_release[sid] = 0.0
            return 0.0

        if self.erq_remaining[sid] <= 0.0:
            self.today_release[sid] = 0.0
            return 0.0

        # Daily release = annual ERQ / 120, capped at remaining balance
        release = min(self.erq_daily_rate[sid], self.erq_remaining[sid])
        release = max(release, 0.0)
        self.today_release[sid] = release
        return release

    def after(self):
        """
        Update state after the model solves each timestep:
          1. Decrement erq_remaining by today's committed release.
          2. Accumulate NYC delivery into the annual consumption tracker.
          3. On June 1: compute next seasonal ERQ using prior year's consumption;
             reset the annual accumulator.
        """
        # 1. Subtract today's ERQ release from remaining balance
        self.erq_remaining -= self.today_release
        np.maximum(self.erq_remaining, 0.0, out=self.erq_remaining)

        # 2. Accumulate today's NYC diversion (MGD × 1 day = MG)
        # delivery_nyc.flow is the daily diversion out of the DRB watershed.
        # Art. III-B-1(c) uses "consumption" — for NYC DRB diversions, consumption ≈
        # diversion because diversions leave the watershed entirely.
        try:
            nyc_flow = self.model.nodes[self.nyc_delivery_node_name].flow
            self.current_year_consumption_mg += nyc_flow
        except KeyError:
            pass  # node not in model — consumption tracking silently skipped

        # 3. June 1: compute new seasonal ERQ; reset annual accumulator
        if self.datetime.month == 6 and self.datetime.day == 1:
            safe_yield_annual_mg = self.NYC_SAFE_YIELD_FLOOR_MGD * 365.25
            new_erq = self.erq_fraction * np.maximum(
                0.0,
                safe_yield_annual_mg - self.current_year_consumption_mg,
            )
            np.minimum(new_erq, self.erq_cap_mg, out=new_erq)  # Art. III-B-1(d) cap
            self.erq_remaining[:] = new_erq
            # Daily rate for the new seasonal period (June 15 – Mar 15)
            self.erq_daily_rate[:] = np.where(
                new_erq > 0.0,
                new_erq / self.SEASONAL_DAYS,
                0.0,
            )
            self.current_year_consumption_mg[:] = 0.0

        self.datetime += pd.Timedelta(1, "d")

    @classmethod
    def load(cls, model, data):
        """Load ERQRelease from model JSON dict. Called by pywr at model load time."""
        erq_fraction           = data.pop("erq_fraction",           0.83)
        erq_cap_mg             = data.pop("erq_cap_mg",             70_000.0)
        nyc_delivery_node_name = data.pop("nyc_delivery_node_name", "delivery_nyc")
        return cls(model,
                   erq_fraction=erq_fraction,
                   erq_cap_mg=erq_cap_mg,
                   nyc_delivery_node_name=nyc_delivery_node_name,
                   **data)


ERQRelease.register()


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