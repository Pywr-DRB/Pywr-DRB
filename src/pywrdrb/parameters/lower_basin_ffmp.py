"""
Used to determine the Trenton flow target contributions from the lower basin reservoirs.

Overview:
Lower basin reservoirs (blueMarsh, beltzvilleCombined, nockaixon) can be used 
to help NYC meet the Trenton flow target. These parameters are used to determine if and how 
much these reservoirs should release. 

Technical Notes:
- The lower basin reservoirs use STARFIT as the 'baseline' release. Trenton contributions are added to this baseline release.
- NYC Trenton equivalent flow bank (see pywrdrb.parameters.banks) are used before lower basin reservoirs.
- During "Normal" NYC conditions, only blueMarsh and beltzvilleCombined are used to meet the Trenton target.
- During "Drought" NYC conditions, additional lower basin reservoirs can be used to meet the Trenton target.
- #TODO: account for additional lower basin reservoirs:
    - FE Walter, 
    - Prompton 
    - Wallenpaupack

Links:
NA

Change Log:
TJA, 2025-05-07, Add docs.
"""
from pywr.parameters import Parameter, load_parameter
import numpy as np
from pywrdrb.utils.constants import cfs_to_mgd, epsilon
from pywrdrb.utils.lists import drbc_lower_basin_reservoirs


# During Normal conditions, lower basin reservoirs
# are used to meet the Trenton target, but only BlueMarsh and Beltzville
reservoirs_used_during_normal_conditions = [
    "beltzvilleCombined", "blueMarsh"
]

# During NYC drought conditions, 
# additional lower basin reservoirs can be used
# TODO: Add FE Walter, Prompton, Wallenpaupack
reservoirs_used_during_drought_conditions = [
    "beltzvilleCombined", "blueMarsh", "nockamixon"
]


# Drought emergency lower basin staging levels
# Source: Water Code §2.5.5 Table 1 "Priority of Use for Existing Lower Basin
#         Reservoirs During Drought"
# Items are: [priority_level, reservoir, storage_fraction_lower_bound]
#
# What it says (Table 1 columns: Priority | Operation | Remaining Storage (%) | cfs-days Used):
#   Priority 3: Beltzville to Elev. 615  → 73.7% / 9.89 bg  → 5,475 cfs-days
#   Priority 3: Blue Marsh  to Elev. 283 → 68.9% / 5.13 bg  → 3,595 cfs-days
#   Priority 5: Beltzville to Elev. 590  → 38.0% / 5.10 bg  → 6,364 cfs-days
#   Priority 6: Blue Marsh  to Elev. 273 → 36.8% / 2.74 bg  → 7,411 cfs-days
#   Priority 7: Beltzville to Elev. 537  →  3.4% / 0.45 bg  → 3,700 cfs-days
#   Priority 7: Blue Marsh  to Elev. 261 → 13.0% / 0.97 bg  → 7,198 cfs-days
#   Priority 8: Nockamixon  to Elev. 325.5 → 1.0% / 0.13 bg → 13,745 cfs-days
#   (Prompton and F.E. Walter at priorities 1–2; not currently in model)
#
# Model interpretation: each [priority, reservoir, fraction] entry defines the storage
# floor below which that reservoir stops being considered for Trenton MRF contribution
# at that priority level. LowerBasinMaxMRFContribution iterates this list to determine
# whether the current reservoir has usable storage above the relevant priority floor.
#
# Considerations: priorities 1-2 (Prompton, F.E. Walter) are excluded — see TODO in
# module header. The priority numbers here are re-indexed from the Table 1 values to
# a 1-6 internal scale; the absolute priority level matters only for sequencing.
priority_use_during_drought = [
    [1, "beltzvilleCombined", 0.737],
    [1, "blueMarsh", 0.689],
    [2, "nockamixon", 0.687],
    [3, "beltzvilleCombined", 0.380],
    [4, "blueMarsh", 0.368],
    [5, "beltzvilleCombined", 0.034],
    [5, "blueMarsh", 0.130],
    [6, "nockamixon", 0.010],
]

## Max storage for lower basin reservoirs
# STARFIT relevant max storage often corresponds to flood storage
# Some reservoirs (notably blueMarsh) are operated at much lower % of max flood storage
# These are chosen to match prescribed priority staging levels 
# in the Water Code relative to "usable storage"
# Taken from https://drbc.maps.arcgis.com/apps/dashboards/690464a9958b49e5b49550964641ffd7
drbc_max_usable_storages = {
    "beltzvilleCombined": 13500,
    "blueMarsh": 7450,
    "nockamixon": 13000,
}

# Max discharges at lower reservoirs
# NOTE: Used in the STARFIT parameter
max_discharges = {
    "blueMarsh": 1500 * cfs_to_mgd,
    "beltzvilleCombined": 1500 * cfs_to_mgd,
    "nockamixon": 1000 * cfs_to_mgd,
    "fewalter": 2000 * cfs_to_mgd,
}


## Max daily MRF contributions from lower basin reservoirs for trenton target flow
# Rough estimate based on observed release patterns during periods where lower basin was used
max_mrf_daily_contributions = {
    "blueMarsh": 300,
    "beltzvilleCombined": 300,
    "nockamixon": 300,
    "fewalter": 2000 * cfs_to_mgd,
}

### lag days from Trenton for each drbc lower basin reservoir
lag_days_from_Trenton = {
    "blueMarsh": 2,
    "beltzvilleCombined": 2,
    "nockamixon": 1,
    "fewalter": 2,
}

## Conservation releases at lower reservoirs
# Source: Water Code §2.5.5 Table 4 "Conservation Releases"
#
# What it says (verbatim table):
#
#   Reservoir    Normal (cfs)    Drought Warning & Drought (cfs)
#   ─────────────────────────────────────────────────────────────
#   F.E. Walter     50               43
#   Prompton        inflow-outflow    6
#   Beltzville      35               15
#   Blue Marsh      41 (50*)         21 (30*)
#   Nockamixon      11                7
#
#   * With Western Berks Water Authority release included.  As the future needs
#     of the Authority increase, the release will correspondingly increase.
#
# Trigger: §2.5.6.C.4 — "conservation releases from the listed lower basin reservoirs
#   shall be modified as set forth in Table 4 of Section 2.5.5 commencing with the
#   declaration of a lower basin 'drought warning' condition"
#
# Model interpretation: uses WBWA-inclusive values (50/30 cfs Blue Marsh) throughout,
#   matching current operational practice.  drought_level_agg_lb >= 1 (any drought
#   stage active, including Warning) triggers the switch to drought releases.
#
# Considerations: same reduced releases apply to both Drought Warning (stage 1) and
#   Drought (stage 2), consistent with "commencing with the declaration of a lower
#   basin 'drought warning' condition."  Blue Marsh base-table value is 41/21 cfs;
#   model uses 50/30 cfs (WBWA-inclusive) — correct for post-2017 operations.
#   LB drought stage switching NOW IMPLEMENTED via drought_level_agg_lb parameter.
conservation_releases = {
    "blueMarsh": 50 * cfs_to_mgd,           # Table 4: 41 cfs + WBWA → 50 cfs
    "beltzvilleCombined": 35 * cfs_to_mgd,  # Table 4: 35 cfs (normal)
    "nockamixon": 11 * cfs_to_mgd,          # Table 4: 11 cfs (normal)
    "fewalter": 50 * cfs_to_mgd,            # Table 4: 50 cfs (normal)
}

# Drought Warning & Drought releases — Water Code §2.5.5 Table 4
# Applied when drought_level_agg_lb >= 1 (stage 1 or 2 active).
# What it says: "Drought Warning and Drought Conservation Releases (cfs)"
#
# Model interpretation: LowerBasinMaxMRFContribution.value() switches to this dict
#   whenever drought_level_agg_lb.get_value(scenario_index) >= 1.
#
# Considerations: F.E. Walter and Prompton included for completeness; these reservoirs
#   are not in reservoirs_used_during_drought_conditions so their entries are unused
#   in practice (STARFIT handles their releases independently).
lower_basin_drought_conservation_releases = {
    "blueMarsh": 30 * cfs_to_mgd,           # Table 4: 21 cfs + WBWA → 30 cfs
    "beltzvilleCombined": 15 * cfs_to_mgd,  # Table 4: 15 cfs (drought warning/drought)
    "nockamixon": 7 * cfs_to_mgd,           # Table 4:  7 cfs (drought warning/drought)
    "fewalter": 43 * cfs_to_mgd,            # Table 4: 43 cfs (drought warning/drought)
}

class LowerBasinMaxMRFContribution(Parameter):
    """Used to track the max of allowable release from a lower basin reservoir based on policy and current conditions.

    If predicted Trenton flow > flow target, then max allowable release is 0.0 (no contribution needed).
    If predicted Trenton flow < flow target, then max allowable release is determined by:
    - the current storage of the reservoir 
    - the priority level of the reservoir, as described in WaterCode.

    This is provided to VolBalanceLowerBasinMRFAggregate to determine the aggregate
    MRF contribution from the lower basin reservoirs.
    """
    def __init__(self, model, reservoir, 
                 step, nodes, parameters, **kwargs):
        """Initialize the LowerBasinMaxMRFContribution parameter.
        
        Parameters
        ----------
        model : Model
            The pywrdrb.Model instance.
        reservoir : str
            The name of the reservoir, e.g. "blueMarsh", "beltzvilleCombined", "nockamixon".
        step : int
            The step of the Trenton flow calculation (1-4). See pywrdrb.parameters.ffmp for more.
        nodes : dict
            Dictionary of pywrdrb.nodes.Reservoir instances for each lower basin reservoir. Format {"reservoir_name": pywr.Node}.
        parameters : dict
            Dictionary of pywrdrb.parameters.Parameter instances for each lower basin reservoir. Format {"parameter_name": pywr.Parameter}.
        **kwargs : dict
            Additional keyword arguments to pass to the pywr.Parameter class.
            cap_multiplier (float): scales max_mrf_daily_contribution (default 1.0).
            
        Returns
        -------
        None
        """
        cap_multiplier = float(kwargs.pop("cap_multiplier", 1.0))
        super().__init__(model, **kwargs)
        self.debugging = False
        self.reservoir = reservoir
        self.step = step
        self.days_ahead_prediction = 5 - self.step
        self.nodes = nodes
        self.parameters = parameters
        self.cap_multiplier = cap_multiplier

        # Reservoirs considered during this step
        self.consider_reservoirs = (
            drbc_lower_basin_reservoirs if (self.step > 1) else ["nockamixon"]
        )

        # Current NYC drought level
        self.drought_level_agg_nyc = self.parameters["drought_level_agg_nyc"]
        self.children.add(self.drought_level_agg_nyc)

        # Current LB drought level (Water Code §2.5.6; None if not yet in model)
        self.drought_level_agg_lb = self.parameters.get("drought_level_agg_lb", None)
        if self.drought_level_agg_lb is not None:
            self.children.add(self.drought_level_agg_lb)

        # Trenton flow requirement
        self.release_needed_mrf_trenton = self.parameters["release_needed_mrf_trenton"]
        self.children.add(self.release_needed_mrf_trenton)

        # Max storage scaling to match priority staging
        self.max_volumes = drbc_max_usable_storages

        # R_min (conservation release) is now dynamic — computed per-timestep in value()
        # based on drought_level_agg_lb.  Keep references to both release tables.
        self.conservation_releases_normal  = conservation_releases
        self.conservation_releases_drought = lower_basin_drought_conservation_releases

        self.max_mrf_daily_contribution = max_mrf_daily_contributions[self.reservoir]

        assert (
            self.max_mrf_daily_contribution is not None
        ), f"LowerBasinMaxMRFContribution: max_mrf_daily_contribution is None"

    def _print_allowable_storage_info(self, 
                                     max_allowable, 
                                     current_reservoir_priority, 
                                     percent_storage,
                                     ):
        """
        Print info about the allowable storage for this reservoir of interest for debugging purposes.
        Info includes max allowable contribution, relevant priority level, and current storage.
        """
        if max_allowable > 0.0:
            print(
                f"{self.reservoir} max allowed contr {max_allowable}, priority: {current_reservoir_priority}, S_hat: {percent_storage}"
            )
        else:
            print(
                f"{self.reservoir} not allowed for MRF, priority: {current_reservoir_priority}, S_hat: {percent_storage}"
            )

    def get_current_usable_reservoirs(self, scenario_index):
        """
        Get lower basin reservoirs that are allowed to be used for Trenton flow target currently.

        During NYC Normal conditions, only BlueMarsh and Beltzville are used.
        Additional reservoirs become available when either NYC is in drought emergency
        (drought_level_agg_nyc == 6) OR the LB is in any drought stage
        (drought_level_agg_lb >= 1, i.e. LB Drought Warning or LB Drought).

        Parameters
        ----------
        scenario_index : ScenarioIndex
            The index of the simulation scenario.

        Returns
        -------
        list
            Names of currently usable lower basin reservoirs.
        """
        # NYC drought emergency check
        current_nyc_drought_level = self.drought_level_agg_nyc.get_value(scenario_index)
        is_nyc_drought_emergency = current_nyc_drought_level >= 6

        # LB drought check (any stage above Normal activates expanded reservoir list)
        is_lb_drought = False
        if self.drought_level_agg_lb is not None:
            current_lb_drought_level = self.drought_level_agg_lb.get_value(scenario_index)
            is_lb_drought = current_lb_drought_level >= 1

        if is_nyc_drought_emergency or is_lb_drought:
            usable_reservoirs = reservoirs_used_during_drought_conditions
        else:
            usable_reservoirs = reservoirs_used_during_normal_conditions
        return usable_reservoirs


    def value(self, timestep, scenario_index):
        """Get total allowable contributions from lower basin reservoirs.
        
        This is automatically called by pywr during simulation.
        
        Parameters
        ----------
        timestep : Timestep
            The timestep being evaluated.
        scenario_index : ScenarioIndex
            The index of the simulation scenario.
            
        Returns
        -------
        float
            The max allowable contribution from this reservoir to the Trenton flow target.
        """

        # if reservoir is further lag from Trenton than our current step is working on
        # (eg Beltzville & BLue Marsh in Step 4), return 0
        if lag_days_from_Trenton[self.reservoir] > (5 - self.step):
            return 0.0
        
        # If no added flow is needed for Trenton,
        # then return 0.0
        trenton_requirement = self.release_needed_mrf_trenton.get_value(scenario_index)
        if trenton_requirement < epsilon:
            return 0.0

        # Determine which reservoirs are usable
        # If this reservoir is not in the usable reservoirs, then return 0
        usable_reservoirs = self.get_current_usable_reservoirs(scenario_index)
        if self.reservoir not in usable_reservoirs:
            return 0.0

        ### Determine the max allowable contribution from this reservoir

        # Determine conservation release (R_min) based on current LB drought stage
        # Water Code Table 4: drought-warning / drought releases are lower than normal releases,
        # conserving storage for later use.
        if (self.drought_level_agg_lb is not None and
                self.drought_level_agg_lb.get_value(scenario_index) >= 1):
            R_min = self.conservation_releases_drought[self.reservoir]
        else:
            R_min = self.conservation_releases_normal[self.reservoir]

        # We want to return the max allowable contribution from this reservoir
        # But need to consider storages and priority staging of each lower basin reservoir
        percent_storages = {}
        max_allowable_releases = {}
        max_allowable = 0.0
        already_used_reservoirs = []

        # Find what priority level each lower basin reservoir is in
        # And corresponding usable storage
        for res in usable_reservoirs:
            # Get current storage percentage
            S_max = self.max_volumes[res]
            S_t = self.nodes[res].volume[scenario_index.indices]

            # Add inflow and remove required conservation releases from storage
            inflow = self.parameters[f"flow_{res}"].get_value(scenario_index)
            S_t += inflow
            S_t -= R_min

            # Storage as fraction of max storage
            S_hat_t = min(S_t / S_max, 1.0)
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
                available_mrf_daily_release = usable_storage / max(
                    self.days_ahead_prediction - lag_days_from_Trenton[res] + 1, 1
                )

                if available_mrf_daily_release > 0.0:
                    max_allowable_releases[res] = [
                        priority_level,
                        available_mrf_daily_release,
                    ]
                    already_used_reservoirs.append(res)
                else:
                    max_allowable_releases[res] = [999, 0.0]

        for res in usable_reservoirs:
            assert (
                res in max_allowable_releases.keys()
            ), f"LowerBasinMaxMRFContribution: {res} not in max_allowable_releases.keys()"

        # Find highest priority level to use
        all_priority_levels = [
            max_allowable_releases[res][0] for res in max_allowable_releases.keys()
        ]
        active_priority = min(all_priority_levels)

        # if step 4, then only nockamixon is able to be used
        if self.step == 4 and 'nockamixon' in usable_reservoirs:
            active_priority = max_allowable_releases["nockamixon"][0]

        # Now set max allowable for this specific reservoir depending if it is highest priority
        current_reservoir_priority = max_allowable_releases[self.reservoir][0]
        if current_reservoir_priority == active_priority:
            max_allowable = max_allowable_releases[self.reservoir][1]

            # Constraints :
            # Dont release more than historically used for MRF (estimated from obs)
            if max_allowable >= self.max_mrf_daily_contribution * self.cap_multiplier:
                max_allowable = self.max_mrf_daily_contribution * self.cap_multiplier

            if self.debugging:
                self._print_allowable_storage_info(
                    max_allowable,
                    current_reservoir_priority,
                    percent_storages[self.reservoir],
                )

            # Return max allowable (ensure scalar for Pywr parameter API)
            assert (
                max_allowable is not None
            ), f"LowerBasinMaxMRFContribution: max_allowable is None"
            x = np.asarray(max_allowable).ravel()
            return float(x[0])
        else:
            return 0.0


    @classmethod
    def load(cls, model, data):
        """Load the LowerBasinMaxMRFContribution parameter.
        
        Parameters
        ----------
        model : Model
            The pywrdrb.Model instance.
        data : dict
            Dictionary of data to load the parameter from. Must include "node" and "step".
        
        Returns
        -------
        LowerBasinMaxMRFContribution
            The loaded LowerBasinMaxMRFContribution parameter.
        """

        reservoir = data.pop("node")
        step = data.pop("step")
        reservoir = reservoir.split("_")[1]
        nodes = {}
        parameters = {}
        for res in drbc_lower_basin_reservoirs:
            nodes[res] = model.nodes[f"reservoir_{res}"]
            parameters[f"flow_{res}"] = load_parameter(model, f"flow_{res}")
        parameters["drought_level_agg_nyc"] = load_parameter(
            model, f"drought_level_agg_nyc"
        )

        # LB drought level (Water Code §2.5.6) — graceful fallback if not yet in model
        try:
            parameters["drought_level_agg_lb"] = load_parameter(
                model, "drought_level_agg_lb"
            )
        except KeyError:
            parameters["drought_level_agg_lb"] = None

        # Trenton MRF contributions required by NYC and Lower Basin
        # (beyond releases already needed for Montague)
        parameters["release_needed_mrf_trenton"] = load_parameter(
            model, f"release_needed_mrf_trenton_step{step}"
        )
        return cls(model, reservoir, step, nodes, parameters, **data)


LowerBasinMaxMRFContribution.register()


class VolBalanceLowerBasinMRFAggregate(Parameter):
    """
    Calculates total Trenton contribution from all lower basin reservoirs.
    
    Methods
    -------
    value(timestep, scenario_index)
        Checks how much lower basin reservoir release are needed & allowed to contribute to Trenton.
    load(model, data)
        Load the VolBalanceLowerBasinMRFAggregate parameter.
    
    Attributes
    ----------
    release_needed_mrf_trenton : Parameter
        Required MRF contributions for Trenton.
    lower_basin_max_mrf_contributions : dict
        Maximum MRF contributions from each lower basin reservoir.
    drbc_lower_basin_reservoirs : list
        List of lower basin reservoirs.
    """

    def __init__(self,
                model,
                release_needed_mrf_trenton,
                lower_basin_max_mrf_contributions,
                **kwargs,):
        """
        Initialize the VolBalanceLowerBasinMRFAggregate parameter.
        
        Parameters
        ----------
        model : Model
            The pywrdrb.Model instance.
        release_needed_mrf_trenton : Parameter
            Required contributions to maintain Trenton target.
        lower_basin_max_mrf_contributions : dict
            Maximum contributions from each lower basin reservoir.
        **kwargs
            Additional keyword arguments.
        """
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
        Calculate aggregate Trenton contribution from lower basin reservoirs.
        
        Automatically called by pywr during simulation.
        
        Parameters
        ----------
        timestep : Timestep
            The current timestep.
        scenario_index : ScenarioIndex
            The index of the scenario.

        Returns
        -------
        float
            Aggregate contribution in MGD.
        """
        # If Montague/Trenton MRF target is 0, then return 0
        if self.release_needed_mrf_trenton.get_value(scenario_index) < epsilon:
            return 0.0

        # If Montague/Trenton MRF target is not 0, then handle partitioning
        else:
            # Check if lower basin releases are allowed (must be FFMP drought)
            max_aggregate_allowed = sum(
                [
                    self.lower_basin_max_mrf_contributions[reservoir].get_value(
                        scenario_index
                    )
                    for reservoir in self.drbc_lower_basin_reservoirs
                ]
            )

            assert (
                max_aggregate_allowed is not None
            ), f"VolBalanceLowerBasinMRFAggregate: max_aggregate_allowed is None"

            # If no lower basin releases are allowed, then return 0
            if max_aggregate_allowed < epsilon:
                return 0.0
            # Otherwise determine the total lower basin contribution to MRF
            else:
                return min(
                    self.release_needed_mrf_trenton.get_value(scenario_index),
                    max_aggregate_allowed,
                )

    @classmethod
    def load(cls, model, data):
        """
        Load the parameter from model data.
        
        Parameters
        ----------
        model : Model
            The pywrdrb.Model instance.
        data : dict
            Parameter data from the model configuration.
            
        Returns
        -------
        VolBalanceLowerBasinMRFAggregate
            Initialized parameter instance.
        """

        step = data.pop("step")

        # Trenton MRF contributions required by NYC and Lower Basin (beyond releases already needed for Montague)
        release_needed_mrf_trenton = load_parameter(
            model, f"release_needed_mrf_trenton_step{step}"
        )

        # Lower basin reservoir max Trenton contributions
        lower_basin_max_mrf_contributions = {}
        for r in drbc_lower_basin_reservoirs:
            lower_basin_max_mrf_contributions[r] = load_parameter(
                model, f"max_mrf_trenton_step{step}_{r}"
            )
        return cls(
            model, release_needed_mrf_trenton, lower_basin_max_mrf_contributions, **data
        )


VolBalanceLowerBasinMRFAggregate.register()


class VolBalanceLowerBasinMRFIndividual(Parameter):
    """
    Distributes Trenton contributions among individual lower basin reservoirs.
    
    Allocates the required aggregate contribution to individual
    reservoirs based on their capacity and priority staging.
    
    Methods
    -------
    split_lower_basin_mrf_contributions(scenario_index)
        Split the MRF contributions from the lower basin reservoirs into individual reservoir contributions.
    value(timestep, scenario_index)
        Return the contribution from this specific reservoir.
    load(model, data)
        Load and initialize the VolBalanceLowerBasinMRFIndividual parameter.
    
    Attributes
    ----------
    reservoir : str
        Name of the reservoir.
    lower_basin_agg_mrf_trenton : Parameter
        Aggregate MRF contribution required from lower basin.
    lower_basin_max_mrf_contributions : dict
        Maximum MRF contributions from each lower basin reservoir.
    drbc_lower_basin_reservoirs : list
        List of lower basin reservoirs.
    """
    def __init__(self,
                model,
                reservoir,
                lower_basin_agg_mrf_trenton,
                lower_basin_max_mrf_contributions,
                **kwargs,
                ):
        """
        Initialize the VolBalanceLowerBasinMRFIndividual parameter.
        
        Parameters
        ----------
        model : Model
            The pywrdrb.Model instance.
        reservoir : str
            Name of the reservoir.
        lower_basin_agg_mrf_trenton : Parameter
            Aggregate MRF contribution required from lower basin.
        lower_basin_max_mrf_contributions : dict
            Maximum MRF contributions from each lower basin reservoir.
        **kwargs
            Additional keyword arguments.
        """
        super().__init__(model, **kwargs)
        self.debugging = False
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
        Split aggregate Trenton contribution among individual reservoirs.
                
        Parameters
        ----------
        scenario_index : ScenarioIndex
            The index of the scenario.
            
        Returns
        -------
        dict
            Dictionary mapping reservoir names to their MRF contributions.
        """
        # Get total allowable MRF contribution from lower basin reservoirs
        requirement_total = self.lower_basin_agg_mrf_trenton.get_value(scenario_index)

        # Get individual allowable contributions
        # Currently naive handling of priority ordering of lower basin reservoirs
        individual_contributions = {}
        requirement_remaining = requirement_total

        total_max_contribution = sum(
            [
                self.lower_basin_max_mrf_contributions[reservoir].get_value(
                    scenario_index
                )
                for reservoir in self.drbc_lower_basin_reservoirs
            ]
        )
        assert (
            total_max_contribution >= requirement_total
        ), f"VolBalanceLowerBasinMRFIndividual: total_max_contribution < requirement_total"

        for reservoir in drbc_lower_basin_reservoirs:
            if requirement_remaining > 0.0:
                individual_max = self.lower_basin_max_mrf_contributions[
                    reservoir
                ].get_value(scenario_index)
                individual_contributions[reservoir] = min(
                    individual_max, requirement_remaining
                )
                requirement_remaining -= individual_contributions[reservoir]
            else:
                individual_contributions[reservoir] = 0.0

        assert (
            requirement_remaining < epsilon
        ), f"VolBalanceLowerBasinMRFIndividual: requirement_remaining is not 0: {requirement_remaining}"
        diff = sum(individual_contributions.values()) - requirement_total
        if abs(diff) > epsilon:
            print(f"diff: {diff}")
            print(
                f"sum individual_contributions: {sum(individual_contributions.values())}"
            )
            print(f"requirement_total: {requirement_total}")
            print(f"requirement_remaining: {requirement_remaining}")
            print(
                f'Blue Marsh max: {self.lower_basin_max_mrf_contributions["blueMarsh"].get_value(scenario_index)}'
            )
            print(
                f'Beltzville max: {self.lower_basin_max_mrf_contributions["beltzvilleCombined"].get_value(scenario_index)}'
            )
            print(
                f'Nockamixon max: {self.lower_basin_max_mrf_contributions["nockamixon"].get_value(scenario_index)}'
            )
            print(f'BlueMarsh contribution: {individual_contributions["blueMarsh"]}')
            print(
                f'Beltzville contribution: {individual_contributions["beltzvilleCombined"]}'
            )
            print(f'Nockamixon contribution: {individual_contributions["nockamixon"]}')

        assert (
            abs(diff) < epsilon
        ), f"VolBalanceLowerBasinMRFIndividual: sum(individual_constributions) != requirement_total"
        return individual_contributions

    def value(self, timestep, scenario_index):
        """
        Calculate Trenton contribution for this specific reservoir.
        
        Automatically called by pywr during simulation.
                
        Parameters
        ----------
        timestep : Timestep
            The current timestep.
        scenario_index : ScenarioIndex
            The index of the scenario.

        Returns
        -------
        float
            Individual reservoir's contribution in MGD. Added to baseline starfit release.
        """

        # Check if lower basin is used for MRF contribution
        if self.lower_basin_agg_mrf_trenton.get_value(scenario_index) < epsilon:
            return 0.0

        # Otherwise, split required contributions
        else:
            # Get individual reservoir contributions
            individual_targets = self.split_lower_basin_mrf_contributions(
                scenario_index
            )
            # Return the contribution from the reservoir of interest
            release = individual_targets[self.reservoir]
            if self.debugging:
                print(
                    f"S{scenario_index}: {timestep} Lower Basin MRF releases: {individual_targets}"
                )
            return release

    @classmethod
    def load(cls, model, data):
        """
        Load the parameter from model data.
        
        Parameters
        ----------
        model : Model
            The pywrdrb.Model instance.
        data : dict
            Parameter data from the model configuration.
            
        Returns
        -------
        VolBalanceLowerBasinMRFIndividual
            Initalized parameter instance.
        """
        reservoir = data.pop("node")
        reservoir = reservoir.split("_")[1]
        step = data.pop("step")

        lower_basin_agg_mrf_trenton = load_parameter(
            model, f"lower_basin_agg_mrf_trenton_step{step}"
        )

        # Lower basin reservoir max Trenton contributions
        lower_basin_max_mrf_contributions = {}
        for r in drbc_lower_basin_reservoirs:
            lower_basin_max_mrf_contributions[r] = load_parameter(
                model, f"max_mrf_trenton_step{step}_{r}"
            )

        return cls(
            model,
            reservoir,
            lower_basin_agg_mrf_trenton,
            lower_basin_max_mrf_contributions,
            **data,
        )


VolBalanceLowerBasinMRFIndividual.register()
