"""
Flood stage monitoring parameters for Pywr-DRB.

Provides parameters for:
1. Converting discharge to stage using USGS rating curves
2. Tracking flood conditions based on stage height thresholds

Flood Categories:
    0 = Normal (below action stage)
    1 = Action stage (flood operations triggered)
    2 = Minor flood (flood stage reached)
    3 = Moderate flood
    4 = Major flood

Change Log:
TJA, 2026-01-08, Initial implementation for flood monitoring.
TJA, 2026-01-08, Consolidated rating_curves.py and flood.py into single module.
TJA, 2026-01-08, Added warning suppression for rating curve extrapolation.
"""

from pywr.parameters import Parameter, load_parameter
from pywrdrb.utils.constants import cfs_to_mgd
from pywrdrb.utils.rating_curves import load_all_flood_monitoring_curves
from pywrdrb.flood_thresholds import flood_stage_thresholds
import warnings


class StageFromDischargeParameter(Parameter):
    """
    Converts discharge to stage using USGS rating curve.

    Uses log-log interpolation of rating curve data for accurate stage
    estimation across the full range of flows.

    Parameters
    ----------
    model : Model
        Pywr model instance
    node : str
        Node name to monitor (e.g., '01426500')
    site_no : str, optional
        USGS site number (defaults to node name)

    Returns
    -------
    float
        Stage in feet above gage datum

    Notes
    -----
    Rating curves are loaded once at class level and shared across all
    instances for efficiency. Extrapolation warnings are issued only once
    per site to avoid log clutter in ensemble simulations.
    """

    # Class-level cache for rating curves (loaded once)
    _rating_curves = None

    # Class-level set to track which sites have issued extrapolation warnings
    _extrapolation_warned = set()

    def __init__(self, model, node, site_no=None, **kwargs):
        super().__init__(model, **kwargs)
        self.node_name = node
        self.site_no = site_no if site_no else node

        # Load rating curves on first instantiation
        if StageFromDischargeParameter._rating_curves is None:
            StageFromDischargeParameter._rating_curves = load_all_flood_monitoring_curves()

        # Get rating curve for this site
        if self.site_no not in self._rating_curves:
            raise ValueError(
                f"No rating curve available for site {self.site_no}. "
                f"Available sites: {list(self._rating_curves.keys())}"
            )

        self.rating_curve = self._rating_curves[self.site_no]
        self._node = None

        # Flag to track if this instance has checked extrapolation
        self._has_extrapolated = False

    def setup(self):
        super().setup()
        # Get reference to the link node
        try:
            self._node = self.model.nodes[f"link_{self.node_name}"]
        except KeyError:
            raise KeyError(
                f"Node 'link_{self.node_name}' not found in model. "
                f"Ensure flood monitoring nodes are added to the network."
            )

    def value(self, timestep, scenario_index):
        """
        Compute stage from current discharge.

        Returns
        -------
        float
            Stage in feet above gage datum

        Notes
        -----
        Extrapolation warnings are issued only once per site to avoid
        excessive warnings during ensemble simulations.
        """
        # Get flow in MGD
        flow_mgd = self._node.prev_flow[scenario_index.global_id]

        # Convert to CFS
        flow_cfs = flow_mgd / cfs_to_mgd

        # Check if we need to warn about extrapolation
        # Only warn once per site (not once per parameter instance)
        if not self._has_extrapolated:
            if (flow_cfs < self.rating_curve.discharge_min or
                flow_cfs > self.rating_curve.discharge_max):

                # Check if we've already warned for this site
                if self.site_no not in StageFromDischargeParameter._extrapolation_warned:
                    warnings.warn(
                        f"Site {self.site_no}: Discharge values outside rating curve range "
                        f"[{self.rating_curve.discharge_min:.1f}, {self.rating_curve.discharge_max:.1f}] cfs. "
                        f"Using endpoint extrapolation. (This warning will only appear once per site.)",
                        UserWarning
                    )
                    # Mark this site as having been warned
                    StageFromDischargeParameter._extrapolation_warned.add(self.site_no)

                # Mark this instance as having checked extrapolation
                self._has_extrapolated = True

        # Convert to stage using rating curve
        # Note: The rating curve utility handles extrapolation internally
        stage_ft = self.rating_curve.discharge_to_stage(flow_cfs)

        return stage_ft

    @classmethod
    def load(cls, model, data):
        node = data.pop("node")
        site_no = data.pop("site_no", None)
        return cls(model, node, site_no=site_no, **data)


class FloodLevelIndicator(Parameter):
    """
    Returns integer flood level (0-4) based on stage at a monitoring location.

    Uses stage-based thresholds from FFMP and NWS flood categories to
    classify current flood conditions.

    Parameters
    ----------
    model : Model
        Pywr model instance
    stage_parameter : str or Parameter
        Parameter that provides stage in feet
    location : str
        Location identifier (e.g., '01426500', 'delMontague')

    Returns
    -------
    int
        Flood level category:
            0 = Normal (below action stage)
            1 = Action stage (flood operations may be triggered)
            2 = Minor flood stage
            3 = Moderate flood stage
            4 = Major flood stage

    Notes
    -----
    Thresholds are defined in pywrdrb.flood_thresholds and sourced from
    USGS/NWS Advanced Hydrologic Prediction Service (AHPS) and FFMP provisions.
    """

    def __init__(self, model, stage_parameter, location, **kwargs):
        super().__init__(model, **kwargs)

        # Handle stage parameter
        if isinstance(stage_parameter, str):
            self.stage_param_name = stage_parameter
            self.stage_parameter = None
        else:
            self.stage_parameter = stage_parameter
            self.children.add(stage_parameter)

        # Get thresholds for this location
        if location not in flood_stage_thresholds:
            raise ValueError(
                f"No flood thresholds defined for location '{location}'. "
                f"Available locations: {list(flood_stage_thresholds.keys())}"
            )

        self.location = location
        self.thresholds = flood_stage_thresholds[location]

    def setup(self):
        super().setup()
        if self.stage_parameter is None:
            self.stage_parameter = self.model.parameters[self.stage_param_name]
            self.children.add(self.stage_parameter)

    def value(self, timestep, scenario_index):
        """
        Determine flood category based on current stage.

        Returns
        -------
        int
            Flood level category (0-4)
        """
        stage = self.stage_parameter.get_value(scenario_index)

        if stage >= self.thresholds["major"]:
            return 4
        elif stage >= self.thresholds["moderate"]:
            return 3
        elif stage >= self.thresholds["minor"]:
            return 2
        elif stage >= self.thresholds["action"]:
            return 1
        else:
            return 0

    @classmethod
    def load(cls, model, data):
        stage_parameter = data.pop("stage_parameter")
        location = data.pop("location")
        return cls(model, stage_parameter, location, **data)


# Register parameters with Pywr
StageFromDischargeParameter.register()
FloodLevelIndicator.register()
