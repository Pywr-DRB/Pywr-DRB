import numpy as np
import pandas as pd
import math

from pywr.parameters import Parameter, load_parameter

from pywrdrb.utils.directories import model_data_dir
from pywrdrb.utils.constants import cfs_to_mgd
from pywrdrb.utils.lists import modified_starfit_reservoir_list
from pywrdrb.parameters.lower_basin_ffmp import conservation_releases, max_discharges


class STARFITReservoirRelease(Parameter):
    """
    Custom Pywr Parameter used to implement the STARFIT-inferred reservoir operations policy at non-NYC reservoirs following Turner et al. (2021).

    Attributes:
        model (Model): The PywrDRB model.
        storage_node (str): The storage node associated with the reservoir.
        flow_parameter: The PywrDRB catchment inflow parameter corresponding to the reservoir.

    Methods:
        value(timestep, scenario_index): returns the STARFIT-inferred reservoir release for the current timestep and scenario index
    """

    def __init__(
        self,
        model,
        reservoir_name,
        storage_node,
        flow_parameter,
        run_starfit_sensitivity_analysis,
        sensitivity_analysis_scenarios,
        **kwargs,
    ):
        super().__init__(model, **kwargs)

        self.node = storage_node
        self.reservoir_name = reservoir_name
        self.inflow = flow_parameter

        # Add children
        self.children.add(flow_parameter)

        # Check if parameters have been loaded
        self.parameters_loaded = False
        # Load the sample scenario IDs
        self.sample_scenario_index = None
        self.run_sensitivity_analysis = run_starfit_sensitivity_analysis
        self.sensitivity_analysis_scenarios = sensitivity_analysis_scenarios

        # Modifications to
        self.remove_R_max = False
        self.linear_below_NOR = False
        use_adjusted_storage = True
        self.WATER_YEAR_OFFSET = 0

    def load_default_starfit_params(self, model_data_dir):
        """
        Load default STARFIT parameters from istarf_conus.csv

        Args:
        model_data_dir (str): The path to the model data directory.

        Returns:
        pd.DataFrame: The default STARFIT parameters.
        """

        return pd.read_csv(
            f"{model_data_dir}drb_model_istarf_conus.csv", sep=",", index_col=0
        )

    def load_starfit_sensitivity_samples(self, sample_scenario_id):
        """
        Load STARFIT sensitivity samples from an HDF5 file.

        Args:
        sample_scenario_id (int): The sample scenario ID.

        Returns:
        pd.DataFrame: The STARFIT sensitivity samples for the given scenario ID.
        """
        samples = f"/starfit/scenario_{sample_scenario_id}"

        # Load the data from the HDF5 file using pandas
        df = pd.read_hdf(f"{model_data_dir}scenarios_data.h5", key=samples)
        df.set_index("reservoir", inplace=True)

        return df

    def assign_starfit_param_values(self, starfit_params):
        """
        Assign STARFIT parameter values to the reservoir.

        Args:
            name (str): The name of the reservoir.
            starfit_params (pd.DataFrame): The STARFIT parameters.
        """
        ## Load STARFIT parameters

        use_adjusted_storage = True

        # Use modified storage parameters for DRBC relevant reservoirs
        if self.name in modified_starfit_reservoir_list:
            self.starfit_name = "modified_" + self.name
        else:
            self.starfit_name = self.reservoir_name

        # Check if parameters are available
        if self.starfit_name not in starfit_params.index:
            print(f"Warning: No STARFIT parameters found for '{self.starfit_name}'.")
            return

        # Pull data from node
        if use_adjusted_storage:
            self.S_cap = starfit_params.loc[self.starfit_name, "Adjusted_CAP_MG"]
            self.I_bar = starfit_params.loc[self.starfit_name, "Adjusted_MEANFLOW_MGD"]

        else:
            self.S_cap = starfit_params.loc[self.starfit_name, "GRanD_CAP_MG"]
            self.I_bar = starfit_params.loc[self.starfit_name, "GRanD_MEANFLOW_MGD"]

        # Store STARFIT parameters
        self.NORhi_mu = starfit_params.loc[self.starfit_name, "NORhi_mu"]
        self.NORhi_min = starfit_params.loc[self.starfit_name, "NORhi_min"]
        self.NORhi_max = starfit_params.loc[self.starfit_name, "NORhi_max"]
        self.NORhi_alpha = starfit_params.loc[self.starfit_name, "NORhi_alpha"]
        self.NORhi_beta = starfit_params.loc[self.starfit_name, "NORhi_beta"]

        self.NORlo_mu = starfit_params.loc[self.starfit_name, "NORlo_mu"]
        self.NORlo_min = starfit_params.loc[self.starfit_name, "NORlo_min"]
        self.NORlo_max = starfit_params.loc[self.starfit_name, "NORlo_max"]
        self.NORlo_alpha = starfit_params.loc[self.starfit_name, "NORlo_alpha"]
        self.NORlo_beta = starfit_params.loc[self.starfit_name, "NORlo_beta"]

        self.Release_alpha1 = starfit_params.loc[self.starfit_name, "Release_alpha1"]
        self.Release_alpha2 = starfit_params.loc[self.starfit_name, "Release_alpha2"]
        self.Release_beta1 = starfit_params.loc[self.starfit_name, "Release_beta1"]
        self.Release_beta2 = starfit_params.loc[self.starfit_name, "Release_beta2"]

        self.Release_c = starfit_params.loc[self.starfit_name, "Release_c"]
        self.Release_p1 = starfit_params.loc[self.starfit_name, "Release_p1"]
        self.Release_p2 = starfit_params.loc[self.starfit_name, "Release_p2"]

        # Override STARFIT max releases at DRBC lower reservoirs
        if self.reservoir_name in list(max_discharges.keys()):
            self.R_max = max_discharges[self.reservoir_name]

        else:
            self.R_max = (
                999999
                if self.remove_R_max
                else (
                    (starfit_params.loc[self.starfit_name, "Release_max"] + 1)
                    * self.I_bar
                )
            )

        # Override STARFIT min releases at DRBC lower reservoirs
        if self.reservoir_name in list(conservation_releases.keys()):
            self.R_min = conservation_releases[self.reservoir_name]
        else:
            self.R_min = (
                starfit_params.loc[self.starfit_name, "Release_min"] + 1
            ) * self.I_bar

    def setup(self):
        """
        Set up the parameter.
        """
        super().setup()
        self.N_SCENARIOS = len(self.model.scenarios.combinations)
        self.releases = np.empty([self.N_SCENARIOS], np.float64)

    def standardize_inflow(self, inflow):
        """
        Standardize the current reservoir inflow based on historic average.

        Args:
            inflow (float): The inflow value (MGD).

        Returns:
            float: The standardized inflow value.
        """
        return (inflow - self.I_bar) / self.I_bar

    def calculate_percent_storage(self, storage):
        """
        Calculate the reservoir's current percentage of storage capacity.

        Args:
            storage (float): The storage value (MG).

        Returns:
            float: The percentage of storage capacity.
        """
        return storage / self.S_cap

    def get_NORhi(self, timestep):
        """
        Get the upper-bound normalized reservoir storage of the Normal Operating Range (NORlo) for a given timestep.

        Args:
            timestep: The timestep.

        Returns:
            float: The NORhi value.
        """
        c = math.pi * (timestep.dayofyear + self.WATER_YEAR_OFFSET) / 365
        NORhi = (
            self.NORhi_mu
            + self.NORhi_alpha * math.sin(2 * c)
            + self.NORhi_beta * math.cos(2 * c)
        )
        if (NORhi <= self.NORhi_max) and (NORhi >= self.NORhi_min):
            return NORhi / 100
        elif NORhi > self.NORhi_max:
            return self.NORhi_max / 100
        else:
            return self.NORhi_min / 100

    def get_NORlo(self, timestep):
        """
        Get the lower-bound normalized reservoir storage of the Normal Operating Range (NORlo) for a given timestep.

        Args:
            timestep: The timestep.

        Returns:
            float: The NORlo value.
        """
        c = math.pi * (timestep.dayofyear + self.WATER_YEAR_OFFSET) / 365
        NORlo = (
            self.NORlo_mu
            + self.NORlo_alpha * math.sin(2 * c)
            + self.NORlo_beta * math.cos(2 * c)
        )
        if (NORlo <= self.NORlo_max) and (NORlo >= self.NORlo_min):
            return NORlo / 100
        elif NORlo > self.NORlo_max:
            return self.NORlo_max / 100
        else:
            return self.NORlo_min / 100

    def get_harmonic_release(self, timestep):
        """
        Get the harmonic release for a given timestep.

        Args:
            timestep: The timestep.

        Returns:
            float: The seasonal harmonic reservoir release (MGD).
        """
        c = math.pi * (timestep.dayofyear + self.WATER_YEAR_OFFSET) / 365
        R_avg_t = (
            self.Release_alpha1 * math.sin(2 * c)
            + self.Release_alpha2 * math.sin(4 * c)
            + self.Release_beta1 * math.cos(2 * c)
            + self.Release_beta2 * math.cos(4 * c)
        )
        return R_avg_t

    def calculate_release_adjustment(self, S_hat, I_hat, NORhi_t, NORlo_t):
        """
        Calculate the release adjustment.

        Args:
            S_hat (float): The standardized storage value.
            I_hat (float): The standardized inflow value.
            NORhi_t (float): The upper bound of normal operation range for the current timestep.
            NORlo_t (float): The lower bound of normal operation range for the current timestep.

        Returns:
            float: The release adjustment value.
        """
        # Calculate normalized value within NOR
        A_t = (S_hat - NORlo_t) / (NORhi_t)
        epsilon_t = self.Release_c + self.Release_p1 * A_t + self.Release_p2 * I_hat
        return epsilon_t

    def calculate_target_release(
        self, harmonic_release, epsilon, NORhi, NORlo, S_hat, I
    ):
        """
        Calculate the target release under current inflow and storage.

        Args:
            harmonic_release (float): The harmonic release for the current day.
            epsilon (float): The release adjustment value.
            NORhi_t (float): The upper bound of normal operation range for the current timestep.
            NORlo_t (float): The lower bound of normal operation range for the current timestep.
            S_hat (float): The standardized storage value.
            I (float): The inflow value.

        Returns:
            float: The target release value.
        """
        if (S_hat <= NORhi) and (S_hat >= NORlo):
            target = min(
                (self.I_bar * (harmonic_release + epsilon) + self.I_bar), self.R_max
            )
        elif S_hat > NORhi:
            target = min((self.S_cap * (S_hat - NORhi) + I * 7) / 7, self.R_max)
        else:
            if self.linear_below_NOR:
                target = (self.I_bar * (harmonic_release + epsilon) + self.I_bar) * (
                    S_hat / NORlo
                )  # (1 - (NORlo - S_hat)/NORlo)
                target = max(target, self.R_min)
            else:
                target = self.R_min
        return target

    def value(self, timestep, scenario_index):
        """
        Get the reservoir release for a given timestep and scenario index.

        Args:
            timestep: The timestep.
            scenario_index: The scenario index.

        Returns:
            float: The STARFIT prescribed reservoir release (MGD).
        """

        # Check if parameters have been loaded
        if not self.parameters_loaded and self.run_sensitivity_analysis:
            self.pywr_scenario_index = scenario_index
            self.sample_scenario_index = self.sensitivity_analysis_scenarios[
                self.pywr_scenario_index.indices[0]
            ]

            # load values from file
            self.starfit_params = self.load_starfit_sensitivity_samples(
                self.sample_scenario_index
            )

            print(f"Loading STARFIT parameters for {self.reservoir_name}")

            self.assign_starfit_param_values(self.starfit_params)
            # change bool to prevent re-loading
            self.parameters_loaded = True

        elif not self.parameters_loaded and not self.run_sensitivity_analysis:
            self.starfit_params = self.load_default_starfit_params(model_data_dir)
            # print(f"Assigning STARFIT parameters for {self.reservoir_name}")
            self.assign_starfit_param_values(self.starfit_params)
            self.parameters_loaded = True

        # Get current storage and inflow conditions
        I_t = self.inflow.get_value(scenario_index)
        S_t = self.node.volume[scenario_index.indices]

        I_hat_t = self.standardize_inflow(I_t)
        S_hat_t = self.calculate_percent_storage(S_t)

        NORhi_t = self.get_NORhi(timestep)
        NORlo_t = self.get_NORlo(timestep)

        seasonal_release_t = self.get_harmonic_release(timestep)

        # Get adjustment from seasonal release
        epsilon_t = self.calculate_release_adjustment(
            S_hat_t, I_hat_t, NORhi_t, NORlo_t
        )

        # Get target release
        target_release = self.calculate_target_release(
            S_hat=S_hat_t,
            I=I_t,
            NORhi=NORhi_t,
            NORlo=NORlo_t,
            epsilon=epsilon_t,
            harmonic_release=seasonal_release_t,
        )

        # Get actual release subject to constraints
        release_t = max(min(target_release, I_t + S_t), (I_t + S_t - self.S_cap))

        return max(0, release_t)

    @classmethod
    def load(cls, model, data):
        """Set up the parameter."""
        reservoir_name = data.pop("node")
        storage_node = model.nodes[f"reservoir_{reservoir_name}"]
        flow_parameter = load_parameter(model, f"flow_{reservoir_name}")
        run_starfit_sensitivity_analysis = data.pop("run_starfit_sensitivity_analysis")
        sensitivity_analysis_scenarios = data.pop("sensitivity_analysis_scenarios")
        return cls(
            model,
            reservoir_name,
            storage_node,
            flow_parameter,
            run_starfit_sensitivity_analysis,
            sensitivity_analysis_scenarios,
            **data,
        )


# Register the parameter for use with Pywr
STARFITReservoirRelease.register()
