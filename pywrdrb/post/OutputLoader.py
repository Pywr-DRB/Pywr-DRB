"""
The PywrDRBOutput class is used to load and store pywrdrb simulation results.

Methods include:
load()
    Loads pywrdrb simulation results.
"""
import os
from pywrdrb.post.get_results import get_pywrdrb_results, get_base_results
from pywrdrb.utils.results_sets import pywrdrb_results_set_opts
from pywrdrb.utils.directories import output_dir
from pywrdrb.utils.hdf5 import get_hdf5_realization_numbers

default_kwargs = {
    "output_dir": output_dir,
    "results_sets": [],
    "datetime_index": None,
    "scenarios": None,
    "units": "MG",
    "print_status": False,
}


class Output:
    default_kwargs = default_kwargs

    def __parse_kwargs__(self, **kwargs):
        """
        Parses and sets the provided keyword arguments as attributes,
        using the provided kwargs, existing attributes, or default values in that order.
        """
        for key, default_value in self.default_kwargs.items():
            # Set attribute based on order of precedence:
            # kwargs > existing attribute > default value
            setattr(self, key, kwargs.get(key, getattr(self, key, default_value)))

    def __init__(self, models, **kwargs):
        """
        Args:
            models (list): A list of models/datasets to load results for.

        Keyword Args:
            output_dir (str): The directory where the output files are stored.
            results_sets (list): A list of results sets to load.
            datetime_index (bool): If True, use a datetime index for the results.
            start_date (str): The start date to load results for.
            end_date (str): The end date to load results for.
            scenarios (list): A list of scenarios to load results for.
            units (str): The units to use for the results. Options are 'MG' or 'MCM'.

        Attributes:


        Methods:
            load()
                Load the results for the specified models and results sets.
        """

        self.models = models
        self.pywrdrb_results_set_opts = pywrdrb_results_set_opts

        self.__parse_kwargs__(**kwargs)

    def _validate_results_sets(self):
        """
        Validate that results_sets list contains valid options.
        """
        for s in self.results_sets:
            if s not in self.pywrdrb_results_set_opts:
                err_msg = (
                    f"Specified results set {s} in results_sets is not a valid option. "
                )
                err_msg += f"Valid options are: {self.pywrdrb_results_set_opts}"
                raise ValueError(err_msg)
        return

    def _validate_output_files_exists(self):
        """
        Validate that the output files exist for all models requested.
        """
        for m in self.models:
            fname = f"{self.output_dir}drb_output_{m}.hdf5"
            if not os.path.exists(fname):
                err_msg = f"Output file for model {m} does not exist. "
                err_msg += f"Expected file: {fname}"
                raise FileNotFoundError(err_msg)

    def _get_scenario_ids_for_models(self):
        """
        Get the scenario numbers for the given model.
        """
        scenario_ids = {}

        for m in self.models:
            if "ensemble" not in m:
                scenario_ids[m] = [0]
            else:
                fname = f"{self.output_dir}drb_output_{m}.hdf5"
                scenario_ids[m] = get_hdf5_realization_numbers(fname)
        self.scenarios = scenario_ids
        return

    def load(self, **kwargs):
        self.__parse_kwargs__(**kwargs)

        self._validate_results_sets()
        self._validate_output_files_exists()
        self._get_scenario_ids_for_models()

        # Load the results
        all_results_data = {}

        datetime = None
        for s in self.results_sets:
            all_results_data[s] = {}
            for m in self.models:
                if self.print_status:
                    print(f"Loading {s} data for {m}")

                all_results_data[s][m], datetime = get_pywrdrb_results(
                    model=m,
                    results_set=s,
                    scenarios=self.scenarios[m],
                    output_dir=self.output_dir,
                    datetime_index=datetime,
                    units=self.units,
                )

        # Now save results as attributes using results_set names
        for s in self.pywrdrb_results_set_opts:
            if s in all_results_data.keys():
                setattr(self, s, all_results_data[s])
            else:
                setattr(self, s, None)

        # delete the all_results_data dictionary
        del all_results_data
        return
