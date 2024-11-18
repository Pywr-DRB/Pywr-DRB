"""
The PywrDRBOutput class is used to load and store pywrdrb simulation results.

Methods include:
load()
    Loads pywrdrb simulation results.
"""
import os
from pywrdrb.post.get_results import get_pywrdrb_results, get_base_results
from pywrdrb.utils.results_sets import pywrdrb_results_set_opts, base_results_set_opts
from pywrdrb.utils.directories import output_dir, input_dir
from pywrdrb.utils.hdf5 import get_hdf5_realization_numbers

default_kwargs = {
    "output_dir": output_dir,
    "input_dir": input_dir,
    "results_sets": [],
    "datetime_index": None,
    "scenarios": None,
    "units": "MG",
    "print_status": False,
}


class Output:
    """
    The pywrdrb.Output class is used to load and store data from:
    - Observations
    - Pywr-DRB simulations with different input datasets
    - NHM output data
    - NWM output data
    - WRF-Hydro output data

    Attributes:
        models (list): A list of models/datasets to load results for.
        base_results (bool): If True, load 'base' results which are not pywrdrb simulations (eg, observations, pure NHM/NWM).
        output_dir (str): The directory where the output files are stored.
        input_dir (str): The directory where the input files are stored.
        results_sets (list): A list of results sets to load.
        datetime_index (bool): If True, use a datetime index for the results.
        start_date (str): The start date to load results for.
        end_date (str): The end date to load results for.
        scenarios (list): A list of scenarios to load results for.
        units (str): The units to use for the results. Options are 'MG' or 'MCM'.
        print_status (bool): If True, print status messages.

    Methods:
        load()
            Load the results for the specified `models` and `results_sets`.

    Examples:
    ```python
    from pywrdrb import Output

    models = ["nhmv10", "nwmv21"]
    results_sets = ["reservoir_downstream_gage", "major_flow"]

    output = Output(models=models,
                    results_sets=results_sets,
                    base_results=False)
    output.load()

    # Access the data using format:
    # output.results_set[model][0]
    ```
    """

    def __parse_kwargs__(self, **kwargs):
        """
        Parses and sets the provided keyword arguments as attributes,
        using the provided kwargs, existing attributes, or default values in that order.
        """
        for key, default_value in self.default_kwargs.items():
            # Set attribute based on order of precedence:
            # kwargs > existing attribute > default value
            setattr(self, key, kwargs.get(key, getattr(self, key, default_value)))

    def __init__(self, models, base_results=False, **kwargs):
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
        self.base_results = base_results
        self.pywrdrb_results_set_opts = pywrdrb_results_set_opts
        self.base_results_set_opts = base_results_set_opts

        self.default_kwargs = default_kwargs
        self.__parse_kwargs__(**kwargs)

    def _validate_results_sets(self):
        """
        Validate that results_sets list contains valid options.
        """
        if self.base_results:
            self.valid_results_set_opts = self.base_results_set_opts
        else:
            self.valid_results_set_opts = self.pywrdrb_results_set_opts

        for s in self.results_sets:
            if s not in self.valid_results_set_opts:
                err_msg = (
                    f"Specified results set {s} in results_sets is not a valid option. "
                )
                err_msg += f"Valid options are: {self.valid_results_set_opts}"
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
        """
        Loads data of interest, corresponding to the specified models and results_sets and
        consideration of the base_results flag.

        Data are then stored as attributes of the Output object.

        Returns:
            None
        """
        self.__parse_kwargs__(**kwargs)

        self._validate_results_sets()

        if not self.base_results:
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

                if not self.base_results:
                    all_results_data[s][f"pywr_{m}"], datetime = get_pywrdrb_results(
                        model=m,
                        results_set=s,
                        scenarios=self.scenarios[m],
                        output_dir=self.output_dir,
                        datetime_index=datetime,
                        units=self.units,
                    )
                else:
                    all_results_data[s][m], datetime = get_base_results(
                        input_dir=self.input_dir,
                        model=m,
                        datetime_index=datetime,
                        results_set=s,
                        ensemble_scenario=self.scenarios[m]
                        if "ensemble" in m
                        else None,
                        units=self.units,
                    )

        # Now save results as attributes using results_set names
        # if the results_set is already an attribute, combine into a single dict and save
        for s in self.valid_results_set_opts:
            # if the results_set is already a realvalued attribute,
            # combine into a single dict and save
            if hasattr(self, s) and (s in all_results_data.keys()):
                if getattr(self, s) is not None:
                    getattr(self, s).update(all_results_data[s])

            # if not an attribute, save as an attribute
            elif not hasattr(self, s):
                if s in all_results_data.keys():
                    setattr(self, s, all_results_data[s])
                else:
                    setattr(self, s, None)

        # delete the all_results_data dictionary
        del all_results_data
        return
