import numpy as np

from pywrdrb.load.abstract_loader import AbstractDataLoader
from pywrdrb.load.get_results import get_pywrdrb_results
from pywrdrb.utils.results_sets import pywrdrb_results_set_opts
from pywrdrb.utils.directories import output_dir
from pywrdrb.utils.hdf5 import get_n_scenarios_from_pywrdrb_output_file

default_kwargs = {
    "output_dir": output_dir,
    "results_sets": [],
    "units": "MG",
    "print_status": False,
}


class Output(AbstractDataLoader):
    def __init__(self, 
                 output_filenames, 
                 **kwargs):
        """
        Initialize the Output loader with filenames and other options.

        Args:
            output_filenames (list): List of output files to load.

        Keyword Args:
            results_sets (list): Results sets to load.
            units (str): Units for the results (default 'MG').
            print_status (bool): Whether to print status updates (default False).
        """

        # Save output filenames with and without filetype
        self.output_filenames_with_filetype = []
        self.output_labels_and_files = {}
        for f in output_filenames:
            if "." in f:
                self.output_filenames_with_filetype.append(f)
            else:
                self.output_filenames_with_filetype.append(f"{f}.hdf5")

            self.output_labels_and_files[self.__get_output_label_from_filename__(f)] = f
        self.output_labels = list(self.output_labels_and_files.keys())
        
        self.valid_results_set_opts = pywrdrb_results_set_opts
        
        self.default_kwargs = default_kwargs
        super().__parse_kwargs__(self.default_kwargs, 
                              **kwargs)



    def __get_output_label_from_filename__(self, filename):
        """
        Extract the label from a full filename.
        The label is the output filename (<path>/<filename>.hdf5) without filetype or path.
        
        Args:
            filename (str): Full path of the filename.

        Returns:
            str: Extracted label (filename without path or extension).
        """
        if "/" in filename:
            filename = filename.split("/")[-1]
        elif "\\" in filename:
            filename = filename.split("\\")[-1]
        
        if "." in filename:
            filename = filename.split(".")[0]
        
        return filename
            
        
    def __get_scenario_ids_for_output_filenames__(self):
        """
        Get the scenario indices for each output file in the output_filenames list.
        """
        self.scenarios = {
            label: np.arange(get_n_scenarios_from_pywrdrb_output_file(file))
            for label, file in self.output_labels_and_files.items()
        }

    def load(self, **kwargs):
        """
        Load output data based on filenames and results sets.
        Data are stored as attributes of the Output object.

        Keyword Args:
            results_sets (list): Results sets to load.
            print_status (bool): Whether to print status updates.

        Returns:
            None
        """

        super().__parse_kwargs__(default_kwargs=self.default_kwargs,
                              **kwargs)

        super().__validate_results_sets__(valid_results_set_opts=self.valid_results_set_opts)

        super().__verify_files_exist__(files=self.output_filenames_with_filetype)

        self.__get_scenario_ids_for_output_filenames__()

        # Load the results
        all_results_data = {}

        datetime = None
        for s in self.results_sets:
            all_results_data[s] = {}
            
            for label, file in self.output_labels_and_files.items():
                if self.print_status:
                    print(f"Loading {s} data from {label}")


                all_results_data[s][label], datetime = get_pywrdrb_results(
                    output_filename=file,
                    results_set=s,
                    scenarios=self.scenarios[label],
                    datetime_index=datetime,
                    units=self.units,
                )
            
            super().set_data(data = all_results_data[s],
                             name = s)
