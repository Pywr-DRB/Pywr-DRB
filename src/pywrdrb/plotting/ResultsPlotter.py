import matplotlib.pyplot as plt

# Plotting functions available in the ResultsPlotter
from pywrdrb.plotting.nyc_storage import plot_nyc_storage
from pywrdrb.plotting.plot_reservoir_dynamics import plot_reservoir_dynamics
from pywrdrb.plotting.standard_plots.timeseries import timeseries


# Dict of all standard kwargs and option format
default_kwargs = {
    "models": [],
    "node": None,
    "start_date": None,
    "end_date": None,
    "save_fig": False,
    "fig_dir": None,
    "log_scale": False,
    "plot_observed": False,
    "legend": True,
    "colordict": None,
    "labeldict": None,
}


std_kwargs = {
    "start_date": "YYYY-MM-DD",
    "end_date": "YYYY-MM-DD",
    "node": "str",
    "save_fig": "bool",
    "fig_dir": "str",
    "log_scale": "bool",
    "plot_percent": "bool",
    "plot_observed": "bool",
    "legend": "bool",
}

plot_type_options = [
    "timeseries",
    "reservoir_dynamics",
]


class Plotter:
    """
    Plotter class is used to generate figures of pywrdrb simulation results.

    Methods
    -------
    plot(type='timeseries', **kwargs)
        Generates a figure of the specified type.

    Attributes
    ----------

    Example Usage
    -------------
    import pywrdrb.Plotter
    plotter = Plotter(output)
    plotter.plot(type='reservoir_dynamics', node='cannonsville')

    """

    default_kwargs = default_kwargs

    def __validate_kwargs__(self, **kwargs):
        """
        Validate keyword arguments.
        """
        # make sure kwarg is valid name
        for key, value in kwargs.items():
            if key not in list(self.default_kwargs.keys()):
                raise ValueError(f"Invalid keyword argument: {key}.")

        # make sure kwarg value is correct type - TODO
        pass

    def __parse_kwargs__(self, **kwargs):
        """
        Parses and sets the provided keyword arguments as attributes,
        using the provided kwargs, existing attributes, or default values in that order.
        """

        self.__validate_kwargs__(**kwargs)

        for key, default_value in self.default_kwargs.items():
            # Set attribute based on order of precedence:
            # kwargs > existing attribute > default value
            setattr(self, key, kwargs.get(key, getattr(self, key, default_value)))

    def __init__(self, output, **kwargs):
        """
        Args:
            output (Output): An Output object containing the simulation results.
        """

        # Output data object
        self.output = output

        # Currently available results_set data
        self.avail_results_sets = output.results_sets.copy()

        self.__parse_kwargs__(**kwargs)
        self.plot_type_options = plot_type_options

    def _validate_plot_type(self, type):
        """
        Validate that plot type is a valid option.
        """
        if type not in self.plot_type_options:
            err_msg = f"Specified plot type {type} is not a valid option. "
            err_msg += f"Valid options are: {self.plot_type_options}"
            raise ValueError(err_msg)
        return

    def _initalize_figure(self):
        """
        Initialize a figure and axis for plotting.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        return fig, ax

    def plot(
        self, results_set, models, scenarios, variable, type="timeseries", **kwargs
    ):
        """
        Generate a figure of the specified type.

        Args:
            type (str): The type of plot to generate. Options are 'timeseries' or 'reservoir_dynamics'.
            **kwargs: Additional keyword arguments for the plot.

        Returns:
            None
        """

        self._validate_plot_type(type)
        self.__parse_kwargs__(**kwargs)

        fig, ax = self._initalize_figure()

        if type == "timeseries":
            timeseries(
                ax,
                output=self.output,
                results_set=results_set,
                models=self.models,
                variable=variable,
                **kwargs,
            )
        elif type == "reservoir_dynamics":
            pass
        else:
            raise ValueError(
                f"Invalid plot type: {type}. Must be 'timeseries' or 'reservoir_dynamics'"
            )
