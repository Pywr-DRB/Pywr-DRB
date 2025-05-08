"""
General custom parameters for Pywr-DRB.

Overview
--------
This module defines general-purpose custom `pywr.parameters.Parameter` subclasses
used within the Pywr-DRB modeling framework. These include parameters for calculating 
rolling averages, flow adjustments, and other generic behaviors that can be applied 
across reservoirs, nodes, or time series.

These parameters extend base Pywr functionality to support reservoir operations,
temporal smoothing, and rule-based logic needed for DRB system simulation.

Key Steps
---------
1. Define reusable custom Pywr parameter classes (e.g., rolling average, min flows).
2. Register each parameter with Pywr for YAML-based model loading.
3. Support time- and node-based parameter logic in scenarios where Pywr native logic is insufficient.

Technical Notes
---------------
- Designed to work with Pywr model YAML loading and custom node logic.
- Parameters are typically referenced by name in the model JSON/YAML configuration.
- Some parameters expect associated data columns or precomputed series to be provided.
- These general parameters are often used in STARFIT, FFMP, or other policy implementations.
- Interacts with Pywr core via `Parameter.get_value()` at runtime.

Links
-----
- https://github.com/Pywr-DRB/Pywr-DRB
- See `model_builder.py` and `starfit.py` for examples of usage

Change Log
----------
Marilyn Smith, 2025-05-07, Added module-level docstring and cleaned to DRB documentation standard.
"""
from pywr.parameters import Parameter, load_parameter


class LaggedReservoirRelease(Parameter):
    """
    Computes historical release using rolling averages of outflow and spill from past timesteps.

    This parameter is useful for policies or metrics that depend on past reservoir behavior,
    particularly where only rolling means are available (e.g., in observational datasets or
    model parameters). It estimates the release `N` timesteps ago using a simple linear
    reconstruction based on two rolling mean values.

    Attributes
    ----------
    lag : int
        Number of timesteps to lag (i.e., N).
    roll_mean_lag_outflow : Parameter
        Rolling mean of outflow `N` timesteps ago.
    roll_mean_lagMinus1_outflow : Parameter
        Rolling mean of outflow `N-1` timesteps ago. Only used if `lag > 1`.
    roll_mean_lag_spill : Parameter
        Rolling mean of spill `N` timesteps ago.
    roll_mean_lagMinus1_spill : Parameter
        Rolling mean of spill `N-1` timesteps ago. Only used if `lag > 1`.

    Methods
    -------
    value(timestep, scenario_index)
        Calculates the reconstructed release value from `lag` timesteps ago.
    load(model, data)
        Loads and instantiates the parameter from model config.
    """

    def __init__(
        self,
        model,
        lag,
        roll_mean_lag_outflow,
        roll_mean_lagMinus1_outflow,
        roll_mean_lag_spill,
        roll_mean_lagMinus1_spill,
        **kwargs,
    ):
        """
        Initialize a LaggedReservoirRelease parameter instance.

        Parameters
        ----------
        model : pywr.core.Model
            The Pywr model object.
        lag : int
            Number of timesteps to lag (must be >= 1).
        roll_mean_lag_outflow : Parameter
            Rolling mean outflow for `lag` timesteps ago.
        roll_mean_lagMinus1_outflow : Parameter or None
            Rolling mean outflow for `lag - 1` timesteps ago.
        roll_mean_lag_spill : Parameter
            Rolling mean spill for `lag` timesteps ago.
        roll_mean_lagMinus1_spill : Parameter or None
            Rolling mean spill for `lag - 1` timesteps ago.
        **kwargs
            Additional keyword arguments passed to `Parameter.__init__()`.

        Notes
        -----
        If `lag == 1`, only the lag outflow and spill values are used.
        """
        super().__init__(model, **kwargs)
        self.lag = lag
        self.roll_mean_lag_outflow = roll_mean_lag_outflow
        self.roll_mean_lag_spill = roll_mean_lag_spill

        self.children.add(roll_mean_lag_outflow)
        self.children.add(roll_mean_lag_spill)

        if lag > 1:
            self.roll_mean_lagMinus1_outflow = roll_mean_lagMinus1_outflow
            self.roll_mean_lagMinus1_spill = roll_mean_lagMinus1_spill
            self.children.add(roll_mean_lagMinus1_outflow)
            self.children.add(roll_mean_lagMinus1_spill)

    def value(self, timestep, scenario_index):
        """
        Estimate the release value `lag` timesteps ago.

        For lag > 1, this uses a linear difference of rolling averages:
        `value ≈ lag * roll_mean_lag - (lag - 1) * roll_mean_lagMinus1`

        Parameters
        ----------
        timestep : int
            Current model timestep.
        scenario_index : int
            Scenario index for ensemble or stochastic runs.

        Returns
        -------
        float
            Reconstructed release value from `lag` timesteps ago.
        """
        if self.lag == 1:
            value = self.roll_mean_lag_outflow.get_value(
                scenario_index
            ) + self.roll_mean_lag_spill.get_value(scenario_index)
        else:
            value = self.lag * (
                self.roll_mean_lag_outflow.get_value(scenario_index)
                + self.roll_mean_lag_spill.get_value(scenario_index)
            ) - (self.lag - 1) * (
                self.roll_mean_lagMinus1_outflow.get_value(scenario_index)
                + self.roll_mean_lagMinus1_spill.get_value(scenario_index)
            )
        return max(value, 0.0)

    @classmethod
    def load(cls, model, data):
        """
        Load and configure the LaggedReservoirRelease parameter from YAML.

        Parameters
        ----------
        model : pywr.core.Model
            Pywr model instance.
        data : dict
            Dictionary from the YAML config containing `lag` and `node` keys, 
            along with any other parameter arguments.

        Returns
        -------
        LaggedReservoirRelease
            Configured instance of the parameter.
        """

        lag = data.pop("lag")
        node = data.pop("node")

        roll_mean_lag_outflow = load_parameter(model, f"outflow_{node}_rollmean{lag}")
        roll_mean_lag_spill = load_parameter(model, f"spill_{node}_rollmean{lag}")

        if lag > 1:
            roll_mean_lagMinus1_outflow = load_parameter(
                model, f"outflow_{node}_rollmean{lag - 1}"
            )
            roll_mean_lagMinus1_spill = load_parameter(
                model, f"spill_{node}_rollmean{lag-1}"
            )
        else:
            roll_mean_lagMinus1_outflow = None
            roll_mean_lagMinus1_spill = None

        return cls(
            model,
            lag,
            roll_mean_lag_outflow,
            roll_mean_lagMinus1_outflow,
            roll_mean_lag_spill,
            roll_mean_lagMinus1_spill,
            **data,
        )


# register the custom parameter so Pywr recognizes it
LaggedReservoirRelease.register()
