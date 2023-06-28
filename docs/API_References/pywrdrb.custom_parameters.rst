Custom Pywr Parameters
========================================

The following custom parameters are built from the `pywr` Parameter class and used to track various model state variables throughout the simulation period. 

.. currentmodule:: pywrdrb.custom_parameters

FFMP Parameters
--------------------

These custom parameters are used to implement the Flexible Flow Management Program (FFMP).

.. autosummary::
   :toctree: generated/
   :recursive:

   ffmp_parameters


STARFIT Reservoir Release Parameter
--------------------

This parameter is used to simulate STARFIT reservoir release operations ([Turner et al. (2021)](https://www.sciencedirect.com/science/article/pii/S0022169421008933?via%3Dihub)) at non-NYC reservoirs.

.. autosummary::
   :toctree: generated/

   starfit_parameter


Flow Ensemble Parameter
--------------------

.. autosummary::
   :toctree: generated/

   flow_ensemble_parameter

