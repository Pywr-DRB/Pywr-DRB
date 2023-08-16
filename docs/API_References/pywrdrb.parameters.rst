Custom Pywr Parameters
========================================

The following custom parameters are built from the `pywr Parameter class <https://pywr.github.io/pywr/api/pywr.parameters.html>`_ and used to track various model state variables throughout the simulation period.

.. currentmodule:: pywrdrb.parameters

FFMP Parameters
------------------------------------------------------------

These custom parameters are used to implement the Flexible Flow Management Program (FFMP).

.. autosummary::
   :toctree: generated/

   FfmpNycRunningAvgParameter
   FfmpNjRunningAvgParameter
   NYCCombinedReleaseFactor
   NYCFloodRelease
   VolBalanceNYCDownstreamMRFTargetAgg_step1CanPep
   VolBalanceNYCDownstreamMRF_step1CanPep
   VolBalanceNYCDownstreamMRF_step2Nev
   VolBalanceNYCDemand

STARFIT Reservoir Release Parameter
------------------------------------------------------------

.. autosummary::
   :toctree: generated/
   
   STARFITReservoirRelease


Flow Ensemble Parameter
------------------------------------------------------------

.. autosummary::
   :toctree: generated/

   FlowEnsemble

