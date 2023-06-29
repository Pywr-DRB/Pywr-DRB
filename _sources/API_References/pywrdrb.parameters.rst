Custom Pywr Parameters
========================================

The following custom parameters are built from the `pywr` Parameter class and used to track various model state variables throughout the simulation period. 

.. currentmodule:: pywrdrb.parameters

FFMP Parameters
------------------------------------------------------------

These custom parameters are used to implement the Flexible Flow Management Program (FFMP).

.. autosummary::
   :toctree: generated/
   :inherited-members: False

   FfmpNjRunningAvgParameter
   VolBalanceNYCDemandTarget
   VolBalanceNYCDemandFinal
   VolBalanceNYCDownstreamMRFTargetAgg
   VolBalanceNYCDownstreamMRFTarget
   VolBalanceNYCDownstreamMRFFinal
   NYCCombinedReleaseFactor


STARFIT Reservoir Release Parameter
------------------------------------------------------------

.. autosummary::
   :toctree: generated/
   :inherited-members: False
   
   STARFITReservoirRelease


STARFIT Reservoir Release Parameter (recursive)
------------------------------------------------------------

.. autosummary::
   :toctree: generated/
   :inherited-members: False
   
   STARFITReservoirRelease


Flow Ensemble Parameter
------------------------------------------------------------

.. autosummary::
   :toctree: generated/

   FlowEnsemble

