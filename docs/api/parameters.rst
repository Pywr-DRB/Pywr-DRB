pywrdrb.parameters
====================

.. currentmodule:: pywrdrb.parameters

Parameters are used to define operations within the pywrdrb model.


NYC Reservoir Releases
----------------------
Used to calculate NYC reservoir releases based on minimum daily release requirements and flood conditions, prior to accounting for Montague or other downstream releases. Assumes NYC reservoirs have equal percentage storage.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :recursive:

    NYCCombinedReleaseFactor
    NYCFloodRelease
    VolBalanceNYCDemand


NYC and NJ Diversions
---------------------
Used to keep track of annual average NYC and NJ diversions, which is limited by regulations.

.. autosummary::
   :toctree: generated/
   :nosignatures:

    FfmpNycRunningAvgParameter
    FfmpNjRunningAvgParameter


Flexible Flow Management Program (FFMP)
---------------------------------------
Used to implement various aspects of the FFMP including minimum required flow (MRF) at Montague, and other excess release quantity banks.

.. autosummary::
   :toctree: generated/
   :nosignatures:

    TotalReleaseNeededForDownstreamMRF
    VolBalanceNYCDownstreamMRF_step1
    VolBalanceNYCDownstreamMRF_step2
    LowerBasinMaxMRFContribution
    VolBalanceLowerBasinMRFAggregate
    VolBalanceLowerBasinMRFIndividual
    IERQRelease_step1


STARFIT Reservoir Operations
----------------------------
Used to determine reservoir release volume for STARFIT reservoirs.

.. autosummary::
   :toctree: generated/
   :nosignatures:

    STARFITReservoirRelease


Temperature LSTM Prediction
----------------------------
Used to predict and track water temperature at Lordville using an LSTM model. Currently, no thermal control policy is implemented.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   TemperatureModel
   Estimated_Q_C
   Estimated_Q_i
   ThermalReleaseRequirement
   ForecastedTemperatureBeforeThermalRelease
   UpdateTemperatureAtLordville
   TemperatureAfterThermalRelease


Salt Front LSTM Prediction
--------------------------
Used to predict and track the salt front location in the lower basin using an LSTM model.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   SalinityModel
   UpdateSaltFrontLocation
   SaltFrontLocation


Ensemble Specific
-----------------
Used to handle ensemble simulation data, accounting for specific subsets of realizations. 

.. autosummary::
   :toctree: generated/
   :nosignatures:

    FlowEnsemble
    PredictionEnsemble


General
-------
General purpose parameters.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   LaggedReservoirRelease
