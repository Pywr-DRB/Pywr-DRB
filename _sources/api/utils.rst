pywrdrb.utils
================

pywrdrb.utils.directories
----------------------------

.. automodule:: pywrdrb.utils.directories

This module contains string variables that provide absolut paths to the following directories:

.. py:data:: input_dir
   :module: pywrdrb.utils.directories

   The folder where all input data is stored. 

.. py:data:: output_dir
   :module: pywrdrb.utils.directories

   The folder where output data will be stored during and after simulation. 


pywrdrb.utils.hdf5
--------------------
.. automodule:: pywrdrb.utils.hdf5

.. autosummary::
   :toctree: generated/
   :nosignatures:

   pywrdrb.utils.hdf5.get_hdf5_realization_numbers
   pywrdrb.utils.hdf5.extract_realization_from_hdf5
   pywrdrb.utils.hdf5.combine_batched_hdf5_outputs


pywrdrb.utils.timeseries
-------------------------
.. automodule:: pywrdrb.utils.timeseries
    
.. autosummary::
   :toctree: generated/
   :nosignatures:

   pywrdrb.utils.timeseries.subset_timeseries
   pywrdrb.utils.timeseries.get_rollmean_timeseries