import pytest
import pandas as pd

from pywrdrb.pre import ExtrapolatedDiversionPreprocessor

def test_extrapolated_diversion_preprocessor():
    
    processor = ExtrapolatedDiversionPreprocessor(loc='nj')

    # load() function should return two DataFrames
    hist_diversions, hist_flows = processor.load()

    assert isinstance(hist_diversions, pd.DataFrame), "hist_diversions is not a DataFrame"
    assert isinstance(hist_flows, pd.DataFrame), "hist_flows is not a DataFrame"

    # after process() function, should have processor.processed_data as a DataFrame
    processor.process()
    assert hasattr(processor, 'processed_data'), "processed_data attribute not found in ExtrapolatedDiversionPreprocessor after process()"
    assert isinstance(processor.processed_data, pd.DataFrame), "ExtrapolatedDiversionPreprocessor.processed_data is not a DataFrame"
