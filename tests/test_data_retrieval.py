import pytest
import os
import pandas as pd
from pywrdrb.pre import ObservationalDataRetriever
from pywrdrb.pywr_drb_node_data import storage_curves, storage_gauge_map
from pywrdrb.pywr_drb_node_data import all_flow_gauges, nyc_reservoirs

@pytest.mark.network
def test_get_single_gauge():
    retriever = ObservationalDataRetriever(start_date="2020-01-01", end_date="2020-12-31")
    df = retriever.get(["01428750"], param_cd="00060")
    assert not df.empty, "Failed to retrieve data for gauge 01428750"
    assert "01428750" in df.columns, "Gauge ID not found in columns"


@pytest.mark.network
def test_get_gauge_list():
    retriever = ObservationalDataRetriever(start_date="2020-01-01", end_date="2020-12-31")
    df = retriever.get(all_flow_gauges, param_cd="00060")
    assert not df.empty, "No inflow data retrieved"
    retrieved = set(df.columns)
    missing = set(all_flow_gauges) - retrieved
    # Some gauges legitimately have no record for the request window, but the
    # majority of the network should be retrieved.
    assert len(missing) < 0.5 * len(all_flow_gauges), (
        f"Too many gauges missing from retrieval: {sorted(missing)}"
    )


def test_elevation_to_storage_conversion():
    # Choose a known gauge that has a valid curve in storage_curves
    gauge = "01428900"
    curve_path = storage_curves[gauge]
    assert os.path.exists(curve_path), f"Storage curve for {gauge} not found: {curve_path}"
    

    sample_df = pd.DataFrame({gauge: [1300.0]}, index=[pd.Timestamp("2020-01-01")])
    retriever = ObservationalDataRetriever()
    converted = retriever.elevation_to_storage(sample_df, storage_curves)
    assert not converted.empty, "Converted storage data is empty"
    assert gauge in converted.columns, f"{gauge} not in storage conversion result"
    assert not pd.isna(converted[gauge].iloc[0]), "Conversion result is NaN"

