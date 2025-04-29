import os
import time
import pandas as pd
from ..pre.observations import DataRetriever
from ..utils.pywr_drb_node_data import (
    inflow_gauge_map,
    release_gauge_map,
    storage_gauge_map,
    storage_curves,
    nyc_reservoirs
)

import pywrdrb
from pywrdrb import get_pn_object
pn = get_pn_object()

RAW_DATA_DIR = pn.observations.get_str() + os.sep + "_raw"
PROCESSED_DATA_DIR = pn.observations.get_str()
FIG_DIR = pn.figures.get_str()


# === Settings ===
start_date = "1980-01-01"
retriever = DataRetriever(start_date=start_date)

# === Flatten gauges ===
def flatten_gauges(gauge_dict):
    return sorted({g for gauges in gauge_dict.values() for g in gauges})

inflow_gauges = flatten_gauges(inflow_gauge_map)
release_gauges = flatten_gauges(release_gauge_map)
storage_gauges_nyc = [g for k, v in storage_gauge_map.items() if k in nyc_reservoirs for g in v]
storage_gauges_std = [g for k, v in storage_gauge_map.items() if k not in nyc_reservoirs for g in v]

# === Retrieve Data ===
start_time = time.time()
print(" Starting data retrieval...")

inflows = retriever.get(inflow_gauges, type="flow")
releases = retriever.get(release_gauges, type="flow")
retriever.save_raw_gauge_data(inflows, releases)

# === Save named gauge-level inflows (optional if needed) ===
retriever.save_to_csv(inflows, "inflow_raw")

# === Elevation Data ===
elev_std = retriever.get(storage_gauges_std, type="elevation_std")
elev_nyc = retriever.get(storage_gauges_nyc, type="elevation_nyc")
elev_all = pd.concat([elev_std, elev_nyc], axis=1)
retriever.save_to_csv(elev_all, "elevation_raw")

# === Convert to storage ===
storage_converted = retriever.elevation_to_storage(elev_all, storage_curves, nyc_reservoirs)
retriever.save_to_csv(storage_converted, "storage_raw")

# === Aggregate and Save ===
retriever.postprocess_and_save(inflows, inflow_gauge_map, "catchment_inflow_mgd.csv")
retriever.postprocess_and_save(storage_converted, storage_gauge_map, "reservoir_storage_mg.csv")

# Done
elapsed = time.time() - start_time
print(f"\n Data retrieval complete in {elapsed:.2f} seconds.")
