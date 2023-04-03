# `prep_input_data.py`

This script is used to prepare all of the necessary inflow datasets used in Pywr-DRB. For more information on different inflow datasets, see the [Data Summary page.](../Supplemental/data_summary.md)

**Functions:**
- `read_modeled_estimates()`
This function loads available data.

- `match_gages()`
This function processes streamflow data to account for spatial relationships among nodes.


***

### `read_modeled_estimates()`

This function reads input streamflows from modeled NHM/NWM estimates, preps them for Pywr, and returns the resulting dataframe.

The function first reads in the data from the specified file and filters it based on the start and end dates. It then restructures the data to have gages as columns, and converts the streamflow values from cubic meters per second (cms) to million gallons per day (mgd).
The function returns a dataframe with the restructured data.

#### Syntax
```python
read_modeled_estimates(filename: str, sep: str, date_label: str, site_label: str, streamflow_label: str, start_date: str, end_date: str) -> pd.DataFrame
```

**Parameters:**
- `filename` (str): The name of the file containing the modeled estimates.
- `sep` (str): The delimiter used in the file.
- `date_label` (str): The label for the date column.
- `site_label` (str): The label for the site or node.
- `streamflow_label` (str): The label for the streamflow dataset.
- `start_date` (str): The start date for the data.
- `end_date` (str): The end date for the data.

**Returns:**
- `df_gages` (pd.DataFrame): The resulting dataframe after reading and processing the modeled estimates.



***

### `match_gages()`

This is a function for matching USGS gage sites to nodes in Pywr-DRB and handling cases where the gage is located downstream of a reservoir or other nodes. It takes as input a pandas dataframe `df` containing the USGS gage data, a string `dataset_label` to label the output file, and a boolean `use_pub` that determines whether to use additional modeled gage data. The function returns a pandas dataframe whose columns are the names of Pywr-DRB nodes.

The function first reads in additional modeled gage data if `use_pub` is True. Then, it defines a dictionary `site_matches_reservoir` that matches gages to reservoir catchments and a list of lists `site_matches_link` containing mainstem nodes, matching USGS gages, and upstream nodes to subtract.

#### Syntax
```python
match_gages(df: pandas.DataFrame, dataset_label: str, use_pub: bool = False) -> pandas.DataFrame
```

**Parameters:**
- `df` (pandas.DataFrame): The input dataframe containing the data to match gages.
- `dataset_label` (str): A label to use in the output filename.
- `use_pub` (bool): If True, uses PUB-modeled streamflows' see [Streamflow Prediction in Ungauged Basins](../Supplemental/pub.md) for more information on these PUB streamflows. Default is False.

**Returns:**
- `inflow` (pandas.DataFrame): The dataframe containing modified streamflow records.
