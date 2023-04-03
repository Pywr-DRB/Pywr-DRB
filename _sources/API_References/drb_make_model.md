# `drb_make_model.py`

This script as an executable used to generate the Pywr-DRB model.

**Functions:**
- `create_starfit_params()`
This function creates and returns a dictionary of parameters required to simulate starfit reservoir.

- `add_major_node()`
This function adds nodes (reservoirs and river nodes) to the Pywr model.

- `drb_make_model()`
This function constructs the JSON file which defines the Pywr mdoel.


***

### `create_starfit_params()`
This function creates and returns a dictionary of parameters required to simulate starfit reservoir. It first initializes some constants associated with the STARFIT rule type. It then retrieves the required constant parameters for the given reservoir from a csv file and saves them to the dictionary. Finally, it aggregates some of the constant parameters using a specified function and saves the aggregated values to the dictionary.

#### Syntax
```python
create_starfit_params(d: dict, r: str, starfit_remove_Rmax=False, starfit_linear_below_NOR=False) -> None
```

**Parameters:**
- `d` (dict): A dictionary which will store the parameters for the reservoir.
- `r` (str): The name of the reservoir.
- `starfit_remove_Rmax` (bool): Whether to remove the Rmax parameter from starfit.
- `starfit_linear_below_NOR` (bool): Whether to model below the NOR with linear behavior.

**Returns:**
- `d` (dict): The dictionary containing the required parameters for simulating the starfit reservoir.


***

### `add_major_node()`

Add a major node to the model. Major nodes types include reservoir & river nodes. This function will add the major node and all standard minor nodes that belong to each major node ( i.e., catchment, withdrawal, consumption, outflow), along with their standard parameters and edges. All nodes, edges, and parameters are added to the model dict, which is then returned.

#### Syntax
```python
add_major_node(model, name, node_type, inflow_type, backup_inflow_type=None, outflow_type=None, downstream_node=None, initial_volume=None, initial_volume_perc=None, variable_cost=None) -> dict
```

**Parameters:**
-   `model` (dict): the dict holding all model elements, which will be written to JSON file at completion.
-   `name` (str): name of major node
-   `node_type` (str): type of major node - either 'reservoir' or 'river'
-   `inflow_type` (str): 'nhmv10', etc
-   `backup_inflow_type` (str): 'nhmv10', etc. only active if inflow_type is a WEAP series - backup used to fill inflows for non-WEAP reservoirs.
-   `outflow_type` (str): define what type of outflow node to use (if any) - either 'starfit' or 'regulatory'
-   `downstream_node` (str): name of node directly downstream, for writing edge network.
-   `initial_volume` (float): (reservoirs only) starting volume of reservoir in MG. Must correspond to "initial_volume_perc" times total volume, as pywr doesnt calculate this automatically in time step 0.
-   `initial_volume_perc` (float): (reservoirs only) fraction full for reservoir initially (note this is fraction, not percent, a confusing pywr convention)
-   `variable_cost` (bool): (reservoirs only) If False, cost is fixed throughout simulation. If True, it varies according to state-dependent parameter.

**Returns:**
-   `model` (dict): the updated model dict, with all nodes, edges, and parameters added.


***

### `drb_make_model()`

This function creates a JSON file used by Pywr to define the DRB (Delaware River Basin) model, including all nodes, edges, and parameters.This function depends on `add_major_node()` function.

#### Syntax
```python
drb_make_model(inflow_type: str, backup_inflow_type: str, start_date: str, end_date: str, use_hist_NycNjDeliveries: bool=True) -> None
```

**Parameters:**
- `inflow_type` (str): The type of inflow to the reservoir.
- `backup_inflow_type` (str): The type of backup inflow to the reservoir.
- `start_date` (str): The start date of the simulation period.
- `end_date` (str): The end date of the simulation period.
- `use_hist_NycNjDeliveries` (bool): Whether to use historical New York City and New Jersey deliveries data. Default value is True.

**Returns:**
- `None`. The JSON file representing the model is saved as `./model_data/drb_model_full.json`
