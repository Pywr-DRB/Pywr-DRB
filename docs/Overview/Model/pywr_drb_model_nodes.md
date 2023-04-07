# Pywr-DRB Model nodes

There are 32 major internal nodes in the Pywr-DRB model consisting of 18 reservoirs and 14 locations along the across the stream network. Each node is connected to one or more other nodes via an *edge*.

More information about the default Pywr node classes can be found [in the Pywr documentation.](https://pywr.github.io/pywr/api/pywr.nodes.html#nodes-classes)

## Major nodes

Each major node in the Pywr-DRB model is created using the [`add_major_node()`](../../API_References/drb_make_model.md) function. This function adds the major node to the model (stored in `./model_data/drb_model_full.json`) and also constructs five corresponding *minor* nodes connected to the major node representing: withdrawals, consumption, outflow, and a link which connects to the downstream node. These minor nodes are described in {numref}`reservoir-node-table`

Major nodes in the Pywr-DRB model are classified as either `"reservoir"` or `"river"` types.

Reservoir nodes are defined as `"storage"` node classes in `pywr` and contain information about the volume of storage in the reservoir at each timestep along with the reservoir release policy. River nodes are defined as `"link"` classes in `pywr` and record simulated streamflow timeseries at that location.

```{list-table} Summary of major nodes and corresponding minor nodes.
:header-rows: 1
:name: reservoir-node-table

* - Representation
  - Node Name
  - Description
* - Major Node
  - `reservoir_{name}` or `link_{name}`
  - A major node representing either a reservoir or river location. Major reservoir nodes are Pywr 'storage' node types which include information on storage and release policies. Major river nodes are Pywr 'link' node types which only record simulated streamflow timeseries at that location.
* - Catchment
  - `catchment_{name}`
  - The basin upstream of the reservoir or river location, where inflows are derived.
* - Withdrawal
  - `catchmentWithdrawal_{name}`
  - The water withdrawal from the catchment basin.
* - Consumption
  - `catchmentConsumption_{name}`
  - The consumptive water use from the catchment basin.
* - Outflow
  - `ouflow_{name}`
  - Outflows or releases from the major node, after reservoir and withdrawal impacts.
* - Link
  - `link_{downstream node}`
  - This is a link which connects the outflow of the major node to the next downstream node.
```

### Reservoir nodes

Each reservoir in the model is defined as a major node, using the `add_major_node()` function, and is also assigned the five corresponding minor nodes outlined in {numref}`reservoir-node-table`.

The figure below shows the relationship between a reservoir node and its corresponding minor nodes:

```{figure} ../../images/reservoir_node_schematic.png
:name: reservoir-node-schematic
:scale: 50%

Graphical representation of reservoir nodes and corresponding minor nodes used in the Pywr-DRB model.
```
