# Pywr-DRB Model Structure

Pywr-DRB is an integrated water resource model of the Delaware River Basin (DRB) designed to assist in water resource decision making within the basin.

The model is build using [Pywr](https://pywr.github.io/pywr/index.html), an open-source Python package for constructing resource allocation models.

## Model Goals

The Pywr-DRB model was constructed with the following goals:

1. Accurately represent water management policy across the DRB
2. Allow for simulation of historic and future water supply conditions


## Model design

The Pywr-DRB model is designed to receive catchment inflow timeseries for 18 reservoirs and 14 other locations of interest such as USGS gages, minimum flow targets, and tributaries. The model simulates and outputs water supply diversions, reservoir releases and storage levels, and streamflows throughout the basin on a daily timestep.


```{figure} ../images/pywr_structure.jpg
:name: pywr-structure
:height: 350

Graphical representation of Pywr model structure; not specific to the Delaware River Basin. **Figure source:** Tomlinson, Arnott, & Harou (2020)
```

### Represented components

The Pywr-DRB model includes representation of the following aspects of the Basin:

- Historic and future streamflow throughout the basin
- Distributed water deliveries within the basin
- Water deliveries to NYC and NJ
- Reservoirs across the basin
- Regulation governing reservoir operations
