# Pywr-DRB

Pywr-DRB is a water resources model of the Delaware River Basin (DRB) designed to improve our understanding of water availability, drought risk, and long-term water supply planning within the Basin.

Pywr-DRB builds on the open-source Python package Pywr, which provides a modular, flexible structure for building and simulating complex water resources networks. Pywr uses a linear program to allocate water flows throughout the network within each time step, and updates the flow constraints each time step based on parameterized rules and evolving internal states of the model (e.g., seasonal reservoir releases based on current storages and downstream environmental targets).

Pywr-DRB is a node-based network simulation model of the DRB system that includes 18 major reservoirs, 14 important river gage flow locations, two major out-of-basin water supply transfers (NYC and NJ), as well as various in-basin water demands. Pywr-DRB represents all reservoir operations and water diversions using state-aware rules that respond to changing conditions within the system. A critical component of the model is its representation of the 2017 FFMP, which defines adaptive rules governing the operations of NYC’s three major water supply reservoirs, the allowable diversions to NYC and NJ, and the minimum flow rules at downstream locations meant to protect habitat, water supply, and water quality. Fifteen other reservoirs in the basin are represented using the CONUS-scale data-driven reservoir operating rules derived in the ISTARF-CONUS dataset.

Another advantage of Pywr-DRB is its flexibility with respect to input streamflow forcings. Many water management models such as WEAP are built to use meteorological inputs, which are internally converted to streamflows using an embedded hydrologic model. Pywr-DRB, on the other hand, ingests streamflow inputs directly, allowing for alternative scenarios to be run with multiple streamflow estimates from different hydrologic models, observational datasets, or statistical reanalysis datasets. This ensemble modeling approach improves the characterization of uncertainty by decoupling hydrologic modeling errors from water management modeling errors. Pywr-DRB currently has support for modeled streamflow estimates from NHM v1.0, NWM v2.1, and WEAP. Additionally, we have developed a new dataset (obs-pub) that combines historical gaged streamflow records with statistically reconstructed estimates for ungaged locations. As other streamflow modeling datasets become available, they will be integrated into Pywr-DRB as well (e.g., our colleagues at USGS/NCAR are developing WRF-Hydro simulations for the 1960’s drought of record and various future climate scenarios).

A graphical representation of the Pywr-DRB model is shown below, where every octogon represents a reservoir in the Basin:

<div style="padding-bottom:75%; position:relative; display:block; width: 100%">
  <iframe src="drb_model_map.html"
  height = "100%" width = "100%"
  title = "Graphical Representation of Pywr-DRB Model"
  frameborder="0" allowfullscreen="" style="position:absolute; top:0; left: 0">
  </iframe>
</div>

## Contents
```{tableofcontents}
```

## License
This program is licensed under the MIT License. See the full license [here](https://github.com/Pywr-DRB/Pywr-DRB/blob/master/LICENSE).

Copyright (c) 2023 Andrew Hamilton, Trevor Amestoy, Patrick Reed
