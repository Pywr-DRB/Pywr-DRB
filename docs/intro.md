# Pywr-DRB

Pywr-DRB is a water resources model of the Delaware River Basin (DRB) designed to improve our understanding of water availability, drought risk, and long-term water supply planning within the Basin.

Pywr-DRB is an open-source Python model for exploring the role of reservoir operations, transbasin diversions, minimum flow targets, and other regulatory rules on water availability and drought risk in the DRB. Pywr-DRB is designed to flexibly draw on streamflow estimates from a variety of emerging data resources, such as the National Water Model, the National Hydrologic Model, and hybrid datasets blending modeled and observed data. Pywr-DRB bridges state-of-the-art advances in large-scale hydrologic modeling with an open-source representation of the significant role played by the basin's evolving water infrastructure and management institutions.

Pywr-DRB builds on the open-source Python package Pywr, which provides a modular, flexible structure for building and simulating complex water resources networks.

You can access a pre-print of the Pywr-DRB model release publication here:

> Hamilton, Andrew L., Amestoy, Trevor J. and Reed, Patrick M., Pywr-Drb: An Open-Source Python Model for Water Availability and Drought Risk Assessment in the Delaware River Basin (Under Review). Available at SSRN: https://ssrn.com/abstract=4765247 or http://dx.doi.org/10.2139/ssrn.4765247


A graphical representation of the Pywr-DRB model is shown below, where every octogon represents a reservoir in the Basin:

<div style="padding-bottom:75%; position:relative; display:block; width: 100%">
  <iframe src="drb_model_map.html"
  height = "100%" width = "100%"
  title = "Graphical Representation of Pywr-DRB Model"
  frameborder="0" allowfullscreen="" style="position:absolute; top:0; left: 0">
  </iframe>
</div>


## Funding & Acknowledgements
This research was funded by the U.S. Geological Survey (USGS) Water Availability and Use Science Program as part of the Water Resources Mission Area Predictive Understanding of Multiscale Processes Project (USGS Grant Number G21AC10668). The authors thank Hedeff Essaid and Noah Knowles from USGS and Aubrey Dugger and David Yates from the National Center for Atmospheric Research (NCAR) for providing data and feedback that improved this work. The views expressed in this work are those of the authors and do not reflect the views or policies of the USGS or NCAR.

## License
This program is licensed under the MIT License. See the full license [here](https://github.com/Pywr-DRB/Pywr-DRB/blob/master/LICENSE).

Copyright (c) 2023 Andrew Hamilton, Trevor Amestoy, Patrick Reed.
