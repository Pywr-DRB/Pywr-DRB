# Pywr-DRB


:::{admonition} **New Release:**
:class: tip

Pywr-DRB v2.0.0-beta is now available [on GitHub!](https://github.com/Pywr-DRB/Pywr-DRB)
:::

Pywr-DRB is a water resources model of the Delaware River Basin (DRB) designed to improve our understanding of water availability, drought risk, and long-term water supply planning within the Basin.

Pywr-DRB is an open-source Python model for exploring the role of reservoir operations, transbasin diversions, minimum flow targets, and other regulatory rules on water availability and drought risk in the DRB. Pywr-DRB is designed to flexibly draw on streamflow estimates from a variety of emerging data resources, such as the National Water Model, the National Hydrologic Model, and hybrid datasets blending modeled and observed data. Pywr-DRB bridges state-of-the-art advances in large-scale hydrologic modeling with an open-source representation of the significant role played by the basin's evolving water infrastructure and management institutions.

Pywr-DRB builds on the open-source Python package [Pywr](https://github.com/pywr/pywr), which provides a flexible structure for building and simulating water resource networks.

A graphical representation of the Pywr-DRB model is shown below, where every octogon represents a reservoir in the DRB:

<div style="padding-bottom:75%; position:relative; display:block; width: 100%">
  <iframe src="drb_model_map.html"
  height = "100%" width = "100%"
  title = "Graphical Representation of Pywr-DRB Model"
  frameborder="0" allowfullscreen="" style="position:absolute; top:0; left: 0">
  </iframe>
</div>


## Training Resources

The following Jupyter Notebooks are designed to serve as training material for the `pywrdrb` package. In the future, we will share more notebooks to highlight advanced workflows or model features.

##### Basic Tutorials
- [Tutorial 01 Installation](../notebooks/Tutorial%2001%20Installation.ipynb)
- [Tutorial 02 Run Simulation](../notebooks/Tutorial%2002%20Run%20Simulation.ipynb)
- [Tutorial 03 Load Results](../notebooks/Tutorial%2003%20Load%20Results.ipynb)

For more general setup and background, see the  
- [Getting Started (GitHub README)](https://github.com/Pywr-DRB/Pywr-DRB)

##### Advanced Tutorials
[Tutorial 04 Using Customized Data To Run Model](../notebooks/Tutorial%2004%20Using%20Customized%20Data%20To%20Run%20Model.ipynb)

## Publications

- Hamilton, A. L., Amestoy, T. J., & Reed, P. M. (2024). Pywr-DRB: An open-source Python model for water availability and drought risk assessment in the Delaware River Basin. Environmental Modelling & Software, 181, 106185. https://doi.org/10.1016/j.envsoft.2024.106185 
  - [Associated Pywr-DRB v1.0.2](https://zenodo.org/records/13214630)
  - [GitHub Repository](https://github.com/Pywr-DRB/Pywr-DRB/tree/diagnostic_paper)



- Amestoy, T. J. & Reed, P. M., (In Review) Integrated River Basin Assessment Framework Combining Probabilistic Streamflow Reconstruction, Bayesian Bias Correction, and Drought Storyline Analysis. Available at SSRN: https://ssrn.com/abstract=5240633 or http://dx.doi.org/10.2139/ssrn.5240633
  - [Streamflow Reconstruction Dataset](https://zenodo.org/records/15101164)
  - [GitHub Repository](https://github.com/Pywr-DRB/DRB-Historic-Reconstruction)



## Citation and DOI
If you are using the package, we kindly ask that you acknowledge both the model and the associated publication.

1. Model: Click [![DOI](https://zenodo.org/badge/479150651.svg)](https://doi.org/10.5281/zenodo.10720011) to view the citation for the latest version of Pywr-DRB, as well as citations for previous versions.

2. Paper: Hamilton, A. L., Amestoy, T. J., & Reed, Patrick. M. (2024). Pywr-DRB: An open-source Python model for water availability and drought risk assessment in the Delaware River Basin. Environmental Modelling & Software, 106185. https://doi.org/10.1016/j.envsoft.2024.106185


## Acknowledgements
This research was funded by the U.S. Geological Survey (USGS) as part of the Water Resources Mission area (USGS Grant Numbers G21AC10668, G23AC00678, and G24AC00124). The authors thank Hedeff Essaid and Noah Knowles from USGS and Aubrey Dugger and David Yates from the National Center for Atmospheric Research (NCAR) for providing data and feedback that improved this work. The views expressed in this work are those of the authors and do not reflect the views or policies of the USGS or NCAR.

## License
This program is licensed under the MIT License. See the full license [here](https://github.com/Pywr-DRB/Pywr-DRB/blob/master/LICENSE).

Copyright (c) 2025 Chung-Yi Lin, Trevor Amestoy, Marilyn Smith, Andrew Hamilton, Patrick Reed.

Copyright (c) 2023-2024 Andrew Hamilton, Trevor Amestoy, Patrick Reed.