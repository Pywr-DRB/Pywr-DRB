# Hydrology of the Delaware River Basin

The Delaware River Basin is a complex system that spans over four states and provides water for millions of people, industries, and ecosystems. Understanding the hydrology of the Basin is essential for effective water management and environmental protection.


## Environmental Challenges
The Delaware River Basin faces numerous environmental challenges, including water pollution, habitat loss, and climate change. The potential impacts of climate change are of particular concern as changes in precipitation patterns, increasing temperatures, and increased frequency of extreme weather events may pose challenges for the water supply system.

### Sensitive Ecosystems
The Delaware River Basin contains a wide range of sensitive ecosystems. Aquatic species are particularly vulnerable to changes in the hydrologic cycle in the Basin. For example, cold-water fish species are vulnerable to rising stream temperatures at times of low-flow and high temperature. 

To learn more about how the US Geologic Survey is working understand temperature impacts in the Basin, take a look at their dataviz narrative on the topic: [*How We Monitor Stream Temperature in the DRB* (USGS).](https://labs.waterdata.usgs.gov/visualizations/temperature-prediction/index.html#/monitoring)

### Salinity Intrusion

As sea levels rise in the DRB estuary, salt water is able to push further up-stream along the river, particularly during low-flow periods. 

```{figure} ../../images/phili_sea_level.png
:scale: 75%
:name: philadelphia-sea-level

Relative sea level trend at Philadelphia, PA. **Figure source:** [National Oceanic and Atmospheric Administration (2023)](https://tidesandcurrents.noaa.gov/sltrends/sltrends_station.shtml?id=8545240)
```

Monitoring and managing salinity intrusion is necessary to ensure that salt water does not reach urban drinking water treatment plants upstream. To do this, managers coordinate water releases from the different reservoirs within the Basin to try and maintain sufficient streamflow at the estuary and repell the salt front down out of the basin.


## Reservoirs

The Delaware River Basin is home to several large reservoirs that play a critical role in managing the Basin's water resources. These reservoirs are used for a variety of purposes including water supply, flood control, and power generation, and they are located throughout the Basin.

The Pywr-DRB model simulates streamflow across the reservoirs throughout the Basin, and attempts to capture their impact on the downstream streamflow conditions.

```{figure} ../../images/drb_reservoir_schematic.png
:name: drb-reservoir-schematic
:scale: 55%
:alt: A graphic which shows a network of reservoirs in the DRB and how they are connected. 

Representative illustration of reservoirs within the DRB. **Figure source:** [Delaware River Basin Commission](https://www.nj.gov/drbc/basin/map/)
```

The reservoirs shown in the above illustration have been represented within the Pywr-DRB model, allowing for thier relative impacts on downstream flow conditions to be assessed. 


### Reservoirs that Provide Water to New York City
The New York City Department of Environmental Protection owns and operates three large reservoirs in the DRB that provide water to New York City: the Cannonsville Reservoir, the Pepacton Reservoir, and the Neversink Reservoir. These reservoirs, located primarily in the upper basin, have a combined storage capacity of over 570 billion gallons of water.

Water from these reservoirs is conveyed to New York City through the Delaware Aqueduct, which is the longest tunnel in the world. The aqueduct spans over 85 miles and can deliver up to 600 million gallons of water per day to the city.

```{figure} ../../images/nyc_reservoirs.png
:name: nyc-reservoirs
:scale: 40%

Graphical depiction of NYC water supplies, showing contribution from upper DRB reservoirs. **Figure source:** [New York City Environmental Protection](https://www.nyc.gov/site/dep/water/reservoir-levels.page)
```

In addition to the New York City water supply, an average of 1 million gallons per day are diverted out of the Basin to New Jersey during normal conditions. 

The amount of water diverted out of the Basin is subject to regulatory oversight, and changes depending upon hydrologic conditions in the Basin. For more information about the water management regulation in the Basin, see [Planning & Management.](./drb_planning_management.md)
