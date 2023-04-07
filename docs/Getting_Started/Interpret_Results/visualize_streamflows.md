# Visualize Streamflows

The figures below provide for interactive visualization of the Pywr-DRB simulated streamflow timeseries at Trenton resulting for each of the different inflow datasets. Click on the tabs above the figure to change the dataset being shown.

The upper-panel of the figure shows the total observed and Pywr-DRB simulated streamflows in the Delaware River at Trenton. The [Flexible Flow Management Program (FFMP)](../../Overview/DRB/drb_planning_management.md) is designed to ensure that streamflow at this location remains above 1,614 MGD, which is indicated by the black line in the top panel.

The lower panel in the figure shows Pywr-DRB simulated **percentage of total observed flow at Trenton** based upon where the relative flow contribution was provided by (1) NYC reservoir releases, (2) releases from other reservoirs in the Basin, or (3) if the flow did not pass through any of the reservoirs in the Basin (labeled "Unmanaged Flows Upstream").

If the simulation was perfect, these streamflow contributions would stack-up to 100%, indicated by the horizontal line in the lower panel. Consequently, deviations from 100% in the lower panel resemble errors in the simulated flow for that specific inflow dataset.

```{admonition} Interactive plot
:class: tip
Zoom in on the upper-panel of the figure to look more closely at a specific time period. 

Notice how the relative streamflow contributions from the NYC reservoirs is greatest during periods when the total flow at Trenton at is near the minimum flow target; this is due to the FFMP and the way in which this policy is intended to maintain the minimum flow target.
```


````{tab-set}
```{tab-item} Historic Reconstruction
:sync: key_obs_pub

<div style="padding-bottom:75%; position:relative; display:block; width: 100%">
  <iframe src="../../_images/obs_pub_interactive_streamflow_stack.html"
  height = "100%" width = "100%"
  title = "Percentage flow contributions at Trenton"
  frameborder="0" allowfullscreen="" style="position:absolute; top:0; left: 0">
  </iframe>
</div>
```

```{tab-item} NHMv10
:sync: key_nhm

<div style="padding-bottom:80%; position:relative; display:block; width: 100%">
  <iframe src="../../_images/nhmv10_interactive_streamflow_stack.html"
  height = "100%" width = "100%"
  title = "Percentage flow contributions at Trenton"
  frameborder="0" allowfullscreen="" style="position:absolute; top:0; left: 0">
  </iframe>
</div>
```

```{tab-item} NWMv2.1
:sync: key_nwm

<div style="padding-bottom:85%; position:relative; display:block; width: 100%">
  <iframe src="../../_images/nwmv21_interactive_streamflow_stack.html"
  height = "100%" width = "100%"
  title = "Percentage flow contributions at Trenton"
  frameborder="0" allowfullscreen="" style="position:absolute; top:0; left: 0">
  </iframe>
</div>
```

```{tab-item} WEAP
:sync: key_weap

<div style="padding-bottom:60%; position:relative; display:block; width: 100%">
  <iframe src="../../_images/WEAP_23Aug2022_gridmet_nhmv10_interactive_streamflow_stack.html"
  height = "100%" width = "100%"
  title = "Percentage flow contributions at Trenton"
  frameborder="0" allowfullscreen="" style="position:absolute; top:0; left: 0">
  </iframe>
</div>
```
````
