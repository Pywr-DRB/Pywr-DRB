# Prediction in ungauged basins

## Summary

The Pywr-DRB model requires input streamflow timeseries at each reservoir in the basin, however many of these reservoirs lack historic inflow observation data. To overcome this, and generate input streamflows at these locations consistent with the historic record, methods for predicting streamflow at ungauged locations were explored and implemented. The methods used, a combination of machine learning methods, are presented here.

The latest version of the streamflow predictions at these locations are available for simulation in the [`input_data/modeled_gages/` folder of the repository](https://github.com/ahamilton144/DRB_water_management/tree/master/input_data/modeled_gages).

The code used to generate these predictions, is available in [Trevor Amestoy's Streamflow_PUB GitHub repository.](https://github.com/TrevorJA/Streamflow_PUB)

## Methods

First, flow duration curves (FDCs) are predicted at each of the ungauged locations using a multi-output neural network (NN) trained on a set of geophysical and hydroclimatic variables.

Next, streamflow timeseries are generated using the QPPQ method, which takes an inverse-weighted aggregation of the K-nearest streamflow timeseries observations and converts that to flow timeseries using the NN predicted FDC.

The methods used and presented here are described in detail in [Worland et al (2019)(1)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018WR024463) and [Worland et al (2020)(2)] respectively.

### Neural Net Prediction of Flow Duration Curves

```{note}
Details on neural net to be added later.
```

### QPPQ-Method for Streamflow Timeseries Generation

[Fennessey (1994)](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=efFhgZ8AAAAJ&citation_for_view=efFhgZ8AAAAJ:zYLM7Y9cAGgC) introduced the *QPPQ method* for streamflow estimation at ungauged locations.

The QPPQ method is commonly used and encouraged by the USGS, and is described at length in their publication [*Estimation of Daily Mean Streamflow for Ungaged Stream locations...* (2016)](https://pubs.usgs.gov/sir/2015/5157/sir20155157.pdf), and is summarized well in a figure taken from the same publication:

```{figure} ../images/QPPQ_method.png
:name: QPPQ-method

Graphical representation of the QPPQ methodology, showing A, observed daily mean streamflow
at a reference streamgage, B, flow-duration curve at the reference streamgage, C, constructed flow-duration curve at the ungaged stream location, and D, estimated daily mean streamflow at the ungaged
stream location. (Modified from Archfield and others, 2010; Stuckey and others, 2014.)
```

**QPPQ consists of four key steps:**
1. Estimating an FDC for the target catchment of interest, $\hat{FDC}_{pred}$.
2. Identify $K$ donor locations, nearest to the target point.
3. Transferring the timeseries of nonexceedance probabilities ($\mathbf{P}$) from the donor site(s) to the target.
4. Using estimated FDC for the target to map the donated nonexceedance timeseries, $\mathbf{P}$ back to streamflow.

To limit the scope of this tutorial, let's assume that an estimate of the FDC at the target site, $\hat{FDC}_{pred}$, has already been determined through some other statistical or observational study.

Then the QPPQ method can be described more formally. Given an ungauged location with an estimated FDC, $\hat{FDC}_{pred}$, and set of observed streamflow timeseries $\mathbf{q_i}$ at $K$ neighboring sites, such that:

$$
Q_{obs} = \set{\mathbf{q_1}, \mathbf{q_2}, ..., \mathbf{q_k}}
$$

With corresponding $K$ FDCs at the observation locations:

$$
FDC_{obs} = \set{FDC_1, FDC_2, ... , FDC_k}
$$

The FDCs are used to convert the observed streamflow timeseries, $\mathbf{q_{obs, i}}$, to non-exceedance probability timeseries, $\mathbf{p_{obs, i}}$.

$$
FDC_i : \mathbf{q_{i}} \to \mathbf{p_i}
$$

We can then perform a weighted-aggregation of the non-exceedance probability timeseries to estimate the non-exceedance timeseries at the ungauged location. It is most common to apply an inverse-squared-distance weight to each observed timeseries such that:

$$
\mathbf{p_{pred}} = \sum^k (\mathbf{p_i}w_i)
$$

Where $w_i = 1 / d_i^2$ where $d_i$ is the distance from the observation $i$ to the ungauged location, and $\sum^k w_i = 1$.

Finally, the estimated FDC at the ungauged location, $\hat{FDC}_{pred}$, is used to convert the non-exceedance timeseries to streamflow timeseries:

$$
\hat{FDC}_{pred} : \mathbf{p_{pred}} \to \mathbf{q_{pred}}, \, \mathbf{P} = \set{\mathbf{p}_1, ..., \mathbf{p}_k}
$$

Looking at this formulation, and the sequence of transformations that take place, I hope it is clear why the method is rightfully called the *QPPQ method*.



## References

(1) Worland, S. C., Steinschneider, S., Asquith, W., Knight, R., & Wieczorek, M. (2019). Prediction and inference of flow duration curves using multioutput neural networks. _Water Resources Research_, _55_(8), 6850-6868.

(2) Worland, S. C., Steinschneider, S., Farmer, W., Asquith, W., & Knight, R. (2019). Copula theory as a generalized framework for flow‐duration curve based streamflow estimates in ungaged and partially gaged catchments. _Water Resources Research_, _55_(11), 9378-9397.
