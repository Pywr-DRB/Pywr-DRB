# Summary of data

## Model data requirements

The Pywr-DRB model is designed to receive streamflow timeseries at 18 upstream catchments and 3 mainstem locations as inputs.

## Data sources

```{list-table} Streamflow Data Sources Available
:header-rows: 1

* - Data Source
  - Reference Name
  - Description
* - Reconstructed Historic
  - `obs_pub`
  - A combination of historic streamflow observations from USGS gauge stations and predictions of streamflow at ungauged locations. For more detail on ungauged predictions, see [Supplemental: Streamflow Prediction in Ungauged Basins.](./pub.md)
* - National Water Model (NWM)
  - `nwmv21`, `nwmv21_withLakes`
  - The NWM is a forecasting system developed by the National Oceanic and Atmospheric Administration (NOAA) that predicts water availability, movement, and flooding across the United States.
* - National Hydrological Model (NHM)
  - `nhmv10`
  - The NHM is a hydrological model developed by the US Geological Survey (USGS) that simulates water movement and availability across the United States.
* - Water Evaluation and Planning (WEAP) Model
  - `WEAP_gridmet`
  - Streamflows throughout the basin are generated within a [WEAP model](https://www.weap21.org/index.asp) model of the basin. These streamflows have been gathered from the model and are available as comparative inputs.
```


### Reconstructed historic record
A combination of historic streamflow observations from USGS gauge stations and predictions of streamflow at ungauged locations. For more detail on ungauged predictions, see [Supplemental: Streamflow Prediction in Ungauged Basins.](./pub.md)

### [National Water Model (NWM)](https://water.noaa.gov/about/nwm)
The NWM is a forecasting system developed by the National Oceanic and Atmospheric Administration (NOAA) that predicts water availability, movement, and flooding across the United States.

### [National Hydrologic Model (NHM)](https://www.sciencebase.gov/catalog/item/4f4e4773e4b07f02db47e234)
The NHM is a hydrological model developed by the US Geological Survey (USGS) that simulates water movement and availability across the United States.

### [Water Evaluation and Planning System (WEAP)](https://www.weap21.org/) Streamflows
Streamflows throughout the basin are generated within a [WEAP model](https://www.weap21.org/index.asp) model of the basin. These streamflows have been gathered from the model and are available as comparative inputs.  

## Data Availability

Streamflow data needed to run the Pywr-DRB model is available in the [`input_data/` folder within the project repository.](https://github.com/ahamilton144/DRB_water_management/tree/master/input_data)
