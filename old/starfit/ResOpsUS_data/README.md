# ResOpsUS
*A dataset of historical reservoir operations in the US*

## Description
This dataset contains time series of historical reservoir operations contiguous United States (CONUS) (and areas draining to the contiguous US). The database includes: inflows, outflows, change in storage and evaporative losses as available. Data was assembled directly from reservoir operators and spans 1974 to 2020.  

## Citation
Steyaert, J., Condon, L., Turner, S., Voisin, N., (2021) *Historical Reservoir Operations in the Contiguous United States*, In Review.

## Dataset Contents
The database contains four main folders:

1. attributes
2. time_series_all
3. time_series_including_duplicate_records
4. time_series_master_tables

 ### 1. The Attributes Folder contains:
  1. reservoir_attributes,
  2. agency_attributes,
  3. time_series_inventory,
  4. time_series_variables,
  5. time_series_inventory_variables,
  6. agency_attributes_variables, and
  7. reservoir_attributes_variables.

    (1) The reservoir_attributes are from the Global Reservoir and Dams (GRanD) dataset compiled by Lehner et al, 2011.

    (2) agency_attributes contain relevant information about the different agencies that provided data to the dataset. It  contains an agency code as well as any details about data processing.

    (3) time_series_inventory describes the length of record for each variable in the dataset as well as the data source for each variable. This is important as we combined records for different variables in the time_series_all
    folder from multiple agencies (ie inflow from Bureau of Reclamation and outflow from US Army Corps of Engineers) to ensure the longest period of record. We did not combine data from different sources for the same variable (ie storage from US Army Corps
    and Bureau of Reclamation).

    (4) The time_series_variables file describe the units of measurements for the time series files.

    (5) time_series_inventory_variables has a column that contains all the variables in time_series_inventory and a brief explanation of what each variable means.

    (6) agency_attributes_variables contains all the different variables in agency_attributes and a brief explanation of what that variable means.

    (7) reservoir_attribute_variables contains all the different variables in reservoir_attributes and a brief explanation of what that variable means.

 ### 2. The time_series_all folder contains all the processed time series data for reservoirs in the reservoir_attributes.csv file.
 The file name corresponds to the name of the dataset 'ResOpsUS' and the DAM_ID seperated by a '_' to allow for linkage back to the reservoir_attributes.csv and time_series_inventory.csv. Each
 time_series file has 6 columns: (1) date, (2) storage, (3) inflow, (4) outflow, (5) elevation, (6) evaporation.

 All units are in metric with storage in million cubic meters, inflow and outflow in meters cubed per second,
 elevation in meters and evaporation in meters cubed per second or million cubic meters depending on the agency. The agency_attributes file has a note on the specific units of evaporation for each agency that contains evaporation data.

 ### 3. The raw_time_series folder contains all the the time series records that we have gathered in our data search.
 This includes multiple records for the different dams in our dataset. All files are labeled DAM_ID_"SOURCE". The source refers to the agency if the data was downloaded directly from the agency's portal. If the source is JCS, this means that the data was collected by personal communication by Jennie Steyaert. In these cases, the agency code from time_series_inventory will denote which agency the data came from.


### 4. The time_series_master_tables folder contains three main files:
1. DAILY_AV_INFLOW_CUMECS.csv,
2. DAILY_AV_OUTFLOW_CUMECS.csv,
3.  DAILY_AV_STORAGE_MCM.csv.

  (1) DAILY_AV_INFLOW_CUMECS.csv contains daily records for all the dams (columns) from 1980-01-01 to 2020-12-31. All the records are in cubic meters per second.

  (2) DAILY_AV_OUTFLOW_CUMECS.csv contains daily records for all the dams (columns) from 1980-01-01 to 2020-12-31. All the records are in cubic meters per second.
  
  (3) DAILY_AV_STORAGE_MCM.csv contains daily records for all the dams (columns) from 1980-01-01 to 2020-12-31. All the records are in million cubic meters.

## Contacts:
For questions please contact:
- steyaertj@email.arizona.edu
- lecondon@arizona.edu
- sean.turner@pnnl.gov

## Funding Sources:
Dr. Laura Condon and Jennie Steyaert's work was supported by the U.S. Department of Energy, Interoperable Design of Extreme-scale Application Software (IDEAS) Project under Award Number DE‐AC02‐05CH11231.

This research was supported by the US Department of Energy, Office of Science, as part of research in the MultiSector Dynamics, Earth and Environmental System Modeling Program for Dr. Sean Turner and Dr. Nathalie Voisin.


## References:
*Lehner, B., Liermann, C. R., Revenga, C., Vörösmarty, C., Fekete, B., Crouzet, P., . . . Wisser, D. (2011). High‐resolution mapping of the world's reservoirs and dams for sustainable river‐flow management. Frontiers in Ecology and the Environment, 9(9), 494-502. doi:10.1890/100125*


*Patterson, L. A., & Doyle, M. W. (2018). A Nationwide Analysis of U.S. Army Corps of Engineers Reservoir Performance in Meeting Operational Targets. JAWRA Journal of the American Water Resources Association, 54(2), 543-564. doi:10.1111/1752-1688.12622*
