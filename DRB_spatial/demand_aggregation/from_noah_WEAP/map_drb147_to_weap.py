import pandas as pd
import numpy as np
import geopandas as gpd

# catchments from model (two different sets)
# g1 = gpd.GeoDataFrame.from_file('WEAP_catchments_fixed.zip')
# model_basin_id_name = 'ObjID'
g1 = gpd.GeoDataFrame.from_file("huc10_drb.zip")
model_basin_id_name = "HUC10"

# catchments from DRBC 2021 report
g2 = gpd.GeoDataFrame.from_file("drb147.zip")  # 147 basins used in report

g1 = g1.to_crs(g2.crs)
indx_list = []
shpdata = []
data = []
for index, model_basin in g1.iterrows():
    for index2, drbc_basin in g2.iterrows():
        if model_basin["geometry"].intersects(drbc_basin["geometry"]):
            intersect_area = (
                model_basin["geometry"].intersection(drbc_basin["geometry"]).area
            )
            frac_area_model_basin = intersect_area / model_basin["geometry"].area
            frac_area_drbc_basin = intersect_area / drbc_basin["geometry"].area
            if model_basin_id_name == "HUC10":
                print(model_basin.huc10, drbc_basin.BASIN_ID)
                indx_list.append((model_basin.huc10, drbc_basin.BASIN_ID))
            else:
                print(model_basin.Name, drbc_basin.BASIN_ID)
                indx_list.append((model_basin.ObjID, drbc_basin.BASIN_ID))
            data.append(
                {
                    "frac_area_model_basin": frac_area_model_basin,
                    "frac_area_drbc_basin": frac_area_drbc_basin,
                }
            )

indx = pd.MultiIndex.from_tuples(
    indx_list, names=[model_basin_id_name, "DRBC_BASIN_ID"]
)
df_areas = pd.DataFrame(
    data, columns=["frac_area_model_basin", "frac_area_drbc_basin"], index=indx
)


# first, load GW and SW for WD and CU for each DB basin and category (2 dataframes: sw and gw; index: db BASIN_ID; column levels: category, WD_or_CU); leave out self-supplied domestic (no gw/sw/designation).

usecolnames = ["BASIN_ID", "YEAR", "DESIGNATION", "WD_MGD", "CU_MGD"]
# historical (1990-2018)
sheetnames = {
    "PWS": "A-1",
    "PWR_THERM": "A-6",
    "PWR_HYDRO": "A-9",
    "IND": "A-11",
    "MIN": "A-14",
    "IRR": "A-17",
    "OTH": "A-22",
}
yrbeg, yrend = 2000, 2018  # avg over years from 2000 on

##projected (2010-2060)
# sheetnames = { 'PWS':'A-2',
#'PWR_THERM':'A-7',
#'PWR_HYDRO':'A-10',
#'IND':'A-12',
#'MIN':'A-15',
#'IRR':'A-19',  # 18=RCP4.5, 19=RCP8.5
#'OTH':'A-23'
# }
# yrbeg,yrend = 2010,2060


def mean_over_period(s):
    t = pd.PeriodIndex(
        [pd.Period(r) for r in s.droplevel(level=0).index.values.astype(int).tolist()]
    )
    s.index = t
    v = s  # comment this and uncomment next line to fill in zeros for missing years
    # v=s.resample('A').asfreq().fillna(0)
    w = v.loc[str(yrbeg) : str(yrend), :].mean()
    return w


sw_list = []
gw_list = []
for cat in sheetnames:
    sheet = sheetnames[cat]
    df = pd.read_excel(
        "DRBCreport_data-release_v2110.xlsx",
        sheet,
        engine="openpyxl",
        usecols=usecolnames,
    )
    df = (
        df.set_index("DESIGNATION", append=True)
        .unstack()
        .swaplevel(1, 0, axis=1)
        .sort_index(axis=1)
    )

    if "SW" in df.columns.levels[0].values:
        sw = (
            df.loc[:, ("SW", slice(None))]
            .droplevel(axis=1, level=0)
            .dropna(subset=["BASIN_ID"])
        )
        sw = sw.groupby(
            ["BASIN_ID", "YEAR"]
        ).sum()  # sum is to aggregate over the Pennsylvania GWPA subbasins
        sw.columns.name = "WD_or_CU"
        sw = pd.concat({cat: sw}, names=["Category"], axis=1)
        sw = sw.groupby("BASIN_ID").apply(mean_over_period)
        sw_list.append(sw)

    if "GW" in df.columns.levels[0].values:
        gw = (
            df.loc[:, ("GW", slice(None))]
            .droplevel(axis=1, level=0)
            .dropna(subset=["BASIN_ID"])
        )
        gw = gw.groupby(["BASIN_ID", "YEAR"]).sum()
        gw.columns.name = "WD_or_CU"
        gw = pd.concat({cat: gw}, names=["Category"], axis=1)
        gw = gw.groupby("BASIN_ID").apply(mean_over_period)
        gw_list.append(gw)

df_sw = pd.concat(sw_list, axis=1).fillna(
    0
)  # fillna(0) assumes no data at a site for a category means 0 MGD
df_gw = pd.concat(gw_list, axis=1).fillna(0)

# Unclear whether to fill missing years and missing basins with 0. Currently implemented: years no, basins yes. This was based on data. E.g., sometimes a whole decade is missing in a time series-- unlikely to be all zeros. But some basins have, eg, all their demand in gw and no entries for sw, so it makes sense to assume sw=0 for those basins.

# now use frac_area_drbc_basin to calculate weighted sums of sw and gw for model catchments (WEAP or HUC10)
model_basin_ids = df_areas.index.get_level_values(level=0).unique()
sw_list = []
gw_list = []
for model_basin_id in model_basin_ids:
    frac_areas = df_areas.loc[(model_basin_id, slice(None)), :].droplevel(level=0)[
        "frac_area_drbc_basin"
    ]
    sw_model0 = (
        df_sw.reindex(index=frac_areas.index)
        .fillna(0)
        .multiply(frac_areas, axis=0)
        .sum()
    )  # fillna(0) assumes no data at a site for a category means 0 MGD
    sw_model0.name = model_basin_id
    gw_model0 = (
        df_gw.reindex(index=frac_areas.index)
        .fillna(0)
        .multiply(frac_areas, axis=0)
        .sum()
    )
    gw_model0.name = model_basin_id
    sw_list.append(sw_model0)
    gw_list.append(gw_model0)

# results in MGD
sw_model = pd.concat(sw_list, axis=1).T
sw_model[("Total", "CU_MGD")] = sw_model.loc[:, (slice(None), "CU_MGD")].sum(axis=1)
sw_model[("Total", "WD_MGD")] = sw_model.loc[:, (slice(None), "WD_MGD")].sum(axis=1)
gw_model = pd.concat(gw_list, axis=1).T
gw_model[("Total", "CU_MGD")] = gw_model.loc[:, (slice(None), "CU_MGD")].sum(axis=1)
gw_model[("Total", "WD_MGD")] = gw_model.loc[:, (slice(None), "WD_MGD")].sum(axis=1)

if model_basin_id_name == "HUC10":
    xlsname = "sw_gw_avg_wateruse_HUC10_tmp.xlsx"
else:
    xlsname = "sw_gw_avg_wateruse_WEAPCatchments_tmp.xlsx"

writer = pd.ExcelWriter(xlsname, engine="xlsxwriter")
sw_model.to_excel(writer, sheet_name="Surface Water")
gw_model.to_excel(writer, sheet_name="Ground Water")
writer.save()
