import pandas as pd
import rasterio
from rasterstats import zonal_stats
import geopandas as gpd

v = gpd.GeoDataFrame.from_file("WEAP_catchments_fixed.zip")
fname='mirad250_17v4.tif'
r = rasterio.open("mirad250m_DRB/{}".format(fname))
v = v.to_crs(r.crs)

fnames=['mirad250_02v4.tif','mirad250_07v4.tif','mirad250_12v4.tif','mirad250_17v4.tif']
years=[2002,2007,2012,2017]
frac_list = []
for fname,yr in zip(fnames,years):
    r = rasterio.open("mirad250m_DRB/{}".format(fname))
    arr=r.read(1)
    affine=r.transform
    zs=zonal_stats(v,arr,affine=affine,categorical=True)
    stats = pd.DataFrame(zs).fillna(0).drop(255,axis=1)
    f=stats[1]/stats.sum(axis=1)
    f.name=yr
    frac_list.append(f)
frac = pd.concat(frac_list,axis=1)
frac.index=v['ObjID'] # or 'BasinID' or 'Name'
frac.to_excel('irrigated_fraction.xlsx')