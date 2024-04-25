"""
Generates an interactive HTML map of the Pywr-DRB model and model sources.
"""

import os
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
from shapely import ops
from shapely.geometry import Point, LineString, MultiLineString
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import folium
from folium.plugins import MarkerCluster
import sys

sys.path.append('../')
from pywrdrb.pywr_drb_node_data import obs_pub_site_matches, obs_site_matches, nhm_site_matches


crs = 4386


### PLot options
## Plot options
plot_gauges = True
plot_outputs = True
plot_catchments = True
plot_storage = True
plot_links = True
plot_reservoir_types = False  # Allows options to toggle reservoir color by regulation types
plot_nhm_hru = True
plot_nwm_gauges = True
PLOT_ALL_USGS_GAGES= False

popup_width = 300

## Specify COLORS
# Link colors and dimensions
tributary_color = '#9fc5e8'
mainstem_color = '#397aa7'
diversion_color = '#f0b237'

tributary_weight = 4
mainstem_weight = 4
diversion_weight = 1

edge_opacity = 0.9
basin_opacity = 0.2

# Node colors
storage_colors = ["#505376", "#353856"]
gauge_colors = ['#78b72c', '#78b72c']
output_colors = ['#b72c78', '#b72c78']
catchment_colors = ["#3186cc", "#3186cc"]
link_colors = ['#5e5e5e', '#5e5e5e']

# Node dimensions
output_size  = 10
storage_size = 5
catchment_size = 30
link_size = 7

# Reservoir sizes
scale_reservoir_size = False
volume_scale = 1/15
max_radius = 25
min_radius = 10
fixed_reservoir_size = 15


# Fill opacity
fop = 0.9

# Initialize the map
start_coords = [40.7, -75]
geomap = folium.Map(location = start_coords, 
                    zoom_start = 7.35,
                   tiles = 'cartodbpositron',
                   control_scale = True)

# Start a feature group for toggle functionality
reservoir_layer = folium.FeatureGroup(name='Reservoirs', show=True)
output_layer = folium.FeatureGroup(name='Outputs & Diversions', show=True)
flow_target_layer = folium.FeatureGroup(name='DRBC Flow Target Locations', show=True)
gauge_layer = folium.FeatureGroup(name='USGS Gauges', show=False)
if plot_catchments:
    catchment_layer = folium.FeatureGroup(name='Reservoir Catchments', show=False)
basin_layer = folium.FeatureGroup(name='DRB Boundary', show=True)
if plot_nhm_hru:
    nhm_hru_layer = folium.FeatureGroup(name = 'NHM HRUs', show = False)
if plot_nwm_gauges:
    nwm_gauge_layer = folium.FeatureGroup(name = 'NWM Unmanaged Gauges', show = True)








#####################################################################################
### Load data, and usgs gauge data
node_geodata = pd.read_csv('./model_components/drb_model_major_nodes.csv', sep = ',')
all_usgs_gauges = pd.read_csv(f'./model_components/drb_usgs_gauges.csv', sep = ',')

### Load drb shapefiles
drb_boundary = gpd.read_file('DRB_shapefiles/drb_bnd_polygon.shp').to_crs(crs)
drb_river = gpd.read_file('DRB_shapefiles/delawareriver.shp').to_crs(crs)

# Reservoir data
reservoir_data = pd.read_csv('../pywrdrb/model_data/drb_model_istarf_conus.csv', sep = ',')

# Node catchments
node_basins = gpd.read_file('DRB_shapefiles/node_basin_geometries.shp').to_crs(crs)

# NHM Points of Interest
nhm_poi = pd.read_csv('./model_components/nhm_poi_ids.csv', sep =',', index_col=0)

# NWM unmanaged USGS gauges
nwm_gauges = pd.read_csv('./nwmv21_unmanaged_gauge_metadata.csv', sep =',')

# Filter drb gauges
all_gauges = gpd.GeoDataFrame(
    all_usgs_gauges.drop('geometry', axis = 1), geometry=gpd.points_from_xy(all_usgs_gauges.long, all_usgs_gauges.lat, crs = crs))
drb_gauges = gpd.clip(all_gauges, drb_boundary)

# Load the model
drb_model = json.load(open('../pywrdrb/model_data/drb_model_full_nhmv10.json'))
all_nodes = drb_model['nodes']
all_edges = drb_model['edges']

# Get all the available node type prefixes
all_node_types = []
for i in range(len(all_nodes)):
    all_node_types.append(all_nodes[i]['name'].split('_')[0])
all_node_types = list(set(all_node_types))


# Specify node and edge types to be plotted
plot_node_types = ['reservoir', 'delivery', 'link', 'outflow', 'output', 'delay']

# Filter for just types of interest
node_names = []
node_types = []
node_lat = []
node_long = []
node_descriptions = []
for i in range(len(all_nodes)):
    node_name = all_nodes[i]['name']
    if node_name.split('_')[0] in plot_node_types:
        node_names.append(node_name)
        node_types.append(all_nodes[i]['type'])
    
        for j, full_name in enumerate(node_geodata.name):
            if node_name == full_name:
                geodata_index = j
            elif node_name.split('_') == full_name.split('_'):
                geodata_index = j
            else:
                pass
            
        node_lat.append(node_geodata.lat.iloc[geodata_index])
        node_long.append(node_geodata.long.iloc[geodata_index])
        desc = node_geodata.description.iloc[geodata_index]
        if pd.isna(desc):
            node_descriptions.append(node_name)
        else:
            node_descriptions.append(desc)
            

nodes = pd.DataFrame({'name':node_names, 
                      'type':node_types, 
                      'lat':node_lat, 
                      'long':node_long,
                      'description': node_descriptions})


### Prepare edges
all_node1 = []
all_node2 = []
for i in range(len(all_edges)):
    if (all_edges[i][0] in nodes.name.values) and (all_edges[i][1] in nodes.name.values):
        all_node1.append(all_edges[i][0])
        all_node2.append(all_edges[i][1])

node1_coord = nodes.set_index('name').loc[all_node1][['lat','long']]
node2_coord = nodes.set_index('name').loc[all_node2][['lat','long']]

edges = pd.DataFrame({'node1': all_node1, 
         'node2': all_node2,
         'node1_lat': node1_coord.lat.values,
         'node1_long': node1_coord.long.values,
         'node2_lat': node2_coord.lat.values,
         'node2_long':node2_coord.long.values})

n_nodes = nodes.shape[0]
n_edges = edges.shape[0]

## Separate reservoirs for later color coding
nyc_main = ['reservoir_cannonsville', 'reservoir_neversink', 'reservoir_pepacton']
normal_supplemental = ['reservoir_blueMarsh', 'reservoir_beltzvilleCombined']
emergency = ['reservoir_wallenpaupack', 'reservoir_mongaupeCombined', 'reservoir_fewalter', 'reservoir_nockamixon', 'reservoir_prompton']
consumptive_makeup = ['reservoir_merrillCreek']
docket = ['reservoir_marshCreek']

regulated = nyc_main + normal_supplemental + emergency + consumptive_makeup + docket
all_reservoirs = nodes[nodes['type'] == 'storage'].name.to_list()

s = set(regulated)
unregulated = [x for x in all_reservoirs if x not in s]



### Add elements to map ###########################################################

# DRB Boundary
for _, r in drb_boundary.iterrows():
    sim_geo = gpd.GeoSeries(r['geometry']).simplify(tolerance=0.001)
    geo_j = sim_geo.to_json()
    geo_j = folium.GeoJson(data=geo_j,
                           style_function=lambda x: {'fillColor': 'none',
                                                    'weight': 1,
                                                    'color':'black', 
                                                    'opacity':1,
                                                     'fill_opacity': 0.8,
                                                    'fill': False})
    folium.Popup('Delaware River Basin Boundary', 
                 min_width = 200, 
                 max_width = 200).add_to(geo_j)
    geo_j.add_to(basin_layer)

# Mainstem River
for _, r in drb_river.iterrows():
    sim_geo = gpd.GeoSeries(r['geometry']).simplify(tolerance=0.001)
    geo_j = sim_geo.to_json()
    geo_j = folium.GeoJson(data=geo_j,
                           style_function=lambda x: {'fillColor': 'none',
                                                    'weight': 1,
                                                    'color':'black', 
                                                    'opacity':1,
                                                    'fill': 'none'})
    folium.Popup('Delaware River Main Stem', 
                 min_width = 200, 
                 max_width = 200).add_to(geo_j)
    geo_j.add_to(basin_layer)

if plot_catchments:
    node_basins = node_basins.iloc[::-1]
    node_basins['area'] = node_basins.area

    def basin_colormap(area):
        if area > 0.9:
            return '#8ec2ed'
        elif area > 0.5:
            return '#376387'
        elif area > 0.1:
            return '#32857e'
        elif area < 0.1: 
            return '#1b6339'

    for _, r in node_basins.iterrows():
        node_name = r['node']
        sim_geo = gpd.GeoSeries(r['geometry']).simplify(tolerance=0.001)
        geo_j = sim_geo.to_json()
        geo_j = folium.GeoJson(data=geo_j,
                            style_function=lambda x: {'fillColor': basin_colormap(r['area']),
                                                        'weight': 1.0,
                                                        'color': storage_colors[0], 
                                                        'fillOpacity': basin_opacity,
                                                        'fill': True})
        folium.Popup(f'Catchment basin for {node_name}', 
                    min_width = 200, 
                    max_width = 200).add_to(geo_j)
        geo_j.add_to(catchment_layer)


### Add ALL USGS Gauges

if PLOT_ALL_USGS_GAGES:
    for n in range(drb_gauges.shape[0]):
        coords = [drb_gauges.lat.iloc[n], drb_gauges.long.iloc[n]]
        
        disp = f'Gauge: 0{drb_gauges.site_id.iloc[n]} <br>Start: {drb_gauges.start_date.iloc[n]} <br>End: {drb_gauges.end_date.iloc[n]}'
        
        pop = folium.Popup(disp, min_width = popup_width, max_width = popup_width)
        
        folium.CircleMarker(coords, 
                                popup = pop,
                                fill_color = gauge_colors[1],
                                fill = True,
                                fill_opacity = 0.5,
                                radius = 3,
                                color = gauge_colors[0]).add_to(gauge_layer)

### Add USGS Gauges specific to Pywrdrb
for node, sites in obs_site_matches.items():
    if s:
        for s in sites:
            s_i = drb_gauges[drb_gauges.site_id == int(s[1:])].index
            
            if len(s_i)>0:
                coords = [drb_gauges.loc[s_i,'lat'], 
                        drb_gauges.loc[s_i,'long']]
        
                if s == node:
                    disp_text = f'Gauge {s} downstream of reservoir. <br>Start: {drb_gauges.loc[s_i,"start_date"]} <br>End: {drb_gauges.loc[s_i, "end_date"]}'
                else:
                    disp_text = f'Inflow gauge for {node.capitalize()}: {s} <br>Start: {drb_gauges.loc[s_i,"start_date"]} <br>End: {drb_gauges.loc[s_i, "end_date"]}'
                    
                
                pop = folium.Popup(disp_text, min_width = popup_width, max_width = popup_width)
                
                folium.CircleMarker(coords, 
                                        popup = pop,
                                        fill_color = gauge_colors[1],
                                        fill = True,
                                        fill_opacity = 1.0,
                                        radius = 5,
                                        color = gauge_colors[0]).add_to(gauge_layer)
                
## Plot edges
for l in range(n_edges):
    coord_1 = [edges['node1_lat'][l], edges['node1_long'][l]]
    coord_2 = [edges['node2_lat'][l], edges['node2_long'][l]]
    line = [coord_1, coord_2]
    
    folium.PolyLine(line, 
                       weight = tributary_weight,
                       color = tributary_color,
                       opacity = edge_opacity).add_to(geomap)


# Add nodes
for n in range(n_nodes):
    coords = [nodes.lat.iloc[n], nodes.long.iloc[n]]
    
    disp = nodes['description'].iloc[n]
    if pd.isna(disp):
        disp = nodes.name[n]
    
    node_type = nodes.type[n]
    name = nodes.name[n]
    
    pop = folium.Popup(disp, min_width = popup_width, max_width = popup_width)
        
    if node_type == 'storage':
        
        res_name = nodes.name.iloc[n].split('_')[1]
        volume = reservoir_data[reservoir_data['reservoir'] == res_name]['GRanD_CAP_MCM'].iloc[0]
        if scale_reservoir_size:
            if (volume*volume_scale > max_radius):
                s = max_radius
            elif (volume*volume_scale < min_radius):
                s = min_radius
            else:
                s = volume*volume_scale
        else:
            s = fixed_reservoir_size
            
        pop = folium.Popup(f'{disp} <br> Capacity: {volume} MCM', min_width = popup_width, max_width = popup_width)


        folium.RegularPolygonMarker(coords,
                                   popup = pop,
                                   number_of_sides = 8,
                                   radius = s,
                                    weight = 0.75,
                                   fill_color = storage_colors[1],
                                   fill_opacity = fop,
                                   color = storage_colors[0],
                                   rotation = 90).add_to(reservoir_layer)
        
    elif node_type == 'rivergauge':
        if name == 'outflow_delTrenton':
            folium.Marker(coords, 
                        popup = pop,
                        icon=folium.Icon(color="green"),
                        radius =100).add_to(flow_target_layer)
        elif name == 'outflow_delMontague':
            folium.Marker(coords, 
                        popup = pop,
                        icon=folium.Icon(color="purple"),
                        radius =60).add_to(flow_target_layer)
                
    elif node_type == 'output':
        if plot_outputs:
            if name == 'output_del':
                c = '#EAC100'
            else:
                c = output_colors[1]
            folium.RegularPolygonMarker(coords,
                                       popup = pop,
                                       number_of_sides = 4, 
                                       radius = output_size,
                                       fill_color= c,
                                        fill_opacity = fop,
                                       color = c).add_to(output_layer)
        
    elif node_type == 'link':
        
        if plot_links:
            if name.split('_')[0] != 'outflow':
                folium.CircleMarker(coords, 
                                    popup = pop,
                                    fill_color = link_colors[1],
                                    fill = True,
                                    fill_opacity = 0.5,
                                    color = link_colors[0],
                                radius = link_size).add_to(geomap)
        else:
            print(f'Warning: {name} node of type {node_type} not plotted.')
            

# Add NHM HRUs/Points of Interest
if plot_nhm_hru:
    
    for i in range(nhm_poi.shape[0]):
        coords = [nhm_poi.lat[i], nhm_poi.long[i]]
        disp = f'COMID: {nhm_poi.comid[i]}<br>NHM ID: {nhm_poi.nhm_id[i]}'
        pop = folium.Popup(disp, min_width = popup_width, max_width = popup_width)
    
        folium.CircleMarker(coords, 
                        popup = pop,
                        fill_color = 'purple',
                        fill = True,
                        fill_opacity = fop,
                        radius = 3,
                        color = 'purple').add_to(nhm_hru_layer)

# Add unmanged NWM gauges

if plot_nwm_gauges:
    for i in range(nwm_gauges.shape[0]):
        coords = [nwm_gauges.lat[i], nwm_gauges.long[i]]
        disp = f'NWM modeled flow at unmanaged USGS site: 0{nwm_gauges.site_no[i]}\nCOMID:{nwm_gauges.comid[i]}'
        pop = folium.Popup(disp, min_width = popup_width, max_width = popup_width)
    
        folium.CircleMarker(coords, 
                        popup = pop,
                        fill_color = 'darkgreen',
                        fill = True,
                        fill_opacity = fop,
                        radius = 3,
                        color = 'darkpurple').add_to(nwm_gauge_layer)
        
        
## Add reservoir layers based on regulation level
if plot_reservoir_types:
    nyc_reservoir_layer = folium.FeatureGroup(name='NYC', show=False)
    normal_reservoir_layer = folium.FeatureGroup(name='Normal Non-NYC', show=False)
    emergency_reservoir_layer = folium.FeatureGroup(name='Emergency', show=False)
    docket_reservoir_layer = folium.FeatureGroup(name='Docket', show=False)
    consumptive_makeup_reservoir_layer = folium.FeatureGroup(name='Consumptive Make-Up', show=False)

    ## Color codes
    nyc_main_color = '#3D9F2A'
    normal_supplemental_color = '#03D19C'
    emergency_color = '#E38B14'
    consumptive_makeup_color = '#6E6E6E'
    docket_color = '#FFEB90'
    unregulated_color = '#A4A4A4'

    s2_opacity = 0.9

    # Add nodes
    for n in range(n_nodes):
        coords = [nodes.lat.iloc[n], nodes.long.iloc[n]]
        
        disp = nodes['description'].iloc[n]
        if pd.isna(disp):
            disp = nodes.name[n]
        
        node_type = nodes.type[n]
        name = nodes.name[n]
        
        pop = folium.Popup(disp, min_width = popup_width, max_width = popup_width)
            
        if node_type == 'storage':
            
            res_name = nodes.name.iloc[n].split('_')[1]
            volume = reservoir_data[reservoir_data['reservoir'] == res_name]['GRanD_CAP_MCM'].iloc[0]
            if scale_reservoir_size:
                if (volume*volume_scale > max_radius):
                    s = max_radius
                elif (volume*volume_scale < min_radius):
                    s = min_radius
                else:
                    s = volume*volume_scale
            else:
                s = fixed_reservoir_size

            pop = folium.Popup(f'{disp} <br> Capacity: {volume} MCM', min_width = popup_width, max_width = popup_width)
            
            if name in nyc_main:
                folium.RegularPolygonMarker(coords,
                                        popup = pop,
                                        number_of_sides = 8,
                                        radius = s,
                                        fill_color = nyc_main_color,
                                        fill_opacity = s2_opacity,
                                        color = nyc_main_color,
                                        rotation = 90).add_to(nyc_reservoir_layer)
            elif name in normal_supplemental:
                folium.RegularPolygonMarker(coords,
                                        popup = pop,
                                        number_of_sides = 8,
                                        radius = s,
                                        fill_color = normal_supplemental_color,
                                        fill_opacity = s2_opacity,
                                        color = normal_supplemental_color,
                                        rotation = 90).add_to(normal_reservoir_layer)

            elif name in emergency:
                folium.RegularPolygonMarker(coords,
                                        popup = pop,
                                        number_of_sides = 8,
                                        radius = s,
                                        fill_color = emergency_color,
                                        fill_opacity = s2_opacity,
                                        color = emergency_color,
                                        rotation = 90).add_to(emergency_reservoir_layer)
            elif name in docket:
                folium.RegularPolygonMarker(coords,
                                        popup = pop,
                                        number_of_sides = 8,
                                        radius = s,
                                        fill_color = docket_color,
                                        fill_opacity = s2_opacity,
                                        color = docket_color,
                                        rotation = 90).add_to(docket_reservoir_layer)

            elif name in consumptive_makeup:
                folium.RegularPolygonMarker(coords,
                                        popup = pop,
                                        number_of_sides = 8,
                                        radius = s,
                                        fill_color = consumptive_makeup_color,
                                        fill_opacity = s2_opacity,
                                        color = consumptive_makeup_color,
                                        rotation = 90).add_to(consumptive_makeup_reservoir_layer)

            else:
                print(f'Warning: {name} node of type {node_type} not plotted.')


### Finqlize and save
basin_layer.add_to(geomap)
gauge_layer.add_to(geomap)
reservoir_layer.add_to(geomap)
output_layer.add_to(geomap)
flow_target_layer.add_to(geomap)

if plot_nhm_hru:
    nhm_hru_layer.add_to(geomap)

if plot_nwm_gauges:
    nwm_gauge_layer.add_to(geomap)

if plot_catchments:
    catchment_layer.add_to(geomap)

if plot_reservoir_types:
    nyc_reservoir_layer.add_to(geomap)
    normal_reservoir_layer.add_to(geomap)
    emergency_reservoir_layer.add_to(geomap)
    docket_reservoir_layer.add_to(geomap)
    consumptive_makeup_reservoir_layer.add_to(geomap)

folium.LayerControl().add_to(geomap)
geomap.keep_in_front(reservoir_layer)
geomap.save("drb_model_map.html")