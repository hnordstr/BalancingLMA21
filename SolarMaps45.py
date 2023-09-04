"""
This file is used to set up wind maps considering a higher share of offshore wind in future LMA scenarios
"""

import json
import geopandas as gpd
import geojson_rewind
import pandas as pd
from shapely.geometry import shape, Point
import warnings
from pandas.core.common import SettingWithCopyWarning
import geopy.distance
from shapely.ops import unary_union
import fiona
import folium
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle as pkl

scenario = 'EP45'
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

areas = ('SE1', 'SE2', 'SE3', 'SE4', 'NO1', 'NO2', 'NO3', 'NO4', 'DK1', 'DK2', 'FI')
data_areas = ('SE1', 'SE2', 'SE3', 'SE4', 'NO1', 'NO2', 'NO3', 'NO4', 'DK1', 'DK2', 'FI')
lma_areas = ('SE1', 'SE2', 'SE3', 'SE4', 'NO1', 'NO2', 'NO3', 'NO4', 'NO5','DK2', 'FI')# , 'NO5'
solar_lma_areas = ('SE1', 'SE2', 'SE3', 'SE4', 'NO1', 'NO2', 'NO3', 'NO5','DK2', 'FI')
wind_current = pd.read_csv(
'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\wind_nordic.csv', index_col=0)

onsh_cap_current = {}

for a in areas:
    df = wind_current.loc[wind_current['Area'] == a]
    onsh_cap_current[a] = df['Wind'].sum()

#wind_current = wind_current.loc[wind_current['Area'] != 'DK1']

name_list = []
lon_list = []
lat_list = []
area_list = []


def wind_map45():
    wind_45 = pd.DataFrame(columns=['Name', 'Wind', 'Longitude', 'Latitude', 'Area'])
    name_list = []
    wind_list = []
    area_list = []
    lon_list = []
    lat_list = []

    for _,f in wind_current.iterrows():
        name_list.append(f['Kommun'])
        area_list.append(f['Area'])
        wind_list.append(f['Wind'])
        lon_list.append(f['Longitude'])
        lat_list.append(f['Latitude'])


    wind_45['Name'] = name_list
    wind_45['Area'] = area_list
    wind_45['Longitude'] = lon_list
    wind_45['Latitude'] = lat_list
    wind_45['Wind'] = wind_list
    distance_metrics = pd.DataFrame(columns=['Center Latitude', 'Center Longitude', 'Average Distance'], index=data_areas)

    for a in data_areas:
        df = wind_45.loc[wind_45['Area'] == a]
        distance_metrics['Center Longitude'][a] = (df['Longitude'] * df['Wind']).sum() / df['Wind'].sum()
        distance_metrics['Center Latitude'][a] = (df['Latitude'] * df['Wind']).sum() / df['Wind'].sum()
        wind_tot = df['Wind'].sum()
        cent_coord = (distance_metrics['Center Latitude'][a], distance_metrics['Center Longitude'][a])
        dist = []
        for _, i in df.iterrows():
            coord = (i['Latitude'], i['Longitude'])
            dist.append(geopy.distance.geodesic(cent_coord, coord).km * i['Wind'] / wind_tot)
            distance_metrics['Average Distance'][a] = sum(dist)
    return distance_metrics

solar_areas = ('SE1', 'SE2', 'SE3', 'SE4', 'DK1', 'DK2')
with open('solar_forecasts.pickle', 'rb') as handle:
    solar_dict = pkl.load(handle)
solar_dist = pd.read_csv('SolarDistance.csv', index_col=0)
wind_dist = wind_map45()
# print(solar_dist)

#Used to calculate solar distance
# df = pd.DataFrame(index=solar_areas, columns=['WC Lon', 'WC Lat', 'WC Dist', 'SC Lon', 'SC Lat', 'SC Dist', 'SCap', 'WCap', 'Relation'])
# for a in solar_areas:
#     df['WC Lon'][a] = wind_dist['Center Longitude'][a]
#     df['WC Lat'][a] = wind_dist['Center Latitude'][a]
#     df['WC Dist'][a] = wind_dist['Average Distance'][a]
#     df['SC Lon'][a] = solar_dist['Center Longitude'][a]
#     df['SC Lat'][a] = solar_dist['Center Latitude'][a]
#     df['SC Dist'][a] = solar_dist['Average Distance'][a]
#     df['SCap'][a] = solar_dict[a]['Generation'].max()
#     df['WCap'][a] = onsh_cap_current[a]
#     df['Relation'][a] = df['SC Dist'][a] / df['WC Dist'][a]
#
# factor = 0
# for a in solar_areas:
#     factor += df['Relation'][a] * df['SCap'][a] / df['SCap'].sum()


factor = 0.839303

def solar_map45():
    distance_metrics = pd.DataFrame(columns=['Center Latitude', 'Center Longitude', 'Average Distance'],
                                    index=lma_areas)
    for a in lma_areas:
        if a in solar_areas:
            distance_metrics['Center Latitude'][a] = solar_dist['Center Latitude'][a]
            distance_metrics['Center Longitude'][a] = solar_dist['Center Longitude'][a]
            distance_metrics['Average Distance'][a] = solar_dist['Average Distance'][a]
        elif a != 'NO5' and a not in solar_areas:
            distance_metrics['Center Latitude'][a] = wind_dist['Center Latitude'][a]
            distance_metrics['Center Longitude'][a] = wind_dist['Center Longitude'][a]
            distance_metrics['Average Distance'][a] = wind_dist['Average Distance'][a] * factor
        else:
            distance_metrics['Center Longitude'][a] = 5.64
            distance_metrics['Center Latitude'][a] = 60.67
            distance_metrics['Average Distance'][a] = (distance_metrics['Average Distance']['NO1'] + \
                                                           distance_metrics['Average Distance']['NO2']) / 2
    return distance_metrics


dist = solar_map45()
print(dist)

def compute_solar_target(distance):
    abs_target = 0.12388 - 0.00074 * distance
    std_target = 0.27897 - 0.00141 * distance
    var_target = 0.07476 - 0.00045 * distance
    return abs_target, std_target, var_target

def compute_correlation(distance):
    if distance <= 750:
        corr = 0.54177 - 0.00071 * distance
    else:
        corr = 0.02153 - 0.00003 * distance
    return corr

# #these lines compute the target metrics per area
# targets = pd.DataFrame(columns=['Abs', 'Std', 'Var'], index=lma_areas)
# for a in lma_areas:
#    abs_target, std_target, var_target = compute_solar_target(dist['Average Distance'][a])
#    targets['Abs'][a] = abs_target
#    targets['Std'][a] = std_target
#    targets['Var'][a] = var_target
#
# print(targets)

# these lines compute the correlation between areas
# corr_matrix = pd.DataFrame(index=solar_lma_areas, columns=solar_lma_areas)
# for a in solar_lma_areas:
#     for b in solar_lma_areas:
#         print(a)
#         print(b)
#         if a == b:
#             corr_matrix[a][b] = 1
#         else:
#             coord_1 = (dist['Center Latitude'][a], dist['Center Longitude'][a])
#             coord_2 = (dist['Center Latitude'][b], dist['Center Longitude'][b])
#             distance = geopy.distance.geodesic(coord_1, coord_2).km
#             corr_matrix[a][b] = compute_correlation(distance)
# print(corr_matrix)
# corr_matrix.to_csv(f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\SolarCorr.csv')




#wind_map_ef45()
# def solar_map_ef45():
#     pass
#
# def solar_map_ep45():
#     pass

