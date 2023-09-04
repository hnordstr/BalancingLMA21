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

scenario = 'FM45'
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

areas = ('SE1', 'SE2', 'SE3', 'SE4', 'NO1', 'NO2', 'NO3', 'NO4', 'DK1', 'DK2', 'FI')
data_areas = ('SE1', 'SE2', 'SE3', 'SE4', 'NO1', 'NO2', 'NO3', 'NO4', 'DK2', 'FI')
lma_areas = ('SE1', 'SE2', 'SE3', 'SE4', 'NO1', 'NO2', 'NO3', 'NO4', 'NO5','DK2', 'FI')# , 'NO5'
wind_current = pd.read_csv(
'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\wind_nordic.csv', index_col=0)

onsh_cap_current = {}

for a in areas:
    df = wind_current.loc[wind_current['Area'] == a]
    onsh_cap_current[a] = df['Wind'].sum()

wind_current = wind_current.loc[wind_current['Area'] != 'DK1']

name_list = []
lon_list = []
lat_list = []
area_list = []

offsh_data = gpd.read_file('C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\KTH\\Simuleringsdata\\Maps\\nordic_offshore.geojson')

num_offsh = {
    'SE1': 1,
    'SE2': 2,
    'SE3': 5,
    'SE4': 6,
    'FI': 7,
    'DK2': 3,
    'NO1': 1,
    'NO2': 3,
    'NO3': 2,
    'NO4': 0,
    'NO5': 1
}

if scenario == 'EF45':

    offshore_capacity = {
        'SE1': 4500,
        'SE2': 4500,
        'SE3': 9500,
        'SE4': 10000,
        'FI': 21845,  # NGDP
        'DK2': 24189,
        'NO1': 569,
        'NO2': 1922,
        'NO3': 50,
        'NO4': 0,
        'NO5': 2024
    }

    onshore_capacity = {
        'SE1': 8730,
        'SE2': 10430,
        'SE3': 5408,
        'SE4': 2223,
        'FI': 42671,  # NGDP
        'DK2': 2506,
        'NO1': 722,
        'NO2': 2002,
        'NO3': 2266,
        'NO4': 2957,
        'NO5': 2674
    }

    pv_capacity = {
        'SE1': 866,
        'SE2': 835,
        'SE3': 12129,
        'SE4': 5290,
        'FI': 10938,  # NGDP
        'DK2': 10837,
        'NO1': 26,
        'NO2': 77,
        'NO3': 53,
        'NO4': 0,
        'NO5': 95
    }

elif scenario == 'EP45':

    offshore_capacity = {
        'SE1': 1519,
        'SE2': 1519,
        'SE3': 4838,
        'SE4': 2250,
        'FI': 7761,  # NGDP
        'DK2': 8594,
        'NO1': 202,
        'NO2': 683,
        'NO3': 18,
        'NO4': 0,
        'NO5': 719
    }

    onshore_capacity = {
        'SE1': 5968,
        'SE2': 8760,
        'SE3': 6433,
        'SE4': 2802,
        'FI': 37737,  # NGDP
        'DK2': 2216,
        'NO1': 638,
        'NO2': 1771,
        'NO3': 2004,
        'NO4': 2615,
        'NO5': 2365
    }

    pv_capacity = {
        'SE1': 92,
        'SE2': 501,
        'SE3': 7278,
        'SE4': 3174,
        'FI': 6318,  # NGDP
        'DK2': 6260,
        'NO1': 15,
        'NO2': 45,
        'NO3': 31,
        'NO4': 0,
        'NO5': 55
    }

elif scenario == 'SF45':

    offshore_capacity = {
        'SE1': 0,
        'SE2': 269,
        'SE3': 312,
        'SE4': 794,
        'FI': 1054,  # NGDP
        'DK2': 1167,
        'NO1': 27,
        'NO2': 93,
        'NO3': 2,
        'NO4': 0,
        'NO5': 98
    }

    onshore_capacity = {
        'SE1': 4583,
        'SE2': 7812,
        'SE3': 6798,
        'SE4': 1985,
        'FI': 33733,  # NGDP
        'DK2': 1981,
        'NO1': 571,
        'NO2': 1583,
        'NO3': 1791,
        'NO4': 2338,
        'NO5': 2114
    }

    pv_capacity = {
        'SE1': 1597,
        'SE2': 3049,
        'SE3': 17299,
        'SE4': 7159,
        'FI': 16649,  # NGDP
        'DK2': 16496,
        'NO1': 40,
        'NO2': 118,
        'NO3': 81,
        'NO4': 0,
        'NO5': 145
    }

elif scenario == 'FM45':

    offshore_capacity = {
        'SE1': 0,
        'SE2': 725,
        'SE3': 2900,
        'SE4': 3625,
        'FI': 5557,  # NGDP
        'DK2': 6153,
        'NO1': 145,
        'NO2': 489,
        'NO3': 13,
        'NO4': 0,
        'NO5': 515
    }

    onshore_capacity = {
        'SE1': 5963,
        'SE2': 7978,
        'SE3': 8060,
        'SE4': 2247,
        'FI': 38620,  # NGDP
        'DK2': 2268,
        'NO1': 653,
        'NO2': 1812,
        'NO3': 2051,
        'NO4': 2676,
        'NO5': 2420
    }

    pv_capacity = {
        'SE1': 74,
        'SE2': 402,
        'SE3': 5835,
        'SE4': 2545,
        'FI': 5065,  # NGDP
        'DK2': 5019,
        'NO1': 12,
        'NO2': 36,
        'NO3': 25,
        'NO4': 0,
        'NO5': 44
    }

for _,f in offsh_data.iterrows():
    name_list.append(f['name'])
    area_list.append(f['area'])
    lon_list.append(f['geometry'].x)
    lat_list.append(f['geometry'].y)

offsh = pd.DataFrame(columns=['Name', 'Area', 'Longitude', 'Latitude', 'Wind'])
offsh['Name'] = name_list
offsh['Area'] = area_list
offsh['Longitude'] = lon_list
offsh['Latitude'] = lat_list

wind_list = []
for _,f in offsh.iterrows():
    wind_list.append(1 / num_offsh[f['Area']])

offsh['Wind'] = wind_list

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
        wind_list.append(f['Wind'] * onshore_capacity[f['Area']] / onsh_cap_current[f['Area']])
        lon_list.append(f['Longitude'])
        lat_list.append(f['Latitude'])

    for _,f in offsh.iterrows():
        name_list.append(f['Name'])
        area_list.append(f['Area'])
        wind_list.append(f['Wind'] * offshore_capacity[f['Area']])
        lon_list.append(f['Longitude'])
        lat_list.append(f['Latitude'])

    wind_45['Name'] = name_list
    wind_45['Area'] = area_list
    wind_45['Longitude'] = lon_list
    wind_45['Latitude'] = lat_list
    wind_45['Wind'] = wind_list
    distance_metrics = pd.DataFrame(columns=['Center Latitude', 'Center Longitude', 'Average Distance'], index=lma_areas)

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
    distance_metrics['Center Longitude']['NO5'] = 5.64
    distance_metrics['Center Latitude']['NO5'] = 60.67
    distance_metrics['Average Distance']['NO5'] = (distance_metrics['Average Distance']['NO1'] +\
                                                  distance_metrics['Average Distance']['NO2']) / 2
    return distance_metrics

dist = wind_map45()

def compute_wind_target(distance):
    abs_target = 0.25259 - 0.00082 * distance
    std_target = 0.35232 - 0.00123 * distance
    var_target = 0.10651 - 0.00033 * distance
    return abs_target, std_target, var_target

def compute_correlation(distance):
    if distance <= 700:
        corr = 0.29358 - 0.00039 * distance
    else:
        corr = 0.003545 - 0.00002 * distance
    return corr

#these lines compute the target metrics per area
targets = pd.DataFrame(columns=['Abs', 'Std', 'Var'], index=lma_areas)
for a in lma_areas:
   abs_target, std_target, var_target = compute_wind_target(dist['Average Distance'][a])
   targets['Abs'][a] = abs_target
   targets['Std'][a] = std_target
   targets['Var'][a] = var_target
print(targets)

# these lines compute the correlation between areas
# corr_matrix = pd.DataFrame(index=lma_areas, columns=lma_areas)
# for a in lma_areas:
#     for b in lma_areas:
#         if a == b:
#             corr_matrix[a][b] = 1
#         else:
#             coord_1 = (dist['Center Latitude'][a], dist['Center Longitude'][a])
#             coord_2 = (dist['Center Latitude'][b], dist['Center Longitude'][b])
#             distance = geopy.distance.geodesic(coord_1, coord_2).km
#             corr_matrix[a][b] = compute_correlation(distance)
# print(corr_matrix)
# corr_matrix.to_csv(f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\WindCorr_{scenario}.csv')




#wind_map_ef45()
# def solar_map_ef45():
#     pass
#
# def solar_map_ep45():
#     pass

