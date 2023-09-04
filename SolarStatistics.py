"""
This file was used to read solar forecast data, and includes function to calculate the relevant metrics.
All the solar forecast data is now stored in solar_forecasts.pickle
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import geopandas as gpd
import geojson_rewind
import folium
from shapely.geometry import shape, Point
import warnings
from pandas.core.common import SettingWithCopyWarning
import geopy.distance
import pickle as pkl
import scipy.optimize

warnings.simplefilter(action='ignore',category=SettingWithCopyWarning)

"""Make sure dates are added to determine correlations properly"""

it_areas = ['IT_NORD', 'IT_CNOR', 'IT_CSUD', 'IT_SUD', 'IT_SICI', 'IT_SARD']
tennet_areas = ['Bayern', 'Hessen', 'Nordrhein-Westfalen', 'Bremen/Niedersachsen', 'Schleswig-Holstein']
transnet_areas = ['Baden-Württemberg']
se_areas = ['SE1', 'SE2', 'SE3', 'SE4']
dk_areas = ['DK1', 'DK2']
de_areas = ['Bayern', 'Hessen', 'Nordrhein-Westfalen', 'Bremen/Niedersachsen', 'Schleswig-Holstein', 'Baden-Württemberg']
be_areas = ['Antwerp', 'Brussels', 'East Flanders', 'Flemish Brabant' , 'West Flanders', 'Hainaut', 'Liege',
            'Limburg', 'Luxembourg', 'Namur', 'Walloon Brabant']
nordic_areas = ['SE1', 'SE2', 'SE3', 'SE4', 'NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'DK2', 'FI']
solar_areas = ['IT_NORD', 'IT_CNOR', 'IT_CSUD', 'IT_SUD', 'IT_SICI', 'IT_SARD', 'Bayern', 'Hessen',
               'Nordrhein-Westfalen', 'Bremen/Niedersachsen', 'Schleswig-Holstein', 'Baden-Württemberg', 'DK1', 'DK2'] #'SE1', 'SE2', 'SE3', 'SE4',
solar_countries = {'IT': it_areas, 'SE': se_areas, 'DE': de_areas, 'DK': dk_areas}
da_to_id_factor = 0.63431 # Based on changes in the 11AM forecast and the actual value in Belgian data

bzs_in = gpd.read_file('C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\KTH\\Simuleringsdata\\Maps\\solarmap_complete.geojson')
bzs = json.loads(bzs_in.to_json())
bzs = geojson_rewind.rewind(bzs, rfc7946=False)
# m = folium.Map([60,15], zoom_start=5, tiles='cartodbpositron')
# for _,f in bzs_in.iterrows():
#    # plot polygons with folium
#    polygon = gpd.GeoSeries(f['geometry']).simplify(tolerance=0.001)
#    geo_j = polygon.to_json()
#    geo_j = folium.GeoJson(data=geo_j, style_function=lambda x: {'fillColor': 'orange'})
#    geo_j.add_to(m)
#
# points_in = pd.read_csv(f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\PV_info_IT.csv', index_col=0)
# power_centres = pd.DataFrame(columns=['Longitude', 'Latitude'], index=it_areas)
# for a in it_areas:
#    df = points_in.loc[points_in['Area'] == a]
#    power_centres['Longitude'][a] = (df['Longitude'] * df['Capacity']).sum()/ df['Capacity'].sum()
#    power_centres['Latitude'][a] = (df['Latitude'] * df['Capacity']).sum() / df['Capacity'].sum()
#
# for a in it_areas:
#    #plot power centres
#    folium.Marker(location=[power_centres['Latitude'][a], power_centres['Longitude'][a]]).add_to(m)
#
# for i in range(points_in.__len__()):
#     #plot points with folium
#     folium.CircleMarker(location=[points_in['Latitude'][i], points_in['Longitude'][i]], radius=2, weight=points_in['Capacity'][i]/20).add_to(m)
#
# m.show_in_browser()

def read_belgian_forecast(save=False):
    area__dict = {
        'Antwerp': 'Antwerp',
        'Brussels': 'Brussels',
        'East Flanders': 'East-Flanders',

    }
    errors = pd.DataFrame(columns=['Mean DA', 'Mean ID', 'Mean gen', 'ID/DA'], index=be_areas)
    data = pd.read_csv('C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\belgian_data.csv', delimiter=';', header=0, skiprows=range(1, 221705), nrows=490560) #Reads 2022 data
    for a in be_areas:
        a = a.replace(' ', '-')
        if a == 'Liege':
            a = 'Liège'
        x = data.loc[data['Region'] == a]
        x.reset_index(drop=True, inplace=True)
        x = x[['Datetime', 'Measured & Upscaled', 'Most recent forecast', 'Day Ahead 11AM forecast']] # Forecast error = forecaste - Actual
        x = x.iloc[::-1]
        x = x.dropna()
        x.reset_index(drop=True, inplace=True)
        a = a.replace('-', ' ')
        if a == 'Liège':
            a = 'Liege'
        pv = pd.DataFrame(columns=['DA', 'ID', 'Generation'])
        pv['ID'] = [x['Most recent forecast'][i * 4:(i + 1) * 4].mean() for i in range(int(x.__len__() / 4))]
        pv['DA'] = [x['Day Ahead 11AM forecast'][i * 4:(i + 1) * 4].mean() for i in range(int(x.__len__() / 4))]
        pv['Generation'] = [x['Measured & Upscaled'][i * 4:(i + 1) * 4].mean() for i in range(int(x.__len__() / 4))]
        errors['Mean DA'][a] = (pv['DA'] - pv['Generation']).abs().mean()
        errors['Mean ID'][a] = (pv['ID'] - pv['Generation']).abs().mean()
        errors['Mean gen'][a] = (pv['Generation']).mean()
        errors['ID/DA'][a] = errors['Mean ID'][a] / errors['Mean DA'][a]

def read_it_forecast(save=False):
    it_dict = {
        'IT_NORD': 'IT-North',
        'IT_SUD': 'IT-South',
        'IT_CNOR': 'IT-Centre-North',
        'IT_CSUD': 'IT-Centre-South',
        'IT_SARD': 'IT-Sardinia',
        'IT_SICI': 'IT-Sicily'
    }
    #solar_dict = {}
    for a in it_areas:
        data = pd.read_csv(f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\{a}_2022.csv')
        data = data.drop(index=[2042])
        solar_dict[a] = pd.DataFrame(index=pd.date_range(start='2022-01-01 00:00', end='2022-12-31 23:00', freq='h', tz='Europe/Berlin'),
                                     columns=['Generation', 'DA forecast', 'DA error', 'ID error', 'ID forecast'])
        solar_dict[a]['DA forecast'] = data[f'Generation - Solar  [MW] Day Ahead/ BZN|{it_dict[a]}'].tolist()
        solar_dict[a]['Generation'] = data[f'Generation - Solar  [MW] Current / BZN|{it_dict[a]}'].tolist()
        solar_dict[a]['DA error'] = solar_dict[a]['DA forecast'] - solar_dict[a]['Generation']
        solar_dict[a]['ID error'] = solar_dict[a]['DA error'] * da_to_id_factor
        solar_dict[a]['ID forecast'] = solar_dict[a]['Generation'] + solar_dict[a]['ID error']
        # solar_dict[a] = solar_dict[a].dropna()
        # solar_dict[a].reset_index(drop=True, inplace=True)

def read_se_forecast(save=False):
    #solar_dict = {}
    for a in se_areas:
        data = pd.read_csv(f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\{a}_2022.csv')
        data = data.drop(index=[2042])
        solar_dict[a] = pd.DataFrame(index=pd.date_range(start='2022-01-01 00:00', end='2022-12-31 23:00', freq='h', tz='Europe/Berlin'),
                                     columns=['Generation', 'DA forecast', 'DA error', 'ID error', 'ID forecast'])
        solar_dict[a]['DA forecast'] = data[f'Generation - Solar  [MW] Day Ahead/ BZN|{a}'].astype(float).tolist()
        solar_dict[a]['Generation'] = data[f'Generation - Solar  [MW] Current / BZN|{a}'].astype(float).tolist()
        solar_dict[a]['DA error'] = solar_dict[a]['DA forecast'] - solar_dict[a]['Generation']
        solar_dict[a]['ID error'] = solar_dict[a]['DA error'] * da_to_id_factor
        solar_dict[a]['ID forecast'] = solar_dict[a]['Generation'] + solar_dict[a]['ID error']
        # solar_dict[a] = solar_dict[a].dropna()
        # solar_dict[a].reset_index(drop=True, inplace=True)

def read_dk_forecast(save=False):
    #solar_dict = {}
    for a in dk_areas:
        data = pd.read_csv(f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\{a}_2022.csv')
        data = data.drop(index=[2042])
        solar_dict[a] = pd.DataFrame(index=pd.date_range(start='2022-01-01 00:00', end='2022-12-31 23:00', freq='h',tz='Europe/Berlin'),
                                     columns=['Generation', 'DA forecast', 'DA error', 'ID error', 'ID forecast'])
        solar_dict[a]['DA forecast'] = data[f'Generation - Solar  [MW] Day Ahead/ BZN|{a}'].astype(float).tolist()
        solar_dict[a]['Generation'] = data[f'Generation - Solar  [MW] Current / BZN|{a}'].astype(float).tolist()
        solar_dict[a]['DA error'] = solar_dict[a]['DA forecast'] - solar_dict[a]['Generation']
        solar_dict[a]['ID error'] = solar_dict[a]['DA error'] * da_to_id_factor
        solar_dict[a]['ID forecast'] = solar_dict[a]['Generation'] + solar_dict[a]['ID error']


def read_tennet_forecast(save=False):
    #solar_dict = {}
    for a in tennet_areas:
        if a == 'Bremen/Niedersachsen':
            aa = 'Bremen'
        else:
            aa = a
        data = pd.read_csv(f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\{aa}_2022.csv', delimiter=';')
        solar_dict[a] = pd.DataFrame(
            index=pd.date_range(start='2022-01-01 00:00', end='2022-12-31 23:00', freq='h', tz='Europe/Berlin'),
            columns=['Generation', 'DA forecast', 'DA error', 'ID error', 'ID forecast'])
        solar_dict[a]['DA forecast'] = [data['Forecast in MW'][i*4:(i+1)*4].mean() for i in range(8760)]
        solar_dict[a]['Generation'] = [data['Actual in MW'][i * 4:(i + 1) * 4].mean() for i in range(8760)]
        solar_dict[a]['DA error'] = solar_dict[a]['DA forecast'] - solar_dict[a]['Generation']
        solar_dict[a]['ID error'] = solar_dict[a]['DA error'] * da_to_id_factor
        solar_dict[a]['ID forecast'] = solar_dict[a]['Generation'] + solar_dict[a]['ID error']
        # solar_dict[a] = solar_dict[a].dropna()
        # solar_dict[a].reset_index(drop=True, inplace=True)
        # plt.scatter(solar_dict[a]['DA forecast'], solar_dict[a]['ID error'])
        # plt.title(a)
        # plt.show()

def read_transnet_forecast(save=False):
    #solar_dict = {}
    a = transnet_areas[0]
    da_list = []
    actual_list = []
    for i in range(1,13):
        data = pd.read_csv(f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\BW_2022_{i}.csv', delimiter=';')
        da_list.extend([data['Prognose (MW)'][i*4:(i+1)*4].mean() for i in range(int(data.__len__()/4))])
        actual_list.extend([data['Ist-Wert (MW)'][i*4:(i+1)*4].mean() for i in range(int(data.__len__()/4))])
    solar_dict[a] = pd.DataFrame(
        index=pd.date_range(start='2022-01-01 00:00', end='2022-12-31 23:00', freq='h', tz='Europe/Berlin'),
        columns=['Generation', 'DA forecast', 'DA error', 'ID error', 'ID forecast'])
    solar_dict[a]['DA forecast'] = da_list
    solar_dict[a]['Generation'] = actual_list
    solar_dict[a]['DA error'] = solar_dict[a]['DA forecast'] - solar_dict[a]['Generation']
    solar_dict[a]['ID error'] = solar_dict[a]['DA error'] * da_to_id_factor
    solar_dict[a]['ID forecast'] = solar_dict[a]['Generation'] + solar_dict[a]['ID error']
    # plt.scatter(solar_dict[a]['DA forecast'], solar_dict[a]['ID error'])
    # plt.title(a)
    # plt.show()

def read_de_geography(save=False):
    data = pd.read_csv(f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\renewable_power_plants_DE.csv')
    data = data.loc[data['energy_source_level_2'] == 'Solar']
    data = data.loc[(data['tso'] == 'TenneT') | (data['tso'] == 'TransnetBW')]
    data.reset_index(drop=True, inplace=True)
    de_info = pd.DataFrame(columns=['Capacity', 'Longitude', 'Latitude', 'Area'])
    shape_dict = {}
    for f in bzs['features']:
        shape_dict[f['properties']['zoneName']] = shape(f['geometry'])
    lat_list = []
    lon_list = []
    area_list = []
    capacity_list = []
    for i in range(data.__len__()):
        pnt = Point(data['lon'][i], data['lat'][i])
        for a in de_areas:
            if shape_dict[a].contains(pnt):
                lat_list.append(data['lat'][i])
                lon_list.append(data['lon'][i])
                area_list.append(a)
                capacity_list.append(data['electrical_capacity'][i])
                break
    de_info['Capacity'] = capacity_list
    de_info['Longitude'] = lon_list
    de_info['Latitude'] = lat_list
    de_info['Area'] = area_list
    de_info.to_csv(f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\PV_info_DE.csv')


def read_dk_geography(save=False):
    data = pd.read_csv(f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\renewable_power_plants_DK_filtered.csv')
    data = data.loc[data['energy_source_level_2'] == 'Solar']
    data.reset_index(drop=True, inplace=True)
    dk_info = pd.DataFrame(columns=['Capacity', 'Longitude', 'Latitude', 'Area'])
    shape_dict = {}
    for f in bzs['features']:
        shape_dict[f['properties']['zoneName']] = shape(f['geometry'])
    lat_list = []
    lon_list = []
    area_list = []
    capacity_list = []
    for i in range(data.__len__()):
        pnt = Point(data['lon'][i], data['lat'][i])
        for a in dk_areas:
            if shape_dict[a].contains(pnt):
                lat_list.append(data['lat'][i])
                lon_list.append(data['lon'][i])
                area_list.append(a)
                capacity_list.append(data['electrical_capacity'][i])
                break
    dk_info['Capacity'] = capacity_list
    dk_info['Longitude'] = lon_list
    dk_info['Latitude'] = lat_list
    dk_info['Area'] = area_list
    dk_info.to_csv(f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\PV_info_DK.csv')

def read_se_geography(save=False):
    pv_info = pd.DataFrame(columns=['Kommun', 'Capacity', 'Longitude', 'Latitude', 'Area'])
    pv_in = pd.read_excel('C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\Sol_Sverige_2022.xlsx')

    num = len(pv_in)

    name_list = []
    pv_list = []
    for i in range(num):
       name_list.append(pv_in['Kommun'][i][5:])
       if pv_in['PV'][i] == '-':
          pv_list.append(0)
       else:
          pv_list.append(pv_in['PV'][i])
    pv_info['Kommun'] = name_list
    pv_info['Capacity'] = pv_list

    municip_in = gpd.read_file('C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\KTH\\Simuleringsdata\\Maps\\kommuner.geojson')
    municip_in = json.loads(municip_in.to_json())
    municip = geojson_rewind.rewind(municip_in, rfc7946=False)

    for f in municip['features']:
       if f['properties']['name'][:-7] in name_list:
          idx = pv_info[pv_info['Kommun'] == f['properties']['name'][:-7]].index.values
          pv_info['Longitude'][idx] = f['geometry']['coordinates'][0]
          pv_info['Latitude'][idx] = f['geometry']['coordinates'][1]
       elif f['properties']['name'][:-8] in name_list:
          idx = pv_info[pv_info['Kommun'] == f['properties']['name'][:-8]].index.values
          pv_info['Longitude'][idx] = f['geometry']['coordinates'][0]
          pv_info['Latitude'][idx] = f['geometry']['coordinates'][1]
       elif f['properties']['name'][7:] in name_list:
          idx = pv_info[pv_info['Kommun'] == f['properties']['name'][7:]].index.values
          pv_info['Longitude'][idx] = f['geometry']['coordinates'][0]
          pv_info['Latitude'][idx] = f['geometry']['coordinates'][1]
       elif f['properties']['name'] == 'Falu kommun':
          idx = pv_info[pv_info['Kommun'] == 'Falun'].index.values
          pv_info['Longitude'][idx] = f['geometry']['coordinates'][0]
          pv_info['Latitude'][idx] = f['geometry']['coordinates'][1]
       elif f['properties']['name'] == 'Åsele kommun, 2.875 invånare 31 december 2013 display=inline':
          idx = pv_info[pv_info['Kommun'] == 'Åsele'].index.values
          pv_info['Longitude'][idx] = f['geometry']['coordinates'][0]
          pv_info['Latitude'][idx] = f['geometry']['coordinates'][1]
       elif f['properties']['name'] == 'Robertsfors kommun, 6.738 invånare 31 december 2013':
          idx = pv_info[pv_info['Kommun'] == 'Robertsfors'].index.values
          pv_info['Longitude'][idx] = f['geometry']['coordinates'][0]
          pv_info['Latitude'][idx] = f['geometry']['coordinates'][1]

    bzs_in = gpd.read_file('C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\KTH\\Simuleringsdata\\Maps\\nordicwcoastal.geojson')
    bzs = json.loads(bzs_in.to_json())
    bzs = geojson_rewind.rewind(bzs, rfc7946=False)

    for i in range(num):
        pnt = Point(pv_info['Longitude'][i], pv_info['Latitude'][i])
        for f in bzs['features']:
            polygon = shape(f['geometry'])
            if polygon.contains(pnt):
                pv_info['Area'][i] = f['properties']['name']
                break
    pv_info.to_csv(f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\PV_info_SE.csv')

def read_it_geography(save=False):
    data = pd.read_excel(f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\Italy_2022.xlsx', sheet_name='Potenza')
    for i in range(data.__len__()):
        if data['Region'][i][-2:] == '**':
            data = data.drop(i)
    data.reset_index(drop=True, inplace=True)

    it_info = pd.DataFrame(columns=['Name', 'Capacity', 'Longitude', 'Latitude', 'Area'])
    prov_in = gpd.read_file('C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\KTH\\Simuleringsdata\\Maps\\italy_provinces.geojson')
    prov = json.loads(prov_in.to_json())
    prov = geojson_rewind.rewind(prov, rfc7946=False)
    name_list = []
    lon_list = []
    lat_list = []
    capacity_list = []
    area_list = []
    shape_dict = {}
    for f in bzs['features']:
        shape_dict[f['properties']['zoneName']] = shape(f['geometry'])
    for f in prov['features']:
        df = data.loc[data['Region'] == f['properties']['prov_name']]
        df.reset_index(drop=True, inplace=True)
        polygon = shape(f['geometry'])
        centr = polygon.centroid
        capacity_list.append(df['MW'][0])
        name_list.append(f['properties']['prov_name'])
        lon_list.append(polygon.centroid.x)
        lat_list.append(polygon.centroid.y)
        for a in it_areas:
            if shape_dict[a].contains(centr):
                area_list.append(a)
                break
        if f['properties']['prov_name'] == 'Venezia':
            area_list.append('IT_NORD')
        elif f['properties']['prov_name'] == 'Rimini':
            area_list.append('IT_NORD')
        elif f['properties']['prov_name'] == 'Livorno':
            area_list.append('IT_CNOR')
        elif f['properties']['prov_name'] == 'Cagliari':
            area_list.append('IT-SARD')
    it_info['Capacity'] = capacity_list
    it_info['Longitude'] = lon_list
    it_info['Latitude'] = lat_list
    it_info['Area'] = area_list
    it_info['Name'] = name_list
    it_info.to_csv(f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\PV_info_IT.csv')

solar_dict = {}
# read_it_forecast()
# read_se_forecast()
# read_dk_forecast()
# read_tennet_forecast()
# read_transnet_forecast()
# with open(f'solar_forecasts.pickle', 'wb') as handle:
#     pkl.dump(solar_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)

def compute_distances(save=False):
    distance_metrics = pd.DataFrame(columns=['Center Longitude', 'Center Latitude', 'Average Distance'], index=solar_areas)
    for c in solar_countries.keys():
        points_in = pd.read_csv(f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\PV_info_{c}.csv', index_col=0)
        for a in solar_countries[c]:
            print(a)
            df = points_in.loc[points_in['Area'] == a]
            if a in de_areas or a in dk_areas:
                df = df.loc[df['Capacity'] > 0.05]
                print(df.__len__())
            distance_metrics['Center Longitude'][a] = (df['Longitude'] * df['Capacity']).sum()/ df['Capacity'].sum()
            distance_metrics['Center Latitude'][a] = (df['Latitude'] * df['Capacity']).sum() / df['Capacity'].sum()
            pv_tot = df['Capacity'].sum()
            cent_coord = (distance_metrics['Center Latitude'][a], distance_metrics['Center Longitude'][a])
            dist = []
            for _,i in df.iterrows():
                coord = (i['Latitude'], i['Longitude'])
                dist.append(geopy.distance.geodesic(cent_coord, coord).km * i['Capacity'] / pv_tot)
                distance_metrics['Average Distance'][a] = sum(dist)
            print(distance_metrics)
    #distance_metrics.to_csv('SolarDistance.csv')

def compute_target_metrics():
    with open('solar_forecasts.pickle','rb') as handle:
        solar_dict = pkl.load(handle)
    dist = pd.read_csv('SolarDistance.csv', index_col=0)
    metrics = pd.DataFrame(columns=['Abs', 'Std', 'Var', 'Distance'], index=solar_areas)
    for a in solar_areas:
        metrics['Distance'][a] = dist['Average Distance'][a]
        df = solar_dict[a]
        df = df.dropna()
        df.reset_index(drop=True, inplace=True)
        metrics['Abs'][a] = float(df['ID error'].abs().sum() / df['ID forecast'].sum())
        metrics['Std'][a] = float(df['ID error'].std() / df['ID forecast'].mean())
        metrics['Var'][a] = float(
            np.sum(np.abs(np.subtract(np.array(df['ID error'][1:]), np.array(df['ID error'][:-1])))) /
            df['ID forecast'].sum())
    plt.rcParams.update({'font.size': 12})
    plt.scatter(metrics['Distance'], metrics['Abs'], label='_nolegend_', color='blue', alpha=0.5, s=50)
    a, b = np.polyfit(np.array(metrics['Distance'].tolist()), np.array(metrics['Abs'].tolist()), 1)
    vec = np.array(range(0, int(1.2 * metrics['Distance'].max()), 1))
    plt.plot(vec, a * vec + b, label=r'$\phi^{1}$' + f' = {round(b,5)} - {-round(a,5)}' + r'$\cdot$x', color='blue')
    plt.scatter(metrics['Distance'], metrics['Std'], label='_nolegend_', color='green', alpha=0.5, s=50)
    a, b = np.polyfit(np.array(metrics['Distance'].tolist()), np.array(metrics['Std'].tolist()), 1)
    plt.plot(vec, a * vec + b,  label=r'$\phi^{2}$' + f' = {round(b,5)} - {-round(a,5)}' + r'$\cdot$x', color='green')
    plt.scatter(metrics['Distance'], metrics['Var'], label='_nolegend_', color='red', alpha=0.5, s=50)
    a, b = np.polyfit(np.array(metrics['Distance'].tolist()), np.array(metrics['Var'].tolist()), 1)
    plt.plot(vec, a * vec + b, label=r'$\phi^{3}$' + f' = {round(b,5)} - {-round(a,5)}' + r'$\cdot$x', color='red')
    plt.legend()
    plt.grid()
    plt.xlabel('Area diameter [km]')
    plt.ylabel('Target value')
    #plt.ylim(0, 0.37)
    plt.xlim(-0.01, int(1.2 * metrics['Distance'].max()))
    plt.tight_layout()
    fig = plt.gcf()
    save = True
    if save:
        fig.savefig(
            f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\Figures\\SolarTargets.pdf',
            dpi=fig.dpi, pad_inches=0, bbox_inches='tight')
    plt.show()

def compute_correlation():
    with open('solar_forecasts.pickle','rb') as handle:
        solar_dict = pkl.load(handle)
    dist = pd.read_csv('SolarDistance.csv', index_col=0)
    name_list = []
    dist_list = []
    corr_list = []
    correlation = pd.DataFrame(columns=['Areas', 'Distance', 'Correlation'])
    for a in solar_areas:
        for b in solar_areas:
            if a == b:
                pass
            elif f'{b}-{a}' in name_list:
                pass
            else:
                xx = f'{a}-{b}'
                name_list.append(xx)
                coord_1 = (dist['Center Latitude'][a], dist['Center Longitude'][a])
                coord_2 = (dist['Center Latitude'][b], dist['Center Longitude'][b])
                dist_list.append(geopy.distance.geodesic(coord_1, coord_2).km)
                #corr_list.append(solar_dict[a]['ID error'].corr(solar_dict[b]['ID error']))
                df = pd.DataFrame(columns=['A', 'B'])
                df['A'] = solar_dict[a]['ID error']
                df['B'] = solar_dict[b]['ID error']
                df = df.dropna()
                corr_list.append(df['A'].corr(df['B']))
    correlation['Areas'] = name_list
    correlation['Distance'] = dist_list
    correlation['Correlation'] = corr_list
    bp = 750
    df1 = correlation.loc[correlation['Distance'] <= bp]
    df2 = correlation.loc[correlation['Distance'] > bp]
    a1,b1 = np.polyfit(np.array(df1['Distance']), np.array(df1['Correlation']), 1)
    a2, b2 = np.polyfit(np.array(df2['Distance']), np.array(df2['Correlation']), 1)
    vec1 = np.array(range(0, bp-20, 1))
    vec2 = np.array(range(bp-20, int(1.2 * df2['Distance'].max()),1))
    plt.rcParams.update({'font.size': 12})
    plt.plot(vec1, a1 * vec1 + b1, label=r'$y_{1} = $' + f'{round(b1, 5)} - {-round(a1, 5)}' + r'$\cdot$x', color='blue')
    plt.plot(vec2, a2 * vec2 + b2, label=r'$y_{2} = $' + f'{round(b2, 5)} - {-round(a2, 5)}' + r'$\cdot$x', color='blue')
    plt.scatter(correlation['Distance'], correlation['Correlation'], alpha=0.5, color='blue', s=50, label='_nolegend_')
    plt.xlabel('Distance between power centers [km]')
    plt.ylabel('Correlation')
    plt.xlim(-0.01, int(1.1 * df2['Distance'].max()))
    plt.tight_layout()
    plt.grid()
    plt.legend()
    fig = plt.gcf()
    save = True
    if save:
        fig.savefig(
            f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\Figures\\SolarCorrelation.pdf',
            dpi=fig.dpi, pad_inches=0, bbox_inches='tight')
    plt.show()

compute_correlation()





