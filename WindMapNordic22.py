"""
This file was used to read the geographic location of current Nordic wind farms.
The information needed is saved in csv-file wind_nordic.csv in J3 folder
"""

import json
import geopandas as gpd
import geojson_rewind
import pandas as pd
from shapely.geometry import shape, Point
import warnings
from pandas.core.common import SettingWithCopyWarning
from shapely.ops import unary_union
import fiona
import folium
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

# ##########DENMARK################
# wind_in_dk_raw = pd.read_excel('C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\KTH\\Papers&Projects\\J3 - Balancing analysis\\Vind_Danmark_2022.xlsx', sheet_name='IkkeAfmeldte-Existing turbines')
# wind_in_dk_raw = wind_in_dk_raw.iloc[:,[2,8,9]]
# wind_in_dk_onsh = wind_in_dk_raw.loc[wind_in_dk_raw['Type af placering'] == 'LAND']
# wind_in_dk_offsh = wind_in_dk_raw.loc[wind_in_dk_raw['Type af placering'] == 'HAV']
#
# name_list = []
# wind_list = []
# done_names = []
# for _,f in wind_in_dk_onsh.iterrows():
#     if f['Kommune'] in done_names:
#         pass
#     else:
#         done_names.append(f['Kommune'])
#         if f['Kommune'] == 'Ålborg':
#             name_list.append('Aalborg')
#         elif f['Kommune'] == 'Brønderslev':
#             name_list.append('Brønderslev-Dronninglund')
#         elif f['Kommune'] == 'Vesthimmerlands':
#             name_list.append('Vesthimmerland')
#         else:
#             name_list.append(f['Kommune'])
#         df = wind_in_dk_onsh.loc[wind_in_dk_onsh['Kommune'] == f['Kommune']]
#         wind_list.append(df['Kapacitet (kW)'].sum() / 1000)
# wind_in_dk_onsh = pd.DataFrame(columns=['Kommun', 'Wind'])
# wind_in_dk_onsh['Kommun'] = name_list
# wind_in_dk_onsh['Wind'] = wind_list
# wind_in_dk_onsh = wind_in_dk_onsh.drop([56])
#
# name_list = []
# wind_list = []
# for _,f in wind_in_dk_offsh.iterrows():
#     if f['Kommune'] in name_list:
#         pass
#     else:
#         name_list.append(f['Kommune'])
#         df = wind_in_dk_offsh.loc[wind_in_dk_offsh['Kommune'] == f['Kommune']]
#         wind_list.append(df['Kapacitet (kW)'].sum() / 1000)
# wind_in_dk_offsh = pd.DataFrame(columns=['Kommun', 'Wind'])
# wind_in_dk_offsh['Kommun'] = name_list
# wind_in_dk_offsh['Wind'] = wind_list
#
# wind_in_dk_onsh.to_csv('dk.csv')
# wind_info_dk = pd.DataFrame(columns=['Kommun', 'Wind', 'Longitude', 'Latitude', 'Area'])
# name_list = []
# wind_list = []
# lon_list = []
# lat_list = []
# area_list = []
#
# offsh_municip = gpd.read_file('C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\KTH\\Simuleringsdata\\Maps\\kommuner_danmark_hav.geojson')
#
# for _,f in wind_in_dk_offsh.iterrows():
#     name_list.append(f'{f["Kommun"]}_Hav')
#     wind_list.append(f['Wind'])
#     idx = offsh_municip.loc[offsh_municip['name'] == f'{f["Kommun"]}_Hav'].index.tolist()
#     lon_list.extend(offsh_municip['geometry'][idx].x.values.tolist())
#     lat_list.extend(offsh_municip['geometry'][idx].y.values.tolist())
#
# municip_in_dk = gpd.read_file('C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\KTH\\Simuleringsdata\\Maps\\kommuner_danmark.geojson')
# done_names = []
#
# for _,f in municip_in_dk.iterrows():
#     name = f['label_dk']
#     if name in wind_in_dk_onsh['Kommun'].tolist() and name not in done_names:
#         done_names.append(name)
#         name_list.append(name)
#         wind_list.extend(wind_in_dk_onsh['Wind'][wind_in_dk_onsh.loc[wind_in_dk_onsh['Kommun']==name].index.tolist()].values.tolist())
#         df = municip_in_dk.loc[municip_in_dk['label_dk'] == name]
#         poly_list = []
#         for _,i in df.iterrows():
#             poly_list.append(shape(i['geometry']))
#         polygon = unary_union(poly_list)
#         center = polygon.centroid
#         lon_list.append(center.x)
#         lat_list.append(center.y)
#
# wind_info_dk['Kommun'] = name_list
# wind_info_dk['Longitude'] = lon_list
# wind_info_dk['Latitude'] = lat_list
# wind_info_dk['Wind'] = wind_list
#
# bzs_in = gpd.read_file('C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\KTH\\Simuleringsdata\\Maps\\nordicwcoastal.geojson')
# bzs = json.loads(bzs_in.to_json())
# bzs = geojson_rewind.rewind(bzs, rfc7946=False)
#
# for i in range(len(wind_info_dk)):
#    pnt = Point(wind_info_dk['Longitude'][i], wind_info_dk['Latitude'][i])
#    for f in bzs['features']:
#       polygon = shape(f['geometry'])
#       if polygon.contains(pnt):
#          wind_info_dk['Area'][i] = f['properties']['name']
#
# dk_areas = ['DK1', 'DK2']
# power_centres = pd.DataFrame(columns=['Longitude', 'Latitude'], index=dk_areas)
# for a in dk_areas:
#    df = wind_info_dk.loc[wind_info_dk['Area'] == a]
#    power_centres['Longitude'][a] = (df['Longitude'] * df['Wind']).sum()/ df['Wind'].sum()
#    power_centres['Latitude'][a] = (df['Latitude'] * df['Wind']).sum() / df['Wind'].sum()
# #
# #
# m = folium.Map([60,15], zoom_start=5, tiles='cartodbpositron')
# for _,f in bzs_in.iterrows():
#    # plot polygons with folium
#    polygon = gpd.GeoSeries(f['geometry']).simplify(tolerance=0.001)
#    geo_j = polygon.to_json()
#    geo_j = folium.GeoJson(data=geo_j, style_function=lambda x: {'fillColor': 'orange'})
#    geo_j.add_to(m)
# #
# for a in dk_areas:
#     #plot power centres
#    folium.Marker(location=[power_centres['Latitude'][a], power_centres['Longitude'][a]]).add_to(m)
# #
# for _,i in wind_info_dk.iterrows():
#    # plot weighted point
#    if i['Wind'] == 0:
#       pass
#    else:
#       folium.CircleMarker(location=[i['Latitude'], i['Longitude']], radius=2, weight=i['Wind']/20).add_to(m)
# #
#
#
#
# ########FINLAND###############
# wind_in_fi_raw = pd.read_excel('C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\KTH\\Papers&Projects\\J3 - Balancing analysis\\Vind_Finland_2022.xlsx', sheet_name='Tuulivoimalat')
# wind_in_fi_raw = wind_in_fi_raw.loc[wind_in_fi_raw[wind_in_fi_raw.columns.tolist()[13]]==1]
# wind_in_fi_raw = wind_in_fi_raw.iloc[:,[1,6]]
# name_list = []
# wind_list = []
# for _,f in wind_in_fi_raw.iterrows():
#     if f['Municipality'] in name_list:
#         pass
#     elif f['Municipality'] == '\'Ikaalinen':
#         pass
#     elif f['Municipality'] == 'Ikaalinen':
#         name_list.append(f['Municipality'])
#         df = wind_in_fi_raw.loc[wind_in_fi_raw['Municipality'] == f['Municipality']]
#         df = df.iloc[:,[1]]
#         wind_list.append((df['Nominal power\n(kW)'].sum() + 6000) / 1000)
#     else:
#         name_list.append(f['Municipality'])
#         df = wind_in_fi_raw.loc[wind_in_fi_raw['Municipality'] == f['Municipality']]
#         df = df.iloc[:,[1]]
#         wind_list.append(df['Nominal power\n(kW)'].sum() / 1000)
#
# wind_info_fi = pd.DataFrame(columns=['Kommun', 'Wind', 'Longitude', 'Latitude', 'Area'])
# wind_info_fi['Kommun'] = name_list
# wind_info_fi['Wind'] = wind_list
# wind_info_fi['Area'] = ['FI' for i in range(name_list.__len__())]
# municip_in_fi = gpd.read_file('C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\KTH\\Simuleringsdata\\Maps\\finland_kommuner.geojson')
# municip_in_fi = json.loads(municip_in_fi.to_json())
# municip_in_fi = geojson_rewind.rewind(municip_in_fi, rfc7946=False)
# fi_to_se = {'Ikaalinen': 'Ikalis',
#             'Raahe': 'Brahestad',
#             'Luoto': 'Larsmo',
#             'Hamina': 'Fredrikshamn, Finland',
#             'Närpiö': 'Närpes',
#             'Pori': 'Björneborg',
#             'Liminka': 'Limingo',
#             'Hailuoto': 'Karlö',
#             'Huittinen': 'Vittis',
#             'Kemiönsaari': 'Kimitoön',
#             'Uusikaarlepyy': 'Nykarleby',
#             'Lapua': 'Lappo',
#             'Siikainen': 'Siikais',
#             'Kristiinankaupunki': 'Kristinestad',
#             'Ilmajoki': 'Ilmola',
#             'Tornio': 'Torneå',
#             'Eurajoki': 'Euraåminne',
#             'Ii': 'Ijo',
#             'Karijoki': 'Bötom',
#             'Isojoki': 'Storå, Finland',
#             'Enontekiö': 'Enontekis',
#             'Luhanka': 'Luhango',
#             'Kokkola': 'Karleby',
#             'Maalahti': 'Malax',
#             'Vaasa': 'Vasa',
#             'Uusikaupunki': 'Nystad',
#             'Lappeenranta': 'Villmanstrand',
#             'Teuva': 'Östermark',
#             'Kajaani': 'Kajana',
#             'Sauvo': 'Sagu',
#             'Alavus': 'Alavo',
#             'Oulunsalo': 'Uleåsalo',
#             'Hanko': 'Hangö',
#             'Vöyri': 'Vörå',
#             'Jokioinen': 'Jockis',
#             'Marttila': 'S:t Mårtens',
#             'Oulu': 'Uleåborg',
#             'Urjala': 'Urdiala'
#             }
# key_list = list(fi_to_se.keys())
# val_list = list(fi_to_se.values())
# for f in municip_in_fi['features']:
#     if f['properties']['name'] in name_list:
#         idx = wind_info_fi[wind_info_fi['Kommun']==f['properties']['name']].index.values
#         wind_info_fi['Longitude'][idx] = f['geometry']['coordinates'][0]
#         wind_info_fi['Latitude'][idx] = f['geometry']['coordinates'][1]
#     elif f['properties']['name'][:-9] in name_list:
#         idx = wind_info_fi[wind_info_fi['Kommun']==f['properties']['name'][:-9]].index.values
#         wind_info_fi['Longitude'][idx] = f['geometry']['coordinates'][0]
#         wind_info_fi['Latitude'][idx] = f['geometry']['coordinates'][1]
#     elif f'{f["properties"]["name"]} ' in name_list:
#         idx = wind_info_fi[wind_info_fi['Kommun'] == f'{f["properties"]["name"]} '].index.values
#         wind_info_fi['Longitude'][idx] = f['geometry']['coordinates'][0]
#         wind_info_fi['Latitude'][idx] = f['geometry']['coordinates'][1]
#     elif f["properties"]["name"] in val_list:
#         key = key_list[val_list.index(f["properties"]["name"])]
#         idx = wind_info_fi[wind_info_fi['Kommun'] == key].index.values
#         wind_info_fi['Longitude'][idx] = f['geometry']['coordinates'][0]
#         wind_info_fi['Latitude'][idx] = f['geometry']['coordinates'][1]
#
# bzs_in = gpd.read_file('C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\KTH\\Simuleringsdata\\Maps\\nordicwcoastal.geojson')
# bzs = json.loads(bzs_in.to_json())
# bzs = geojson_rewind.rewind(bzs, rfc7946=False)
# #
# fi_areas = ['FI']
# power_centres = pd.DataFrame(columns=['Longitude', 'Latitude'], index=fi_areas)
# for a in fi_areas:
#    df = wind_info_fi.loc[wind_info_fi['Area'] == a]
#    power_centres['Longitude'][a] = (df['Longitude'] * df['Wind']).sum()/ df['Wind'].sum()
#    power_centres['Latitude'][a] = (df['Latitude'] * df['Wind']).sum() / df['Wind'].sum()
# #
# #
# # m = folium.Map([60,15], zoom_start=5, tiles='cartodbpositron')
# # for _,f in bzs_in.iterrows():
# #    # plot polygons with folium
# #    polygon = gpd.GeoSeries(f['geometry']).simplify(tolerance=0.001)
# #    geo_j = polygon.to_json()
# #    geo_j = folium.GeoJson(data=geo_j, style_function=lambda x: {'fillColor': 'orange'})
# #    geo_j.add_to(m)
#
# for a in fi_areas:
#     #plot power centres
#    folium.Marker(location=[power_centres['Latitude'][a], power_centres['Longitude'][a]]).add_to(m)
# #
# for _,i in wind_info_fi.iterrows():
#    # plot weighted point
#    if i['Wind'] == 0:
#       pass
#    else:
#       folium.CircleMarker(location=[i['Latitude'], i['Longitude']], radius=2, weight=i['Wind']/20).add_to(m)
# #
#
#
#
# ######NORWAY#####
# municip_in_nor = gpd.read_file('C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\KTH\\Simuleringsdata\\Maps\\norge_kommuner.geojson')
# municip_nor = pd.DataFrame(columns=['Name', 'Longitude', 'Latitude'])
# name_list = []
# lat_list = []
# lon_list = []
# for _,f in municip_in_nor.iterrows():
#     if f['geometry'] is None:
#         pass
#     else:
#         name = json.loads(f['navn'])
#         name_list.append(name[0]['navn'])
#         polygon = shape(f['geometry'])
#         center = polygon.centroid
#         lon_list.append(center.x)
#         lat_list.append(center.y)
# municip_nor['Name'] = name_list
# municip_nor['Longitude'] = lon_list
# municip_nor['Latitude'] = lat_list
# #
#
# wind_in_nor = pd.read_excel('C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\Vind_Norge_2021.xlsx')
# wind_info_nor = pd.DataFrame(columns=['Kommun', 'Wind', 'Longitude', 'Latitude', 'Area'])
# wind_info_nor['Kommun'] = wind_in_nor['Kommun'].tolist()
# wind_info_nor['Wind'] = wind_in_nor['Vind'].tolist()
# lat_list = []
# lon_list = []
# area_list = []
# for _,f in wind_in_nor.iterrows():
#     area_list.append(f'NO{f["Prisområde"]}')
#     if f['Kommun'] == 'Sortland – Suortá':
#         idx = municip_nor.loc[municip_nor['Name']=='Sortland'].index.tolist()[0]
#         lat_list.append(municip_nor['Latitude'][idx])
#         lon_list.append(municip_nor['Longitude'][idx])
#     elif f['Kommun'] == 'Våler (Innlandet)':
#         idx = municip_nor.loc[municip_nor['Name'] == 'Våler'].index.tolist()[0]
#         lat_list.append(municip_nor['Latitude'][idx])
#         lon_list.append(municip_nor['Longitude'][idx])
#     else:
#         idx = municip_nor.loc[municip_nor['Name']==f['Kommun']].index.tolist()[0]
#         lat_list.append(municip_nor['Latitude'][idx])
#         lon_list.append(municip_nor['Longitude'][idx])
# wind_info_nor['Longitude'] = lon_list
# wind_info_nor['Latitude'] = lat_list
# wind_info_nor['Area'] = area_list
# #
# bzs_in = gpd.read_file('C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\KTH\\Simuleringsdata\\Maps\\nordicwcoastal.geojson')
# bzs = json.loads(bzs_in.to_json())
# bzs = geojson_rewind.rewind(bzs, rfc7946=False)
# #
# no_areas = ('NO1', 'NO2', 'NO3', 'NO4')#, 'NO5')
# power_centres = pd.DataFrame(columns=['Longitude', 'Latitude'], index=no_areas)
# for a in no_areas:
#    df = wind_info_nor.loc[wind_info_nor['Area'] == a]
#    power_centres['Longitude'][a] = (df['Longitude'] * df['Wind']).sum()/ df['Wind'].sum()
#    power_centres['Latitude'][a] = (df['Latitude'] * df['Wind']).sum() / df['Wind'].sum()
# #
# #
# # m = folium.Map([60,15], zoom_start=5, tiles='cartodbpositron')
# # for _,f in bzs_in.iterrows():
# #    # plot polygons with folium
# #    polygon = gpd.GeoSeries(f['geometry']).simplify(tolerance=0.001)
# #    geo_j = polygon.to_json()
# #    geo_j = folium.GeoJson(data=geo_j, style_function=lambda x: {'fillColor': 'orange'})
# #    geo_j.add_to(m)
#
# for a in no_areas:
#     #plot power centres
#    folium.Marker(location=[power_centres['Latitude'][a], power_centres['Longitude'][a]]).add_to(m)
# #
# for _,i in wind_info_nor.iterrows():
#    # plot weighted point
#    if i['Wind'] == 0:
#       pass
#    else:
#       folium.CircleMarker(location=[i['Latitude'], i['Longitude']], radius=2, weight=i['Wind']/20).add_to(m)
# #
# # m.show_in_browser()
# #
# #### SWEDEN #####
# wind_info = pd.DataFrame(columns=['Kommun', 'Wind', 'Longitude', 'Latitude', 'Area'])
# wind_in = pd.read_excel('C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\Vindinstallationer_2021.xlsx')
#
# num = len(wind_in)
#
# name_list = []
# wind_list = []
# for i in range(num):
#    name_list.append(wind_in['Kommuner'][i][5:])
#    if wind_in['Vind'][i] == '-':
#       wind_list.append(0)
#    else:
#       wind_list.append(wind_in['Vind'][i])
# wind_info['Kommun'] = name_list
# wind_info['Wind'] = wind_list
#
# municip_in = gpd.read_file('C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\KTH\\Simuleringsdata\\Maps\\kommuner.geojson')
# municip_in = json.loads(municip_in.to_json())
# municip = geojson_rewind.rewind(municip_in, rfc7946=False)
#
# id_list = []
# for f in municip['features']:
#    if f['properties']['name'][:-7] in name_list:
#       idx = wind_info[wind_info['Kommun']==f['properties']['name'][:-7]].index.values
#       wind_info['Longitude'][idx] = f['geometry']['coordinates'][0]
#       wind_info['Latitude'][idx] = f['geometry']['coordinates'][1]
#    elif f['properties']['name'][:-8] in name_list:
#       idx = wind_info[wind_info['Kommun']==f['properties']['name'][:-8]].index.values
#       wind_info['Longitude'][idx] = f['geometry']['coordinates'][0]
#       wind_info['Latitude'][idx] = f['geometry']['coordinates'][1]
#    elif f['properties']['name'][7:] in name_list:
#       idx = wind_info[wind_info['Kommun']==f['properties']['name'][7:]].index.values
#       wind_info['Longitude'][idx] = f['geometry']['coordinates'][0]
#       wind_info['Latitude'][idx] = f['geometry']['coordinates'][1]
#    elif f['properties']['name'] == 'Falu kommun':
#       idx = wind_info[wind_info['Kommun']=='Falun'].index.values
#       wind_info['Longitude'][idx] = f['geometry']['coordinates'][0]
#       wind_info['Latitude'][idx] = f['geometry']['coordinates'][1]
#    elif f['properties']['name'] == 'Åsele kommun, 2.875 invånare 31 december 2013 display=inline':
#       idx = wind_info[wind_info['Kommun']=='Åsele'].index.values
#       wind_info['Longitude'][idx] = f['geometry']['coordinates'][0]
#       wind_info['Latitude'][idx] = f['geometry']['coordinates'][1]
#    elif f['properties']['name'] == 'Robertsfors kommun, 6.738 invånare 31 december 2013':
#       idx = wind_info[wind_info['Kommun']=='Robertsfors'].index.values
#       wind_info['Longitude'][idx] = f['geometry']['coordinates'][0]
#       wind_info['Latitude'][idx] = f['geometry']['coordinates'][1]
#
#
# bzs_in = gpd.read_file('C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\KTH\\Simuleringsdata\\Maps\\nordicwcoastal.geojson')
# bzs = json.loads(bzs_in.to_json())
# bzs = geojson_rewind.rewind(bzs, rfc7946=False)
#
# for i in range(num):
#    pnt = Point(wind_info['Longitude'][i], wind_info['Latitude'][i])
#    for f in bzs['features']:
#       polygon = shape(f['geometry'])
#       if polygon.contains(pnt):
#          wind_info['Area'][i] = f['properties']['name']
#
#
# se_areas = ('SE1', 'SE2', 'SE3', 'SE4')
# power_centres = pd.DataFrame(columns=['Longitude', 'Latitude'], index=se_areas)
# for a in se_areas:
#    df = wind_info.loc[wind_info['Area'] == a]
#    power_centres['Longitude'][a] = (df['Longitude'] * df['Wind']).sum()/ df['Wind'].sum()
#    power_centres['Latitude'][a] = (df['Latitude'] * df['Wind']).sum() / df['Wind'].sum()
#
#
# # m = folium.Map([60,15], zoom_start=5, tiles='cartodbpositron')
# # for _,f in bzs_in.iterrows():
# #    # plot polygons with folium
# #    polygon = gpd.GeoSeries(f['geometry']).simplify(tolerance=0.001)
# #    geo_j = polygon.to_json()
# #    geo_j = folium.GeoJson(data=geo_j, style_function=lambda x: {'fillColor': 'orange'})
# #    geo_j.add_to(m)
#
# for a in se_areas:
#    #plot power centres
#    folium.Marker(location=[power_centres['Latitude'][a], power_centres['Longitude'][a]]).add_to(m)
#
# for _,i in wind_info.iterrows():
#    # plot weighted point
#    if i['Wind'] == 0:
#       pass
#    else:
#       folium.CircleMarker(location=[i['Latitude'], i['Longitude']], radius=2, weight=i['Wind']/20).add_to(m)
#
#
# # m.show_in_browser()
#
# wind_nordic = pd.DataFrame(columns=['Kommun', 'Wind', 'Longitude', 'Latitude', 'Area'])
# wind_nordic['Kommun'] = wind_info['Kommun'].tolist() + wind_info_fi['Kommun'].tolist() + wind_info_nor['Kommun'].tolist() + wind_info_dk['Kommun'].tolist()
# wind_nordic['Wind'] = wind_info['Wind'].tolist() + wind_info_fi['Wind'].tolist() + wind_info_nor['Wind'].tolist() + wind_info_dk['Wind'].tolist()
# wind_nordic['Longitude'] = wind_info['Longitude'].tolist() + wind_info_fi['Longitude'].tolist() + wind_info_nor['Longitude'].tolist() + wind_info_dk['Longitude'].tolist()
# wind_nordic['Latitude'] = wind_info['Latitude'].tolist() + wind_info_fi['Latitude'].tolist() + wind_info_nor['Latitude'].tolist() + wind_info_dk['Latitude'].tolist()
# wind_nordic['Area'] = wind_info['Area'].tolist() + wind_info_fi['Area'].tolist() + wind_info_nor['Area'].tolist() + wind_info_dk['Area'].tolist()


# All nordic plot
bzs_in = gpd.read_file('C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\KTH\\Simuleringsdata\\Maps\\nordicwcoastal.geojson')
bzs = json.loads(bzs_in.to_json())
bzs = geojson_rewind.rewind(bzs, rfc7946=False)
wind_info = pd.read_csv('C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\wind_nordic.csv')

areas = ('SE1', 'SE2', 'SE3', 'SE4', 'NO1', 'NO2', 'NO3', 'NO4', 'DK1', 'DK2', 'FI')
power_centres = pd.DataFrame(columns=['Longitude', 'Latitude'], index=areas)
for a in areas:
   df = wind_info.loc[wind_info['Area'] == a]
   power_centres['Longitude'][a] = (df['Longitude'] * df['Wind']).sum()/ df['Wind'].sum()
   power_centres['Latitude'][a] = (df['Latitude'] * df['Wind']).sum() / df['Wind'].sum()


m = folium.Map([60,15], zoom_start=5, tiles='cartodbpositron')
for _,f in bzs_in.iterrows():
   # plot polygons with folium
   polygon = gpd.GeoSeries(f['geometry']).simplify(tolerance=0.001)
   geo_j = polygon.to_json()
   geo_j = folium.GeoJson(data=geo_j, style_function=lambda x: {'fillColor': 'orange'})
   geo_j.add_to(m)

for a in areas:
   #plot power centres
   folium.Marker(location=[power_centres['Latitude'][a], power_centres['Longitude'][a]]).add_to(m)

for _,i in wind_info.iterrows():
   # plot weighted point
   if i['Wind'] == 0:
      pass
   else:
      folium.CircleMarker(location=[i['Latitude'], i['Longitude']], radius=2, weight=i['Wind']/30).add_to(m)

m.show_in_browser()

###Compute average distances