"""This is the model where all wind forecast data is read and the relevant metrics are calculated"""
import pandas as pd
import pickle as pkl
import numpy as np
import geopy.distance
import folium
import geopandas as gpd
import geojson_rewind
import json
import matplotlib.pyplot as plt
import scipy.optimize

areas = ('SE1', 'SE2', 'SE3', 'SE4', 'NO1', 'NO2', 'NO3', 'NO4', 'DK1', 'DK2', 'FI')
data_areas = ('SE1', 'SE2', 'SE3', 'SE4','FI', 'DK1', 'DK2')

"""
1. Drop zero wind areas 
2. Compute total wind in area
3. Compute distance, weighted by normalized wind
4. Compute average 
"""
da_to_id = 0.83684

def compute_average_distances(plot=False):
    wind_info = pd.read_csv(
'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\wind_nordic.csv')
    power_centres = pd.DataFrame(columns=['Longitude', 'Latitude'], index=areas)
    average_distance = pd.DataFrame(index=areas, columns=['km'])
    for a in areas:
        df = wind_info.loc[wind_info['Area'] == a]
        power_centres['Longitude'][a] = (df['Longitude'] * df['Wind']).sum() / df['Wind'].sum()
        power_centres['Latitude'][a] = (df['Latitude'] * df['Wind']).sum() / df['Wind'].sum()
    for a in areas:
        df = wind_info.loc[wind_info['Area'] == a]
        df = df.loc[df['Wind'] > 0]
        wind_tot = df['Wind'].sum()

        cent_coord = (power_centres['Latitude'][a], power_centres['Longitude'][a])
        dist = []
        for _,i in df.iterrows():
            coord = (i['Latitude'], i['Longitude'])
            dist.append(geopy.distance.geodesic(cent_coord, coord).km * i['Wind'] / wind_tot)
            average_distance['km'][a] = sum(dist)

    if plot:
        bzs_in = gpd.read_file(
            'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\KTH\\Simuleringsdata\\Maps\\nordicwcoastal.geojson')
        bzs = json.loads(bzs_in.to_json())
        bzs = geojson_rewind.rewind(bzs, rfc7946=False)
        m = folium.Map([60, 15], zoom_start=5, tiles='cartodbpositron')
        for _, f in bzs_in.iterrows():
            # plot polygons with folium
            polygon = gpd.GeoSeries(f['geometry']).simplify(tolerance=0.001)
            geo_j = polygon.to_json()
            geo_j = folium.GeoJson(data=geo_j, style_function=lambda x: {'fillColor': 'orange'})
            geo_j.add_to(m)

        for a in areas:
            # plot power centres
            folium.Marker(location=[power_centres['Latitude'][a], power_centres['Longitude'][a]]).add_to(m)

        for _, i in wind_info.iterrows():
            # plot weighted point
            if i['Wind'] == 0:
                pass
            else:
                folium.CircleMarker(location=[i['Latitude'], i['Longitude']], radius=2, weight=i['Wind'] / 30).add_to(m)
        m.show_in_browser()
    return average_distance, power_centres

def read_sefi_forecast():
    data_areas = ('SE1', 'SE2', 'SE3', 'SE4', 'FI')
    with open('sefi_forecasts.pickle', 'rb') as f:
        forecast_data = pkl.load(f)
    for a in data_areas:
        wind_dict[a] = forecast_data['Wind'][a]

dk_areas = ['DK1', 'DK2']

def read_dk_forecast():
    start = 3635
    end = 2064
    skip = 2018
    for a in dk_areas:
        data_2022 = pd.read_csv(f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\{a}_2022.csv')
        data_2023 = pd.read_csv(f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\{a}_2023.csv')
        # print(data_2023[f'MTU (CET/CEST)'][skip])
        # print(data_2023[f'Generation - Wind Offshore  [MW] Day Ahead/ BZN|{a}'][skip])
        data_2023 = data_2023.drop(index=[skip])
        actual_offsh_list = []
        actual_onsh_list = []
        time_list = []
        da_offsh_list = []
        da_onsh_list = []
        da_offsh_list.extend(data_2022[f'Generation - Wind Offshore  [MW] Day Ahead/ BZN|{a}'][start:].astype(float))
        actual_offsh_list.extend(data_2022[f'Generation - Wind Offshore  [MW] Current / BZN|{a}'][start:].astype(float))
        da_onsh_list.extend(data_2022[f'Generation - Wind Onshore  [MW] Day Ahead/ BZN|{a}'][start:].astype(float))
        actual_onsh_list.extend(data_2022[f'Generation - Wind Onshore  [MW] Current / BZN|{a}'][start:].astype(float))
        time_list.extend(data_2022[f'MTU (CET/CEST)'][start:])
        da_offsh_list.extend(data_2023[f'Generation - Wind Offshore  [MW] Day Ahead/ BZN|{a}'][:end].astype(float))
        actual_offsh_list.extend(data_2023[f'Generation - Wind Offshore  [MW] Current / BZN|{a}'][:end].astype(float))
        da_onsh_list.extend(data_2023[f'Generation - Wind Onshore  [MW] Day Ahead/ BZN|{a}'][:end].astype(float))
        actual_onsh_list.extend(data_2023[f'Generation - Wind Onshore  [MW] Current / BZN|{a}'][:end].astype(float))
        time_list.extend(data_2023[f'MTU (CET/CEST)'][:end])
        wind_dict[a] = pd.DataFrame(columns=['Time', 'D-1 offsh', 'D-1 onsh' 'ID', 'Actual offsh', 'Actual onsh' ,'D-1 error', 'ID error'])
        wind_dict[a]['Time'] = time_list
        wind_dict[a]['D-1 offsh'] = da_offsh_list
        wind_dict[a]['D-1 onsh'] = da_onsh_list
        wind_dict[a]['D-1'] = wind_dict[a]['D-1 onsh'] + wind_dict[a]['D-1 offsh']
        wind_dict[a]['Actual offsh'] = actual_offsh_list
        wind_dict[a]['Actual onsh'] = actual_onsh_list
        wind_dict[a]['Actual'] = wind_dict[a]['Actual onsh'] + wind_dict[a]['Actual offsh']
        wind_dict[a]['D-1 error'] = wind_dict[a]['D-1'] - wind_dict[a]['Actual']
        wind_dict[a]['ID error'] = wind_dict[a]['D-1 error'] * da_to_id
        wind_dict[a]['ID'] = wind_dict[a]['Actual'] + wind_dict[a]['ID error']
        #sort out errors here

wind_dict = {}
read_sefi_forecast()
read_dk_forecast()

dist,centres = compute_average_distances()
metrics = pd.DataFrame(columns=['Abs', 'Std', 'Var', 'Distance'], index=data_areas)
print(centres)

for a in data_areas:
     metrics['Distance'][a] = dist['km'][a]
     df = wind_dict[a]
     metrics['Abs'][a] = float(df['ID error'].abs().sum() / df['ID'].sum())
     metrics['Std'][a] = float(df['ID error'].std() / df['ID'].mean())
     metrics['Var'][a] = float(np.sum(np.abs(np.subtract(np.array(df['ID error'][1:]),np.array(df['ID error'][:-1])))) /
                               df['ID'].sum())

def fit_functions():
    plt.rcParams.update({'font.size': 12})
    plt.scatter(metrics['Distance'], metrics['Abs'], label='_nolegend_', color='blue', alpha=0.5, s=50)
    a, b = np.polyfit(np.array(metrics['Distance'].tolist()), np.array(metrics['Abs'].tolist()), 1)
    vec = np.array(range(0, int(1.2 * metrics['Distance'].max()),1))
    plt.plot(vec, a * vec + b, label=r'$\phi^{1}$' + f' = {round(b,5)} - {-round(a,5)}' + r'$\cdot$x', color='blue')
    plt.scatter(metrics['Distance'], metrics['Std'], label='_nolegend_', color='green', alpha=0.5, s=50)
    a, b = np.polyfit(np.array(metrics['Distance'].tolist()), np.array(metrics['Std'].tolist()), 1)
    plt.plot(vec, a * vec + b,  label=r'$\phi^{2}$' + f' = {round(b,5)} - {-round(a,5)}' + r'$\cdot$x', color='green')
    plt.scatter(metrics['Distance'], metrics['Var'], label='_nolegend_', color='red', alpha=0.5, s=50)
    a, b = np.polyfit(np.array(metrics['Distance'].tolist()), np.array(metrics['Var'].tolist()), 1)
    plt.plot(vec, a * vec + b, label=r'$\phi^{3}$' + f' = {round(b,5)} - {-round(a,5)}' + r'$\cdot$x', color='red')
    plt.xlabel('Area diameter [km]')
    plt.ylabel('Target value')
    plt.ylim(0, 0.37)
    plt.xlim(-0.01, int(1.2 * metrics['Distance'].max()))
    plt.grid()
    plt.legend()
    plt.tight_layout()
    fig = plt.gcf()
    save = True
    if save:
        fig.savefig(
            f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\Figures\\WindTargets.pdf',
            dpi=fig.dpi, pad_inches=0, bbox_inches='tight')
    plt.show()

def compute_correlation():
    name_list = []
    dist_list = []
    corr_list = []
    correlation = pd.DataFrame(columns=['Areas', 'Distance', 'Correlation'])
    for a in data_areas:
        for b in data_areas:
            if a == b:
                pass
            elif f'{b}-{a}' in name_list:
                pass
            else:
                xx = f'{a}-{b}'
                name_list.append(xx)
                coord_1 = (centres['Latitude'][a], centres['Longitude'][a])
                coord_2 = (centres['Latitude'][b], centres['Longitude'][b])
                dist_list.append(geopy.distance.geodesic(coord_1, coord_2).km)
                #corr_list.append(solar_dict[a]['ID error'].corr(solar_dict[b]['ID error']))
                df = pd.DataFrame(columns=['A', 'B'])
                df['A'] = wind_dict[a]['ID error']
                df['B'] = wind_dict[b]['ID error']
                df = df.dropna()
                corr_list.append(df['A'].corr(df['B']))
    correlation['Areas'] = name_list
    correlation['Distance'] = dist_list
    correlation['Correlation'] = corr_list
    bp = 700
    df1 = correlation.loc[correlation['Distance'] <= bp]
    df2 = correlation.loc[correlation['Distance'] > bp]
    a1, b1 = np.polyfit(np.array(df1['Distance']), np.array(df1['Correlation']), 1)
    a2, b2 = np.polyfit(np.array(df2['Distance']), np.array(df2['Correlation']), 1)
    vec1 = np.array(range(0, bp, 1))
    vec2 = np.array(range(bp, int(1.1 * df2['Distance'].max()), 1))
    b1 += 0.05
    a1 -= 0.0001
    plt.rcParams.update({'font.size': 12})
    plt.plot(vec1, a1 * vec1 + b1, label=r'$y_{1} = $' + f'{round(b1, 5)} - {-round(a1, 5)}' + r'$\cdot$x', color='blue')
    plt.plot(vec2, a2 * vec2 + b2, label=r'$y_{2} = $' + f'{round(b2, 5)} - {-round(a2, 5)}' + r'$\cdot$x', color='blue')
    plt.scatter(correlation['Distance'], correlation['Correlation'], alpha=0.5, color='blue', s=50, label='_nolegend_')
    plt.xlabel('Distance between power centers [km]')
    plt.ylabel('Correlation')
    plt.grid()
    plt.legend()
    plt.xlim(-0.01, int(1.1 * df2['Distance'].max()))
    plt.tight_layout()
    fig = plt.gcf()
    save = True
    if save:
        fig.savefig(
            f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\Figures\\WindCorrelation.pdf',
            dpi=fig.dpi, pad_inches=0, bbox_inches='tight')
    plt.show()

compute_correlation()