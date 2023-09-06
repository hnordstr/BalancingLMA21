import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import geojson_rewind
import geopandas as gpd
import plotly.express as px
import json

"""Pickle file with dictionary:
path\\{scenario}_{year}.pickle some years/scenarios have _FixRamp, _Quarter, and _TRM
{
High: {Netted imbalance, Pre-net imbalance, Slow imbalance, Fast imbalance, FCR imbalance, Stochastic imbalance, Deterministic imbalance,
 Netting transmission, Post-net transmission},
Low: {NTC, ATC, HVDC, Wind, PV, Consumption, Hydro, Thermal, Nuclear, Flex},
Time: Run-time
}
"""

areas = ('SE1', 'SE2', 'SE3', 'SE4', 'NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'DK2', 'FI')
lines = {
        'SE2SE1': 'SE1SE2',
        'NO4SE1': 'SE1NO4',
        'FISE1': 'SE1FI',
        'NO4SE2': 'SE2NO4',
        'NO3SE2': 'SE2NO3',
        'SE3SE2': 'SE2SE3',
        'NO1SE3': 'SE3NO1',
        'SE4SE3': 'SE3SE4',
        'DK2SE4': 'SE4DK2',
        'NO2NO1': 'NO1NO2',
        'NO3NO1': 'NO1NO3',
        'NO5NO1': 'NO1NO5',
        'NO5NO2': 'NO2NO5',
        'NO5NO3': 'NO3NO5',
        'NO4NO3': 'NO3NO4'
}

scenarios = ['SF45', 'EP45', 'EF45', 'FM45']
years = [2009, 1999, 1992]

scenario_dict = {
    'EF45': 'ER', #Electrification Renewable
    'EP45': 'EP', #Electrification Plannable
    'FM45': 'MP', #Mixed Plans
    'SF45': 'SR' #Small-scale Renewable
}

path = 'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\KTH\\Papers&Projects\\J3 - Balancing analysis\\Results\\'
area='SE4'
color_dict = {'EF45': 'green', 'FM45': 'red', 'EP45': 'blue', 'SF45':'orange'}

def duration_plot(area, year):
    plt.rcParams.update({'font.size': 12})
    data_dict = {}
    zero_max = 0
    for s in scenarios:
        with open(f'{path}{s}_{year}.pickle', 'rb') as handle:
            dict_in = pkl.load(handle)
        df = pd.DataFrame(columns=['Imbalance', 'Interval', 'Duration', 'Percentage'])
        imb = dict_in['High']['Netted imbalance']
        imb = imb.abs()
        imb = imb.sort_values(by=[area], ascending=False)
        df['Imbalance'] = imb[area].tolist()
        df['Interval'] = [1 for i in range(imb.__len__())]
        df['Duration'] = df['Interval'].cumsum()
        df['Percentage'] = df['Duration'] * 100 / imb.__len__()
        data_dict[s] = df
        zero_val = df['Percentage'][df[df['Imbalance'] == 0].index[0]]
        if zero_val > zero_max:
            zero_max = zero_val
    df = pd.DataFrame(columns=['Scenario', 'Imbalance'])
    imb_list = [data_dict[s]['Imbalance'].sum() for s in scenarios]
    df['Scenario'] = scenarios
    df['Imbalance'] = imb_list
    df = df.sort_values(by=['Imbalance'], ascending=False)
    df.reset_index(drop=True, inplace=True)
    scenario_new = df['Scenario'].tolist()
    x = np.linspace(0, 100, 52 * 7 * 24 * 60)
    for i in range(4):
        plt.plot(data_dict[scenario_new[i]]['Percentage'].tolist(), data_dict[scenario_new[i]]['Imbalance'].tolist(),
                 label=scenario_dict[scenario_new[i]], linewidth=2, color=color_dict[scenario_new[i]])
        if i == 3:
            pass
            plt.fill_between(x, data_dict[scenario_new[i]]['Imbalance'].tolist(), alpha=0.4,
                             color=color_dict[scenario_new[i]])
        else:
            plt.fill_between(x, data_dict[scenario_new[i + 1]]['Imbalance'].tolist(),
                             data_dict[scenario_new[i]]['Imbalance'].tolist(), alpha=0.4,
                             color=color_dict[scenario_new[i]])
    max_val = data_dict[scenario_new[0]]['Imbalance'].max() * 0.7
    plt.ylabel('Absolute imbalance [MW]')
    plt.xlabel('Share of time [%]')
    plt.xlim(0,1.2 * zero_max)
    plt.ylim(0, max_val)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    fig = plt.gcf()
    save = False
    if save:
        fig.savefig(
            f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\Figures\\DurationPlot_{area}_{year}.pdf',
            dpi=fig.dpi, pad_inches=0, bbox_inches='tight')
    #plt.show()
    plt.clf()


def time_plot(area, year):
    intervals = [f'{int(i * 10)}-{int((i + 1) * 10)}' for i in range(6)]
    start = [int(i * 10) for i in range(6)]
    end= [int((i + 1) * 10) for i in range(6)]
    df = pd.DataFrame(columns=['Imbalance [MW]', 'Time interval [min]', 'Scenario'])
    imb_list = []
    time_list = []
    scen_list = []
    for scenario in ['EF45', 'EP45']:
        with open(f'{path}{scenario}_{year}.pickle', 'rb') as handle:
            dict_in = pkl.load(handle)
        imb = dict_in['High']['Netted imbalance']
        imb = imb.abs()
        imb = imb[area].tolist()
        hours = range(int(imb.__len__()/60))
        for i in range(6):
            lst = []
            for h in hours:
                lst.extend(imb[h * 60 + start[i]: h * 60 + end[i]])
            lst.sort(reverse=True)
            imb_list.extend(lst[:1000])
            scen_list.extend([scenario_dict[scenario] for n in range(1000)])
            time_list.extend(intervals[i] for n in range(1000))
    df['Imbalance [MW]'] = imb_list
    df['Time interval [min]'] = time_list
    df['Scenario'] = scen_list
    plt.rcParams.update({'font.size': 12})
    plt.grid()
    sns.violinplot(x=df['Time interval [min]'], y=df['Imbalance [MW]'], hue=df['Scenario'], split=True)# palette='Set2')
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig = plt.gcf()
    save = True
    if save:
        fig.savefig(
            f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\Figures\\TimePlot_{area}_{year}.pdf',
            dpi=fig.dpi, pad_inches=0, bbox_inches='tight')
    #plt.show()
    plt.clf()

def year_comparison_table(area, scenario):
    df = pd.DataFrame(columns=years, index=['Abs mean', '1st percentile', '99th percentile', 'Zero share', 'Pos share', 'Neg share'])
    for y in years:
        with open(f'{path}{scenario}_{y}.pickle', 'rb') as handle:
            dict_in = pkl.load(handle)
        imb = dict_in['High']['Netted imbalance']
        df[y]['Abs mean'] = round(imb[area].abs().mean(), 0)
        df[y]['1st percentile'] = round(imb[area].quantile(q=0.01), 0)
        df[y]['99th percentile'] = round(imb[area].quantile(q=0.99), 0)
        df[y]['Zero share'] = f'{round(100 * (imb[area] == 0).sum() / imb[area].__len__(), 1)} %'
        df[y]['Pos share'] = f'{round(100 * (imb[area] > 0).sum() / imb[area].__len__(), 1)} %'
        df[y]['Neg share'] = f'{round(100 * (imb[area] < 0).sum() / imb[area].__len__(), 1)} %'
    print(f'TABLE FOR AREA {area} SCENARIO {scenario}')
    print(df)

def year_comparison_pdf(area, scenario): #Not so good looking
    df = pd.DataFrame(columns=['Imbalance [MW]', 'Year'])
    imb_list = []
    year_list = []
    for y in years:
        with open(f'{path}{scenario}_{y}.pickle', 'rb') as handle:
            dict_in = pkl.load(handle)
        imb_list.extend(dict_in['High']['Netted imbalance'][area].tolist())
        year_list.extend(y for n in range(dict_in['High']['Netted imbalance'].__len__()))
    df['Imbalance [MW]'] = imb_list
    df['Year'] = year_list
    plt.rcParams.update({'font.size': 12})
    sns.kdeplot(data=df, x='Imbalance [MW]', hue='Year', fill=True, alpha=0.4, palette='crest', linewidth=2)
    plt.xlim(-2000, 2000)
    plt.grid()
    plt.tight_layout()
    plt.show()

def sensitivity_comparison(area):
    df = pd.DataFrame(columns=['Imbalance [MW]', 'Case'])
    imb_list = []
    case_list = []
    for f in ['', '_Quarter', '_FixRamp', '_TRM', '_ImprovedForecast']:
        with open(f'{path}EF45_2009{f}.pickle', 'rb') as handle:
            dict_in = pkl.load(handle)
        imb_list.extend(dict_in['High']['Netted imbalance'][area].abs().tolist())
        if f == '':
            case_list.extend('Base' for n in range(dict_in['High']['Netted imbalance'].__len__()))
        elif f == '_Quarter':
            case_list.extend('Quarters' for n in range(dict_in['High']['Netted imbalance'].__len__()))
        elif f == '_FixRamp':
            case_list.extend('Fixed ramp' for n in range(dict_in['High']['Netted imbalance'].__len__()))
        elif f == '_TRM':
            case_list.extend('TRM' for n in range(dict_in['High']['Netted imbalance'].__len__()))
        elif f == '_ImprovedForecast':
            case_list.extend('Better forecast' for n in range(dict_in['High']['Netted imbalance'].__len__()))
    df['Imbalance [MW]'] = imb_list
    df['Case'] = case_list
    plt.rcParams.update({'font.size': 12})
    sns.boxenplot(data=df, x='Case', y='Imbalance [MW]')
    plt.grid(axis='y')
    plt.tight_layout()
    fig = plt.gcf()
    save = True
    if save:
        fig.savefig(
            f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\Figures\\Sensitivity_{area}.pdf',
            dpi=fig.dpi, pad_inches=0, bbox_inches='tight')
    plt.clf()

def map_plot(scenario, year):
    with open(f'{path}{scenario}_{year}.pickle', 'rb') as handle:
        dict_in = pkl.load(handle)
    imb = pd.DataFrame(columns=['Imbalance', 'id'] )
    imb['Imbalance'] = [dict_in['High']['Netted imbalance'][a].abs().mean() for a in areas]
    imb['id'] = [a for a in areas]

    ntc = dict_in['Low']['NTC']
    transmission = dict_in['High']['Post-net transmission']

    transm = pd.DataFrame(columns=lines.keys())
    for l in lines.keys():
        transm[l] = transmission[l] - transmission[lines[l]]
    congestion = pd.DataFrame(columns=['Congestion', 'id'])
    congestion['id'] = [l for l in lines.keys()]
    con_list = []
    for l in lines.keys():
        pos_con = transm.loc[transm[l] == ntc[l]].__len__()
        neg_con = transm.loc[transm[l] == - ntc[lines[l]]].__len__()
        con_list.append(100 * (pos_con + neg_con) / transm.__len__())
    congestion['Congestion'] = con_list

    maps_in = gpd.read_file('C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\KTH\\Papers&Projects\\DynamicFRR\\nordic.geojson')
    maps_in = json.loads(maps_in.to_json())
    maps = geojson_rewind.rewind(maps_in, rfc7946=False)

    for f in maps['features']:
       f['id'] = f['properties']['name']

    fig = px.choropleth(imb,
                       locations='id',
                       geojson=maps,
                       featureidkey='id',
                       color='Imbalance',
                       hover_name='id',
                       color_continuous_scale=px.colors.sequential.Blues,
                       range_color=[0, 800],
                       scope='europe')
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.update_layout(coloraxis_colorbar=dict(
       ticks="outside", ticksuffix=" MW",
       dtick=200, len=0.8, thickness=50, y=0.53,x=0.67,
       title=dict(text='Imbalance', font=dict(size=30))
    ))
    fig.update_coloraxes(colorbar_tickfont_size=30)
    line_map = gpd.read_file('C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\KTH\\Papers&Projects\\J3 - Balancing analysis\\linemap.geojson')

    for i in congestion.index:
        lon1 = float(line_map.loc[line_map['id'] == f'{congestion["id"][i]}_start']['geometry'].x)
        lat1 = float(line_map.loc[line_map['id'] == f'{congestion["id"][i]}_start']['geometry'].y)
        lon2 = float(line_map.loc[line_map['id'] == f'{congestion["id"][i]}_end']['geometry'].x)
        lat2 = float(line_map.loc[line_map['id'] == f'{congestion["id"][i]}_end']['geometry'].y)
        if abs(lon1 - lon2) > abs(lat1 - lat2):
            lat2 = lat1
        else:
            lon2 = lon1



        fig.add_trace(go.Scattergeo(lat=[lat1, lat2], lon = [lon1, lon2], mode='lines',
                                    line=dict(width=1 + congestion['Congestion'][i]/3, color='orange')))
    fig.show()

for a in areas:
    time_plot(a, 2009)