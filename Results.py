import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns

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
scenarios = ['SF45', 'EP45', 'EF45', 'FM45']

scenario_dict = {
    'EF45': 'ER', #Electrification Renewable
    'EP45': 'EP', #Electrification Plannable
    'FM45': 'MP', #Mixed Plans
    'SF45': 'SR' #Small-scale Renewable
}

path = 'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\KTH\\Papers&Projects\\J3 - Balancing analysis\\Results\\'
area='SE4'

def duration_curve(area, year):
    for s in scenarios:
        with open(f'{path}{s}_{year}.pickle', 'rb') as handle:
            dict = pkl.load(handle)

        df = pd.DataFrame(columns=['Imbalance', 'Interval', 'Duration', 'Percentage'])
        imb = dict['High']['Netted imbalance']
        imb = imb.abs()
        imb = imb.sort_values(by=[area], ascending=False)
        df['Imbalance'] = imb[area].tolist()
        df['Interval'] = [1 for i in range(imb.__len__())]
        df['Duration'] = df['Interval'].cumsum()
        df['Percentage'] = df['Duration'] * 100 / imb.__len__()
        plt.plot(df['Percentage'].tolist(), df['Imbalance'].tolist(), label=scenario_dict[s], linewidth=2, linestyle='--')
    plt.ylabel('Absolute imbalance [MW]')
    plt.xlabel('Share of time [%]')
    plt.title(f'Duration curve {area}')
    plt.xlim(-1,101)
    plt.grid()
    plt.legend()
    plt.show()


def time_plot(area, scenario, year):
    intervals = [f'{int(i * 5)}-{int((i + 1) * 5)}' for i in range(12)]
    start = [int(i * 5) for i in range(12)]
    end= [int((i + 1) * 5) for i in range(12)]
    df = pd.DataFrame(columns=intervals)
    with open(f'{path}{scenario}_{year}.pickle', 'rb') as handle:
        dict = pkl.load(handle)
    imb = dict['High']['Netted imbalance']
    imb = imb.abs()
    imb = imb[area].tolist()
    hours = range(int(imb.__len__()/60))
    for i in range(12):
        lst = []
        for h in hours:
            lst.extend(imb[h * 60 + start[i]: h * 60 + end[i]])
        df[intervals[i]] = lst
    sns.boxplot(data=df)
    plt.ylabel('Imbalance [MW]')
    plt.xlabel('Time of hour [min]')
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()


duration_curve('FI', 2009)