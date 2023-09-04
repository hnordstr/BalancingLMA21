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
years = [2009, 1999, 1992]

scenario_dict = {
    'EF45': 'ER', #Electrification Renewable
    'EP45': 'EP', #Electrification Plannable
    'FM45': 'MP', #Mixed Plans
    'SF45': 'SR' #Small-scale Renewable
}

path = 'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\KTH\\Papers&Projects\\J3 - Balancing analysis\\Results\\'
area='SE4'
color_dict = {0: 'green', 1: 'red', 2:'blue', 3:'orange'}

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
                 label=scenario_dict[scenario_new[i]], linewidth=2, color=color_dict[i])
        if i == 3:
            pass
            plt.fill_between(x, data_dict[scenario_new[i]]['Imbalance'].tolist(), alpha=0.4, color=color_dict[i])
        else:
            plt.fill_between(x, data_dict[scenario_new[i + 1]]['Imbalance'].tolist(),
                             data_dict[scenario_new[i]]['Imbalance'].tolist(), alpha=0.4, color=color_dict[i])
    max_val = data_dict[scenario_new[0]]['Imbalance'].max() * 0.7
    plt.ylabel('Absolute imbalance [MW]')
    plt.xlabel('Share of time [%]')
    plt.xlim(0,1.2 * zero_val)
    plt.ylim(0, max_val)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    fig = plt.gcf()
    save = True
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
    for scenario in scenarios:
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
    sns.boxplot(x=df['Time interval [min]'], y=df['Imbalance [MW]'], hue=df['Scenario'], palette='Set2')
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig = plt.gcf()
    save = True
    if save:
        fig.savefig(
            f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\Figures\\TimePlot_{area}_{year}.pdf',
            dpi=fig.dpi, pad_inches=0, bbox_inches='tight')
    plt.show()
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




print(year_comparison_table('SE2', 'SF45'))