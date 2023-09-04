"""This file reads demand data to find the appropriate target metrics"""
import pandas as pd
import requests
import json
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
from datetime import datetime, timedelta
areas = ('SE1', 'SE2', 'SE3', 'SE4', 'NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'DK2', 'FI')

def read_forecasts():
    demand_dict = {}
    data_areas = ['FI']
    def read_sefi_forecast():
        with open('sefi_forecasts.pickle', 'rb') as f:
            forecast_data = pkl.load(f)
        for a in data_areas:
            demand_dict[a] = forecast_data['Load'][a]

    read_sefi_forecast()
    for a in data_areas:
        demand_dict[a] = demand_dict[a][:][:6541]
        factor = demand_dict[a]["ID error"].abs().sum() / demand_dict[a]["D-1 error"].abs().sum()


    other_areas = ('SE1', 'SE2', 'SE3', 'SE4', 'NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'DK2')
    def read_other_forecasts():
        for a in other_areas:
            demand_dict[a] = pd.DataFrame(columns=['Time', 'Actual', 'ID', 'D-1', 'ID error', 'D-1 error'])
            demand_dict[a]['Time'] = demand_dict['FI']['Time'].tolist()
            path = 'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\Demand\\'
            dem_in_1 = pd.read_csv(f'{path}{a}_2022.csv')
            dem_in_2 = pd.read_csv(f'{path}{a}_2023.csv')
            dem_in_1 = dem_in_1.drop(index=[7251])
            dem_in_1 = dem_in_1[3635:]
            dem_in_2 = dem_in_2[:1416]
            da_list = []
            actual_list = []
            da_list.extend(dem_in_1[f'Day-ahead Total Load Forecast [MW] - BZN|{a}'].astype(float))
            da_list.extend(dem_in_2[f'Day-ahead Total Load Forecast [MW] - BZN|{a}'].astype(float))
            actual_list.extend(dem_in_1[f'Actual Total Load [MW] - BZN|{a}'].astype(float))
            actual_list.extend(dem_in_2[f'Actual Total Load [MW] - BZN|{a}'].astype(float))
            demand_dict[a]['Actual'] = actual_list
            demand_dict[a]['D-1'] = da_list
            demand_dict[a]['D-1 error'] = demand_dict[a]['D-1'] - demand_dict[a]['Actual']
            demand_dict[a]['ID error'] = demand_dict[a]['D-1 error'] * factor
            demand_dict[a]['ID'] = demand_dict[a]['Actual'] + demand_dict[a]['ID error']
    read_other_forecasts()
    return demand_dict

demand = read_forecasts()
metrics = pd.DataFrame(columns=['Abs', 'Std', 'Var', 'Mean'], index=areas)
error_df = pd.DataFrame(columns=areas)
for a in areas:
     df = demand[a]
     metrics['Abs'][a] = float(df['ID error'].abs().sum() / df['ID'].sum())
     metrics['Std'][a] = float(df['ID error'].std() / df['ID'].mean())
     metrics['Var'][a] = float(np.sum(np.abs(np.subtract(np.array(df['ID error'][1:]),np.array(df['ID error'][:-1])))) /
                               df['ID'].sum())
     metrics['Mean'][a] = float(df['ID'].mean())
     error_df[a] = df['ID error'].tolist()

###SCATTER PLOT
plt.rcParams.update({'font.size': 12})
plt.scatter(metrics['Mean'].tolist(), metrics['Abs'].tolist(), label=r'$\phi^{1}$', color='blue', alpha=0.5, s=50)
plt.scatter(metrics['Mean'].tolist(), metrics['Std'].tolist(), label=r'$\phi^{2}$', color='green', alpha=0.5, s=50)
plt.scatter(metrics['Mean'].tolist(), metrics['Var'].tolist(), label=r'$\phi^{3}$', color='red', alpha=0.5, s=50)
plt.grid()
plt.legend()
plt.xlabel('Mean forecasted demand [MWh]')
plt.ylabel('Target value')
plt.xlim(0, 10000)
plt.tight_layout()
fig = plt.gcf()
save = True
if save:
    fig.savefig(
        f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\Figures\\DemandTargets.pdf',
        dpi=fig.dpi, pad_inches=0, bbox_inches='tight')
plt.show()

# for a in ['NO1', 'NO2', 'NO3', 'NO4', 'NO5']:
#     plt.plot(demand[a]['ID'], label='Forecast')
#     plt.plot(demand[a]['Actual'], label='Actual')
#     plt.title(a)
#     plt.grid()
#     plt.legend()
#     plt.show()
# with open(f'demand_forecasts.pickle', 'wb') as handle:
#     pkl.dump(demand, handle, protocol=pkl.HIGHEST_PROTOCOL)
corr = error_df.corr()
# print(corr)
# corr.to_csv('demandcorr.csv')
#corr.to_csv('C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\DemandCorr.csv')