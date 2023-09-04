"""This is a script where LMA simulation data can be analysed if needed"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta, datetime

areas = ['SE3'] #, 'SE2', 'SE3', 'SE4', 'NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'DK2', 'FI']
path = 'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Simuleringsdata\\LMA21\\EF45\\'

year = 2010
start_time = pd.Timestamp(f'{year}0101', tz='UTC')
end_time = pd.Timestamp(f'{year}1230', tz='UTC')
dates = pd.date_range(start=start_time, end=end_time, freq='D')
if dates.__len__() > 52 * 7:
    dates = dates.delete(31 + 29 - 1)
hours = []
for d in dates:
    for h in range(24):
        hours.append(d + timedelta(hours=h))

df_dict = {}
for a in areas:
    # Read the csv-file
    df_dict[a] = pd.DataFrame(columns=['Consumption', 'Flexibility', 'Demand response', 'Price'], index=hours)
    csv_in = pd.read_csv(f'{path}{a}.csv', skiprows=range(1, (year - 1982) * 24 * 7 * 52 + 1),
                         skipfooter=24 * 7 * 52 * (35 - (year + 1 - 1982)), engine='python')
    df_dict[a]['Consumption'] = csv_in['Tot_cons'].tolist()
    df_dict[a]['Flexibility'] = csv_in['Flex'].tolist()
    df_dict[a]['Demand response'] = csv_in['DemandResponse'].tolist()
    df_dict[a]['Price'] = csv_in['Price'].tolist()

### SCATTER PLOT
# plt.scatter(df_dict['SE1']['Price'].tolist(), df_dict['SE1']['Consumption'].tolist(), alpha=0.5)
# plt.xlabel('Price SEK/MWh')
# plt.ylabel('Consumption [MWh]')
# plt.show()

### CONSUMPTION AND PRICE
# ax1 = plt.subplot()
# l1 = ax1.plot(df_dict['SE3'].index.tolist(), df_dict['SE3']['Price'].tolist(), color='red', label='Price')
# ax2 = ax1.twinx()
# l2 = ax2.plot(df_dict['SE3'].index.tolist(), df_dict['SE3']['Consumption'].tolist(), color='blue', label='Consumption')
# ax1.tick_params(axis='x', rotation=30)
# ax1.set_xlabel('Time')
# ax1.set_ylabel('Price [SEK/MWh]')
# ax2.set_ylabel('Consumption [MWh]')
# ls = l1+l2
# labs = [l.get_label() for l in ls]
# ax1.legend(ls, labs, loc=0)
# plt.grid()
# plt.show()

### DR AND PRICE
# ax1 = plt.subplot()
# l1 = ax1.plot(df_dict[a].index.tolist(), df_dict[a]['Price'].tolist(), color='red', label='Price')
# ax2 = ax1.twinx()
# l2 = ax2.plot(df_dict[a].index.tolist(), df_dict[a]['Demand response'].tolist(), color='blue', label='Demand response')
# ax1.tick_params(axis='x', rotation=30)
# ax1.set_xlabel('Time')
# ax1.set_ylabel('Price [SEK/MWh]')
# ax2.set_ylabel('Demand response [MWh]')
# ls = l1+l2
# labs = [l.get_label() for l in ls]
# ax1.legend(ls, labs, loc=0)
# plt.grid()
# plt.show()

### FLEXIBILITY AND PRICE
# ax1 = plt.subplot()
# l1 = ax1.plot(df_dict[a].index.tolist(), df_dict[a]['Price'].tolist(), color='red', label='Price')
# ax2 = ax1.twinx()
# l2 = ax2.plot(df_dict[a].index.tolist(), df_dict[a]['Flexibility'].tolist(), color='blue', label='Flexibility')
# ax1.tick_params(axis='x', rotation=30)
# ax1.set_xlabel('Time')
# ax1.set_ylabel('Price [SEK/MWh]')
# ax2.set_ylabel('Flexibility [MWh]')
# ls = l1+l2
# labs = [l.get_label() for l in ls]
# ax1.legend(ls, labs, loc=0)
# plt.grid()
# plt.show()