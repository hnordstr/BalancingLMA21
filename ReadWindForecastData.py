"""
This file was used to read wind and demand forecast data from Sweden and Finland.
All data is now saved in "sefi_forecasts.pickle"
"""

import pandas as pd
import requests
import json
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
from datetime import datetime, timedelta

areas = ['SE1', 'SE2', 'SE3', 'SE4']

def wind():
    #Read swedish data
    data = pd.read_excel(
        'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Simuleringsdata\\Vind_intradag\\ForecastData.xlsx',
        sheet_name='Wind')
    wind_data = {}

    for a in areas:
        wind_data[a] = pd.DataFrame(columns=['Time', 'Capacity', 'Actual', 'ID', 'D-1'])
        area_data = data.loc[data['Source.Name'] == f'Vind_{a}.txt']
        area_data.reset_index(drop=True, inplace=True)
        wind_data[a]['Time'] = pd.to_datetime(area_data['Column1'], utc=True)
        wind_data[a]['Capacity'] = area_data['Installed-kap']
        wind_data[a]['Actual'] = area_data['History']
        wind_data[a]['ID'] = area_data['ID']
        wind_data[a]['D-1'] = area_data['DA2']
        wind_data[a] = wind_data[a].dropna(axis=0)
        wind_data[a].reset_index(drop=True, inplace=True)
        wind_data[a]['ID error'] = wind_data[a]['ID'] - wind_data[a]['Actual']
        wind_data[a]['D-1 error'] = wind_data[a]['D-1'] - wind_data[a]['Actual']

    #Read Finnish data
    wind_data['FI'] = pd.DataFrame(columns=['Time', 'Capacity', 'Actual', 'ID', 'D-1'])
    start = '2022-06-01T11:00:00Z'
    end = '2023-03-28T00:00:00Z'
    key = '17yYALnDed5KPth0qIaqg966KUvcuQBE7EPp7Kzr'
    headers = {'x-api-key': key}
    params = {'start_time': start,
              'end_time': end}

    #Actual value
    url = 'https://api.fingrid.fi/v1/variable/75/events/json'
    response = requests.get(url, headers=headers, params=params)
    data = json.loads(response.text)
    df = pd.DataFrame(data)
    df['Timestamp'] = pd.to_datetime(df['start_time'], utc=True, format='%Y-%m-%d')
    df = df.loc[:, ['Timestamp', 'value']]
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], )
    wind_data['FI']['Time'] = df['Timestamp'].tolist()
    wind_data['FI']['Actual'] = df['value']

    #Installed capacity
    url = 'https://api.fingrid.fi/v1/variable/268/events/json'
    response = requests.get(url, headers=headers, params=params)
    data = json.loads(response.text)
    df = pd.DataFrame(data)
    wind_data['FI']['Capacity'] = df['value'].tolist()

    #ID
    url = 'https://api.fingrid.fi/v1/variable/245/events/json'
    response = requests.get(url, headers=headers, params=params)
    data = json.loads(response.text)
    df = pd.DataFrame(data)
    wind_data['FI']['ID'] = df['value'].tolist()

    #DA
    url = 'https://api.fingrid.fi/v1/variable/246/events/json'
    response = requests.get(url, headers=headers, params=params)
    data = json.loads(response.text)
    df = pd.DataFrame(data)
    df['Timestamp'] = pd.to_datetime(df['start_time'], utc=True, format='%Y-%m-%d')
    df = df.set_index('Timestamp')
    df = df.reindex(pd.date_range(start=df.index[0], end=df.index[7165], freq='1h'))
    wind_data['FI']['D-1'] = df['value'].tolist()
    wind_data['FI'].reset_index(drop=True, inplace=True)
    wind_data['FI']['ID error'] = wind_data['FI']['ID'] - wind_data['FI']['Actual']
    wind_data['FI']['D-1 error'] = wind_data['FI']['D-1'] - wind_data['FI']['Actual']
    wind_data['FI']['D-1 error'].fillna(0, inplace=True)
    wind_data['FI']['D-1'].fillna(wind_data['FI']['D-1'].mean(), inplace=True)
    return wind_data

def load():
    # Read swedish data
    data = pd.read_excel(
        'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Simuleringsdata\\Vind_intradag\\ForecastData.xlsx',
        sheet_name='Load')

    load_data = {}

    for a in areas:
        load_data[a] = pd.DataFrame(columns=['Time', 'Actual', 'ID', 'D-1'])
        area_data = data.loc[data['Source.Name'] == f'Load_{a}.txt']
        area_data.reset_index(drop=True, inplace=True)
        load_data[a]['Time'] = pd.to_datetime(area_data['STARTTIME'])
        load_data[a]['Actual'] = area_data['History']
        load_data[a]['ID'] = area_data['ID']
        load_data[a]['D-1'] = area_data['DA']
        load_data[a] = load_data[a].dropna(axis=0)
        load_data[a].reset_index(drop=True, inplace=True)
        load_data[a]['ID error'] = load_data[a]['ID'] - load_data[a]['Actual']
        load_data[a]['D-1 error'] = load_data[a]['D-1'] - load_data[a]['Actual']
        # plt.plot(load_data[a]['Time'].tolist(), load_data[a]['ID error'].tolist(), label='ID')
        # plt.plot(load_data[a]['Time'].tolist(), load_data[a]['D-1 error'].tolist(), label='DA')
        # plt.legend()
        # plt.title(a)
        # plt.show()

    #Read Finnish data
    load_data['FI'] = pd.DataFrame(columns=['Time', 'Actual', 'ID', 'D-1'])
    start = '2022-06-01T11:00:00Z'
    end = '2023-03-28T00:00:00Z'
    key = '17yYALnDed5KPth0qIaqg966KUvcuQBE7EPp7Kzr'
    headers = {'x-api-key': key}
    params = {'start_time': start,
              'end_time': end}

    #Actual value
    url = 'https://api.fingrid.fi/v1/variable/124/events/json'
    response = requests.get(url, headers=headers, params=params)
    data = json.loads(response.text)
    df = pd.DataFrame(data)
    df['Timestamp'] = pd.to_datetime(df['start_time'], utc=True, format='%Y-%m-%d')
    df = df.loc[:, ['Timestamp', 'value']]
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], )
    load_data['FI']['Time'] = df['Timestamp'].tolist()
    load_data['FI']['Actual'] = df['value']

    #ID
    url = 'https://api.fingrid.fi/v1/variable/166/events/json'
    response = requests.get(url, headers=headers, params=params)
    data = json.loads(response.text)
    df = pd.DataFrame(data)
    df['Timestamp'] = pd.to_datetime(df['start_time'], utc=True, format='%Y-%m-%d')
    df = df.set_index('Timestamp')
    df = df.reindex(pd.date_range(start=df.index[0], end=df.index[86268], freq='1h'))
    load_data['FI']['ID'] = df['value'].tolist()

    #DA
    url = 'https://api.fingrid.fi/v1/variable/165/events/json'
    response = requests.get(url, headers=headers, params=params)
    data = json.loads(response.text)
    df = pd.DataFrame(data)
    df['Timestamp'] = pd.to_datetime(df['start_time'], utc=True, format='%Y-%m-%d')
    df = df.set_index('Timestamp')
    df = df.reindex(pd.date_range(start=df.index[0], end=df.index[85980], freq='1h'))
    load_data['FI']['D-1'] = df['value'].tolist()
    load_data['FI'].reset_index(drop=True, inplace=True)
    load_data['FI']['ID error'] = load_data['FI']['ID'] - load_data['FI']['Actual']
    load_data['FI']['D-1 error'] = load_data['FI']['D-1'] - load_data['FI']['Actual']
    load_data['FI']['D-1 error'].fillna(0, inplace=True)
    load_data['FI']['D-1'].fillna(load_data['FI']['D-1'].mean(), inplace=True)
    return load_data

data_dict = {}
data_dict['Wind'] = wind()
data_dict['Load'] = load()

with open(f'sefi_forecasts.pickle', 'wb') as handle:
    pkl.dump(data_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)