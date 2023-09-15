"""This is the main model from where all the generation of imbalance data should be run"""

import time
import pandas as pd
import sqlite3
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from Gurobi_Model import GurobiModel, GurobiModel_Alt
from datetime import datetime, timedelta
from RESshares import res_split
import calendar
from WindModel import Wind_Model
from SolarModel import Solar_Model
from DemandModel import Demand_Model
import sys
import pickle as pkl


class Model:
    def __init__(self, name='Model', scenario='EF45', start_date='2010-01-01',
                 simulated_days=52*7, save=False, quarters=False, fixed_ramp=False, trm=False):
        """"
        Here the setup of the model is done. One can choose between  having fast ramps and quarter-hourly markets.
        The start date should be set as yy-mm-dd. Data available for 1982-2016. Scenarios are SF45, EF45, EP45 and FM45.
        The number of simulated days must be set. Data is available 01/01 to 30/12 for all years, excluding leap days.
        IMPORTANT: Model cannot be run over consecutive years!!!
        Path should be set to folder were results should be saved.
        If results are to be saved, save parameter should be true.
        Potentially allow for running lower resolution than one minute?
        Netting as an option?
        """
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.year = self.start_date.year
        if self.year not in range(1982, 2017):
            print('NON-VALID YEAR')
            print('RUN TERMINATED')
            sys.exit()
        self.path = 'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\Results\\'
        self.save = save
        self.name = name
        self.scenario = scenario
        if self.scenario not in ['EF45', 'FM45', 'EP45', 'SF45']:
            print('NON-VALID SCENARIO')
            print('RUN TERMINATED')
            sys.exit()
        self.db_path = f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Simuleringsdata\\LMA21_{scenario}.db'
        self.fixed_ramp = fixed_ramp
        self.quarters = quarters
        self.trm = trm
        self.lowres = 60
        self.highres = 1
        self.days = simulated_days
        self.timestamps = [self.start_date + timedelta(hours=24*d + h) for d in range(self.days) for h in range(24)]
        if calendar.isleap(self.year) and self.days > 31 + 28:
            del self.timestamps[(31+28)*24:(31+29)*24]
            self.timestamps.extend([self.timestamps[-1] + timedelta(hours=h) for h in range(1,25)])
        if self.timestamps[0].year != self.timestamps[-1].year:
            print('INVALID TIME RANGE: MULTIPLE YEARS')
            print('RUN TERMINATED')
            sys.exit()
        elif datetime.strptime(f'{self.year}-12-31 00:00', '%Y-%m-%d %H:%M') in self.timestamps:
            print('INVALID TIME RANGE: DECEMBER 31ST INCLUDED')
            print('RUN TERMINATED')
            sys.exit()
        self.timestamps_high = [h + timedelta(minutes=m) for h in self.timestamps for m in range(60) ]
        self.timestamps_high_str = [datetime.strftime(d, '%Y-%m-%d %H:%M') for d in self.timestamps_high]
        if self.quarters:
            self.timestamps_q = [h + timedelta(minutes=q * 15) for h in self.timestamps for q in range(4)]
            self.timestamps_q_str = [datetime.strftime(d, '%Y-%m-%d %H:%M') for d in self.timestamps_q]
        self.timestamps_str = [datetime.strftime(d, '%Y-%m-%d %H:%M') for d in self.timestamps]
        self.areas = ('SE1', 'SE2', 'SE3', 'SE4', 'NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'DK2', 'FI')
        self.dc_lines = (
        ('DESE4', 'SE4', 'Ext'),
        ('DK1SE3','SE3', 'Ext'),
        ('LTSE4','SE4', 'Ext'),
        ('PLSE4', 'SE4', 'Ext'),
        ('FISE3', 'SE3', 'FI'),
        ('FISE2', 'SE2', 'FI'),
        ('RUFI', 'FI', 'Ext'),
        ('EEFI', 'FI', 'Ext'),
        ('DEDK2', 'DK2', 'Ext'),
        ('EODK2DK2', 'DK2', 'Ext'),
        ('DK1DK2', 'DK2', 'Ext'),
        ('GBNO2', 'NO2', 'Ext'),
        ('DK1NO2', 'NO2', 'Ext'),
        ('DENO2', 'NO2', 'Ext'),
        ('NLNO2', 'NO2', 'Ext'),
        ('GBNO5', 'NO5', 'Ext'),
        ('KFDK2', 'DK2', 'Ext')
        )
        self.dcindx = [x[0] for x in self.dc_lines]

        """Line name, receiving, sending"""
        self.ac_lines = (
        ('SE2SE1', 'SE1', 'SE2'),
        ('NO4SE1', 'SE1', 'NO4'),
        ('FISE1', 'SE1', 'FI'),
        ('NO4SE2', 'SE2', 'NO4'),
        ('NO3SE2', 'SE2', 'NO3'),
        ('SE3SE2', 'SE2', 'SE3'),
        ('NO1SE3', 'SE3', 'NO1'),
        ('SE4SE3', 'SE3', 'SE4'),
        ('DK2SE4', 'SE4', 'DK2'),
        ('NO2NO1', 'NO1', 'NO2'),
        ('NO3NO1', 'NO1', 'NO3'),
        ('NO5NO1', 'NO1', 'NO5'),
        ('NO5NO2', 'NO2', 'NO5'),
        ('NO5NO3', 'NO3', 'NO5'),
        ('NO4NO3', 'NO3', 'NO4')
        )

        self.ac_unidir = (
        ('SE2SE1', 'SE1', 'SE2'),
        ('SE1SE2', 'SE2', 'SE1'),
        ('NO4SE1', 'SE1', 'NO4'),
        ('SE1NO4', 'NO4', 'SE1'),
        ('FISE1', 'SE1', 'FI'),
        ('SE1FI', 'FI', 'SE1'),
        ('NO4SE2', 'SE2', 'NO4'),
        ('SE2NO4', 'NO4', 'SE2'),
        ('NO3SE2', 'SE2', 'NO3'),
        ('SE2NO3', 'NO3', 'SE2'),
        ('SE3SE2', 'SE2', 'SE3'),
        ('SE2SE3', 'SE3', 'SE2'),
        ('NO1SE3', 'SE3', 'NO1'),
        ('SE3NO1', 'NO1', 'SE3'),
        ('SE4SE3', 'SE3', 'SE4'),
        ('SE3SE4', 'SE4', 'SE3'),
        ('DK2SE4', 'SE4', 'DK2'),
        ('SE4DK2', 'DK2', 'SE4'),
        ('NO2NO1', 'NO1', 'NO2'),
        ('NO1NO2', 'NO2', 'NO1'),
        ('NO3NO1', 'NO1', 'NO3'),
        ('NO1NO3', 'NO3', 'NO1'),
        ('NO5NO1', 'NO1', 'NO5'),
        ('NO1NO5', 'NO5', 'NO1'),
        ('NO5NO2', 'NO2', 'NO5'),
        ('NO2NO5', 'NO5', 'NO2'),
        ('NO5NO3', 'NO3', 'NO5'),
        ('NO3NO5', 'NO5', 'NO3'),
        ('NO4NO3', 'NO3', 'NO4'),
        ('NO3NO4', 'NO4', 'NO3')
        )
        self.acindx = [x[0] for x in self.ac_unidir]

        self.TRM = {
            'SE2SE1': 100,
            'SE1SE2': 100,
            'NO4SE1': 25,
            'SE1NO4': 25,
            'FISE1': 100,
            'SE1FI': 100,
            'NO3SE2': 25,
            'SE2NO3': 25,
            'NO4SE2': 25,
            'SE2NO4': 25,
            'SE3SE2': 300,
            'SE2SE3': 300,
            'SE4SE3': 150,
            'SE3SE4': 150,
            'NO1SE3': 150,
            'SE3NO1': 150,
            'DK2SE4': 50,
            'SE4DK2': 50,
            'NO2NO1': 75,
            'NO1NO2': 75,
            'NO3NO1': 0,
            'NO1NO3': 0,
            'NO5NO1': 75,
            'NO1NO5': 75,
            'NO5NO2': 50,
            'NO2NO5': 50,
            'NO4NO3': 25,
            'NO3NO4': 25,
            'NO5NO3': 50,
            'NO3NO5': 50
        }

        self.ac_pair = {}
        for i in range(self.ac_lines.__len__()):
            self.ac_pair[self.ac_unidir[2 * i][0]] = self.ac_unidir[2 * i + 1][0]
            self.ac_pair[self.ac_unidir[2 * i + 1][0]] = self.ac_unidir[2 * i][0]

    def setup_data(self):
        print('READING INPUT DATA')
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        area_dict = {}

        ###READ FROM DB-files
        for area in self.areas:
            ###Generation and demand
            table = f'Generation_{area}_{self.year}'
            c.execute("SELECT * FROM {0}".format(table))
            area_dict[f'{area}_Gen'] = c.fetchall()
            area_dict[f'{area}_GenCols'] = list(map(lambda x: x[0], c.description))

            ###HVDC
            table = f'HVDC_{area}_{self.year}'
            # if c.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name={0}".format(table)).fetchone()[0] != 1:
            #     pass
            # else:
            try:
                c.execute("SELECT * FROM {0}".format(table))
                area_dict[f'{area}_HVDC'] = c.fetchall()
                area_dict[f'{area}_HVDCCols'] = list(map(lambda x: x[0], c.description))
            except sqlite3.OperationalError:
                pass

            ###AC
            table = f'AC_{area}_{self.year}'
            try:
                c.execute("SELECT * FROM {0}".format(table))
                area_dict[f'{area}_AC'] = c.fetchall()
                area_dict[f'{area}_ACCols'] = list(map(lambda x: x[0], c.description))
            except sqlite3.OperationalError:
                pass

        c.close()
        ### READ INTO SEPARATE DATAFRAMES

        #Hydro
        self.hydro_low = pd.DataFrame(columns=self.areas, index=self.timestamps_str)
        for area in self.areas:
            idx = area_dict[f'{area}_GenCols'].index('Hydro')
            self.hydro_low[area] = [area_dict[f'{area}_Gen'][i][idx] for i in range(self.timestamps_str.__len__())]

        #Available RES
        self.resav_low = pd.DataFrame(columns=self.areas, index=self.timestamps_str)
        for area in self.areas:
            idx = area_dict[f'{area}_GenCols'].index('RES')
            self.resav_low[area] = [area_dict[f'{area}_Gen'][i][idx] for i in range(self.timestamps_str.__len__())]

        #Curtailed RES
        self.curtail_low = pd.DataFrame(columns=self.areas, index=self.timestamps_str)
        for area in self.areas:
            idx = area_dict[f'{area}_GenCols'].index('Curtailments')
            self.curtail_low[area] = [area_dict[f'{area}_Gen'][i][idx] for i in range(self.timestamps_str.__len__())]

        #RES generation
        self.res_low = pd.DataFrame(columns=self.areas, index=self.timestamps_str)
        self.hydro_curtail_low = pd.DataFrame(columns=self.areas, index=self.timestamps_str)
        for area in self.areas:
            self.res_low[area] = self.resav_low[area] + self.curtail_low[area]
            self.hydro_curtail_low[area] = self.resav_low[area] + self.curtail_low[area]
        self.res_low[self.res_low < 0] = 0
        self.hydro_curtail_low[self.hydro_curtail_low > 0] = 0
        for area in self.areas:
            self.hydro_low[area] = self.hydro_low[area] + self.hydro_curtail_low[area]


        #Split RES
        res_dict = res_split(data=self.res_low, scenario=self.scenario, days=self.days, year=self.year)
        self.pv_low = pd.DataFrame(columns=self.areas, index=self.timestamps_str)
        self.offsh_low = pd.DataFrame(columns=self.areas, index=self.timestamps_str)
        self.onsh_low = pd.DataFrame(columns=self.areas, index=self.timestamps_str)
        self.wind_low = pd.DataFrame(columns=self.areas, index=self.timestamps_str)
        for a in self.areas:
            self.pv_low[a] = res_dict[a]['SolarPV'].tolist()
            self.offsh_low[a] = res_dict[a]['WindOffshore'].tolist()
            self.onsh_low[a] = res_dict[a]['WindOnshore'].tolist()
            self.wind_low[a] = self.offsh_low[a] + self.onsh_low[a]

        #Nuclear
        self.nuclear_low = pd.DataFrame(columns=self.areas, index=self.timestamps_str)
        for area in self.areas:
            idx = area_dict[f'{area}_GenCols'].index('Nuclear')
            self.nuclear_low[area] = [area_dict[f'{area}_Gen'][i][idx] for i in range(self.timestamps_str.__len__())]

        #Thermal
        self.thermal_low = pd.DataFrame(columns=self.areas, index=self.timestamps_str)
        for area in self.areas:
            idx = area_dict[f'{area}_GenCols'].index('Thermal')
            self.thermal_low[area] = [area_dict[f'{area}_Gen'][i][idx] for i in range(self.timestamps_str.__len__())]

        #Flex
        self.flex_low = pd.DataFrame(columns=self.areas, index=self.timestamps_str)
        for area in self.areas:
            idx = area_dict[f'{area}_GenCols'].index('Flexibility')
            self.flex_low[area] = [area_dict[f'{area}_Gen'][i][idx] for i in range(self.timestamps_str.__len__())]

        #Total Demand
        self.demand_low = pd.DataFrame(columns=self.areas, index=self.timestamps_str)
        for area in self.areas:
            idx = area_dict[f'{area}_GenCols'].index('Demand')
            self.demand_low[area] = [area_dict[f'{area}_Gen'][i][idx] for i in range(self.timestamps_str.__len__())]

        #Demand response
        self.demandresp_low = pd.DataFrame(columns=self.areas, index=self.timestamps_str)
        for area in self.areas:
            idx = area_dict[f'{area}_GenCols'].index('DemandResponse')
            self.demandresp_low[area] = [area_dict[f'{area}_Gen'][i][idx] for i in range(self.timestamps_str.__len__())]

        #Actual consumption
        self.consumption_low = pd.DataFrame(columns=self.areas, index=self.timestamps_str)
        for area in self.areas:
            self.consumption_low[area] = self.demand_low[area] - self.demandresp_low[area]

        #AC data
        self.ac_low = pd.DataFrame(columns=self.acindx, index=self.timestamps_str)
        for l in self.ac_lines:
            if l[0] == 'SE4SE3':
                idx1 = area_dict[f'{l[1]}_ACCols'].index('SE4SE3_SV')
                idx2 = area_dict[f'{l[1]}_ACCols'].index('SE4SE3_AC')
                self.ac_low[l[0]] = [area_dict[f'{l[1]}_AC'][i][idx1] + area_dict[f'{l[1]}_AC'][i][idx2] \
                                     for i in range(self.timestamps_str.__len__())]
                self.ac_low[self.ac_pair[l[0]]] = [-area_dict[f'{l[1]}_AC'][i][idx1] - area_dict[f'{l[1]}_AC'][i][idx2] \
                                     for i in range(self.timestamps_str.__len__())]
            else:
                idx = area_dict[f'{l[1]}_ACCols'].index(f'{l[0]}')
                self.ac_low[l[0]] = [area_dict[f'{l[1]}_AC'][i][idx] for i in range(self.timestamps_str.__len__())]
                self.ac_low[self.ac_pair[l[0]]] = [- area_dict[f'{l[1]}_AC'][i][idx] for i in range(self.timestamps_str.__len__())]
        self.ac_low[self.ac_low < 0] = 0

        #NTC
        self.ntc = {}
        for l in self.acindx:
            self.ntc[l] = self.ac_low[l].max()

        #HVDC data
        self.hvdc_low = pd.DataFrame(columns=self.dcindx, index=self.timestamps_str)
        for l in self.dc_lines:
            idx = area_dict[f'{l[1]}_HVDCCols'].index(f'{l[0]}')
            self.hvdc_low[l[0]] = [area_dict[f'{l[1]}_HVDC'][i][idx] for i in range(self.timestamps_str.__len__())]

    def fix_error(self):
        balance = pd.DataFrame(columns=self.areas)
        for a in self.areas:
            balance[a] = self.hydro_low[a] + self.wind_low[a] + self.nuclear_low[a] + self.thermal_low[a] + self.pv_low[a] \
                         + self.flex_low[a] - self.consumption_low[a]
            for j in self.dc_lines:
                if j[1] == a:
                    balance[a] = balance[a] + self.hvdc_low[j[0]]
                elif j[2] == a:
                    balance[a] = balance[a] - self.hvdc_low[j[0]]
            for k in self.ac_unidir:
                if k[1] == a:
                    balance[a] = balance[a] + self.ac_low[k[0]]
                elif k[2] == a:
                    balance[a] = balance[a] - self.ac_low[k[0]]
            if a in ['NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'SE1', 'SE2']:
                self.hydro_low[a] = self.hydro_low[a] - balance[a]
            else:
                self.consumption_low[a] = self.consumption_low[a] + balance[a]

    def check_balance(self, plot=False):
        self.setup_data()
        self.fix_error()
        balance = pd.DataFrame(columns=self.areas, index=self.timestamps_str)
        for a in self.areas:
            balance[a] = self.hydro_low[a] + self.wind_low[a] + self.nuclear_low[a] + self.thermal_low[a] \
                         + self.flex_low[a] + self.pv_low[a] - self.consumption_low[a]
            for j in self.dc_lines:
                if j[1] == a:
                    balance[a] = balance[a] + self.hvdc_low[j[0]]
                elif j[2] == a:
                    balance[a] = balance[a] - self.hvdc_low[j[0]]
            for k in self.ac_unidir:
                if k[1] == a:
                    balance[a] = balance[a] + self.ac_low[k[0]]
                elif k[2] == a:
                    balance[a] = balance[a] - self.ac_low[k[0]]
            if plot:
                balance[a].plot()
                plt.title(f'Balance {a}')
                plt.grid()
                plt.show()

    def quarter_interpolation(self, data, type=''):
        x_pts = []
        x_pts.append(0)
        x_pts.extend(np.linspace(60 / 2, 60 / 2 + (24 * self.days  - 1) * 60, 24 * self.days))
        x_pts.append(60 * 24 * self.days)
        x_vals = np.linspace(0, 24 * self.days * 60, int(24 * 60 * self.days / 15))
        y_pts = []
        y_pts.append(data[0] - (data[1] - data[0]) / 2)
        y_pts.extend(data)
        y_pts.append(data[24 * self.days - 1] + (data[int(24 * self.days - 1)] - data[24 * self.days - 2]) / 2)
        f = interpolate.PchipInterpolator(x_pts, y_pts)
        new_list = f(x_vals)
        if type == 'Generation':
            new_list = [n - min(0, n) for n in new_list]
        return new_list

    def quarter_hour_energy_error(self, data, type=''):
        x_ny = []
        LD_ref = []
        h = []
        error = []
        x_ny.extend(data)
        LD_ref.extend(data)
        HD = self.quarter_interpolation(x_ny, type=type)
        for lowres_step in range(int(24 * self.days)):
            h.append(LD_ref[lowres_step] - sum(HD[int(lowres_step * 4):int((lowres_step + 1) * 4)]) / 4)
            error.append(0.5 * h[lowres_step] ** 2)
        error_new = sum(error)
        error_old = np.infty
        while error_old > error_new and error_new > 0.001 * self.days:
            error_old = sum(error)
            for i in range(int(24 * self.days)):
                x_ny[i] = x_ny[i] + h[i]
            HD = self.quarter_interpolation(x_ny, type=type)
            for i in range(int(24 * self.days)):
                h[i] = LD_ref[i] - sum(HD[int(i * 4): int((i + 1) * 4)]) / 4
                error[i] = 0.5 * h[i] ** 2
            error_new = sum(error)
        if error_new > 0.1 * self.days:
            print('!!!TP ENERGY ERROR OCCURRED!!!')
            print(error_new)
        return self.quarter_interpolation(x_ny, type=type)

    def make_quarters(self):
        print('GENERATING QUARTER-HOURLY DATA')
        q = pd.DataFrame(columns=self.areas, index=self.timestamps_q_str)
        for area in self.areas:
            q[area] = self.quarter_hour_energy_error(data=self.hydro_low[area].tolist(), type='Generation')
        self.hydro_old = self.hydro_low
        self.hydro_low = q

        q = pd.DataFrame(columns=self.areas, index=self.timestamps_q_str)
        for area in self.areas:
            q[area] = self.quarter_hour_energy_error(data=self.nuclear_low[area].tolist(), type='Generation')
        self.nuclear_old = self.nuclear_low
        self.nuclear_low = q

        q = pd.DataFrame(columns=self.areas, index=self.timestamps_q_str)
        for area in self.areas:
            q[area] = self.quarter_hour_energy_error(data=self.flex_low[area].tolist(), type='Flex')
        self.flex_old = self.flex_low
        self.flex_low = q

        q = pd.DataFrame(columns=self.areas, index=self.timestamps_q_str)
        for area in self.areas:
            q[area] = self.quarter_hour_energy_error(data=self.thermal_low[area].tolist(), type='Generation')
        self.thermal_old = self.thermal_low
        self.thermal_low = q

        q = pd.DataFrame(columns=self.areas, index=self.timestamps_q_str)
        for area in self.areas:
            q[area] = self.quarter_hour_energy_error(data=self.consumption_low[area].tolist(), type='Generation')
        self.consumption_old = self.consumption_low
        self.consumption_low = q

        q = pd.DataFrame(columns=self.areas, index=self.timestamps_q_str)
        for area in self.areas:
            q[area] = self.quarter_hour_energy_error(data=self.wind_low[area].tolist(), type='Generation')
        self.wind_old = self.wind_low
        self.wind_low = q

        q = pd.DataFrame(columns=self.areas, index=self.timestamps_q_str)
        for area in self.areas:
            q[area] = self.quarter_hour_energy_error(data=self.pv_low[area].tolist(), type='Generation')
        self.pv_old = self.pv_low
        self.pv_low = q

        q = pd.DataFrame(columns=self.dcindx, index=self.timestamps_q_str)
        for line in self.dcindx:
            q[line] = self.quarter_hour_energy_error(data=self.hvdc_low[line].tolist(), type='HVDC')
        self.hvdc_old = self.hvdc_low
        self.hvdc_low = q

    def varying_simulation(self):
        wind_model = Wind_Model(wind_in=self.wind_low, scenario=self.scenario, year_start=self.year,
                                sim_days=self.days, improvement_percentage=20)
        print('RUNNING WIND POWER SIMULATION')
        wind_model.make_multi_area_highres_scenario()

        self.wind_actual = wind_model.dict
        self.wind_high = pd.DataFrame(columns=self.areas, index=self.timestamps_high_str)
        self.wind_actual_low = pd.DataFrame(columns=self.areas, index=self.timestamps_str)
        for area in self.areas:
            self.wind_high[area] = self.wind_actual[area]['Minute']['Generation'].tolist()
            self.wind_actual_low[area] = self.wind_actual[area]['Hourly']['Actual'].tolist()

        pv_model = Solar_Model(pv_in=self.pv_low, scenario=self.scenario, year_start=self.year,
                                sim_days=self.days, improvement_percentage=20)
        print('RUNNING SOLAR POWER SIMULATION')
        pv_model.make_multi_area_highres_scenario()
        self.pv_actual = pv_model.dict
        self.pv_high = pd.DataFrame(columns=self.areas, index=self.timestamps_high_str)
        self.pv_actual_low = pd.DataFrame(columns=self.areas, index=self.timestamps_str)
        for area in self.areas:
            if area == 'NO4':
                self.pv_high[area] = [0 for i in range(self.timestamps_high_str.__len__())]
                self.pv_actual_low[area] = [0 for i in range(self.timestamps_str.__len__())]
            else:
                self.pv_high[area] = self.pv_actual[area]['Minute']['Generation'].tolist()
                self.pv_actual_low[area] = self.pv_actual[area]['Hourly']['Actual'].tolist()

        demand_model = Demand_Model(demand_in=self.consumption_low, scenario=self.scenario, year_start=self.year,
                                sim_days=self.days, improvement_percentage=20)
        print('RUNNING DEMAND SIMULATION')
        demand_model.make_multi_area_highres_scenario()
        self.consumption_actual = demand_model.dict
        self.consumption_high = pd.DataFrame(columns=self.areas, index=self.timestamps_high_str)
        self.consumption_actual_low = pd.DataFrame(columns=self.areas, index=self.timestamps_str)
        for area in self.areas:
            self.consumption_high[area] = self.consumption_actual[area]['Minute']['Generation'].tolist()
            self.consumption_actual_low[area] = self.consumption_actual[area]['Hourly']['Actual'].tolist()

    def ramping_generation(self, data, type):
        highres = 1
        if self.quarters:
            lowres = 15
            max_period = 5
        else:
            lowres = 60
            max_period = 20
        if type == 'Hydro' or type == 'Flex':
            RR = 0.05
        elif type == 'Thermal':
            RR = 0.03
        elif type == 'Nuclear':
            RR = 0.015
        else:
            print('Wrong "type" input')
            return
        ramp_speed = max(max(data), 1) * RR #Max generation time %of max/min
        x_pts = []
        lowres_step = 0
        while lowres_step < self.days * 24 * 60 / lowres:
            if lowres_step == 0:
                x_pts.append(0)
            elif lowres_step == 24 * self.days * 60 / lowres - 1:
                if self.fixed_ramp:
                    ramping_period = max_period
                else:
                    ramping_period = abs(data[lowres_step] - data[lowres_step - 1]) / ramp_speed
                x_pts.append(lowres * lowres_step - min(ramping_period / 2, max_period / 2))
                x_pts.append(lowres * lowres_step + min(ramping_period / 2, max_period / 2))
                x_pts.append(lowres * (lowres_step + 1))
            else:
                if self.fixed_ramp:
                    ramping_period = max_period
                else:
                    ramping_period = abs(data[lowres_step] - data[lowres_step - 1]) / ramp_speed
                x_pts.append(lowres * lowres_step - min(ramping_period / 2, max_period / 2))
                x_pts.append(lowres * lowres_step + min(ramping_period / 2, max_period / 2))
            lowres_step = lowres_step + 1
        quota = int(60 / highres)
        x_vals = np.linspace(0, 24 * self.days * 60, 24 * self.days * quota)
        y_pts = []
        lowres_step = 0
        while lowres_step < 24 * self.days * 60 / lowres:
            y_pts.append(data[lowres_step])
            y_pts.append(data[lowres_step])
            lowres_step = lowres_step + 1
        f = interpolate.interp1d(x_pts, y_pts)
        return f(x_vals)

    def ramping_hvdc(self, data):
        highres = 1
        if self.quarters:
            lowres = 15
            max_period = 5
        else:
            lowres = 60
            max_period = 20
        ramp_speed = 30 #Currently 30 MW/min
        x_pts = []
        lowres_step = 0
        while lowres_step < 24 * self.days * 60 / lowres:
            if lowres_step == 0:
                x_pts.append(0)
            elif lowres_step == 24 * self.days * 60 / lowres - 1:
                if self.fixed_ramp:
                    ramping_period = max_period
                else:
                    ramping_period = abs(data[lowres_step] - data[lowres_step - 1]) / ramp_speed
                x_pts.append(
                    lowres * lowres_step - min(ramping_period / 2, max_period / 2))
                x_pts.append(
                    lowres * lowres_step + min(ramping_period / 2, max_period / 2))
                x_pts.append(lowres * (lowres_step + 1))
            else:
                if self.fixed_ramp:
                    ramping_period = max_period
                else:
                    ramping_period = abs(data[lowres_step] - data[lowres_step - 1]) / ramp_speed
                x_pts.append(
                    lowres * lowres_step - min(ramping_period / 2, max_period / 2))
                x_pts.append(
                    lowres * lowres_step + min(ramping_period / 2, max_period / 2))
            lowres_step = lowres_step + 1
        x_vals = np.linspace(0, 24 * self.days * 60, int(24 * self.days * 60 / highres))
        y_pts = []
        lowres_step = 0
        while lowres_step < 24 * self.days * 60 / lowres:
            y_pts.append(data[lowres_step])
            y_pts.append(data[lowres_step])
            lowres_step = lowres_step + 1
        f = interpolate.interp1d(x_pts, y_pts)
        return f(x_vals)

    def create_high_resolution(self):
        self.setup_data()
        self.fix_error()
        self.varying_simulation()
        if self.quarters:
            self.make_quarters()
            #Call function to rewrite low parameters to quarter-hourly values
        self.hydro_high = pd.DataFrame(columns=self.areas, index=self.timestamps_high_str)
        self.thermal_high = pd.DataFrame(columns=self.areas, index=self.timestamps_high_str)
        self.nuclear_high = pd.DataFrame(columns=self.areas, index=self.timestamps_high_str)
        self.flex_high = pd.DataFrame(columns=self.areas, index=self.timestamps_high_str)
        print("RUNNING RAMPING SIMULATION")
        for area in self.areas:
            self.hydro_high[area] = self.ramping_generation(data=self.hydro_low[area].tolist(), type='Hydro')
            self.thermal_high[area] = self.ramping_generation(data=self.thermal_low[area].tolist(), type='Thermal')
            self.nuclear_high[area] = self.ramping_generation(data=self.nuclear_low[area].tolist(), type='Nuclear')
            self.flex_high[area] = self.ramping_generation(data=self.flex_low[area].tolist(), type='Flex')
        self.hvdc_high = pd.DataFrame(columns=self.dcindx, index=self.timestamps_high_str)
        for line in self.dcindx:
            self.hvdc_high[line] = self.ramping_hvdc(self.hvdc_low[line].tolist())
        print('CALCULATING ATCS')
        self.ac_pre_high = pd.DataFrame(columns=self.acindx, index=self.timestamps_high_str)
        self.atc_high = pd.DataFrame(columns=self.acindx, index=self.timestamps_high_str)
        self.atc_low = pd.DataFrame(columns=self.acindx, index=self.timestamps_str)
        for line in self.acindx:
            ac_list = self.ac_low[line].tolist()
            self.ac_pre_high[line] = [ac_list[t] for t in range(self.timestamps_str.__len__()) for i in range(60)]
        for line in self.acindx:
            self.atc_low[line] = self.ntc[line] - self.ac_low[line] + self.ac_pre_high[self.ac_pair[line]]
            if self.trm:
                self.atc_high[line] = self.ntc[line] + self.TRM[line] - self.ac_pre_high[line] + self.ac_pre_high[self.ac_pair[line]]
            else:
                self.atc_high[line] = self.ntc[line] - self.ac_pre_high[line] + self.ac_pre_high[self.ac_pair[line]]

    def compute_area_imbalance(self):
        if self.quarters:
            period = 15
        else:
            period = 60
        self.stochastic_imbalances = pd.DataFrame(columns=self.areas)
        self.deterministic_imbalances = pd.DataFrame(columns=self.areas)
        self.imbalances = pd.DataFrame(columns=self.areas)
        #positive imbalance = surplus
        #negative imbalance = deficit
        #Add the indexes in the end
        #Add all the other technologies

        print(f'COMPUTING PRE-NETTED IMBALANCES')

        for a in self.areas:
            self.stochastic_imbalances[a] = np.zeros(self.timestamps_high_str.__len__())
            self.deterministic_imbalances[a] = np.zeros(self.timestamps_high_str.__len__())

            ###THERMAL###
            low_list = self.thermal_low[a].tolist()
            high_list = self.thermal_high[a].tolist()
            imb = [high_list[t * period + m] - low_list[t] for t in range(low_list.__len__()) for m in range(period)]
            # Positive thermal imbalance => Positive area imbalance
            self.deterministic_imbalances[a] = self.deterministic_imbalances[a] + pd.Series(imb)

            ###HYDRO###
            low_list = self.hydro_low[a].tolist()
            high_list = self.hydro_high[a].tolist()
            imb = [high_list[t * period + m] - low_list[t] for t in range(low_list.__len__()) for m in range(period)]
            # Positive hydro imbalance => Positive area imbalance
            self.deterministic_imbalances[a] = self.deterministic_imbalances[a] + pd.Series(imb)

            ###NUCLEAR###
            low_list = self.nuclear_low[a].tolist()
            high_list = self.nuclear_high[a].tolist()
            imb = [high_list[t * period + m] - low_list[t] for t in range(low_list.__len__()) for m in range(period)]
            # Positive nuclear imbalance => Positive area imbalance
            self.deterministic_imbalances[a] = self.deterministic_imbalances[a] + pd.Series(imb)

            ###FLEX###
            low_list = self.flex_low[a].tolist()
            high_list = self.flex_high[a].tolist()
            imb = [high_list[t * period + m] - low_list[t] for t in range(low_list.__len__()) for m in range(period)]
            # Positive flex imbalance => Positive area imbalance
            self.deterministic_imbalances[a] = self.deterministic_imbalances[a] + pd.Series(imb)

            ###WIND###
            low_list = self.wind_low[a].tolist()
            high_list = self.wind_high[a].tolist()
            imb = [high_list[t * period + m] - low_list[t] for t in range(low_list.__len__()) for m in range(period)]
            # Positive wind imbalance => Positive area imbalance
            self.stochastic_imbalances[a] = self.stochastic_imbalances[a] + pd.Series(imb)

            ###PV###
            low_list = self.pv_low[a].tolist()
            high_list = self.pv_high[a].tolist()
            imb = [high_list[t * period + m] - low_list[t] for t in range(low_list.__len__()) for m in range(period)]
            # Positive pv imbalance => Positive area imbalance
            self.stochastic_imbalances[a] = self.stochastic_imbalances[a] + pd.Series(imb)

            ###DEMAND###
            low_list = self.consumption_low[a].tolist()
            high_list = self.consumption_high[a].tolist()
            imb = [high_list[t * period + m] - low_list[t] for t in range(low_list.__len__()) for m in range(period)]
            # Positive demand imbalance => Negative area imbalance
            self.stochastic_imbalances[a] = self.stochastic_imbalances[a] - pd.Series(imb)

        ###HVDC###
        for l in self.dc_lines:
            low_list = self.hvdc_low[l[0]].tolist()
            high_list = self.hvdc_high[l[0]].tolist()
            imb = [high_list[t * period + m] - low_list[t] for t in range(low_list.__len__()) for m in range(period)]
            # 0 = line index, #1 = receiving, #2 = sending
            # Positive sending imbalance => Export > Planned => Negative area imbalance
            # Positive receiving imbalance => Import > Planned => Positive area imbalance
            if l[1] in self.areas:
                self.deterministic_imbalances[l[1]] = self.deterministic_imbalances[l[1]] + pd.Series(imb)
            if l[2] in self.areas:
                self.deterministic_imbalances[l[2]] = self.deterministic_imbalances[l[2]] - pd.Series(imb)

        self.deterministic_imbalances = self.deterministic_imbalances.set_index(pd.Series(self.timestamps_high_str))
        self.stochastic_imbalances = self.stochastic_imbalances.set_index(pd.Series(self.timestamps_high_str))
        for a in self.areas:
            self.imbalances[a] = self.deterministic_imbalances[a] + self.stochastic_imbalances[a]
        #     plt.plot(self.imbalances[a].tolist())
        #     plt.title(a)
        #     plt.grid()
        #     plt.show()

    def imbalance_netting(self):
        start = time.time()
        self.cm = GurobiModel(self.name)
        self.cm.setup_problem(self)
        print('SOLVING OPTIMIZATION PROBLEM')
        self.cm.gm.optimize()
        end = time.time()
        print(f'IMBALANCE NETTING COMPLETED IN {round(end - start, 3)} SECONDS')
        self.netted_imbalance = pd.DataFrame(columns=self.areas, index=self.timestamps_high_str)
        self.ac_netting = pd.DataFrame(columns=self.areas, index=self.timestamps_high_str)
        self.ac_post_high = pd.DataFrame(columns=self.areas, index=self.timestamps_high_str)
        for a in self.areas:
            self.netted_imbalance[a] = [- self.cm.var_BALANCINGUP[a, t].X + self.cm.var_BALANCINGDOWN[a, t].X
                                       for t in self.timestamps_high_str]
        for l in self.acindx:
            self.ac_netting[l] = [self.cm.var_AC[l, t].X for t in self.timestamps_high_str]
            self.ac_post_high[l] = self.ac_pre_high[l] + self.ac_netting[l]

    def imbalance_netting_alt(self):
        period_length = 30
        nperiods = np.ceil(self.days / period_length)
        periods = list(range(int(nperiods)))
        imb_list = {}
        ac_list = {}
        start = time.time()
        for a in self.areas:
            imb_list[a] = []
        for l in self.acindx:
            ac_list[l] = []
        for p in periods:
            if p == nperiods - 1:
                idx_range = list(range(p * period_length * 24 * 60, self.timestamps_high_str.__len__()))
            else:
                idx_range = list(range(p * period_length * 24 * 60, (p + 1) * period_length * 24 * 60))
            self.cm = GurobiModel_Alt(self.name, idx_range)
            print(f'SETTING UP IMBALANCE NETTING OPTIMIZATION PERIOD {p + 1}')
            self.cm.setup_problem(self)
            print(f'SOLVING OPTIMIZATION PROBLEM PERIOD {p + 1}')
            self.cm.gm.optimize()
            for a in self.areas:
                imb_list[a].extend([- self.cm.var_BALANCINGUP[a, t].X + self.cm.var_BALANCINGDOWN[a, t].X
                                    for t in self.cm.set_TIME])
            for l in self.acindx:
                ac_list[l].extend([self.cm.var_AC[l, t].X for t in self.cm.set_TIME])
        end = time.time()
        print(f'IMBALANCE NETTING COMPLETED IN {round(end - start, 3)} SECONDS')
        self.netted_imbalance = pd.DataFrame(columns=self.areas, index=self.timestamps_high_str)
        self.ac_netting = pd.DataFrame(columns=self.acindx, index=self.timestamps_high_str)
        self.ac_post_high = pd.DataFrame(columns=self.acindx, index=self.timestamps_high_str)
        for a in self.areas:
            self.netted_imbalance[a] = imb_list[a]
        for l in self.acindx:
            self.ac_netting[l] = ac_list[l]
            self.ac_post_high[l] = self.ac_pre_high[l] + self.ac_netting[l] - self.ac_netting[self.ac_pair[l]]

    def imbalance_filtering(self):
        self.slow_imbalance = pd.DataFrame(columns=self.areas, index=self.timestamps_high_str)
        self.fast_imbalance = pd.DataFrame(columns=self.areas, index=self.timestamps_high_str)
        self.fcr_imbalance = pd.DataFrame(columns=self.areas, index=self.timestamps_high_str)
        five_min = pd.DataFrame(columns=self.areas, index=self.timestamps_high_str)
        for a in self.areas:
            fifteen_min_rolling = self.netted_imbalance[a].rolling(window=15).mean()
            for i in range(15):
                fifteen_min_rolling[i] = fifteen_min_rolling[15]
            five_min_rolling = self.netted_imbalance[a].rolling(window=5).mean()
            for i in range(5):
                five_min_rolling[i] = five_min_rolling[5]
            five_min[a] = five_min_rolling
            self.slow_imbalance[a] = fifteen_min_rolling
            self.fast_imbalance[a] = five_min[a] - self.slow_imbalance[a]
            self.fcr_imbalance[a] = self.netted_imbalance[a] - self.slow_imbalance[a] - self.fast_imbalance[a]
            # fig,ax1 = plt.subplots()
            # ax1.plot(self.netted_imbalance[a].tolist(), label='Imbalance', color='blue')
            # ax1.plot(self.slow_imbalance[a].tolist(), label='Slow imbalance', color='green')
            # ax1.set_ylabel('Total and slow imbalance [MW]')
            # ax2 = ax1.twinx()
            # ax2.plot(self.fast_imbalance[a].tolist(), label='Fast imbalance', color='red')
            # ax2.plot(self.fcr_imbalance[a].tolist(), label='FCR imbalance', color='black')
            # ax2.set_ylabel('Fast and FCR imbalance [MW]')
            # ax1.legend()
            # ax2.legend()
            # plt.grid()
            # plt.title(a)
            # plt.show()

    def create_result_files(self):
        """Pickle file with dictionary:
        {
        High: {Netted imbalance, Pre-net imbalance, Slow imbalance, Fast imbalance, FCR imbalance, Stochastic imbalance, Deterministic imbalance,
         Netting transmission, Post-net transmission},
        Low: {NTC, ATC, HVDC, Wind, PV, Consumption, Hydro, Thermal, Nuclear, Flex},
        Time: Run-time
        }
        """
        print('CREATING RESULT FILES')
        self.results = {}
        self.results['High'] = {}
        self.results['Low'] = {}
        self.results['High']['Netted imbalance'] = self.netted_imbalance
        self.results['High']['Pre-net imbalance'] = self.imbalances
        self.results['High']['Slow imbalance'] = self.slow_imbalance
        self.results['High']['Fast imbalance'] = self.fast_imbalance
        self.results['High']['FCR imbalance'] =self.fcr_imbalance
        self.results['High']['Stochastic imbalance'] = self.stochastic_imbalances
        self.results['High']['Deterministic imbalance'] = self.deterministic_imbalances
        self.results['High']['Netting transmission'] = self.ac_netting
        self.results['High']['Post-net transmission'] = self.ac_post_high
        self.results['High']['Wind'] = self.wind_high
        self.results['High']['Consumption'] = self.consumption_high
        self.results['High']['PV'] = self.pv_high
        self.results['High']['Hydro'] = self.hydro_high
        self.results['Low']['NTC'] = self.ntc
        self.results['Low']['ATC'] = self.atc_low
        self.results['Low']['HVDC'] = self.hvdc_low
        self.results['Low']['Wind'] = self.wind_low
        self.results['Low']['PV'] = self.pv_low
        self.results['Low']['Consumption'] = self.consumption_low
        self.results['Low']['Hydro'] = self.hydro_low
        self.results['Low']['Thermal'] = self.thermal_low
        self.results['Low']['Nuclear'] = self.nuclear_low
        self.results['Low']['Flex'] = self.flex_low
        self.results['Time'] = self.time
        if self.quarters:
            with open(f'{self.path}{self.scenario}_{self.year}_Quarter.pickle', 'wb') as handle:
                pkl.dump(self.results, handle, protocol=pkl.HIGHEST_PROTOCOL)
        elif self.fixed_ramp:
            with open(f'{self.path}{self.scenario}_{self.year}_FixRamp.pickle', 'wb') as handle:
                pkl.dump(self.results, handle, protocol=pkl.HIGHEST_PROTOCOL)
        elif self.trm:
            with open(f'{self.path}{self.scenario}_{self.year}_TRM.pickle', 'wb') as handle:
                pkl.dump(self.results, handle, protocol=pkl.HIGHEST_PROTOCOL)
        else:
            with open(f'{self.path}{self.scenario}_{self.year}.pickle', 'wb') as handle:
                pkl.dump(self.results, handle, protocol=pkl.HIGHEST_PROTOCOL)

    def run(self):
        start = time.time()
        self.create_high_resolution()
        self.compute_area_imbalance()
        self.imbalance_netting_alt()
        self.imbalance_filtering()
        end = time.time()
        self.time = end - start
        print(f'RUN TIME: {self.time} SECONDS')
        if self.save:
            self.create_result_files()

    def run_no_netting(self):
        self.create_high_resolution()
        self.compute_area_imbalance()
        self.results = {}
        self.results['High'] = {}
        self.results['Low'] = {}
        self.results['High']['Pre-net imbalance'] = self.imbalances
        self.results['High']['Stochastic imbalance'] = self.stochastic_imbalances
        self.results['High']['Deterministic imbalance'] = self.deterministic_imbalances
        self.results['Low']['NTC'] = self.ntc
        self.results['Low']['ATC'] = self.atc_low
        self.results['Low']['HVDC'] = self.hvdc_low
        self.results['Low']['Wind'] = self.wind_low
        self.results['Low']['PV'] = self.pv_low
        self.results['Low']['Consumption'] = self.consumption_low
        self.results['Low']['Hydro'] = self.hydro_low
        self.results['Low']['Thermal'] = self.thermal_low
        self.results['Low']['Nuclear'] = self.nuclear_low
        self.results['Low']['Flex'] = self.flex_low
        with open(f'{self.path}{self.scenario}_{self.year}_Test.pickle', 'wb') as handle:
            pkl.dump(self.results, handle, protocol=pkl.HIGHEST_PROTOCOL)


m = Model(start_date='2009-01-01', scenario='EP45', simulated_days=52*7, save=True, quarters=False, fixed_ramp=False, trm=False)
m.run()




