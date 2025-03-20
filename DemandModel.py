"""
This is the file where the demand forecast model should be set up

Steps:
1: Find target values based on historical data - same for all? -done
2: Find correlation matrix based on historical data - done
3: Based on map, compute target value for all areas - done
4: Based on map, compute correlation matrix - done
5: Find function of error vs relative forecast - one similar for all areas. Use many areas -doen
6: Tune forecast error model per area - done
7: Tune multi-area model - done
8: Tune variability model per area based on historical data - done
9: Connect to main model
"""

import pandas as pd
import numpy as np
import scipy.interpolate as interpolate
from datetime import datetime, timedelta
import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class Demand_Model:

    def __init__(self, demand_in, scenario, year_start, sim_days, improvement_percentage=0,seed=1):
        self.start_idx = (year_start - 1982) * 52 * 7 * 24
        start_point = datetime.strptime(f'{year_start}-01-01 00:00', '%Y-%m-%d %H:%M')
        self.end_idx = self.start_idx + sim_days * 24
        self.time_list = []
        self.improvement_percentage = improvement_percentage
        n = -1
        for i in range(self.end_idx - self.start_idx):
            n = n + 1
            date = start_point + timedelta(hours=n)
            if date.strftime('%m-%d') =='02-29':
                n = n + 24
            elif date.strftime('%m-%d') == '12-31':
                n = n + 24
            self.time_list.append(start_point + timedelta(hours=n))
        self.areas = ('SE1', 'SE2', 'SE3', 'SE4', 'NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'DK2', 'FI')
        self.demand = demand_in
        self.scenario = scenario
        self.seed = seed
        self.area_to_idx = {
            'SE1': 0,
            'SE2': 1,
            'SE3': 2,
            'SE4': 3,
            'NO1': 4,
            'NO2': 5,
            'NO3': 6,
            'NO4': 7,
            'NO5': 8,
            'DK2': 9,
            'FI': 10
        }
        if self.scenario == 'SF45':
            self.alpha = [0.736542, 0.913625, 0.896951, 0.873720, 0.871768, 0.962010, 0.660962,
                          0.905162, 0.915939, 0.957910, 0.940291]
            self.beta = [1.563695, 1.588418, 0.181474, 0.357493, 0.112962, 1.029788, 0.666024,
                         0.702192, 1.290930, 0.271646, 0.538252]
            self.gamma = [0.183615, 0.144085, 0.156435, 0.279134, -0.31043, -0.39419, 0.049421,
                          -0.22776, 0.404255, -0.17402, -0.31358]
            self.cmatrix = pd.read_csv(
                f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\ModelTuning\\Demand{self.scenario}\\cmatrix.csv')
            self.ar_params = [1.07, 1.09, 1.07, 1.07, 1.02, 1.1, 1.11, 1.13, 1.13, 1.09, 1.16]
        elif self.scenario == 'EP45':
            self.alpha = [0.641495, 0.909485, 0.964196, 0.974551, 0.986374, 0.855296, 0.975123,
                          0.903479, 0.912537, 0.985667, 0.963835]
            self.beta = [1.382725, 1.704548, 0.275539, 0.457097, 0.250349, 1.217295, 0.612612,
                         0.717005, 1.391857, 0.222432, 0.591016]
            self.gamma = [0.557864, 0.241913, -0.09456, -0.16318, -0.31711, 0.238511, -0.43236,
                          -0.22807, 0.483206, -0.28334, -0.39506]
            self.cmatrix = pd.read_csv(
               f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\ModelTuning\\Demand{self.scenario}\\cmatrix.csv')
            self.ar_params = [1.04, 1.13, 1.09, 1.09, 1.04, 1.16, 1.12, 1.15, 1.17, 1.14, 1.21]
        elif self.scenario == 'EF45':
            self.alpha = [0.678316, 0.958931, 0.918267, 0.941388, 0.98199, 0.866677, 0.850672,
                          0.860216, 0.913245, 0.930651, 0.951562]
            self.beta = [1.400777, 1.589709, 0.226213, 0.452629, 0.205694, 1.053756, 0.640583,
                         0.675414, 1.311474, 0.282139, 0.52164]
            self.gamma = [0.35105, -0.40514, 0.092866, -0.02626, -0.28174, 0.140956, -0.22774,
                          -0.12644, 0.474238, -0.16045, -0.32539]
            self.cmatrix = pd.read_csv(
               f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\ModelTuning\\Demand{self.scenario}\\cmatrix.csv')
            self.ar_params = [1.03, 1.11, 1.10, 1.07, 1.01, 1.10, 1.11, 1.12, 1.15, 1.10, 1.14]
        elif self.scenario == 'FM45':
            self.alpha = [0.758325, 0.961205, 0.893878, 0.958868, 0.983817, 0.959377, 0.897846,
                          0.895254, 0.915061, 0.928819, 0.867309]
            self.beta = [1.592265, 1.584032, 0.181225, 0.438407, 0.239078, 1.032334, 0.629537,
                         0.696366, 1.274361, 0.287463, 0.587926]
            self.gamma = [0.090487, -0.42043, 0.157495, -0.08788, -0.30507, -0.37331, -0.27305,
                          -0.20986, 0.416563, -0.15683, -0.23394]
            self.cmatrix = pd.read_csv(
               f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\ModelTuning\\Demand{self.scenario}\\cmatrix.csv')
            self.ar_params = [1.08, 1.10, 1.06, 1.07, 1.02, 1.11, 1.11, 1.14, 1.13, 1.10, 1.17]

    def stdev_function(self, norm_dem):
        return 0.037172938443793255 - 0.02745556240448081 * norm_dem

    def forecast_errors_single_area(self, area):
        a = area
        demand_in = self.demand[a]
        length = len(demand_in)
        max_demand = demand_in.max()
        rel_demand_in = demand_in / max_demand
        std = []
        for w in rel_demand_in.tolist():
            std.append(self.stdev_function(w))
        random.seed(self.seed + self.area_to_idx[a])
        rndm = []
        rndm.append(0)
        for i in range(length):
            rndm.append(random.gauss(0, float(std[i])) * demand_in[i])
        alpha = self.alpha[self.area_to_idx[a]]
        beta = self.beta[self.area_to_idx[a]]
        gamma = self.gamma[self.area_to_idx[a]]
        simulated_error = []
        simulated_error.append(0)
        for i in range(len(demand_in)):
            err = min(demand_in[i], alpha * simulated_error[i] + beta * rndm[i + 1] + gamma * rndm[i])
            simulated_error.append(err)
        simulated_error = simulated_error[1:]
        self.sim_error = simulated_error
        self.sim_hourly = np.subtract(np.array(demand_in), np.array(self.sim_error))


    def forecast_errors_multi_area(self):
        cmatrix = []
        for a in self.areas:
            cmatrix.extend(self.cmatrix[a].tolist())

        demand_in = self.demand
        stds = pd.DataFrame(columns=self.areas)
        rndms = pd.DataFrame(columns=self.areas)
        max_demand = pd.DataFrame(columns=self.areas, index=[0])
        rel_demand_in = pd.DataFrame(columns=self.areas)

        for a in self.areas:
            max_demand[a][0] = demand_in[a].max()
            rel_demand_in[a] = demand_in[a] / max_demand[a][0]
            length = len(demand_in[a])
            rndm = []
            std = []

            for w in rel_demand_in[a].tolist():
                std.append(self.stdev_function(w))

            random.seed(self.seed + self.area_to_idx[a])
            rndm.append(0)
            for i in range(length):
                rndm.append(random.gauss(0, 1))

            std_real = [0]
            std_real.extend([a * b for a,b in zip(std, self.demand[a].tolist())])
            stds[a] = std_real
            rndms[a] = rndm

        self.sim_errors = pd.DataFrame(columns=self.areas)
        corr_rndms = pd.DataFrame(columns=self.areas)
        self.actual_hourly = pd.DataFrame(columns=self.areas, index=self.time_list)
        for a in self.areas:
            lst = []
            for i in range(len(rndms[a])):
                val = 0
                for aa in self.areas:
                    val = val + rndms[aa][i] * cmatrix[self.area_to_idx[a] * self.areas.__len__() + self.area_to_idx[aa]]
                lst.append(val * stds[a][i])
            corr_rndms[a] = lst
            simulated_error = [0]
            for i in range(len(demand_in[a])):
                err = min(0.5 * demand_in[a][i],
                          self.alpha[self.area_to_idx[a]] * simulated_error[i] + self.beta[self.area_to_idx[a]] * corr_rndms[a][i + 1] +
                          self.gamma[self.area_to_idx[a]] * corr_rndms[a][i])
                simulated_error.append(err)
                # simulated_error.append(self.alpha[self.area_to_idx[a]] * simulated_error[i] + self.beta[self.area_to_idx[a]] * corr_rndms[a][i + 1] +
                #           self.gamma[self.area_to_idx[a]] * corr_rndms[a][i])
            self.sim_errors[a] = simulated_error[1:]
            self.sim_errors[a] = self.sim_errors[a] * (100 - self.improvement_percentage) / 100
            self.actual_hourly[a] = np.subtract(np.array(demand_in[a].tolist()), np.array(self.sim_errors[a].tolist()))

    def energy_error(self, data):
        x_ny = []
        LD_ref = []
        h = []
        error = []
        x_ny.extend(data)
        LD_ref.extend(data)
        HD = self.spline(x_ny)
        for lowres_step in range(data.__len__()):
            h.append(LD_ref[lowres_step] - sum(HD[int(lowres_step * 60):int((lowres_step + 1) * 60)]) / 60)
            error.append(np.sqrt(h[lowres_step] ** 2))
        error_new = sum(error)
        error_old = np.infty
        while error_old > error_new and error_new > 0.001 * data.__len__():
            error_old = sum(error)
            for i in range(int(data.__len__())):
                x_ny[i] = x_ny[i] + h[i]
            HD = self.spline(x_ny)
            for i in range(int(data.__len__())):
                h[i] = LD_ref[i] - sum(HD[int(i * 60): int((i + 1) * 60)]) / 60
                error[i] = np.sqrt(h[i] ** 2)
            error_new = sum(error)
        if error_new > 0.1 * data.__len__():
            print('!!!TP ENERGY ERROR OCCURRED!!!')
            print(error_new)
        return self.spline(x_ny)

    def spline(self, data):
        length = len(data)
        x_pts = []
        x_pts.append(0)
        x_pts.extend(np.linspace(30, 30 + 60 * length - 60, int(length)))
        x_pts.append(60 * length)
        x_vals = np.linspace(0, 60 * length, 60 * length)
        y_pts = []
        y_pts.append(data[0] - (data[1] - data[0]) / 2)
        y_pts.extend(data)
        y_pts.append(data[length - 1] + (data[length - 1] - data[length - 2]) / 2)
        f = interpolate.CubicSpline(x_pts, y_pts)
        new_list = f(x_vals)
        return new_list

    def spline_hour_2_min(self, time, data):
        """For hourly data"""
        demand_spline = self.energy_error(data)
        # Gör datumnkolumn med splines
        a = time[0]
        highres_time = []
        for i in range(60 * data.__len__()):
            highres_time.append(a + timedelta(minutes=i))
        wind_interpol = pd.DataFrame(columns=['Timestamp', 'Generation'])
        wind_interpol['Timestamp'] = highres_time
        wind_interpol['Generation'] = demand_spline
        wind_interpol['Timestamp'] = pd.to_datetime(wind_interpol['Timestamp'], utc=True, format='%Y-%m-%d')
        return wind_interpol

    def variability_simulation(self, data, area):
        x = self.time_list
        y = data
        spline = self.spline_hour_2_min(x, y.tolist())
        spline.loc[spline['Generation']<0, 'Generation'] = 0
        spline_list = spline[spline.index % 3 == 0]
        spline_list = spline_list['Generation'].tolist()
        y_norm = spline['Generation'] / (data.max() * 1.1)
        y_norm = y_norm[y_norm.index % 3 ==0]
        y_norm = y_norm.tolist()

        a = area

        """Gör här en dataframe med 3-minutsupplösning"""
        xx = []
        for i in range(len(y) * 20):
            xx.append(x[0] + timedelta(minutes=i*3))

        xxx = spline['Timestamp'].tolist()
        ar = []
        beta = self.ar_params
        rhos = [0.749842, 0.022939, -0.051539, -0.043106, 0.001220, -0.016049, 0.011089, -0.002796, -0.027262, -0.039671]
        for i in range(len(xx)):
            std = 0.00533 - 0.00239 * y_norm[i]
            if i == 0:
                ar.append(0)
            elif i <= 10 and i  >0:
                ar_sum = 0
                for n in range(i):
                    ar_sum = ar_sum + rhos[n] * ar[i - n - 1]
                ar.append(ar_sum + random.gauss(0, std) * beta[self.area_to_idx[a]] * spline_list[i])
            else:
                ar_sum = 0
                for n in range(10):
                    ar_sum = ar_sum + rhos[n] * ar[i - n - 1]
                ar.append(ar_sum + random.gauss(0, std) * beta[self.area_to_idx[a]] * spline_list[i])

        minute_highfreq = []
        for i in range(len(xx)):
            if i == len(xx) - 1:
                minute_highfreq.append(ar[i])
                minute_highfreq.append(ar[i] + (ar[i] - ar[i - 1]) * 1/3)
                minute_highfreq.append(ar[i] + (ar[i] - ar[i - 1]) * 2/3)
            else:
                minute_highfreq.append(ar[i])
                minute_highfreq.append(ar[i] + (ar[i + 1] - ar[i]) * 1/3)
                minute_highfreq.append(ar[i] + (ar[i + 1] - ar[i]) * 2/3)
        highres = pd.DataFrame(columns=['Time', 'Spline', 'Noise', 'Generation'])
        highres['Time'] = xxx
        highres['Generation'] = spline['Generation'] + pd.Series(minute_highfreq)
        highres.loc[highres['Generation'] < 0, 'Generation'] = 0
        highres['Spline'] = spline['Generation'].tolist()
        highres['Noise'] = highres['Generation'] - highres['Spline']
        return highres

    def make_multi_area_highres_scenario(self):
        self.dict = {} # Areas -> Hourly: Time, Forecasted, Forecast error, Actual; Minutely: Time, Spline, Noise, Actual
        self.forecast_errors_multi_area()
        for a in self.areas:
            self.dict[a] = {}
            self.dict[a]['Hourly'] = pd.DataFrame(columns=['Time', 'Forecasted', 'Forecast error', 'Actual'])
            self.dict[a]['Hourly']['Time'] = self.time_list
            self.dict[a]['Hourly']['Forecasted'] = self.demand[a].tolist()
            self.dict[a]['Hourly']['Forecast error'] = self.sim_errors[a].tolist()
            self.dict[a]['Hourly']['Actual'] = self.actual_hourly[a].tolist()
            self.dict[a]['Minute'] = self.variability_simulation(self.dict[a]['Hourly']['Actual'], a)

    def make_single_area_highres_scenario(self, area):
        self.dict = {} # Hourly: Time, Forecasted, Forecast error, Actual; Minutely: Time, Spline, Noise, Actual
        self.forecast_errors_single_area(area=area)
        self.dict['Hourly'] = pd.DataFrame(columns=['Time', 'Forecasted', 'Forecast error', 'Actual'])
        self.dict['Hourly']['Time'] = self.time_list
        self.dict['Hourly']['Forecasted'] = self.demand[area].tolist()
        self.dict['Hourly']['Forecast error'] = self.sim_error
        self.dict['Hourly']['Actual'] = self.sim_hourly.tolist()
        self.dict['Minute'] = self.variability_simulation(self.dict['Hourly']['Actual'], area)



