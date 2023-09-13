"""
This file should be used as the in-operation solar forecast model
Steps:
1: Find the function for targets vs geographical property - done
2: Find the function for correlation vs geographical distance - done
3: Based on map, compute target value for all areas - done
4: Based on map, compute correlation matrix - done
5: Find function of error vs relative forecast - one similar for all areas. Use many areas - done
6: Tune forecast error model per area - done
7: Tune multi-area model - done
8: Tune variability model per area based on historical data - done
9: Connect to imbalance model - done

abs-func: 0.12388 - 0.00074x
std-func: 0.27897 - 0.00141x
var-func: 0.07476 - 0.00045x

correlation func: <= 750 km: 0.54177 - 0.00071x,  >750 km: 0.02153 - 0.00003x
"""

import pandas as pd
import numpy as np
import scipy.interpolate as interpolate
from datetime import datetime, timedelta
import random

class Solar_Model:

    def __init__(self, pv_in, scenario, year_start, sim_days, improvement_percentage=0):
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

        self.areas = ('SE1', 'SE2', 'SE3', 'SE4', 'NO1', 'NO2', 'NO3', 'NO5', 'DK2', 'FI')
        self.lma_areas = ('SE1', 'SE2', 'SE3', 'SE4', 'NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'DK2', 'FI')
        self.pv = pv_in
        self.scenario = scenario
        self.area_to_idx = {
            'SE1': 0,
            'SE2': 1,
            'SE3': 2,
            'SE4': 3,
            'NO1': 4,
            'NO2': 5,
            'NO3': 6,
            'NO5': 7,
            'DK2': 8,
            'FI': 9
        }
        if self.scenario == 'SF45':
            self.alpha = [1.174164, 1.284138, 1.310644, 1.277100, 1.157306, 1.156060, 1.220144,
                        0.860298, 1.211929, 1.345592]
            self.beta = [0.543005, 0.078956, 0.091883, 0.415955, 0.797041, 0.506341, 0.110968,
                        0.353165, 0.71152, 0.056123]
            self.gamma = [-0.40763, -0.15829, -0.14027, -0.38779, -0.47645, -0.30305, -0.18341,
                        0.190096, -0.49164, -0.10196]
            self.cmatrix = pd.read_csv(
                f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\ModelTuning\\Solar{self.scenario}\\cmatrix.csv')
            self.ar_params = [1.01, 1.16, 1.31, 1.27, 1.29, 0.85, 0.71, 0.63, 1.32, 1.12] #Area parameters
        elif self.scenario == 'EP45':
            self.alpha = [1.019696, 1.328642, 1.325614, 1.273462, 1.20961, 1.187146, 1.221698,
                        1.041255, 1.207969, 1.358245]
            self.beta = [0.352313, 0.052863, 0.010141, 0.360318, 0.659344, 0.496163, 0.125941,
                        0.378936, 0.678156, 0.035475]
            self.gamma = [-0.0677, -0.08996, -0.02263, -0.30129, -0.50812, -0.35108, -0.20044,
                        -0.07937, -0.43148, -0.06417]
            self.cmatrix = pd.read_csv(
                f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\ModelTuning\\Solar{self.scenario}\\cmatrix.csv')
            self.ar_params = [0.64, 0.9, 1.15, 1.07, 1.23, 0.87, 0.74, 0.64, 1.18, 0.88] #Tuned area-dependent parameters
        elif self.scenario == 'EF45':
            self.alpha = [1.187832, 1.330285, 1.31408, 1.231749, 1.215631, 1.173112, 1.180470,
                        0.922086, 1.165346, 1.352798]
            self.beta = [0.450557, 0.045058, 0.006444, 0.303958, 0.651633, 0.559805, 0.088448,
                        0.369799, 0.672136, 0.032356]
            self.gamma = [-0.35045, -0.08116, -0.04793, -0.21835, -0.49159, -0.3693, -0.25263,
                        0.098705, -0.33370, -0.05814]
            self.cmatrix = pd.read_csv(
                f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\ModelTuning\\Solar{self.scenario}\\cmatrix.csv')
            self.ar_params = [0.89, 0.76, 1.11, 0.87, 1.25, 0.92, 0.76, 0.64, 1.09, 0.905] #Tuned area-dependent parameters
        elif self.scenario == 'FM45':
            self.alpha = [1.111576, 1.336988, 1.335642, 1.265049, 1.204272, 1.157307, 1.221833,
                          1.063324, 1.187082, 1.363785]
            self.beta = [0.313465, 0.074815, 0.019138, 0.312214, 0.664302, 0.519000, 0.200627,
                         0.368674, 0.709711, 0.03755]
            self.gamma = [-0.14482, -0.08029, -0.06014, -0.24554, -0.49497, -0.3146, -0.23514,
                          -0.10938, -0.39961, -0.06652]
            self.cmatrix = pd.read_csv(
                f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\ModelTuning\\Solar{self.scenario}\\cmatrix.csv')
            self.ar_params = [0.6, 0.89, 1.2, 0.93, 1.22, 0.85, 0.715, 0.64, 1.18, 0.87] #Tuned area-dependent parameters

    def stdev_function(self, norm_gen):
        if norm_gen <= 0.035:
            return 0.8096173357897626 - 16.595799287953355 * norm_gen
        elif norm_gen > 0.035 and norm_gen <= 0.18:
            return 0.2549277645110786 - 0.8922911241645132 * norm_gen
        elif norm_gen > 0.18 and norm_gen <= 0.5:
            return 0.12026767628481086 - 0.15221697836807174 * norm_gen
        elif norm_gen > 0.5:
            return 0.07762027143228016 - 0.072312283765212 * norm_gen

    def forecast_errors_single_area(self, area):
        a = area
        pv_in = self.pv[a]
        length = len(pv_in)
        max_pv = pv_in.max()
        rel_pv_in = pv_in / max_pv
        std = []
        for w in rel_pv_in.tolist():
            std.append(self.stdev_function(w))
        rndm = []
        rndm.append(0)
        for i in range(length):
            rndm.append(random.gauss(0, float(std[i])) * pv_in[i])
        alpha = self.alpha[self.area_to_idx[a]]
        beta = self.beta[self.area_to_idx[a]]
        gamma = self.gamma[self.area_to_idx[a]]
        simulated_error = []
        simulated_error.append(0)
        for i in range(len(pv_in)):
            if pv_in[i] <= 0.02 * max_pv:
                simulated_error.append(0)
            else:
                err = min(pv_in[i], alpha * simulated_error[i] + beta * rndm[i + 1] + gamma * rndm[i])
                simulated_error.append(max(err, pv_in[i] - max_pv))
        simulated_error = simulated_error[1:]
        self.sim_error = simulated_error
        self.sim_hourly = np.subtract(np.array(pv_in), np.array(self.sim_error))


    def forecast_errors_multi_area(self):
        cmatrix = []
        for a in self.areas:
            cmatrix.extend(self.cmatrix[a].tolist())

        pv_in = self.pv
        pv_in.drop(['NO4'], axis=1)
        stds = pd.DataFrame(columns=self.areas)
        rndms = pd.DataFrame(columns=self.areas)
        max_pv = pd.DataFrame(columns=self.areas, index=[0])
        rel_pv_in = pd.DataFrame(columns=self.areas)

        for a in self.areas:
            max_pv[a][0] = pv_in[a].max() * 1.1
            rel_pv_in[a] = pv_in[a] / max_pv[a][0]
            length = len(pv_in[a])
            rndm = []
            std = []

            for w in rel_pv_in[a].tolist():
                std.append(self.stdev_function(w))

            rndm.append(0)
            for i in range(length):
                rndm.append(random.gauss(0, 1))

            std_real = [0]
            std_real.extend([a * b for a,b in zip(std, self.pv[a].tolist())])
            stds[a] = std_real
            rndms[a] = rndm
        self.sim_errors = pd.DataFrame(columns=self.areas)
        self.sim_errors_norm = pd.DataFrame(columns=self.areas)
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
            max_val = float(max_pv[a][0])
            for i in range(len(pv_in[a])):
                if pv_in[a][i] <= 0.02 * max_val:
                    simulated_error.append(0)
                else:
                    err = min(pv_in[a][i],
                              self.alpha[self.area_to_idx[a]] * simulated_error[i] + self.beta[self.area_to_idx[a]] * corr_rndms[a][i + 1] +
                              self.gamma[self.area_to_idx[a]] * corr_rndms[a][i])
                    simulated_error.append(max(err, pv_in[a][i] - max_pv[a][0]))
                    # simulated_error.append(self.alpha[self.area_to_idx[a]] * simulated_error[i] + self.beta[self.area_to_idx[a]] * corr_rndms[a][i + 1] +
                    #           self.gamma[self.area_to_idx[a]] * corr_rndms[a][i])
            self.sim_errors[a] = simulated_error[1:]
            self.sim_errors[a] = self.sim_errors[a] * (100 - self.improvement_percentage) / 100
            self.actual_hourly[a] = np.subtract(np.array(pv_in[a].tolist()), np.array(self.sim_errors[a].tolist()))

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
        pv_spline = self.energy_error(data)
        # Gör datumnkolumn med splines
        a = time[0]
        highres_time = []
        for i in range(60 * data.__len__()):
            highres_time.append(a + timedelta(minutes=i))
        pv_interpol = pd.DataFrame(columns=['Timestamp', 'Generation'])
        pv_interpol['Timestamp'] = highres_time
        pv_interpol['Generation'] = pv_spline
        pv_interpol['Timestamp'] = pd.to_datetime(pv_interpol['Timestamp'], utc=True, format='%Y-%m-%d')
        return pv_interpol

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
        rhos = [0.832152, 0.04151, -0.029447, -0.021741, -0.012194, -0.016159, -0.014461, 0.002568, -0.015159, -0.041042]
        for i in range(len(xx)):
            if y_norm[i] < 0.05:
                std = 0.045 - 0.616 * y_norm[i]
            else:
                std = 0.015 - 0.016 * y_norm[i]

            if i == 0:
                ar.append(0)
            elif i <= 10 and i > 0:
                ar_sum = 0
                if spline_list[i] == 0:
                    ar.append(0)
                else:
                    for n in range(i):
                        ar_sum = ar_sum + rhos[n] * ar[i - n - 1]
                    ar.append(ar_sum + random.gauss(0, std) * beta[self.area_to_idx[a]] * spline_list[i])
            else:
                if spline_list[i] == 0:
                    ar.append(0)
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
            self.dict[a]['Hourly']['Forecasted'] = self.pv[a].tolist()
            self.dict[a]['Hourly']['Forecast error'] = self.sim_errors[a].tolist()
            self.dict[a]['Hourly']['Actual'] = self.actual_hourly[a].tolist()
            self.dict[a]['Minute'] = self.variability_simulation(self.dict[a]['Hourly']['Actual'], a)

    def make_single_area_highres_scenario(self, area):
        self.dict = {} # Hourly: Time, Forecasted, Forecast error, Actual; Minutely: Time, Spline, Noise, Actual
        self.forecast_errors_single_area(area=area)
        self.dict['Hourly'] = pd.DataFrame(columns=['Time', 'Forecasted', 'Forecast error', 'Actual'])
        self.dict['Hourly']['Time'] = self.time_list
        self.dict['Hourly']['Forecasted'] = self.pv[area].tolist()
        self.dict['Hourly']['Forecast error'] = self.sim_error
        self.dict['Hourly']['Actual'] = self.sim_hourly.tolist()
        self.dict['Minute'] = self.variability_simulation(self.dict['Hourly']['Actual'], area)