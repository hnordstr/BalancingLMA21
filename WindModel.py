"""
Steps:
1: Find the function for targets vs geographical property - done
2: Find the function for correlation vs geographical distance - done
3: Based on map, compute target value for all areas - done
4: Based on map, compute correlation matrix - done
5: Find function of error vs relative forecast - one similar for all areas. Use many areas - done
6: Tune forecast error model per area - done
7: Tune multi-area model - done
8: Tune variability model per area based on historical data - done
9: Connect to Imbalance model - done

abs-func: 0.25259 - 0.00082x
std-func: 0.35232 - 0.00123x
var-func: 0.10651 - 0.00033x

correlation func: <= 700 km: 0.29358 - 0.00039x,  >700 km: 0.003545 - 0.00002x
"""

import pandas as pd
import numpy as np
import scipy.interpolate as interpolate
from datetime import datetime, timedelta
import random


class Wind_Model:

    def __init__(self, wind_in, scenario, year_start, sim_days, improvement_percentage=0):
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
        self.wind = wind_in
        self.scenario = scenario
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
            self.alpha = [0.934895, 0.831286, 0.942514, 0.840097, 0.853451, 0.929322, 0.872953,
                          0.950803, 0.979522, 0.969189, 0.941636] #Real NO4 0.738853
            self.beta = [1.291769, 0.861607, 0.806465, 0.930376, 0.864449, 1.305166, 0.841179,
                         0.745798, 1.452029, 1.408425, 0.738612] #Real NO4 0.052328
            self.gamma = [-0.15392, 0.650779, -0.18537, 0.533644, 0.47935, -0.08849, 0.244863,
                          -0.21114, -0.60998, -0.44410, -0.16041] #Real NO4 -0.09625
            self.cmatrix = pd.read_csv(
                f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\ModelTuning\\Wind{self.scenario}\\cmatrix.csv')
            self.ar_params = [1.4, 1.35, 1.38, 1.44, 1.04, 1.31, 1.2, 1.23, 1.54, 1.6, 1.22]  # Area parameters
        elif self.scenario == 'EP45':
            self.alpha = [0.843239, 0.973591, 0.950928, 0.934952, 0.933028, 0.912425, 0.829976,
                          0.949770, 0.946069, 0.843861, 0.960735]
            self.beta = [1.207018, 1.022957, 0.676614, 1.170874, 0.950841, 1.365844, 0.888299,
                         0.649912, 1.502175, 0.847627, 0.613006]
            self.gamma = [0.726314, -0.42158, -0.22389, -0.15960, -0.12175, 0.011031, 0.521714,
                          -0.22420, -0.26599, 0.503617, -0.25484]
            self.cmatrix = pd.read_csv(
                f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\ModelTuning\\Wind{self.scenario}\\cmatrix.csv')
            self.ar_params = [1.43, 1.4, 1.51, 1.63, 1.08, 1.45, 1.33, 1.28, 1.64, 1.05, 1.2]  # Tuned area-dependent parameters
        elif self.scenario == 'EF45':
            self.alpha = [0.933949, 0.918861, 0.957918, 0.847212, 0.840020, 0.930442, 0.837370,
                          0.850354, 0.965198, 0.938693, 0.967665]
            self.beta = [1.19578, 1.000932, 0.738011, 1.026096, 0.663519, 1.313487, 0.857446,
                         0.701406, 1.453206, 1.011808, 0.657482]
            self.gamma = [-0.1383, -0.06379, -0.26098, 0.471461, 0.444006, -0.10419, 0.476352,
                          0.18516, -0.44781, -0.14949, -0.26736]
            self.cmatrix = pd.read_csv(
                f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\ModelTuning\\Wind{self.scenario}\\cmatrix.csv')
            self.ar_params = [1.46, 1.3, 1.44, 1.27, 1.01, 1.35, 1.36, 1.26, 1.53, 1, 1.3]  # Tuned area-dependent parameters
        elif self.scenario == 'FM45':
            self.alpha = [0.933949, 0.918861, 0.957918, 0.847212, 0.936454, 0.940539, 0.921079,
                          0.898305, 0.962937, 0.938693, 0.967665]
            self.beta = [1.19578, 1.000932, 0.738011, 1.026096, 0.94883, 1.240763, 0.972874,
                         0.67440, 1.447215, 1.011808, 0.657482]
            self.gamma = [-0.1383, -0.06379, -0.26098, 0.471461, -0.12233, -0.18141, -0.07203,
                          -0.00066, -0.41931, -0.14949, -0.26736]
            self.cmatrix = pd.read_csv(
                f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\ModelTuning\\Wind{self.scenario}\\cmatrix.csv')
            self.ar_params = [1.28, 1.31, 1.46, 1.52, 1.06, 1.28, 1.26, 1.25, 1.57, 1.06, 1.21]  # Tuned area-dependent parameters

    def stdev_function(self, norm_gen):
        if norm_gen <= 0.0265:
            return 1.3980518092749208 - 39.88632824668849 * norm_gen
        elif norm_gen > 0.0265 and norm_gen <= 0.12:
            return 0.38382298051111785 - 1.8451357727662174 * norm_gen
        elif norm_gen > 0.12 and norm_gen <= 0.59:
            return 0.1991465324867623 - 0.2533242669906466 * norm_gen
        elif norm_gen > 0.59:
            return 0.11779102907699517 - 0.1054201676089161 * norm_gen


    def forecast_errors_single_area(self, area):
        a = area
        wind_in = self.wind[a]
        length = len(wind_in)
        max_wind = wind_in.max() * 1.1
        rel_wind_in = wind_in / max_wind
        std = []
        for w in rel_wind_in.tolist():
            std.append(self.stdev_function(w))
        rndm = []
        rndm.append(0)
        for i in range(length):
            rndm.append(random.gauss(0, float(std[i])) * wind_in[i])
        alpha = self.alpha[self.area_to_idx[a]]
        beta = self.beta[self.area_to_idx[a]]
        gamma = self.gamma[self.area_to_idx[a]]
        simulated_error = []
        simulated_error.append(0)
        for i in range(len(wind_in)):
            err = min(wind_in[i], alpha * simulated_error[i] + beta * rndm[i + 1] + gamma * rndm[i])
            simulated_error.append(max(err, wind_in[i] - max_wind))
        simulated_error = simulated_error[1:]
        self.sim_error = simulated_error
        self.sim_hourly = np.subtract(np.array(wind_in), np.array(self.sim_error))


    def forecast_errors_multi_area(self):
        cmatrix = []
        for a in self.areas:
            cmatrix.extend(self.cmatrix[a].tolist())

        wind_in = self.wind
        stds = pd.DataFrame(columns=self.areas)
        rndms = pd.DataFrame(columns=self.areas)
        max_wind = pd.DataFrame(columns=self.areas, index=[0])
        rel_wind_in = pd.DataFrame(columns=self.areas)

        for a in self.areas:
            max_wind[a][0] = wind_in[a].max() * 1.1
            rel_wind_in[a] = wind_in[a] / max_wind[a][0]
            length = len(wind_in[a])
            rndm = []
            std = []

            for w in rel_wind_in[a].tolist():
                std.append(self.stdev_function(w))

            rndm.append(0)
            for i in range(length):
                rndm.append(random.gauss(0, 1))

            std_real = [0]
            std_real.extend([a*b for a,b in zip(std, self.wind[a].tolist())])
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
            for i in range(len(wind_in[a])):
                err = min(wind_in[a][i],
                          self.alpha[self.area_to_idx[a]] * simulated_error[i] + self.beta[self.area_to_idx[a]] * corr_rndms[a][i + 1] +
                          self.gamma[self.area_to_idx[a]] * corr_rndms[a][i])
                simulated_error.append(max(err, wind_in[a][i] - max_wind[a][0]))
                # simulated_error.append(self.alpha[self.area_to_idx[a]] * simulated_error[i] + self.beta[self.area_to_idx[a]] * corr_rndms[a][i + 1] +
                #           self.gamma[self.area_to_idx[a]] * corr_rndms[a][i])
            self.sim_errors[a] = simulated_error[1:]
            self.sim_errors[a] = self.sim_errors[a] * (100 - self.improvement_percentage) / 100
            self.actual_hourly[a] = np.subtract(np.array(wind_in[a].tolist()), np.array(self.sim_errors[a].tolist()))

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
        wind_spline = self.energy_error(data)
        # Gör datumnkolumn med splines
        a = time[0]
        highres_time = []
        for i in range(60 * data.__len__()):
            highres_time.append(a + timedelta(minutes=i))
        wind_interpol = pd.DataFrame(columns=['Timestamp', 'Generation'])
        wind_interpol['Timestamp'] = highres_time
        wind_interpol['Generation'] = wind_spline
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
        rhos = [0.832152, 0.04151, -0.029447, -0.021741, -0.012194, -0.016159, -0.014461, 0.002568, -0.015159, -0.041042]
        for i in range(len(xx)):
            if y_norm[i] < 0.05:
                std = 0.045 - 0.616 * y_norm[i]
            else:
                std = 0.015 - 0.016 * y_norm[i]
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
            self.dict[a]['Hourly']['Forecasted'] = self.wind[a].tolist()
            self.dict[a]['Hourly']['Forecast error'] = self.sim_errors[a].tolist()
            self.dict[a]['Hourly']['Actual'] = self.actual_hourly[a].tolist()
            self.dict[a]['Minute'] = self.variability_simulation(self.dict[a]['Hourly']['Actual'], a)

    def make_single_area_highres_scenario(self, area):
        self.dict = {} # Hourly: Time, Forecasted, Forecast error, Actual; Minutely: Time, Spline, Noise, Actual
        self.forecast_errors_single_area(area=area)
        self.dict['Hourly'] = pd.DataFrame(columns=['Time', 'Forecasted', 'Forecast error', 'Actual'])
        self.dict['Hourly']['Time'] = self.time_list
        self.dict['Hourly']['Forecasted'] = self.wind[area].tolist()
        self.dict['Hourly']['Forecast error'] = self.sim_error
        self.dict['Hourly']['Actual'] = self.sim_hourly.tolist()
        self.dict['Minute'] = self.variability_simulation(self.dict['Hourly']['Actual'], area)



