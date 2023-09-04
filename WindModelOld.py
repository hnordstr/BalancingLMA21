"""
This file is an old wind model, used from "Minute resolution
multi-area wind power simulation to estimate future reserve needs"
Used as backup and to learn from other areas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
import scipy.interpolate as interpolate
from datetime import datetime, timedelta
import random
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter
from pandas.plotting import autocorrelation_plot

class Wind_Model:

    def __init__(self, year_start, week_start, day_start, sim_days):
        self.start_idx = (year_start - 1982) * 52 * 7 * 24 + (week_start - 1) * 24 * 7 + day_start * 24
        start_point = datetime.strptime(f'{year_start}-01-01 00:00', '%Y-%m-%d %H:%M')
        self.end_idx = self.start_idx + sim_days * 24
        self.time_list = []
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
        with open('LMA_RES_EF45.pickle', 'rb') as handle:
            wind = pkl.load(handle)
        self.wind = pd.DataFrame(columns=self.areas)
        for a in self.areas:
            self.wind[a] = wind[a]['WindOffshore'][self.start_idx:self.end_idx] + wind[a]['WindOnshore'][self.start_idx:self.end_idx]
        self.wind.reset_index(drop=True, inplace=True)
        self.a = np.array(0.2867689816793785)
        self.b = np.array(0.20991242725821563)
        self.c = np.array(-0.05329247460791067)
        self.alpha = [0.7545, 0.7504, 0.7387, 0.7443, 0.8811, 0.7411, 0.7482, 0.7402, 0.7500, 0.7354, 0.7490]
        self.beta = [0.7590, 0.6493, 0.6019, 0.7649, 0.4900, 0.6198, 0.5711, 0.5984, 0.7650, 0.3870, 0.5421]
        self.gamma = [0.5820, 0.5961, 0.6503, 0.6130, -0.0621, 0.6230, 0.6135, 0.5359, 0.6099, 0.4070, 0.5708]
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


    def forecast_errors_single_area(self, area):
        a = area
        wind_in = self.wind[a]

        length = len(wind_in)
        max_wind = wind_in.max() / 0.9
        rel_wind_in = wind_in / max_wind
        variance = self.a * (self.b ** rel_wind_in.tolist()) + self.c
        rndm1 = []
        rndm1.append(0)
        for i in range(length):
            rndm1.append(random.gauss(0, float(variance[i])) * wind_in[i])

        alpha = self.alpha[self.area_to_idx[a]]
        beta = self.beta[self.area_to_idx[a]]
        gamma = self.gamma[self.area_to_idx[a]]
        simulated_error1 = []
        simulated_error1.append(0)
        for i in range(len(wind_in)):
            err = min(wind_in[i], alpha * simulated_error1[i] + beta * rndm1[i + 1] + gamma * rndm1[i])
            simulated_error1.append(max(err, wind_in[i] - max_wind))
        simulated_error = simulated_error1[1:]
        self.sim_error = simulated_error
        self.sim_hourly = wind_in - pd.Series(self.sim_error)


    def forecast_errors_multi_area(self):
        cmatrix = [0.99207, 0.08265, 0.00983, -0.00010, 0.00930, 0.01556, 0.01235, -0.00722, 0.01644, 0.01020, 0.08057,
                   0.08220, 0.98660, 0.12096, -0.00596, 0.01524, 0.01402, 0.03527, 0.00458, 0.01697, 0.01015, 0.03881,
                   0.00934, 0.11547, 0.94175, 0.30033, 0.03309, 0.01902, 0.01832, 0.01330, 0.01976, 0.00506, 0.07714,
                   -0.00009, -0.00569, 0.30016, 0.94122, 0.01129, 0.02590, 0.01765, 0.03180, 0.01028, 0.11595, 0.07930,
                   0.00891, 0.01471, 0.03360, 0.01150, 0.95704, 0.02008, 0.10739, -0.02383, 0.25388, 0.02550, -0.03658,
                   0.01524, 0.01385, 0.01964, 0.02675, 0.02038, 0.97310, 0.04244, 0.01142, 0.21791, -0.00464, 0.02219,
                   0.01192, 0.03425, 0.01867, 0.01795, 0.10753, 0.04178, 0.95812, -0.00904, 0.24329, 0.04245, -0.06588,
                   -0.00727, 0.00464, 0.01408, 0.03374, -0.02490, 0.01171, -0.00942, 0.99798, -0.00175, -0.00062, 0.00914,
                   0.01521, 0.01579, 0.01924, 0.00999, 0.24336, 0.20546, 0.23294, -0.00161, 0.91742, 0.01587, -0.01476,
                   0.01020, 0.01018, 0.00530, 0.12198, 0.02636, -0.00471, 0.04385, -0.00060, 0.01710, 0.99027, -0.00717,
                   0.07997, 0.03874, 0.08068, 0.08297, -0.03759, 0.02248, -0.06769, 0.00900, -0.01578, -0.00714, 0.98499
                   ]
        areas = ['SE1', 'SE2', 'SE3', 'SE4', 'NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'DK2', 'FI']
        area_to_idx = {
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

        wind_in = self.wind
        variances = pd.DataFrame(columns=areas)
        rndms = pd.DataFrame(columns=areas)
        max_wind = pd.DataFrame(columns=areas, index=[0])
        rel_wind_in = pd.DataFrame(columns=areas)

        for a in areas:
            max_wind[a][0] = wind_in[a].max() / 0.9
            rel_wind_in[a] = wind_in[a] / max_wind[a][0]
            length = len(wind_in[a])
            rndm = []
            variance = []
            variance.append(0)
            variance.extend((self.a * (self.b ** rel_wind_in[a].tolist()) + self.c) * wind_in[a].tolist())
            rndm.append(0)
            for i in range(length):
                rndm.append(random.gauss(0, 1))
            variances[a] = variance
            rndms[a] = rndm

        self.sim_errors = pd.DataFrame(columns=areas)
        self.sim_errors_norm = pd.DataFrame(columns=areas)
        corr_rndms = pd.DataFrame(columns=areas)
        self.actual_hourly = pd.DataFrame(columns=areas)
        for a in areas:
            lst = []
            for i in range(len(rndms[a])):
                val = 0
                for aa in areas:
                    val = val + rndms[aa][i] * cmatrix[self.area_to_idx[a] * 11 + self.area_to_idx[aa]]
                if i == 0:
                    lst.append(val * variances[a][i])
                else:
                    lst.append(val * variances[a][i])
            corr_rndms[a] = lst
            simulated_error = [0]
            for i in range(len(wind_in[a])):
                err = min(wind_in[a][i],
                          self.alpha[self.area_to_idx[a]] * simulated_error[i] + self.beta[self.area_to_idx[a]] * corr_rndms[a][i + 1] +
                          self.gamma[self.area_to_idx[a]] * corr_rndms[a][i])
                simulated_error.append(max(err, wind_in[a][i] - max_wind[a][0]))
            self.sim_errors[a] = simulated_error[1:]
            self.sim_errors_norm[a] = self.sim_errors[a].div(wind_in[a])
            self.actual_hourly[a] = wind_in[a] - self.sim_errors[a]

    def spline_hour_2_min(self, time, data):
        """For hourly data"""
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
        splines = interpolate.splrep(x_pts, y_pts)
        wind_spline = interpolate.splev(x_vals, splines)
        # Gör datumnkolumn med splines
        a = time[0]
        highres_time = []
        for i in range(60 * length):
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
        y_norm = spline['Generation'] / (data.max() / 0.9)
        y_norm = y_norm[y_norm.index % 3 ==0]
        y_norm = y_norm.tolist()

        a = area

        """Gör här en dataframe med 3-minutsupplösning"""
        xx = []
        for i in range(len(y) * 20):
            xx.append(x[0] + timedelta(minutes=i*3))

        xxx = spline['Timestamp'].tolist()
        df = pd.read_csv('noise.csv')
        noise = df['Noise']

        ar = []
        beta = [1.41, 1.38, 1.37, 1.5, 0.87, 1.37, 1.41, 1.27, 1.48, 0.94, 1.29]
        rhos = [0.832152, 0.04151, -0.029447, -0.021741, -0.012194, -0.016159, -0.014461, 0.002568, -0.015159, -0.041042]
        for i in range(len(xx)):
            if y_norm[i] < 0.05:
                std = 0.045 - 0.616 * y_norm[i]
            else:
                std = 0.015 - 0.016 * y_norm[i]
            if i == 0:
                ar.append(0)
            elif i <=10 and i>0:
                summ = 0
                for n in range(i):
                    summ = summ + rhos[n] * ar[i - n - 1]
                ar.append(summ + random.gauss(0, std) * beta[self.area_to_idx[a]] * spline_list[i])
            else:
                summ = 0
                for n in range(10):
                    summ = summ + rhos[n] * ar[i - n - 1]
                ar.append(summ + random.gauss(0, std) * beta[self.area_to_idx[a]] * spline_list[i])

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
            self.dict[a]['Minutely'] = self.variability_simulation(self.dict[a]['Hourly']['Actual'], a)

    def make_single_area_highres_scenario(self, area):
        self.dict = {} # Hourly: Time, Forecasted, Forecast error, Actual; Minutely: Time, Spline, Noise, Actual
        self.forecast_errors_single_area(area=area)
        self.dict['Hourly'] = pd.DataFrame(columns=['Time', 'Forecasted', 'Forecast error', 'Actual'])
        self.dict['Hourly']['Time'] = self.time_list
        self.dict['Hourly']['Forecasted'] = self.wind[area].tolist()
        self.dict['Hourly']['Forecast error'] = self.sim_error
        self.dict['Hourly']['Actual'] = self.sim_hourly.tolist()
        self.dict['Minutely'] = self.variability_simulation(self.dict['Hourly']['Actual'], area)


class Wind_Model_FI:
    def __init__(self):
        with open('fingrid_wind_2021.pickle', 'rb') as handle:
            self.wind = pkl.load(handle)
        self.hourly_wind = self.wind['Hourly']
        self.highres_wind = self.wind['Real time']
        self.a = np.array(0.2867689816793785)
        self.b = np.array(0.20991242725821563)
        self.c = np.array(-0.05329247460791067)
        self.alpha = 0.7421
        self.beta = 0.6359
        self.gamma = 0.5505
        self.time_list = []
        n = -1
        start_point = datetime.strptime(f'2021-01-01 00:00', '%Y-%m-%d %H:%M')
        for i in range(len(self.hourly_wind)):
            n = n + 1
            date = start_point + timedelta(hours=n)
            if date.strftime('%m-%d') == '02-29':
                n = n + 24
            elif date.strftime('%m-%d') == '12-31':
                n = n + 24
            self.time_list.append(start_point + timedelta(hours=n))


    def forecast_errors_single_area(self):
        wind_in = self.hourly_wind['Intra-day forecast']
        length = len(wind_in)
        max_wind = wind_in.max() / 0.9
        rel_wind_in = wind_in / max_wind
        variance = self.a * (self.b ** rel_wind_in.tolist()) + self.c
        rndm1 = []
        rndm1.append(0)
        for i in range(length):
            rndm1.append(random.gauss(0, float(variance[i])) * wind_in[i])
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma
        simulated_error1 = []
        simulated_error1.append(0)
        for i in range(len(wind_in)):
            err = min(wind_in[i], alpha * simulated_error1[i] + beta * rndm1[i + 1] + gamma * rndm1[i])
            simulated_error1.append(max(err, wind_in[i] - max_wind))
        simulated_error = simulated_error1[1:]
        self.sim_error = simulated_error
        self.sim_hourly = wind_in - pd.Series(self.sim_error)

    def spline_hour_2_min(self, time, data):
        """For hourly data"""
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
        splines = interpolate.splrep(x_pts, y_pts)
        wind_spline = interpolate.splev(x_vals, splines)
        # Gör datumnkolumn med splines
        a = time[0]
        highres_time = []
        for i in range(60 * length):
            highres_time.append(a + timedelta(minutes=i))
        wind_interpol = pd.DataFrame(columns=['Timestamp', 'Generation'])
        wind_interpol['Timestamp'] = highres_time
        wind_interpol['Generation'] = wind_spline
        wind_interpol['Timestamp'] = pd.to_datetime(wind_interpol['Timestamp'], utc=True, format='%Y-%m-%d')
        return wind_interpol

    def variability_simulation(self, data):
        x = self.time_list
        y = data
        spline = self.spline_hour_2_min(x, y.tolist())
        spline.loc[spline['Generation'] < 0, 'Generation'] = 0
        y_norm = spline['Generation'] / (data.max() / 0.9)


        """Gör här en dataframe med 3-minutsupplösning"""
        xx = []
        for i in range(len(y) * 20):
            xx.append(x[0] + timedelta(minutes=i * 3))
        xxx = spline['Timestamp'].tolist()
        df = pd.read_csv('noise.csv')
        noise = df['Noise']

        ar = []
        beta = 1.28
        rhos = [0.832152, 0.04151, -0.029447, -0.021741, -0.012194, -0.016159, -0.014461, 0.002568, -0.015159,
                -0.041042]
        for i in range(len(xx)):
            if y_norm[i] < 0.05:
                std = 0.045 - 0.616 * y_norm[i]
            else:
                std = 0.015 - 0.016 * y_norm[i]
            if i == 0:
                ar.append(0)
            elif i <= 10 and i > 0:
                sum = 0
                for n in range(i):
                    sum = sum + rhos[n] * ar[i - n - 1]
                ar.append(sum + random.gauss(0, std) * beta * spline['Generation'][i])
            else:
                sum = 0
                for n in range(10):
                    sum = sum + rhos[n] * ar[i - n - 1]
                ar.append(sum + random.gauss(0, std) * beta * spline['Generation'][i])

        minute_highfreq = []
        for i in range(len(xx)):
            if i == len(xx) - 1:
                minute_highfreq.append(ar[i])
                minute_highfreq.append(ar[i] + (ar[i] - ar[i - 1]) * 1 / 3)
                minute_highfreq.append(ar[i] + (ar[i] - ar[i - 1]) * 2 / 3)
            else:
                minute_highfreq.append(ar[i])
                minute_highfreq.append(ar[i] + (ar[i + 1] - ar[i]) * 1 / 3)
                minute_highfreq.append(ar[i] + (ar[i + 1] - ar[i]) * 2 / 3)
        highres = pd.DataFrame(columns=['Time', 'Spline', 'Noise', 'Generation'])
        highres['Time'] = xxx
        highres['Generation'] = spline['Generation'] + pd.Series(minute_highfreq)
        highres.loc[highres['Generation'] < 0, 'Generation'] = 0
        highres['Spline'] = spline['Generation'].tolist()
        highres['Noise'] = minute_highfreq
        return highres

    def make_highres_scenario(self):
        self.dict = {} # Hourly: Time, Forecasted, Forecast error, Actual; Minutely: Time, Spline, Noise, Actual
        self.forecast_errors_single_area()
        self.dict['Hourly'] = pd.DataFrame(columns=['Time', 'Forecasted', 'Forecast error', 'Actual'])
        self.dict['Hourly']['Time'] = self.time_list
        self.dict['Hourly']['Forecasted'] = self.hourly_wind['Intra-day forecast'].tolist()
        self.dict['Hourly']['Forecast error'] = self.sim_error
        self.dict['Hourly']['Actual'] = self.sim_hourly.tolist()
        self.dict['Minutely'] = self.variability_simulation(self.dict['Hourly']['Actual'])


def make_finland_plot():
    # start_time = datetime.strptime(f'2021-04-30 00:00', '%Y-%m-%d %H:%M')
    # end_time = datetime.strptime(f'2021-05-01 00:00', '%Y-%m-%d %H:%M')
    start_time = pd.Timestamp(f'2021-05-01 00:00', tz='UTC')
    end_time = pd.Timestamp(f'2021-05-03 00:00', tz='UTC')
    for i in range(10):
        hej = Wind_Model_FI()
        hej.make_highres_scenario()
        idx_start = hej.dict['Minutely']['Time'].searchsorted(start_time)
        idx_end = hej.dict['Minutely']['Time'].searchsorted(end_time)
        if i == 0:
            plt.plot(hej.dict['Minutely']['Time'][idx_start:idx_end].tolist(), hej.dict['Minutely']['Generation'][idx_start:idx_end], color='orange', label='Simulated scenarios', linewidth=1)
        else:
            plt.plot(hej.dict['Minutely']['Time'][idx_start:idx_end].tolist(), hej.dict['Minutely']['Generation'][idx_start:idx_end], color='orange', linewidth=1)

    hej = Wind_Model_FI()
    idx_start = hej.highres_wind['Timestamp'].searchsorted(start_time)
    idx_end = hej.highres_wind['Timestamp'].searchsorted(end_time)
    plt.plot(hej.highres_wind['Timestamp'][idx_start:idx_end].tolist(), hej.highres_wind['Real wind'][idx_start:idx_end], color='tab:blue', label='Real wind power', linewidth=1)
    idx_start = hej.hourly_wind['Timestamp'].searchsorted(start_time)
    idx_end = hej.hourly_wind['Timestamp'].searchsorted(end_time)
    plt.step(hej.hourly_wind['Timestamp'][idx_start:idx_end].tolist(), hej.hourly_wind['Intra-day forecast'][idx_start:idx_end], color='black', where='post', linewidth=2, label='Intra-day forecast')
    # Generate confidence intervals
    real_list = []
    forecast_list = []
    for i in range(len(hej.highres_wind)):
        real_list.append(hej.highres_wind['Real wind'][i])
        min = hej.hourly_wind['Timestamp'].searchsorted(hej.highres_wind['Timestamp'][i])
        if min >= len(hej.hourly_wind):
            forecast_list.append(hej.hourly_wind['Intra-day forecast'][len(hej.hourly_wind) - 1])
        else:
            forecast_list.append(hej.hourly_wind['Intra-day forecast'][min - 1])
    df = pd.DataFrame(columns=['real', 'id'])
    df['real'] = real_list
    df['id'] = forecast_list
    df = df.sort_values(by=['id'])
    length = int(len(df) / 10)
    last = []
    first = []
    median = []
    median.append(0)
    first.append(0)
    for i in range(10):
        if i == 9:
            first.append(df['real'][i * length:].quantile(q =0.01))
            last.append(df['real'][i * length:].quantile(q =0.99))
            median.append(df['id'][i * length:].median())
            first.append(df['real'][i * length:].quantile(q =0.01))
            last.append(df['real'][i * length:].quantile(q =0.99))
            median.append(df['id'][i * length:].max())
        elif i == 1:
            first.append(df['real'][i * length: (i + 1) * length].quantile(q =0.01))
            last.append(df['real'][i * length: (i + 1) * length].quantile(q =0.99))
            last.append(df['real'][i * length: (i + 1) * length].quantile(q=0.99))
            median.append(df['id'][i * length: (i + 1) * length].median())
        else:
            first.append(df['real'][i * length: (i + 1) * length].quantile(q =0.01))
            last.append(df['real'][i * length: (i + 1) * length].quantile(q =0.99))
            median.append(df['id'][i * length: (i + 1) * length].median())

    first_spline = interpolate.splrep(median, first)
    last_spline = interpolate.splrep(median, last)
    first = interpolate.splev(hej.hourly_wind['Intra-day forecast'][idx_start:idx_end], first_spline)
    last = interpolate.splev(hej.hourly_wind['Intra-day forecast'][idx_start:idx_end], last_spline)
    plt.plot(hej.hourly_wind['Timestamp'][idx_start:idx_end].tolist(), first, color='black', linewidth=2, linestyle='--', label='1st percentile')
    plt.plot(hej.hourly_wind['Timestamp'][idx_start:idx_end].tolist(), last, color='black', linewidth=2,
             linestyle='--', label='99th percentile')
    plt.xticks(rotation=30, ha="right")
    plt.xlabel('Time')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=4))
    plt.ylabel('Power [MW]')
    plt.xlim(hej.hourly_wind['Timestamp'][idx_start], hej.hourly_wind['Timestamp'][idx_end])
    plt.legend()
    plt.tight_layout()
    plt.grid()
    fig = plt.gcf()
    save = True
    if save:
        fig.savefig(
            f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\Paper2\\scenarios_2.pdf',
            dpi=fig.dpi, pad_inches=0, bbox_inches='tight')

    plt.show()

def make_all_years():
    dict = {}
    for y in range(1982,2017):
        hej = Wind_Model(year_start=y, week_start=1, day_start=0, sim_days=52*7)
        hej.make_multi_area_highres_scenario()
        dict[y] = hej.dict
        print(y)
        with open('C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\Paper2\\year_dic_2.pickle', 'wb') as handle:
            pkl.dump(dict, handle, protocol=pkl.HIGHEST_PROTOCOL)

def make_finland_distributionplot():
    error_list = []
    for i in range(5):
        hej = Wind_Model_FI()
        hej.make_highres_scenario()
        error_list.extend(hej.dict['Hourly']['Forecast error'].tolist())
    error_list = pd.Series(error_list)

    fig, axs = plt.subplots(1,2, sharey=True)
    sns.histplot(error_list, ax=axs[0], stat='probability', bins=30, kde=True, label='nolegend')#, label=f'\u03BC = {round(error_list.mean(),2)} \n' + r'$\sigma^{2} = $' + f'{round(error_list.std(),2)} \nSkew. = {round(error_list.skew(),2)}\nKurt. = {round(error_list.kurt(),2)}')
    axs[0].legend([f'\u03BC = {round(error_list.mean(),2)} \n' + r'$\sigma^{2} = $' + f'{round(error_list.std(),2)} \nSkew. = {round(error_list.skew(),2)}\nKurt. = {round(error_list.kurtosis(),2)}'], fontsize='small', loc='upper left')
    axs[0].set_xlabel('Forecast error [MW]')
    axs[1].set_xlabel('Forecast error [MW]')
    axs[0].set_title('Simulated')
    axs[1].set_title('Actual')
    axs[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    #axs['l'].get_shared_x_axes().join(axs['r'], axs['l'])
    fe = hej.hourly_wind['Forecast error']
    sns.histplot(fe, ax=axs[1], stat='probability', bins=30, kde=True)#, label=f'\u03BC = {round(fe.mean(),2)} \n' + r'$\sigma^{2} = $' + f'{round(fe.std(),2)} \nSkew. = {round(fe.skew(),2)}\nKurt. = {round(fe.kurt(),2)}')
    axs[1].legend([f'\u03BC = {round(fe.mean(),2)} \n' + r'$\sigma^{2} = $' + f'{round(fe.std(),2)} \nSkew. = {round(fe.skew(),2)}\nKurt. = {round(fe.kurtosis(),2)}'], fontsize='small', loc='upper left')
    fig.tight_layout()
    fig = plt.gcf()
    fig.set_figheight(4)
    save = True
    if save:
        fig.savefig(
            f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\Paper2\\distributions.pdf',
            dpi=fig.dpi, pad_inches=0, bbox_inches='tight')

    plt.show()

def analyze_wind_gen():
    mean_df = pd.DataFrame(index=list(range(1982,2017)),columns=('SE1', 'SE2', 'SE3', 'SE4', 'NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'DK2', 'FI'))
    max_df = pd.DataFrame(index=list(range(1982, 2017)),
                          columns=('SE1', 'SE2', 'SE3', 'SE4', 'NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'DK2', 'FI'))
    for a in ('SE1', 'SE2', 'SE3', 'SE4', 'NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'DK2', 'FI'):
        mean_lst = []
        max_lst = []
        for y in range(1982, 2017):
            wind = Wind_Model(year_start=y, week_start=1, day_start=0, sim_days=52*7)
            mean_lst.append(wind.wind[a].mean())
            max_lst.append(wind.wind[a].max())
        mean_df[a] = mean_lst
        max_df[a] = max_lst
    mean_df.to_csv('mean_wind.csv')
    max_df.to_csv('max_wind.csv')

make_finland_plot()
