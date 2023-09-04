"""
This file should be used for tuning of wind forecast models.
File can be changed as the file later in operation should be "WindModel.py"
"""

import pandas as pd
from RESshares import res_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
import scipy.interpolate as interpolate
from datetime import datetime, timedelta
import random
import scipy.optimize as spo
import sqlite3
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter
from pandas.plotting import autocorrelation_plot
import calendar

class Wind_Model_Tuning:
    def __init__(self, scenario='EF45', year=2003):
        self.scenario = scenario
        if self.scenario == 'SF45':
            self.alpha = [0.934895, 0.831286, 0.942514, 0.840097, 0.853451, 0.929322, 0.872953,
                          0.950803, 0.979522, 0.969189, 0.941636] #Real NO4 0.738853
            self.beta = [1.291769, 0.861607, 0.806465, 0.930376, 0.864449, 1.305166, 0.841179,
                         0.745798, 1.452029, 1.408425, 0.738612] #Real NO4 0.052328
            self.gamma = [-0.15392, 0.650779, -0.18537, 0.533644, 0.47935, -0.08849, 0.244863,
                          -0.21114, -0.60998, -0.44410, -0.16041] #Real NO4 -0.09625
            self.year = 1988
            self.abs_target = {
                'SE1': 0.190583,
                'SE2': 0.161775,
                'SE3': 0.119668,
                'SE4': 0.156333,
                'NO1': 0.196868,
                'NO2': 0.208652,
                'NO3': 0.159452,
                'NO4': 0.121213, #Given same as FI - model becomes unrealistic
                'NO5': 0.20276,
                'DK2': 0.196406,
                'FI': 0.121213
            }
            self.std_target = {
                'SE1': 0.259310,
                'SE2': 0.216098,
                'SE3': 0.152938,
                'SE4': 0.207935,
                'NO1': 0.268738,
                'NO2': 0.286414,
                'NO3': 0.212613,
                'NO4': 0.155254,
                'NO5': 0.277576,
                'DK2': 0.268044,
                'FI': 0.155254
            }
            self.var_target = {
                'SE1': 0.081556,
                'SE2': 0.069963,
                'SE3': 0.053017,
                'SE4': 0.067773,
                'NO1': 0.084085,
                'NO2': 0.088828,
                'NO3': 0.069028,
                'NO4': 0.053639,
                'NO5': 0.086457,
                'DK2': 0.083899,
                'FI': 0.053639
            }
            self.cmatrix = pd.read_csv(
                f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\ModelTuning\\Wind{self.scenario}\\cmatrix.csv')
            self.ar_params = [1.4, 1.35, 1.38, 1.44, 1.04, 1.31, 1.2, 1.23, 1.54, 1.6, 1.22] #Area parameters
        elif self.scenario == 'EP45':
            self.alpha = [0.843239, 0.973591, 0.950928, 0.934952, 0.933028, 0.912425, 0.829976,
                          0.949770, 0.946069, 0.843861, 0.960735]
            self.beta = [1.207018, 1.022957, 0.676614, 1.170874, 0.950841, 1.365844, 0.888299,
                         0.649912, 1.502175, 0.847627, 0.613006]
            self.gamma = [0.726314, -0.42158, -0.22389, -0.15960, -0.12175, 0.011031, 0.521714,
                          -0.22420, -0.26599, 0.503617, -0.25484]
            self.year = 1994
            self.abs_target = {
                'SE1': 0.194837,
                'SE2': 0.158342,
                'SE3': 0.095906,
                'SE4': 0.152878,
                'NO1': 0.179793,
                'NO2': 0.196598,
                'NO3': 0.15849,
                'NO4': 0.105514,
                'NO5': 0.188196,
                'DK2': 0.188407,
                'FI': 0.105514
            }
            self.std_target = {
                'SE1': 0.265690,
                'SE2': 0.210948,
                'SE3': 0.117295,
                'SE4': 0.202752,
                'NO1': 0.243124,
                'NO2': 0.268333,
                'NO3': 0.211170,
                'NO4': 0.131706,
                'NO5': 0.255728,
                'DK2': 0.256046,
                'FI': 0.131706
            }
            self.var_target = {
                'SE1': 0.083268,
                'SE2': 0.068581,
                'SE3': 0.043454,
                'SE4': 0.066382,
                'NO1': 0.077214,
                'NO2': 0.083977,
                'NO3': 0.068641,
                'NO4': 0.047321,
                'NO5': 0.080595,
                'DK2': 0.080680,
                'FI': 0.047321
            }
            self.cmatrix = pd.read_csv(
                f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\ModelTuning\\Wind{self.scenario}\\cmatrix.csv')
            self.ar_params = [1.43, 1.4, 1.51, 1.63, 1.08, 1.45, 1.33, 1.28, 1.64, 1.05, 1.2] #Tuned area-dependent parameters
        elif self.scenario == 'EF45':
            self.alpha = [0.951805, 0.944725, 0.979481, 0.929522, 0.949210, 0.857854, 0.879416,
                          0.988339, 0.926101, 0.928169, 0.974702]
            self.beta = [1.398954, 1.002785, 0.591806, 0.958470, 0.851685, 1.109591, 1.006466,
                         0.620911, 1.333323, 0.920377, 0.580100]
            self.gamma = [-0.29505, -0.20017, -0.32964, -0.11249, -0.17821, 0.431928, 0.196483,
                          -0.44510, -0.08643, -0.08927, -0.30336]
            self.year = 2003
            self.abs_target = {
                'SE1': 0.196271,
                'SE2': 0.155335,
                'SE3': 0.088251,
                'SE4': 0.149421,
                'NO1': 0.177006,
                'NO2': 0.184511,
                'NO3': 0.156956,
                'NO4': 0.089685,
                'NO5': 0.180758,
                'DK2': 0.186621,
                'FI': 0.089685
            }
            self.std_target = {
                'SE1': 0.267842,
                'SE2': 0.206438,
                'SE3': 0.105811,
                'SE4': 0.197567,
                'NO1': 0.238944,
                'NO2': 0.250201,
                'NO3': 0.208869,
                'NO4': 0.107963,
                'NO5': 0.244573,
                'DK2': 0.253367,
                'FI': 0.107963
            }
            self.var_target = {
                'SE1': 0.083845,
                'SE2': 0.067371,
                'SE3': 0.040374,
                'SE4': 0.064991,
                'NO1': 0.076092,
                'NO2': 0.079112,
                'NO3': 0.068023,
                'NO4': 0.040951,
                'NO5': 0.077602,
                'DK2': 0.079962,
                'FI': 0.040951
            }
            self.cmatrix = pd.read_csv(
                f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\ModelTuning\\Wind{self.scenario}\\cmatrix.csv')
            self.ar_params = [1.46, 1.3, 1.44, 1.27, 1.01, 1.35, 1.36, 1.26, 1.53, 1, 1.3] #Tuned area-dependent parameters
        elif self.scenario == 'FM45':
            self.alpha = [0.933949, 0.918861, 0.957918, 0.847212, 0.936454, 0.940539, 0.921079,
                          0.898305, 0.962937, 0.938693, 0.967665]
            self.beta = [1.19578, 1.000932, 0.738011, 1.026096, 0.94883, 1.240763, 0.972874,
                         0.67440, 1.447215, 1.011808, 0.657482]
            self.gamma = [-0.1383, -0.06379, -0.26098, 0.471461, -0.12233, -0.18141, -0.07203,
                          -0.00066, -0.41931, -0.14949, -0.26736]
            self.year = 1998
            self.abs_target = {
                'SE1': 0.190583,
                'SE2': 0.160275,
                'SE3': 0.104921,
                'SE4': 0.150474,
                'NO1': 0.183456,
                'NO2': 0.200197,
                'NO3': 0.158810,
                'NO4': 0.110273,
                'NO5': 0.191826,
                'DK2': 0.189520,
                'FI': 0.110273
            }
            self.std_target = {
                'SE1': 0.259310,
                'SE2': 0.213848,
                'SE3': 0.130817,
                'SE4': 0.199146,
                'NO1': 0.248619,
                'NO2': 0.273730,
                'NO3': 0.211650,
                'NO4': 0.138844,
                'NO5': 0.261174,
                'DK2': 0.257715,
                'FI': 0.138844
            }
            self.var_target = {
                'SE1': 0.081556,
                'SE2': 0.069359,
                'SE3': 0.047082,
                'SE4': 0.065414,
                'NO1': 0.078688,
                'NO2': 0.085425,
                'NO3': 0.068769,
                'NO4': 0.049236,
                'NO5': 0.082056,
                'DK2': 0.081128,
                'FI': 0.049236
            }
            self.cmatrix = pd.read_csv(
                f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\ModelTuning\\Wind{self.scenario}\\cmatrix.csv')
            self.ar_params = [1.28, 1.31, 1.46, 1.52, 1.06, 1.28, 1.26, 1.25, 1.57, 1.06, 1.21] #Tuned area-dependent parameters
        self.corr_matrix = pd.read_csv(
            f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\WindCorr_{scenario}.csv',
            index_col=[0])
        self.areas = ('SE1', 'SE2', 'SE3', 'SE4', 'NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'DK2', 'FI')
        #self.year = year
        self.db_path = f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Simuleringsdata\\LMA21_{scenario}.db'
        self.start_date = datetime.strptime(f'{self.year}-01-01', '%Y-%m-%d')
        self.timestamps = [self.start_date + timedelta(hours=24 * d + h) for d in range(52 * 7) for h in range(24)]
        if calendar.isleap(self.year):
            del self.timestamps[(31 + 28) * 24:(31 + 29) * 24]
            self.timestamps.extend([self.timestamps[-1] + timedelta(hours=h) for h in range(1, 25)])
        self.timestamps_str = [datetime.strftime(d, '%Y-%m-%d %H:%M') for d in self.timestamps]
        self.rhos = [0.832152, 0.04151, -0.029447, -0.021741, -0.012194, -0.016159, -0.014461, 0.002568,
                     -0.015159, -0.041042] #AR parameters to have same initial autocorrelation as Finnish WP from 2021

    def read_data(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        area_dict = {}
        for area in self.areas:
            table = f'Generation_{area}_{self.year}'
            c.execute("SELECT * FROM {0}".format(table))
            area_dict[f'{area}_Gen'] = c.fetchall()
            area_dict[f'{area}_GenCols'] = list(map(lambda x: x[0], c.description))

        # Available RES
        self.resav = pd.DataFrame(columns=self.areas, index=self.timestamps_str)
        for area in self.areas:
            idx = area_dict[f'{area}_GenCols'].index('RES')
            self.resav[area] = [area_dict[f'{area}_Gen'][i][idx] for i in range(self.timestamps_str.__len__())]

        # Curtailed RES
        self.curtail = pd.DataFrame(columns=self.areas, index=self.timestamps_str)
        for area in self.areas:
            idx = area_dict[f'{area}_GenCols'].index('Curtailments')
            self.curtail[area] = [area_dict[f'{area}_Gen'][i][idx] for i in
                                  range(self.timestamps_str.__len__())]

        # RES generation
        self.res = pd.DataFrame(columns=self.areas, index=self.timestamps_str)
        for area in self.areas:
            if area in ['NO2', 'NO4', 'NO5']:  # In these areas RES curtail seems to be hydro
                self.res[area] = self.resav[area]
            else:
                self.res[area] = self.resav[area] + self.curtail[area]
        self.res[self.res < 0] = 0

        # Split RES
        res_dict = res_split(data=self.res, scenario=self.scenario, days=52 * 7, year=self.year)
        self.pv = pd.DataFrame(columns=self.areas, index=self.timestamps_str)
        self.offsh = pd.DataFrame(columns=self.areas, index=self.timestamps_str)
        self.onsh = pd.DataFrame(columns=self.areas, index=self.timestamps_str)
        self.wind = pd.DataFrame(columns=self.areas, index=self.timestamps_str)
        for a in self.areas:
            self.pv[a] = res_dict[a]['SolarPV'].tolist()
            self.offsh[a] = res_dict[a]['WindOffshore'].tolist()
            self.onsh[a] = res_dict[a]['WindOnshore'].tolist()
            self.wind[a] = self.offsh[a] + self.onsh[a]

    def stdev_function(self, norm_gen):
        if norm_gen <= 0.0265:
            return 1.3980518092749208 - 39.88632824668849 * norm_gen
        elif norm_gen > 0.0265 and norm_gen <= 0.12:
            return 0.38382298051111785 - 1.8451357727662174 * norm_gen
        elif norm_gen > 0.12 and norm_gen <= 0.59:
            return 0.1991465324867623 - 0.2533242669906466 * norm_gen
        elif norm_gen > 0.59:
            return 0.11779102907699517 - 0.1054201676089161 * norm_gen

    def tune_singlearea_arma_model(self, area):
        a = area
        wind_in = self.wind[a]

        # MAKE 5 SERIES WITH RANDOM VARIABLES
        length = len(wind_in)
        max_wind = wind_in.max() / 0.9
        rel_wind_in = wind_in / max_wind
        std = []
        for w in rel_wind_in.tolist(): # computing a list of standard deviations
            std.append(self.stdev_function(w))

        rndms = {
            0: [0],
            1: [0],
            2: [0],
            3: [0],
            4: [0]
        } #

        for i in range(length): #This is omega PF parameter in Eq (4)
            for n in range(5):
                rndms[n].append(random.gauss(0, float(std[i])) * wind_in[i])

        def f(abg): #This is the optimization function in Eq (7)
            alpha = abg[0]
            beta = abg[1]
            gamma = abg[2]

            simulated_error = {
            0: [0],
            1: [0],
            2: [0],
            3: [0],
            4: [0]
            }

            for i in range(length):
                for n in range(5):
                    error = min(wind_in[i], alpha * simulated_error[n][i] + beta * rndms[n][i + 1] + gamma * rndms[n][i])
                    """Calculating error as minimum of ARMA model and forecasted wind. Overprediction CANNOT be larger
                    than forecasted generation. I.e. negative production not possible"""
                    simulated_error[n].append(max(error, wind_in[i] - max_wind))
                    """Calculating simulated error as maximum of error and difference between forecasted wind
                    and installed capacity. Underprediction CANNOT be smaller than difference between forecasted
                    wind and maximum wind. I.e. higher prod than installed capacity not possible"""

            simulated_error_actual = {}
            target1 = {}
            target2 = {}
            target3 = {}
            mse = 0
            for n in range(5):
                simulated_error_actual[n] = simulated_error[n][1:]
                target1[n] = float(pd.DataFrame(simulated_error_actual[n]).abs().sum() / wind_in.sum()) #Eq (6a)
                target2[n] = float(pd.DataFrame(simulated_error_actual[n]).std() / wind_in.mean()) #Eq (6b)
                target3[n] = float(np.sum(np.abs(np.subtract(np.array(simulated_error[n][:-1]),
                                                             np.array(simulated_error_actual[n])))) / wind_in.sum()) #Eq (6c)
                mse += float(np.sqrt(((target1[n] - self.abs_target[a]) / self.abs_target[a]) ** 2) +
                             np.sqrt(((target2[n] - self.std_target[a]) / self.std_target[a]) ** 2) +
                             np.sqrt(((target3[n] - self.var_target[a]) / self.var_target[a]) ** 2)) #These are normalized with the size of the target
            print(mse)
            return mse

        start = [0.7, 0.75, -0.5]
        result = spo.minimize(f, start, method='Nelder-Mead', options={'disp': True})
        abg = result.x
        self.obj = result.fun
        self.alpha = abg[0]
        self.beta = abg[1]
        self.gamma = abg[2]
        print(f'alpha = {str(self.alpha)}')
        print(f'beta = {str(self.beta)}')
        print(f'gamma = {str(self.gamma)}')
        print(f'obj = {str(self.obj)}')

        simulated_error = {
            0: [0],
            1: [0],
            2: [0],
            3: [0],
            4: [0]
        }

        for i in range(length):
            for n in range(5):
                error = min(wind_in[i], self.alpha * simulated_error[n][i] + self.beta * rndms[n][i + 1] + self.gamma * rndms[n][i])
                """Calculating error as minimum of ARMA model and forecasted wind. Overprediction CANNOT be larger
                than forecasted generation. I.e. negative production not possible"""
                simulated_error[n].append(max(error, wind_in[i] - max_wind))
                """Calculating simulated error as maximum of error and difference between forecasted wind
                and installed capacity. Underprediction CANNOT be smaller than difference between forecasted
                wind and maximum wind. I.e. higher prod than installed capacity not possible"""

        simulated_error_actual = {}
        self.sim_target1 = {}
        self.sim_target2 = {}
        self.sim_target3 = {}
        print(f'Target 1: {self.abs_target[a]}')
        print(f'Target 2: {self.std_target[a]}')
        print(f'Target 3: {self.var_target[a]}')
        for n in range(5):
            simulated_error_actual[n] = simulated_error[n][1:]
            self.sim_target1[n] = float(pd.DataFrame(simulated_error_actual[n]).abs().sum() / wind_in.sum())
            self.sim_target2[n] = float(pd.DataFrame(simulated_error_actual[n]).std() / wind_in.mean())
            self.sim_target3[n] = float(np.sum(np.abs(np.subtract(np.array(simulated_error[n][:-1]),
                                                         np.array(simulated_error_actual[n])))) / wind_in.sum())
        print(f'Simulated Target 1: {self.sim_target1[0]}')
        print(f'Simulated Target 2: {self.sim_target2[0]}')
        print(f'Simulated Target 3: {self.sim_target3[0]}')
            #Distribution plot
            # sns.displot(simulated_error_actual[n], label=f'1 = {round(float(target1[n]), 3)}\n2 = {round(float(target2[n]), 3)}\n3 = {round(float(target3[n]), 3)}')
            # plt.legend()
            # plt.show()

            # #Scatter plot
            # x = wind_in.tolist()
            # y = simulated_error[n][1:]
            # plt.scatter(x, y)
            # plt.grid()
            # plt.show()


    def tune_multiarea_arma_model(self):
        plot_act_corr = False
        make_errors = False
        self.read_data()
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

        correlations = self.corr_matrix

        if plot_act_corr:
            sns.heatmap(correlations, cmap='PuBu', annot=True, center=0.1)

        stds = pd.DataFrame(columns=self.areas)
        rndms = pd.DataFrame(columns=self.areas)
        max_winds = pd.DataFrame(columns=self.areas, index=[0])
        rel_wind_in = pd.DataFrame(columns=self.areas)

        for a in self.areas:
            max_winds[a][0] = self.wind[a].max() / 0.9
            rel_wind_in[a] = self.wind[a] /max_winds[a][0]
            length = rel_wind_in[a].__len__()
            rndm = []
            std = []
            rndm.append(0)

            for w in rel_wind_in[a].tolist():  # computing a list of standard deviations
                std.append(self.stdev_function(w))

            for i in range(length):
                rndm.append(random.gauss(0, 1))

            std_real = [0]
            std_real.extend([a*b for a,b in zip(std, self.wind[a].tolist())])
            stds[a] = std_real
            rndms[a] = rndm

        def f(corrs):
            cmatrix = []
            for a in self.areas:
                sq_sum = 0
                for c in self.areas:
                    # Calculating the squared sum of correlations per row
                    sq_sum = sq_sum + corrs[area_to_idx[a] * self.areas.__len__() + area_to_idx[c]]**2
                for b in self.areas:
                    # Calculating the c-matrix elements as correlation normalized by squared sum of row. Absolute C-matrix elements for each row should add up to one
                    cmatrix.append(corrs[area_to_idx[a] * self.areas.__len__() + area_to_idx[b]]/np.sqrt(sq_sum))
            sim_errors = pd.DataFrame(columns=self.areas)
            sim_errors_norm = pd.DataFrame(columns=self.areas)
            corr_rndms = pd.DataFrame(columns=self.areas)

            # This loop calculates matrix multiplication of cmatrix row times omegas as "val"
            # Then multiplies val with standard deviation in lst which makes the corr_rndms dataframe columns
            for a in self.areas:
                lst = []
                for i in range(len(rndms[a])):
                    val = 0
                    for aa in self.areas:
                        val = val + rndms[aa][i] * cmatrix[area_to_idx[a] * self.areas.__len__() + area_to_idx[aa]]
                    lst.append(val * stds[a][i])
                corr_rndms[a] = lst
                simulated_error = [0]

                for i in range(len(self.wind[a])):
                    err = min(self.wind[a][i],
                              self.alpha[area_to_idx[a]] * simulated_error[i] + self.beta[area_to_idx[a]] * corr_rndms[a][i + 1] +
                              self.gamma[area_to_idx[a]] * corr_rndms[a][i])
                    simulated_error.append(max(err, self.wind[a][i] - max_winds[a][0]))
                sim_errors[a] = simulated_error[1:]
                sim_errors_norm[a] = sim_errors[a].div(self.wind[a].tolist())

            corr = sim_errors.corr()
            target = 0
            for a in self.areas:
                for b in self.areas:
                    target = target + (corr[a][b] - correlations[a][b]) ** 2
            print(target)
            return target, corr

        corrs = []
        for a in self.areas:
            for b in self.areas:
                if a == b:
                    corrs.append(self.corr_matrix[b][a])
                else:
                    corrs.append(0.5 * self.corr_matrix[b][a])

        target = 1000
        while target > 0.001:
            target_prev = target
            target, corr = f(corrs)
            if target_prev == target:
                print('Same target')
                print(corrs)
                break
            diff = corr - correlations
            print(corr)
            for a in self.areas:
                for b in self.areas[area_to_idx[a]:]:
                    if diff[a][b] > 0.001:
                        corrs[area_to_idx[a] * self.areas.__len__() + area_to_idx[b]] = corrs[area_to_idx[a] * self.areas.__len__() + area_to_idx[b]] - 0.001
                        corrs[area_to_idx[b] * self.areas.__len__() + area_to_idx[a]] = corrs[area_to_idx[b] * self.areas.__len__() + area_to_idx[a]] - 0.001
                    elif diff[a][b] < -0.001:
                        corrs[area_to_idx[a] * self.areas.__len__() + area_to_idx[b]] = corrs[area_to_idx[a] * self.areas.__len__() + area_to_idx[b]] + 0.001
                        corrs[area_to_idx[b] * self.areas.__len__() + area_to_idx[a]] = corrs[area_to_idx[b] * self.areas.__len__() + area_to_idx[a]] + 0.001

        ### PRINT OUT CMATRIX ###
        cmatrix = []
        for a in self.areas:
            sq_sum = 0
            for c in self.areas:
                sq_sum = sq_sum + corrs[area_to_idx[a] * self.areas.__len__() + area_to_idx[
                    c]] ** 2
            c_sum = 0
            for b in self.areas:
                cmatrix.append(corrs[area_to_idx[a] * self.areas.__len__() + area_to_idx[b]] / np.sqrt(
                    sq_sum))

        c_matrix = pd.DataFrame(columns=self.areas, index=self.areas)
        for a in self.areas:
            c_matrix[a] = cmatrix[area_to_idx[a] * self.areas.__len__(): (area_to_idx[a] + 1) * self.areas.__len__()]
        c_matrix = c_matrix.transpose()
        self.cmatrix = c_matrix

        if make_errors:
            sim_errors = pd.DataFrame(columns=self.areas)
            corr_rndms = pd.DataFrame(columns=self.areas)
            for a in self.areas:
                lst = []
                for i in range(len(rndms[a])):
                    val = 0
                    for aa in self.areas:
                        val = val + rndms[aa][i] * cmatrix[area_to_idx[a] * self.areas.__len__() + area_to_idx[aa]]
                    if i == 0:
                        lst.append(val * stds[a][i])
                    else:
                        lst.append(val * stds[a][i])
                corr_rndms[a] = lst
                simulated_error = [0]
                for i in range(len(self.wind[a])):
                    err = min(self.wind[a][i],
                              self.alpha[area_to_idx[a]] * simulated_error[i] + self.beta[area_to_idx[a]] * corr_rndms[a][i + 1] +
                              self.gamma[area_to_idx[a]] * corr_rndms[a][i])
                    simulated_error.append(max(err, self.wind[a][i] - max_winds[a][0]))
                sim_errors[a] = simulated_error[1:]
            print(sim_errors)
            print('Actual')
            print(self.corr_matrix)
            print('Simulated')
            print(sim_errors.corr())

    def spline_hour_2_min(self, time, data):
        """Generate a spline interpolation with minutely resolution based on hourly data"""
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
            highres_time.append(datetime.strptime(a, '%Y-%m-%d %H:%M') + timedelta(minutes=i))
        wind_interpol = pd.DataFrame(columns=['Timestamp', 'Generation'])
        wind_interpol['Timestamp'] = highres_time
        wind_interpol['Generation'] = wind_spline
        wind_interpol['Timestamp'] = pd.to_datetime(wind_interpol['Timestamp'], utc=True, format='%Y-%m-%d')
        return wind_interpol

    def variability_simulation(self, area):
        """ Function to simulate minute resolution variability - input is hourly time series + area it belongs to"""
        target_metric = 1.274591896152625 #Summed absolute variability divided by total spline estimation from Finnis WP 2021 - to tune beta
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

        x = self.wind.index.tolist()
        y = self.wind[area]
        spline = self.spline_hour_2_min(x, y.tolist()) #Generate minute resolution spline


        spline.loc[spline['Generation'] < 0, 'Generation'] = 0 #Avoid spline to become negative - set it to zero
        spline_list = spline[spline.index % 3 == 0] #Save a list with spline every third minute
        spline_list = spline_list['Generation'].tolist()

        y_norm = spline['Generation'] / (self.wind[area].max() / 0.9) #Save normalized spline data with 3-minute resolution
        y_norm = y_norm[y_norm.index % 3 ==0]
        y_norm = y_norm.tolist()

        a = area

        three_min_time = []
        for i in range(len(y) * 20):
            three_min_time.append(datetime.strptime(x[0], '%Y-%m-%d %H:%M') + timedelta(minutes=i*3))
        one_min_time = spline['Timestamp'].tolist()

        ar = []
        beta = [1.28, 1.31, 1.46, 1.52, 1.06, 1.28, 1.26, 1.25, 1.57, 1.06, 1.21] #Tuned area-dependent parameters
        rhos = [0.832152, 0.04151, -0.029447, -0.021741, -0.012194, -0.016159, -0.014461, 0.002568, -0.015159, -0.041042] #AR parameters to have same initial autocorrelation as Finnish WP from 2021

        for i in range(len(three_min_time)):
            # Determine standard deviations for random gaussian
            if y_norm[i] < 0.05:
                std = 0.045 - 0.616 * y_norm[i]
            else:
                std = 0.015 - 0.016 * y_norm[i]

            if i == 0:
                ar.append(0)
            elif i <= 10 and i > 0:
                ar_sum = 0
                for n in range(i):
                    ar_sum = ar_sum + rhos[n] * ar[i - n - 1]
                ar.append(ar_sum + random.gauss(0, std) * beta[area_to_idx[a]] * spline_list[i])
            else:
                ar_sum = 0
                for n in range(10):
                    ar_sum = ar_sum + rhos[n] * ar[i - n - 1]
                ar.append(ar_sum + random.gauss(0, std) * beta[area_to_idx[a]] * spline_list[i])
        ar = pd.Series(ar)
        print(f'Target: {target_metric} %')
        print(f'Simulated: {100 * ar.abs().sum()/sum(spline_list)} %')

m = Wind_Model_Tuning(scenario='FM45')
m.read_data()
m.variability_simulation(area='NO5')

# for s in ['EF45', 'EP45', 'SF45', 'FM45']:
#     m = Wind_Model_Tuning(scenario=s)
#     m.read_data()
#     m.tune_multiarea_arma_model()
#     m.cmatrix.to_csv(f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\ModelTuning\\Wind{s}\\cmatrix.csv')


        # Create minute resolution data by linear interpolation of 3_minute resolution data
        # minute_highfreq = []
        # for i in range(len(three_min_time)):
        #     if i == len(three_min_time) - 1:
        #         minute_highfreq.append(ar[i])
        #         minute_highfreq.append(ar[i] + (ar[i] - ar[i - 1]) * 1/3)
        #         minute_highfreq.append(ar[i] + (ar[i] - ar[i - 1]) * 2/3)
        #     else:
        #         minute_highfreq.append(ar[i])
        #         minute_highfreq.append(ar[i] + (ar[i + 1] - ar[i]) * 1/3)
        #         minute_highfreq.append(ar[i] + (ar[i + 1] - ar[i]) * 2/3)
        #
        # #Save data into dataframe
        # highres = pd.DataFrame(columns=['Spline', 'Noise', 'Generation'], index=one_min_time)
        # highres['Generation'] = np.add(spline['Generation'].to_numpy(), np.array(minute_highfreq))
        # highres.loc[highres['Generation'] < 0, 'Generation'] = 0
        # highres['Spline'] = spline['Generation'].tolist()
        # highres['Noise'] = highres['Generation'] - highres['Spline']
        # return highres
    #



def tune_single_area_to_csv(area, scenario):
    csv = pd.DataFrame(columns=['alpha', 'beta', 'gamma', 't1', 't2', 't3'], index=list(range(6)))
    hej = Wind_Model_Tuning(scenario=scenario)
    hej.read_data()

    for i in range(5):
        hej.tune_singlearea_arma_model(area)
        csv['alpha'][i] = hej.alpha
        csv['beta'][i] = hej.beta
        csv['gamma'][i] = hej.gamma
        csv['t1'][i] = hej.sim_target1[0]
        csv['t2'][i] = hej.sim_target2[0]
        csv['t3'][i] = hej.sim_target3[0]
    csv['t1'][5] = hej.abs_target[area]
    csv['t2'][5] = hej.std_target[area]
    csv['t3'][5] = hej.var_target[area]
    print(csv)
    csv.to_csv(f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\ModelTuning\\Wind{scenario}\\results_{area}.csv')

# for a in ['NO1', 'NO2', 'NO3', 'NO4', 'NO5']:
#     tune_single_area_to_csv(a, 'FM45')





# #THESE LINES SORT THE MEAN WIND GENERATION EACH YEAR
# wind_years = pd.DataFrame(columns=['Year', 'Mean gen'])
# years = range(1982, 2017)
# mean_gen = []
# year_list = list(years)
# for y in years:
#     model = Wind_Model_Tuning(scenario='EP45', year=y)
#     model.read_data()
#     mean_gen.append(sum(model.wind[a].mean() for a in model.areas))
# wind_years['Mean gen'] = mean_gen
# wind_years['Year'] = year_list
# wind_years = wind_years.sort_values(by=['Mean gen'])
# print(wind_years)

#     print(csv)
#     csv.to_csv('C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\DynamicFRR\\results.csv')
