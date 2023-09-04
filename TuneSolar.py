"""
This file should be used for tuning of solar forecast models.
File can be changed as the file later in operation should be "SolarModel.py"
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

class Solar_Model_Tuning:
    def __init__(self, scenario='EF45', year=2003):
        self.scenario = scenario
        if self.scenario == 'SF45':
            self.alpha = [1.174164, 1.284138, 1.310644, 1.277100, 1.157306, 1.156060, 1.220144,
                        0.860298, 1.211929, 1.345592]
            self.beta = [0.543005, 0.078956, 0.091883, 0.415955, 0.797041, 0.506341, 0.110968,
                        0.353165, 0.71152, 0.056123]
            self.gamma = [-0.40763, -0.15829, -0.14027, -0.38779, -0.47645, -0.30305, -0.18341,
                        0.190096, -0.49164, -0.10196]
            self.year = 1987
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
            self.year = 1991
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
            self.year = 1988
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
            self.year = 1986
            self.cmatrix = pd.read_csv(
                f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\ModelTuning\\Solar{self.scenario}\\cmatrix.csv')
            self.ar_params = [0.6, 0.89, 1.2, 0.93, 1.22, 0.85, 0.715, 0.64, 1.18, 0.87] #Tuned area-dependent parameters
        self.abs_target = {
            'SE1': 0.079442,
            'SE2': 0.029098,
            'SE3': 0.019411,
            'SE4': 0.063694,
            'NO1': 0.086013,
            'NO2': 0.092826,
            'NO3': 0.053441,
            'NO4': 0,
            'NO5': 0.089419,
            'DK2': 0.089896,
            'FI': 0.027171
        }
        self.std_target = {
            'SE1': 0.194298,
            'SE2': 0.098371,
            'SE3': 0.060860,
            'SE4': 0.164292,
            'NO1': 0.206817,
            'NO2': 0.219799,
            'NO3': 0.144755,
            'NO4': 0,
            'NO5': 0.213308,
            'DK2': 0.214217,
            'FI': 0.094699
        }
        self.var_target = {
            'SE1': 0.047737,
            'SE2': 0.017122,
            'SE3': 0.010150,
            'SE4': 0.038161,
            'NO1': 0.051733,
            'NO2': 0.055876,
            'NO3': 0.031925,
            'NO4': 0,
            'NO5': 0.052804,
            'DK2': 0.054094,
            'FI': 0.015950
        }

        self.corr_matrix = pd.read_csv(
            f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\SolarCorr.csv',
            index_col=[0])
        self.areas = ('SE1', 'SE2', 'SE3', 'SE4', 'NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'DK2', 'FI')
        self.solar_areas = ('SE1', 'SE2', 'SE3', 'SE4', 'NO1', 'NO2', 'NO3', 'NO5', 'DK2', 'FI')
        self.year = year
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
        self.res_low[self.res_low < 0] = 0

        #Split RES
        res_dict = res_split(data=self.res_low, scenario=self.scenario, days=52 * 7, year=self.year)
        self.pv_low = pd.DataFrame(columns=self.areas, index=self.timestamps_str)
        self.offsh_low = pd.DataFrame(columns=self.areas, index=self.timestamps_str)
        self.onsh_low = pd.DataFrame(columns=self.areas, index=self.timestamps_str)
        self.wind_low = pd.DataFrame(columns=self.areas, index=self.timestamps_str)
        for a in self.areas:
            self.pv_low[a] = res_dict[a]['SolarPV'].tolist()
            self.offsh_low[a] = res_dict[a]['WindOffshore'].tolist()
            self.onsh_low[a] = res_dict[a]['WindOnshore'].tolist()
            self.wind_low[a] = self.offsh_low[a] + self.onsh_low[a]

    def stdev_function(self, norm_gen):
        if norm_gen <= 0.035:
            return 0.8096173357897626 - 16.595799287953355 * norm_gen
        elif norm_gen > 0.035 and norm_gen <= 0.18:
            return 0.2549277645110786 - 0.8922911241645132 * norm_gen
        elif norm_gen > 0.18 and norm_gen <= 0.5:
            return 0.12026767628481086 - 0.15221697836807174 * norm_gen
        elif norm_gen > 0.5:
            return 0.07762027143228016 - 0.072312283765212 * norm_gen

    def tune_singlearea_arma_model(self, area):
        a = area
        pv_in = self.pv_low[a]

        # MAKE 5 SERIES WITH RANDOM VARIABLES
        length = len(pv_in)
        max_pv = pv_in.max()
        rel_pv_in = pv_in / max_pv
        std = []
        for w in rel_pv_in.tolist(): # computing a list of standard deviations
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
                rndms[n].append(random.gauss(0, float(std[i])) * pv_in[i])

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

 #Adjust such that forecast error is zero when generation is zero
            for i in range(length):
                for n in range(5):
                    if pv_in[i] <= 0.02 * max_pv:
                        simulated_error[n].append(0)
                        """Unlikely with any forecast error if no/very small PV is forecasted"""
                    else:
                        error = min(pv_in[i], alpha * simulated_error[n][i] + beta * rndms[n][i + 1] + gamma * rndms[n][i])
                        """Calculating error as minimum of ARMA model and forecasted wind. Overprediction CANNOT be larger
                        than forecasted generation. I.e. negative production not possible"""
                        simulated_error[n].append(max(error, pv_in[i] - max_pv))
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
                target1[n] = float(pd.DataFrame(simulated_error_actual[n]).abs().sum() / pv_in.sum()) #Eq (6a)
                target2[n] = float(pd.DataFrame(simulated_error_actual[n]).std() / pv_in.mean()) #Eq (6b)
                target3[n] = float(np.sum(np.abs(np.subtract(np.array(simulated_error[n][:-1]),
                                                             np.array(simulated_error_actual[n])))) / pv_in.sum()) #Eq (6c)
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
                if pv_in[i] < 0.02 * max_pv:
                    simulated_error[n].append(0)
                    """Unlikely with any forecast error if no PV is forecasted"""
                else:
                    error = min(pv_in[i], self.alpha * simulated_error[n][i] + self.beta * rndms[n][i + 1] + self.gamma * rndms[n][i])
                    """Calculating error as minimum of ARMA model and forecasted wind. Overprediction CANNOT be larger
                    than forecasted generation. I.e. negative production not possible"""
                    simulated_error[n].append(max(error, pv_in[i] - max_pv))
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
            self.sim_target1[n] = float(pd.DataFrame(simulated_error_actual[n]).abs().sum() / pv_in.sum())
            self.sim_target2[n] = float(pd.DataFrame(simulated_error_actual[n]).std() / pv_in.mean())
            self.sim_target3[n] = float(np.sum(np.abs(np.subtract(np.array(simulated_error[n][:-1]),
                                                         np.array(simulated_error_actual[n])))) / pv_in.sum())
        print(f'Simulated Target 1: {self.sim_target1[0]}')
        print(f'Simulated Target 2: {self.sim_target2[0]}')
        print(f'Simulated Target 3: {self.sim_target3[0]}')

        # plt.plot(pv_in.tolist())
        # plt.plot(simulated_error_actual[0])
        # plt.show()
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
            'NO5': 7,
            'DK2': 8,
            'FI': 9
        }

        correlations = self.corr_matrix

        if plot_act_corr:
            sns.heatmap(correlations, cmap='PuBu', annot=True, center=0.1)

        stds = pd.DataFrame(columns=self.solar_areas)
        rndms = pd.DataFrame(columns=self.solar_areas)
        max_pv = pd.DataFrame(columns=self.solar_areas, index=[0])
        rel_pv_in = pd.DataFrame(columns=self.solar_areas)

        for a in self.solar_areas:
            max_pv[a][0] = self.pv_low[a].max()
            rel_pv_in[a] = self.pv_low[a] /max_pv[a][0]
            length = rel_pv_in[a].__len__()
            rndm = []
            std = []
            rndm.append(0)

            for w in rel_pv_in[a].tolist():  # computing a list of standard deviations
                std.append(self.stdev_function(w))

            for i in range(length):
                rndm.append(random.gauss(0, 1))

            std_real = [0]
            std_real.extend([a*b for a,b in zip(std, self.pv_low[a].tolist())])
            stds[a] = std_real
            rndms[a] = rndm

        def f(corrs):
            cmatrix = []
            for a in self.solar_areas:
                sq_sum = 0
                for c in self.solar_areas:
                    # Calculating the squared sum of correlations per row
                    sq_sum = sq_sum + corrs[area_to_idx[a] * self.solar_areas.__len__() + area_to_idx[c]]**2
                for b in self.solar_areas:
                    # Calculating the c-matrix elements as correlation normalized by squared sum of row. Absolute C-matrix elements for each row should add up to one
                    cmatrix.append(corrs[area_to_idx[a] * self.solar_areas.__len__() + area_to_idx[b]]/np.sqrt(sq_sum))
            sim_errors = pd.DataFrame(columns=self.solar_areas)
            sim_errors_norm = pd.DataFrame(columns=self.solar_areas)
            corr_rndms = pd.DataFrame(columns=self.solar_areas)

            # This loop calculates matrix multiplication of cmatrix row times omegas as "val"
            # Then multiplies val with standard deviation in lst which makes the corr_rndms dataframe columns
            for a in self.solar_areas:
                lst = []
                for i in range(len(rndms[a])):
                    val = 0
                    for aa in self.solar_areas:
                        val = val + rndms[aa][i] * cmatrix[area_to_idx[a] * self.solar_areas.__len__() + area_to_idx[aa]]
                    lst.append(val * stds[a][i])
                corr_rndms[a] = lst
                simulated_error = [0]

                for i in range(len(self.pv_low[a])):
                    if self.pv_low[a][i] < 0.02 * max_pv[a][0]:
                        simulated_error.append(0)
                    else:
                        err = min(self.pv_low[a][i],
                                  self.alpha[area_to_idx[a]] * simulated_error[i] + self.beta[area_to_idx[a]] * corr_rndms[a][i + 1] +
                                  self.gamma[area_to_idx[a]] * corr_rndms[a][i])
                        simulated_error.append(max(err, self.pv_low[a][i] - max_pv[a][0]))
                sim_errors[a] = simulated_error[1:]
                # sim_errors_norm[a] = sim_errors[a].div(self.pv_low[a].tolist())
                # print(sim_errors_norm[a])
            # corr_rndms.plot()
            # plt.show()
            #self.pv_low['FI'].plot()
            # sim_errors[['SE1', 'SE2', 'FI']].plot()
            # plt.show()
            corr = sim_errors.corr()
            target = 0
            for a in self.solar_areas:
                for b in self.solar_areas:
                    target = target + (corr[a][b] - correlations[a][b]) ** 2
            print(target)
            return target, corr

        dom_matrix = pd.DataFrame(columns=self.solar_areas, index=self.solar_areas)
        for a in self.solar_areas:
            if abs(self.beta[area_to_idx[a]]) > abs(self.gamma[area_to_idx[a]]) / 1.5:
                dom_a = self.beta[area_to_idx[a]]
            else:
                dom_a = self.gamma[area_to_idx[a]]
            for b in self.solar_areas:
                if abs(self.beta[area_to_idx[b]]) > abs(self.gamma[area_to_idx[b]]) / 1.5:
                    dom_b = self.beta[area_to_idx[b]]
                else:
                    dom_b = self.gamma[area_to_idx[b]]
                if a == b:
                    dom_matrix[a][b] = 1
                elif dom_a * dom_b > 0:
                    dom_matrix[a][b] = 1
                    dom_matrix[b][a] = 1
                elif dom_a * dom_b < 0:
                    dom_matrix[a][b] = -1
                    dom_matrix[b][a] = -1
        dom_matrix['SE2']['FI'] = 1
        dom_matrix['FI']['SE2'] = 1


        corrs = []
        for a in self.solar_areas:
            for b in self.solar_areas:
                if a == b:
                    corrs.append(self.corr_matrix[b][a])
                else:
                    #corrs.append(self.corr_matrix[b][a])
                    corrs.append(dom_matrix[a][b] * self.corr_matrix[b][a])

        target = 1000
        corr = []
        switch = False

        while target > 0.001:
            target_prev = target
            target, corr = f(corrs)
            if target_prev < target and not switch:
                switch = True
                print('Switch')
            elif target_prev < target and switch:
                print(f'Ended at target {target}')
                print('Differences')
                print(diff)
                break
            elif target_prev == target:
                print('Same target')
                break
            diff = corr - correlations
            print(diff)
            for a in self.solar_areas:
                for b in self.solar_areas[area_to_idx[a]:]:
                    if not switch:
                        if diff[a][b] > 0.01:
                            corrs[area_to_idx[a] * self.solar_areas.__len__() + area_to_idx[b]] = corrs[area_to_idx[a] * self.solar_areas.__len__() + area_to_idx[b]] - 0.5 * diff[a][b] * dom_matrix[a][b]
                            corrs[area_to_idx[b] * self.solar_areas.__len__() + area_to_idx[a]] = corrs[area_to_idx[b] * self.solar_areas.__len__() + area_to_idx[a]] - 0.5 * diff[a][b] * dom_matrix[a][b]
                        elif diff[a][b] < -0.01:
                            corrs[area_to_idx[a] * self.solar_areas.__len__() + area_to_idx[b]] = corrs[area_to_idx[a] * self.solar_areas.__len__() + area_to_idx[b]] - 0.5 * diff[a][b] * dom_matrix[a][b]
                            corrs[area_to_idx[b] * self.solar_areas.__len__() + area_to_idx[a]] = corrs[area_to_idx[b] * self.solar_areas.__len__() + area_to_idx[a]] - 0.5 * diff[a][b] * dom_matrix[a][b]
                    else:
                        if diff[a][b] > 0.01:
                            corrs[area_to_idx[a] * self.solar_areas.__len__() + area_to_idx[b]] = corrs[area_to_idx[a] * self.solar_areas.__len__() + area_to_idx[b]] - 0.01 * dom_matrix[a][b]
                            corrs[area_to_idx[b] * self.solar_areas.__len__() + area_to_idx[a]] = corrs[area_to_idx[b] * self.solar_areas.__len__() + area_to_idx[a]] - 0.01 * dom_matrix[a][b]
                        elif diff[a][b] < -0.01:
                            corrs[area_to_idx[a] * self.solar_areas.__len__() + area_to_idx[b]] = corrs[area_to_idx[a] * self.solar_areas.__len__() + area_to_idx[b]] + 0.01 * dom_matrix[a][b]
                            corrs[area_to_idx[b] * self.solar_areas.__len__() + area_to_idx[a]] = corrs[area_to_idx[b] * self.solar_areas.__len__() + area_to_idx[a]] + 0.01 * dom_matrix[a][b]


        self.target = target
        ### PRINT OUT CMATRIX ###
        cmatrix = []
        for a in self.solar_areas:
            sq_sum = 0
            for c in self.solar_areas:
                sq_sum = sq_sum + corrs[area_to_idx[a] * self.solar_areas.__len__() + area_to_idx[
                    c]] ** 2
            c_sum = 0
            for b in self.solar_areas:
                cmatrix.append(corrs[area_to_idx[a] * self.solar_areas.__len__() + area_to_idx[b]] / np.sqrt(
                    sq_sum))

        c_matrix = pd.DataFrame(columns=self.solar_areas, index=self.solar_areas)
        for a in self.solar_areas:
            c_matrix[a] = cmatrix[area_to_idx[a] * self.solar_areas.__len__(): (area_to_idx[a] + 1) * self.solar_areas.__len__()]
        c_matrix = c_matrix.transpose()
        self.cmatrix = c_matrix

        if make_errors:
            sim_errors = pd.DataFrame(columns=self.solar_areas)
            corr_rndms = pd.DataFrame(columns=self.solar_areas)
            for a in self.solar_areas:
                lst = []
                for i in range(len(rndms[a])):
                    val = 0
                    for aa in self.solar_areas:
                        val = val + rndms[aa][i] * cmatrix[area_to_idx[a] * self.solar_areas.__len__() + area_to_idx[aa]]
                    if i == 0:
                        lst.append(val * stds[a][i])
                    else:
                        lst.append(val * stds[a][i])
                corr_rndms[a] = lst
                simulated_error = [0]
                for i in range(len(self.pv_low[a])):
                    if self.pv_low[a][i] < 0.02 * max_pv[a][0]:
                        simulated_error.append(0)
                    else:
                        err = min(self.pv_low[a][i],
                                  self.alpha[area_to_idx[a]] * simulated_error[i] + self.beta[area_to_idx[a]] * corr_rndms[a][i + 1] +
                                  self.gamma[area_to_idx[a]] * corr_rndms[a][i])
                        simulated_error.append(max(err, self.pv_low[a][i] - max_pv[a][0]))
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
            'NO5': 7,
            'DK2': 8,
            'FI': 9
        }

        x = self.pv_low.index.tolist()
        y = self.pv_low[area]
        spline = self.spline_hour_2_min(x, y.tolist()) #Generate minute resolution spline


        spline.loc[spline['Generation'] < 0, 'Generation'] = 0 #Avoid spline to become negative - set it to zero
        spline_list = spline[spline.index % 3 == 0] #Save a list with spline every third minute
        spline_list = spline_list['Generation'].tolist()

        y_norm = spline['Generation'] / (self.pv_low[area].max()) #Save normalized spline data with 3-minute resolution
        y_norm = y_norm[y_norm.index % 3 ==0]
        y_norm = y_norm.tolist()

        a = area

        three_min_time = []
        for i in range(len(y) * 20):
            three_min_time.append(datetime.strptime(x[0], '%Y-%m-%d %H:%M') + timedelta(minutes=i*3))
        one_min_time = spline['Timestamp'].tolist()

        ar = []
        beta = [0.6, 0.89, 1.2, 0.93, 1.22, 0.85, 0.715, 0.64, 1.18, 0.87] #Tuned area-dependent parameters
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
                if spline_list[i] == 0:
                    ar.append(0)
                else:
                    for n in range(i):
                        ar_sum = ar_sum + rhos[n] * ar[i - n - 1]
                    ar.append(ar_sum + random.gauss(0, std) * beta[area_to_idx[a]] * spline_list[i])
            else:
                if spline_list[i] == 0:
                    ar.append(0)
                else:
                    ar_sum = 0
                    for n in range(10):
                        ar_sum = ar_sum + rhos[n] * ar[i - n - 1]
                    ar.append(ar_sum + random.gauss(0, std) * beta[area_to_idx[a]] * spline_list[i])
        ar = pd.Series(ar)
        print(f'Target: {target_metric} %')
        print(f'Simulated: {100 * ar.abs().sum()/sum(spline_list)} %')

m = Solar_Model_Tuning(scenario='FM45')
m.read_data()
m.variability_simulation('FI')

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
    hej = Solar_Model_Tuning(scenario=scenario)
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
    csv.to_csv(f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\ModelTuning\\Solar{scenario}\\results_{area}.csv')

# for s in ('EP45', 'SF45', 'FM45'):
#     for a in ('SE1', 'SE2', 'SE3', 'SE4', 'NO1', 'NO2', 'NO3', 'NO5', 'DK2', 'FI'):
#         tune_single_area_to_csv(a, s)



#THESE LINES SORT THE MEAN WIND GENERATION EACH YEAR
# wind_years = pd.DataFrame(columns=['Year', 'Mean gen'])
# years = range(1982, 2017)
# mean_gen = []
# year_list = list(years)
# for y in years:
#     model = Solar_Model_Tuning(scenario='FM45', year=y)
#     model.read_data()
#     mean_gen.append(sum(model.pv_low[a].mean() for a in model.areas))
# wind_years['Mean gen'] = mean_gen
# wind_years['Year'] = year_list
# wind_years = wind_years.sort_values(by=['Mean gen'])
# print(wind_years)


