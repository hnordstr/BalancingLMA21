"""
This file should be used for tuning of demand forecast models.
File can be changed as the file later in operation should be "DemandModel.py"
"""

"""
This file should be used for tuning of wind forecast models.
File can be changed as the file later in operation should be "WindModel.py"
"""

import pandas as pd
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

class Demand_Model_Tuning:
    def __init__(self, scenario='EF45', year=2003):
        self.scenario = scenario
        if self.scenario == 'SF45':
            self.alpha = [0.736542, 0.913625, 0.896951, 0.873720, 0.871768, 0.962010, 0.660962,
                          0.905162, 0.915939, 0.957910, 0.940291]
            self.beta = [1.563695, 1.588418, 0.181474, 0.357493, 0.112962, 1.029788, 0.666024,
                         0.702192, 1.290930, 0.271646, 0.538252]
            self.gamma = [0.183615, 0.144085, 0.156435, 0.279134, -0.31043, -0.39419, 0.049421,
                          -0.22776, 0.404255, -0.17402, -0.31358]
            self.year = 1984
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
            self.year = 1988
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
            self.year = 1986
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
            self.year = 1984
            self.cmatrix = pd.read_csv(
               f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\ModelTuning\\Demand{self.scenario}\\cmatrix.csv')
            self.ar_params = [1.08, 1.10, 1.06, 1.07, 1.02, 1.11, 1.11, 1.14, 1.13, 1.10, 1.17]
        self.abs_target = {
            'SE1': 0.035378,
            'SE2': 0.055814,
            'SE3': 0.010706,
            'SE4': 0.017965,
            'NO1': 0.007545,
            'NO2': 0.031635,
            'NO3': 0.012611,
            'NO4': 0.014709,
            'NO5': 0.052161,
            'DK2': 0.00513,
            'FI': 0.008963
        }
        self.std_target = {
            'SE1': 0.046437,
            'SE2': 0.074895,
            'SE3': 0.013786,
            'SE4': 0.023529,
            'NO1': 0.010038,
            'NO2': 0.041347,
            'NO3': 0.015504,
            'NO4': 0.018883,
            'NO5': 0.066216,
            'DK2': 0.007243,
            'FI': 0.012018
        }
        self.var_target = {
            'SE1': 0.023977,
            'SE2': 0.022491,
            'SE3': 0.00353,
            'SE4': 0.006726,
            'NO1': 0.005826,
            'NO2': 0.014989,
            'NO3': 0.0097,
            'NO4': 0.009594,
            'NO5': 0.017112,
            'DK2': 0.004569,
            'FI': 0.007754
        }
        self.corr_matrix = pd.read_csv(
            f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\DemandCorr.csv',
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
        self.rhos = [0.7498423514885194, 0.022938611682757112, -0.05153934662570188, -0.04310610792562501,
                0.0012196891213756122, -0.016048659633685715, 0.01108932314658005, -0.0027962620143520455,
                -0.027261679919150565, -0.03967111880841876]

    def read_data(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        area_dict = {}
        for area in self.areas:
            table = f'Generation_{area}_{self.year}'
            c.execute("SELECT * FROM {0}".format(table))
            area_dict[f'{area}_Gen'] = c.fetchall()
            area_dict[f'{area}_GenCols'] = list(map(lambda x: x[0], c.description))

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
        self.demand = self.consumption_low

    def stdev_function(self, norm_dem):
        return 0.037172938443793255 - 0.02745556240448081 * norm_dem

    def tune_singlearea_arma_model(self, area):
        a = area
        demand_in = self.demand[a]

        # MAKE 5 SERIES WITH RANDOM VARIABLES
        length = len(demand_in)
        max_demand = demand_in.max()
        rel_demand_in = demand_in / max_demand
        std = []
        for w in rel_demand_in.tolist(): # computing a list of standard deviations
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
                rndms[n].append(random.gauss(0, float(std[i])) * demand_in[i])

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
                    error = min(demand_in[i], alpha * simulated_error[n][i] + beta * rndms[n][i + 1] + gamma * rndms[n][i])
                    """Calculating error as minimum of ARMA model and forecasted demand. Overprediction CANNOT be larger
                    than forecasted demand. I.e. negative demand not possible"""
                    simulated_error[n].append(error)

            simulated_error_actual = {}
            target1 = {}
            target2 = {}
            target3 = {}
            mse = 0
            for n in range(5):
                simulated_error_actual[n] = simulated_error[n][1:]
                target1[n] = round(float(pd.DataFrame(simulated_error_actual[n]).abs().sum() / demand_in.sum()),6) #Eq (6a)
                target2[n] = round(float(pd.DataFrame(simulated_error_actual[n]).std() / demand_in.mean()),6) #Eq (6b)
                target3[n] = round(float(np.sum(np.abs(np.subtract(np.array(simulated_error[n][:-1]),
                                                             np.array(simulated_error_actual[n])))) / demand_in.sum()),6) #Eq (6c)
                mse += float(round(np.abs(target1[n] - self.abs_target[a]), 10) +
                             round(np.abs(target2[n] - self.std_target[a]), 10) +
                             round(np.abs(target3[n] - self.var_target[a]), 10))
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
                error = min(demand_in[i], self.alpha * simulated_error[n][i] + self.beta * rndms[n][i + 1] + self.gamma * rndms[n][i])
                simulated_error[n].append(error)

        simulated_error_actual = {}
        self.sim_target1 = {}
        self.sim_target2 = {}
        self.sim_target3 = {}
        print(f'Target 1: {self.abs_target[a]}')
        print(f'Target 2: {self.std_target[a]}')
        print(f'Target 3: {self.var_target[a]}')
        for n in range(5):
            simulated_error_actual[n] = simulated_error[n][1:]
            self.sim_target1[n] = float(pd.DataFrame(simulated_error_actual[n]).abs().sum() / demand_in.sum())
            self.sim_target2[n] = float(pd.DataFrame(simulated_error_actual[n]).std() / demand_in.mean())
            self.sim_target3[n] = float(np.sum(np.abs(np.subtract(np.array(simulated_error[n][:-1]),
                                                         np.array(simulated_error_actual[n])))) / demand_in.sum())
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
        max_demand = pd.DataFrame(columns=self.areas, index=[0])
        rel_demand_in = pd.DataFrame(columns=self.areas)

        for a in self.areas:
            max_demand[a][0] = self.demand[a].max()
            rel_demand_in[a] = self.demand[a] /max_demand[a][0]
            length = rel_demand_in[a].__len__()
            rndm = []
            std = []
            rndm.append(0)

            for w in rel_demand_in[a].tolist():  # computing a list of standard deviations
                std.append(self.stdev_function(w))

            for i in range(length):
                rndm.append(random.gauss(0, 1))

            std_real = [0]
            std_real.extend([a*b for a,b in zip(std, self.demand[a].tolist())])
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

                for i in range(len(self.demand[a])):
                    err = min(self.demand[a][i],
                              self.alpha[area_to_idx[a]] * simulated_error[i] + self.beta[area_to_idx[a]] * corr_rndms[a][i + 1] +
                              self.gamma[area_to_idx[a]] * corr_rndms[a][i])
                    simulated_error.append(err)
                sim_errors[a] = simulated_error[1:]

            corr = sim_errors.corr()
            target = 0
            for a in self.areas:
                for b in self.areas:
                    target = target + (corr[a][b] - correlations[a][b]) ** 2
            print(target)
            return target, corr

        dom_matrix = pd.DataFrame(columns=self.areas, index=self.areas)
        for a in self.areas:
            if abs(self.beta[area_to_idx[a]]) > abs(self.gamma[area_to_idx[a]]) / 1:
                dom_a = self.beta[area_to_idx[a]]
            else:
                dom_a = self.gamma[area_to_idx[a]]
            for b in self.areas:
                if abs(self.beta[area_to_idx[b]]) > abs(self.gamma[area_to_idx[b]]) / 1:
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


        corrs = []

        for a in self.areas:
            for b in self.areas:
                if a == b:
                    corrs.append(self.corr_matrix[b][a])
                else:
                    corrs.append(0.5 * self.corr_matrix[b][a] * dom_matrix[a][b])

        mov_dict = {}
        mov_matrix = pd.DataFrame(columns=self.areas, index=self.areas)
        target = 1000
        minima = 10000
        i = 0
        while target > 0.01:
            if i > 0:
                diff_old = diff
            target_prev = target
            target, corr = f(corrs)
            if target < minima:
                print('New min')
                minima = target
                min_corrs = corrs
            if target_prev == target:
                print('Same target')
                print(corrs)
                break
            diff = corr - correlations
            print(diff)
            if i > 0:
                for a in self.areas:
                    for b in self.areas:
                        if abs(diff[a][b]) > abs(diff_old[a][b]) + 0.003:
                            mov_matrix[a][b] = dom_matrix[a][b] * (-1)
                            print(f'Change {a}-{b}')
                        else:
                            mov_matrix[a][b] = dom_matrix[a][b]
                mov_dict[i] = mov_matrix
            if i > 1:
                dom_matrix = mov_matrix
                # print('Old')
                # print(mov_dict[i-1])
                # print('New')
                # print(mov_dict[i])
            i += 1
            if i == 100:
                print(f'Minimum at target:  {minima}')
                corrs = min_corrs
                break
            for a in self.areas:
                for b in self.areas[area_to_idx[a]:]:
                    if diff[a][b] > 0.1:
                        corrs[area_to_idx[a] * self.areas.__len__() + area_to_idx[b]] = corrs[area_to_idx[a] * self.areas.__len__() + area_to_idx[b]] - 0.01 * dom_matrix[a][b]
                        corrs[area_to_idx[b] * self.areas.__len__() + area_to_idx[a]] = corrs[area_to_idx[b] * self.areas.__len__() + area_to_idx[a]] - 0.01 * dom_matrix[a][b]
                    elif diff[a][b] < -0.1:
                        corrs[area_to_idx[a] * self.areas.__len__() + area_to_idx[b]] = corrs[area_to_idx[a] * self.areas.__len__() + area_to_idx[b]] + 0.01 * dom_matrix[a][b]
                        corrs[area_to_idx[b] * self.areas.__len__() + area_to_idx[a]] = corrs[area_to_idx[b] * self.areas.__len__() + area_to_idx[a]] + 0.01 * dom_matrix[a][b]
                    elif diff[a][b] > 0.001:
                        corrs[area_to_idx[a] * self.areas.__len__() + area_to_idx[b]] = corrs[area_to_idx[a] * self.areas.__len__() + area_to_idx[b]] - 0.001 * dom_matrix[a][b]
                        corrs[area_to_idx[b] * self.areas.__len__() + area_to_idx[a]] = corrs[area_to_idx[b] * self.areas.__len__() + area_to_idx[a]] - 0.001 * dom_matrix[a][b]
                    elif diff[a][b] < -0.001:
                        corrs[area_to_idx[a] * self.areas.__len__() + area_to_idx[b]] = corrs[area_to_idx[a] * self.areas.__len__() + area_to_idx[b]] + 0.001 * dom_matrix[a][b]
                        corrs[area_to_idx[b] * self.areas.__len__() + area_to_idx[a]] = corrs[area_to_idx[b] * self.areas.__len__() + area_to_idx[a]] + 0.001 * dom_matrix[a][b]



        # target = 1000
        # while target > 0.0001:
        #     target_prev = target
        #     target, corr = f(corrs)
        #     if target_prev == target:
        #         print('Same target')
        #         print(corrs)
        #         break
        #     diff = corr - correlations
        #     print(diff)
        #     for a in self.areas:
        #         for b in self.areas[area_to_idx[a]:]:
        #             if diff[a][b] > 0.1 and target > 0.2:
        #                 corrs[area_to_idx[a] * self.areas.__len__() + area_to_idx[b]] = corrs[area_to_idx[a] * self.areas.__len__() + area_to_idx[b]] - 0.1 * dom_matrix[a][b]
        #                 corrs[area_to_idx[b] * self.areas.__len__() + area_to_idx[a]] = corrs[area_to_idx[b] * self.areas.__len__() + area_to_idx[a]] - 0.1 * dom_matrix[a][b]
        #             elif diff[a][b] < -0.1 and target > 0.2:
        #                 corrs[area_to_idx[a] * self.areas.__len__() + area_to_idx[b]] = corrs[area_to_idx[a] * self.areas.__len__() + area_to_idx[b]] + 0.1 * dom_matrix[a][b]
        #                 corrs[area_to_idx[b] * self.areas.__len__() + area_to_idx[a]] = corrs[area_to_idx[b] * self.areas.__len__() + area_to_idx[a]] + 0.1 * dom_matrix[a][b]
        #             elif diff[a][b] > 0.001:
        #                 corrs[area_to_idx[a] * self.areas.__len__() + area_to_idx[b]] = corrs[area_to_idx[a] * self.areas.__len__() + area_to_idx[b]] - 0.001 * dom_matrix[a][b]
        #                 corrs[area_to_idx[b] * self.areas.__len__() + area_to_idx[a]] = corrs[area_to_idx[b] * self.areas.__len__() + area_to_idx[a]] - 0.001 * dom_matrix[a][b]
        #             elif diff[a][b] < -0.001:
        #                 corrs[area_to_idx[a] * self.areas.__len__() + area_to_idx[b]] = corrs[area_to_idx[a] * self.areas.__len__() + area_to_idx[b]] + 0.001 * dom_matrix[a][b]
        #                 corrs[area_to_idx[b] * self.areas.__len__() + area_to_idx[a]] = corrs[area_to_idx[b] * self.areas.__len__() + area_to_idx[a]] + 0.001 * dom_matrix[a][b]

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
                for i in range(len(self.demand[a])):
                    err = min(self.demand[a][i],
                              self.alpha[area_to_idx[a]] * simulated_error[i] + self.beta[area_to_idx[a]] * corr_rndms[a][i + 1] +
                              self.gamma[area_to_idx[a]] * corr_rndms[a][i])
                    simulated_error.append(err)
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
        target_metric = 0.4671904749768083 #Summed absolute variability divided by total spline estimation from Finnis WP 2021 - to tune beta
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

        x = self.demand.index.tolist()
        y = self.demand[area]
        spline = self.spline_hour_2_min(x, y.tolist()) #Generate minute resolution spline


        spline.loc[spline['Generation'] < 0, 'Generation'] = 0 #Avoid spline to become negative - set it to zero
        spline_list = spline[spline.index % 3 == 0] #Save a list with spline every third minute
        spline_list = spline_list['Generation'].tolist()

        y_norm = spline['Generation'] / (self.demand[area].max()) #Save normalized spline data with 3-minute resolution
        y_norm = y_norm[y_norm.index % 3 ==0]
        y_norm = y_norm.tolist()

        a = area

        three_min_time = []
        for i in range(len(y) * 20):
            three_min_time.append(datetime.strptime(x[0], '%Y-%m-%d %H:%M') + timedelta(minutes=i*3))
        one_min_time = spline['Timestamp'].tolist()

        ar = []
        beta = [1.08, 1.10, 1.06, 1.07, 1.02, 1.11, 1.11, 1.14, 1.13, 1.10, 1.17] #Tuned area-dependent parameters
        rhos = [0.7498423514885194, 0.022938611682757112, -0.05153934662570188, -0.04310610792562501,
                0.0012196891213756122, -0.016048659633685715, 0.01108932314658005, -0.0027962620143520455,
                -0.027261679919150565, -0.03967111880841876]
        #AR parameters to have same initial autocorrelation as Finnish demand from 2021

        for i in range(len(three_min_time)):
            # Determine standard deviations for random gaussian
            std = 0.00533 - 0.00239 * y_norm[i]

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

m = Demand_Model_Tuning(scenario='FM45')
m.read_data()
m.variability_simulation(area='FI')


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
    hej = Demand_Model_Tuning(scenario=scenario)
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
    csv.to_csv(f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\ModelTuning\\Demand{scenario}\\results_{area}.csv')










##THESE LINES SORT THE MEAN WIND GENERATION EACH YEAR
#wind_years = pd.DataFrame(columns=['Year', 'Mean gen'])
#years = range(1982, 2017)
#mean_gen = []
#year_list = list(years)
#for y in years:
#    model = Demand_Model_Tuning(scenario='FM45', year=y)
#    model.read_data()
#    mean_gen.append(sum(model.demand[a].mean() for a in model.areas))
#wind_years['Mean gen'] = mean_gen
#wind_years['Year'] = year_list
#wind_years = wind_years.sort_values(by=['Mean gen'])
#print(wind_years)






