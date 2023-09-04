import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def wind():
    data_areas = ('SE1', 'SE2', 'SE3', 'SE4', 'FI')
    def read_sefi_forecast():
        data_areas = ('SE1', 'SE2', 'SE3', 'SE4', 'FI')
        with open('sefi_forecasts.pickle', 'rb') as f:
            forecast_data = pkl.load(f)
        for a in data_areas:
            wind_dict[a] = forecast_data['Wind'][a]

    wind_dict = {}
    read_sefi_forecast()
    plot_fit = True

    mean_generation = []
    standard_deviation = []
    stats = pd.DataFrame(columns=['Mean generation', 'Standard deviation'])
    t1_list = []
    t_list = []
    gen_list = []
    cap_list = []
    for area in data_areas:
        t1_list.append(0)
        t1_list.extend(wind_dict[area]['ID error'][:-1])
        t_list.extend(wind_dict[area]['ID error'].tolist())
        gen_list.extend(wind_dict[area]['ID'].tolist())
        if area == 'SE3':
            cap_list.extend([wind_dict[area]['Capacity'][i] + 900 for i in range(wind_dict[area].__len__())])
        else:
            cap_list.extend(wind_dict[area]['Capacity'].tolist())
    df = pd.DataFrame(columns=['T', 'T-1', 'Change', 'Generation', 'Relative change', 'Capacity', 'Normalized generation'])
    df['T'] = t_list
    df['T-1'] = t1_list
    df['Change'] = df['T'] - df['T-1']
    df['Generation'] = gen_list
    df['Relative change'] = df['Change'].div(df['Generation'])
    df['Capacity'] = cap_list
    df['Normalized generation'] = df['Generation'].div(df['Capacity'])
    df = df.sort_values(by=['Normalized generation'])
    zero_rows = df.loc[df['Normalized generation'] == 0].index.tolist()
    if zero_rows.__len__() > 0:
        df.drop(zero_rows, axis=0, inplace=True)
    step = int(len(df['Normalized generation']) / 200)
    for i in range(200):
        if i == 199:
            dff = pd.DataFrame(columns=['Normalized generation', 'Relative change'])
            dff['Normalized generation'] = df['Normalized generation'][i * step:]
            dff['Relative change'] = df['Relative change'][i * step:]
        else:
            dff = pd.DataFrame(columns=['Normalized generation', 'Relative change'])
            dff['Normalized generation'] = df['Normalized generation'][i * step: (i + 1) * step]
            dff['Relative change'] = df['Relative change'][i * step: (i + 1) * step]
        dff.reset_index(drop=True, inplace=True)
        mean_generation.append(dff['Normalized generation'].mean())
        standard_deviation.append(dff['Relative change'].std())
    stats['Mean generation'] = mean_generation
    stats['Standard deviation'] = standard_deviation
    stats = stats.sort_values(by=['Mean generation'])
    mean_generation = stats['Mean generation'].tolist()
    standard_deviation = stats['Standard deviation'].tolist()
    plt.scatter(stats['Mean generation'], stats['Standard deviation'], alpha=0.5)
    lim = [0.0265, 0.12, 0.59]

    lim1 = lim[0]
    lim2 = lim[1]
    lim3 = lim[2]
    low = stats.loc[stats['Mean generation'] < lim1]
    med1 = stats.loc[stats['Mean generation'] > lim1]
    med1 = med1.loc[med1['Mean generation'] < lim2]
    med2 = stats.loc[stats['Mean generation'] > lim2]
    med2 = med2.loc[med2['Mean generation'] < lim3]
    high = stats.loc[stats['Mean generation'] > lim3]

    def f(x, a, b):
        return a + b * x

    x = np.array(low['Mean generation'])
    yn = np.array(low['Standard deviation'])
    popt, pcov = curve_fit(f, x, yn)
    a_low = popt[0]
    b_low = popt[1]
    if plot_fit:
        plt.plot(np.linspace(0, lim1, 100), f(np.linspace(0, lim1, 100), a_low, b_low), linewidth=2, color='black')
    x = np.array(med1['Mean generation'])
    yn = np.array(med1['Standard deviation'])
    popt, pcov = curve_fit(f, x, yn)
    a_med1 = popt[0]
    b_med1 = popt[1]
    plt.plot(np.linspace(lim1, lim2, 100), f(np.linspace(lim1, lim2, 100), a_med1, b_med1), linewidth=2,
             color='black')
    x = np.array(med2['Mean generation'])
    yn = np.array(med2['Standard deviation'])
    popt, pcov = curve_fit(f, x, yn)
    a_med2 = popt[0]
    b_med2 = popt[1]
    plt.plot(np.linspace(lim2, lim3, 100), f(np.linspace(lim2, lim3, 100), a_med2, b_med2), linewidth=2,
             color='black')
    x = np.array(high['Mean generation'])
    yn = np.array(high['Standard deviation'])
    popt, pcov = curve_fit(f, x, yn)
    a_high = popt[0]
    b_high = popt[1]
    plt.plot(np.linspace(lim3, 1, 100), f(np.linspace(lim3, 1, 100), a_high, b_high), linewidth=2, color='black')
    plt.xlabel('Normalized generation [p.u.]')
    plt.ylabel('Standard deviation [p.u.]')
    plt.grid()
    plt.show()


def solar():
    with open('solar_forecasts.pickle','rb') as handle:
        solar_dict = pkl.load(handle)
    data_areas = list(solar_dict.keys())
    mean_generation = []
    standard_deviation = []
    stats = pd.DataFrame(columns=['Mean generation', 'Standard deviation'])
    t1_list = []
    t_list = []
    gen_list = []
    cap_list = []
    for area in data_areas:
        t1_list.append(0)
        t1_list.extend(solar_dict[area]['ID error'][:-1])
        t_list.extend(solar_dict[area]['ID error'].tolist())
        gen_list.extend(solar_dict[area]['ID forecast'].tolist())
        cap_list.extend([solar_dict[area]['Generation'].max() for i in range(solar_dict[area].__len__())])
    df = pd.DataFrame(
         columns=['T', 'T-1', 'Change', 'Generation', 'Relative change', 'Capacity', 'Normalized generation'])
    df['T'] = t_list
    df['T-1'] = t1_list
    df['Change'] = df['T'] - df['T-1']
    df['Generation'] = gen_list
    df['Relative change'] = df['Change'].div(df['Generation'])
    df['Capacity'] = cap_list
    df['Normalized generation'] = df['Generation'].div(df['Capacity'])
    df = df.sort_values(by=['Normalized generation'])
    zero_rows = df.loc[df['Normalized generation'] <= 0.01].index.tolist()
    if zero_rows.__len__() > 0:
        df.drop(zero_rows, axis=0, inplace=True)
    step = int(len(df['Normalized generation']) / 200)
    for i in range(200):
        if i == 199:
            dff = pd.DataFrame(columns=['Normalized generation', 'Relative change'])
            dff['Normalized generation'] = df['Normalized generation'][i * step:]
            dff['Relative change'] = df['Relative change'][i * step:]
        else:
            dff = pd.DataFrame(columns=['Normalized generation', 'Relative change'])
            dff['Normalized generation'] = df['Normalized generation'][i * step: (i + 1) * step]
            dff['Relative change'] = df['Relative change'][i * step: (i + 1) * step]
        dff.reset_index(drop=True, inplace=True)
        mean_generation.append(dff['Normalized generation'].mean())
        standard_deviation.append(dff['Relative change'].std())
    stats['Mean generation'] = mean_generation
    stats['Standard deviation'] = standard_deviation
    stats = stats.sort_values(by=['Mean generation'])
    mean_generation = stats['Mean generation'].tolist()
    standard_deviation = stats['Standard deviation'].tolist()
    plt.scatter(stats['Mean generation'], stats['Standard deviation'], alpha=0.5)
    #CONTINUE HERE FITTING THE FUNCTIONS

    lim = [0.035, 0.18, 0.5]

    lim1 = lim[0]
    lim2 = lim[1]
    lim3 = lim[2]
    low = stats.loc[stats['Mean generation'] < lim1]
    med1 = stats.loc[stats['Mean generation'] > lim1]
    med1 = med1.loc[med1['Mean generation'] < lim2]
    med2 = stats.loc[stats['Mean generation'] > lim2]
    med2 = med2.loc[med2['Mean generation'] < lim3]
    high = stats.loc[stats['Mean generation'] > lim3]

    def f(x, a, b):
        return a + b * x

    x = np.array(low['Mean generation'])
    yn = np.array(low['Standard deviation'])
    popt, pcov = curve_fit(f, x, yn)
    a_low = popt[0]
    b_low = popt[1]
    print('Low')
    print(a_low, b_low)
    plt.plot(np.linspace(0, lim1, 100), f(np.linspace(0, lim1, 100), a_low, b_low), linewidth=2, color='black')
    x = np.array(med1['Mean generation'])
    yn = np.array(med1['Standard deviation'])
    popt, pcov = curve_fit(f, x, yn)
    a_med1 = popt[0]
    b_med1 = popt[1]
    print('Med1')
    print(a_med1, b_med1)
    plt.plot(np.linspace(lim1, lim2, 100), f(np.linspace(lim1, lim2, 100), a_med1, b_med1), linewidth=2,
             color='black')
    x = np.array(med2['Mean generation'])
    yn = np.array(med2['Standard deviation'])
    popt, pcov = curve_fit(f, x, yn)
    a_med2 = popt[0]
    b_med2 = popt[1]
    print('Med2')
    print(a_med2, b_med2)
    plt.plot(np.linspace(lim2, lim3, 100), f(np.linspace(lim2, lim3, 100), a_med2, b_med2), linewidth=2,
             color='black')
    x = np.array(high['Mean generation'])
    yn = np.array(high['Standard deviation'])
    popt, pcov = curve_fit(f, x, yn)
    a_high = popt[0]
    b_high = popt[1]
    print('High')
    print(a_high, b_high)
    plt.plot(np.linspace(lim3, 1, 100), f(np.linspace(lim3, 1, 100), a_high, b_high), linewidth=2, color='black')
    plt.xlabel('Normalized generation [p.u.]')
    plt.ylabel('Standard deviation [p.u.]')
    plt.grid()
    #plt.show()

def demand():
    with open('demand_forecasts.pickle','rb') as handle:
        demand_dict = pkl.load(handle)
    data_areas = list(demand_dict.keys())
    mean_demand = []
    standard_deviation = []
    stats = pd.DataFrame(columns=['Mean demand', 'Standard deviation'])
    t1_list = []
    t_list = []
    dem_list = []
    cap_list = []
    for area in data_areas:
        t1_list.append(0)
        t1_list.extend(demand_dict[area]['ID error'][:-1])
        t_list.extend(demand_dict[area]['ID error'].tolist())
        dem_list.extend(demand_dict[area]['ID'].tolist())
        cap_list.extend([demand_dict[area]['ID'].max() for i in range(demand_dict[area].__len__())])
    df = pd.DataFrame(
         columns=['T', 'T-1', 'Change', 'Demand', 'Relative change', 'Max', 'Normalized demand'])
    df['T'] = t_list
    df['T-1'] = t1_list
    df['Change'] = df['T'] - df['T-1']
    df['Demand'] = dem_list
    df['Relative change'] = df['Change'].div(df['Demand'])
    df['Max'] = cap_list
    df['Normalized demand'] = df['Demand'].div(df['Max'])
    df = df.sort_values(by=['Normalized demand'])
    zero_rows = df.loc[df['Normalized demand'] <= 0.01].index.tolist()
    if zero_rows.__len__() > 0:
        df.drop(zero_rows, axis=0, inplace=True)
    step = int(len(df['Normalized demand']) / 200)
    for i in range(200):
        if i == 199:
            dff = pd.DataFrame(columns=['Normalized demand', 'Relative change'])
            dff['Normalized demand'] = df['Normalized demand'][i * step:]
            dff['Relative change'] = df['Relative change'][i * step:]
        else:
            dff = pd.DataFrame(columns=['Normalized demand', 'Relative change'])
            dff['Normalized demand'] = df['Normalized demand'][i * step: (i + 1) * step]
            dff['Relative change'] = df['Relative change'][i * step: (i + 1) * step]
        dff.reset_index(drop=True, inplace=True)
        mean_demand.append(dff['Normalized demand'].mean())
        standard_deviation.append(dff['Relative change'].std())
    stats['Mean demand'] = mean_demand
    stats['Standard deviation'] = standard_deviation
    stats = stats.sort_values(by=['Mean demand'])
    mean_demand = stats['Mean demand'].tolist()
    standard_deviation = stats['Standard deviation'].tolist()
    plt.rcParams.update({'font.size': 12})
    plt.scatter(stats['Mean demand'], stats['Standard deviation'], alpha=0.5, s=50)

    def f(x, a, b):
        return a + b * x

    x = np.array(stats['Mean demand'])
    yn = np.array(stats['Standard deviation'])
    popt, pcov = curve_fit(f, x, yn)
    a_low = popt[0]
    b_low = popt[1]
    plt.plot(np.linspace(0, 1, 100), f(np.linspace(0, 1, 100), a_low, b_low), linewidth=2, color='black', label=f'y = {round(a_low,4)} - {round(- b_low,4)}' + r'$\cdot$x')
    plt.xlabel('Normalized demand [p.u.]')
    plt.ylabel(r'$\sigma^{2}(\cdot)$ [p.u.]')
    plt.xlim(-0.01,1.01)
    plt.ylim(-0.001, 0.04)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    fig = plt.gcf()
    save = True
    if save:
        fig.savefig(
            f'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Papers&Projects\\J3 - Balancing analysis\\Figures\\DemandFunction.pdf',
            dpi=fig.dpi, pad_inches=0, bbox_inches='tight')
    plt.show()


demand()
