"""This file was used to read PECD data into a pickle file.
The data is saved in "C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Simuleringsdata\\ERAA_22\\PECD-MAF2019-Nordic.pickle"
"""
import pandas as pd
import pickle as pkl

maf_areas = ['SE01', 'SE02', 'SE03', 'SE04', 'FI00', 'DKE1', 'NON1', 'NOM1', 'NOS0']
years = list(range(1982,2017))

offshore_pecd = pd.read_csv(
    'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Simuleringsdata\\ERAA_22\\PECD-MAF2019-wide-WindOffshore.csv',
    sep=';', dtype=str)  # , skiprows=lambda x: x not in wind_rows)
onshore_pecd = pd.read_csv(
    'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Simuleringsdata\\ERAA_22\\PECD-MAF2019-wide-WindOnshore.csv',
    sep=';', dtype=str)  # , skiprows=lambda x: x not in wind_rows)
pv_pecd = pd.read_csv(
    'C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Simuleringsdata\\ERAA_22\\PECD-MAF2019-wide-PV.csv',
    sep=';', dtype=str)

maf_dict = {}
maf_dict['WindOffshore'] = {}
maf_dict['WindOnshore'] = {}
maf_dict['SolarPV'] = {}
for a in maf_areas:
    # Offshore wind
    xx = offshore_pecd.loc[offshore_pecd['area'] == a]
    xx = xx.stack().str.replace(',', '.').unstack()
    xx.reset_index(drop=True, inplace=True)
    xx = xx[0:52 * 7 * 24]
    print(xx)
    maf_dict['WindOffshore'][a] = xx

    # Onshore wind
    xx = onshore_pecd.loc[onshore_pecd['area'] == a]
    xx = xx.stack().str.replace(',', '.').unstack()
    xx.reset_index(drop=True, inplace=True)
    xx = xx[0:52 * 7 * 24]
    maf_dict['WindOnshore'][a] = xx

    # PV
    xx = pv_pecd.loc[pv_pecd['area'] == a]
    xx = xx.stack().str.replace(',', '.').unstack()
    xx.reset_index(drop=True, inplace=True)
    xx = xx[0:52 * 7 * 24]
    maf_dict['SolarPV'][a] = xx

with open('C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Simuleringsdata\\ERAA_22\\PECD-MAF2019-Nordic.pickle', 'wb') as handle:
    pkl.dump(maf_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)