import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt

"""This file is used to split res data from LMA into onshore, offshore and PV.
In operation and should not be changed"""

def res_split(data, year=2016, days=52*7, scenario='EF45'):
    #Fragments from ODIN model
    #NO1 = 0,0458123185083295 * NOM1 + 0,124666699093952 * NOS0
    #NO2 = 0,421097781531482 * NOS0
    #NO3 = 0,0303895164207903 * NON1 + 0,753075806205105 * NOM1 + 0,0109663183681411 * NOS0
    #NO4 = 0,96961048357921 * NON1
    #NO5 = 0,201111875286566 * NOM1 + 0,443269201006425 * NOS0

    areas = ['SE1', 'SE2', 'SE3', 'SE4', 'NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'DK2', 'FI']
    maf_areas = ['SE01', 'SE02', 'SE03', 'SE04', 'FI00', 'DKE1', 'NON1', 'NOM1', 'NOS0']
    lma = data


    ### Installed capacity is set according to LMA-data EF45 for SE areas
    ### For other areas it is set according to data for the country in question
    ### Capacity ONLY used to estimate the shares of PV/Offshore/Onshore. Never used to estimate production

    if scenario == 'EF45':

        offshore_capacity = {
            'SE1': 4500,
            'SE2': 4500,
            'SE3': 9500,
            'SE4': 10000,
            'FI': 21845, #NGDP
            'DK2': 24189,
            'NO1': 100,
            'NO2': 500,
            'NO3': 1000,
            'NO4': 0,
            'NO5': 750
        }

        onshore_capacity = {
            'SE1': 8730,
            'SE2': 10430,
            'SE3': 5408,
            'SE4': 2223,
            'FI': 42671, #NGDP
            'DK2': 2506,
            'NO1': 400,
            'NO2': 500,
            'NO3': 2500,
            'NO4': 1509,
            'NO5': 750
        }

        pv_capacity = {
            'SE1': 866,
            'SE2': 835,
            'SE3': 12129,
            'SE4': 5290,
            'FI': 10938, #NGDP
            'DK2': 10837,
            'NO1': 4000,
            'NO2': 500,
            'NO3': 500,
            'NO4': 0,
            'NO5': 100
        }

    elif scenario == 'EP45':

        offshore_capacity = {
            'SE1': 1519,
            'SE2': 1519,
            'SE3': 4838,
            'SE4': 2250,
            'FI': 7761,  # NGDP
            'DK2': 8594,
            'NO1': 500,
            'NO2': 500,
            'NO3': 1000,
            'NO4': 0,
            'NO5': 750
        }

        onshore_capacity = {
            'SE1': 5968,
            'SE2': 8760,
            'SE3': 6433,
            'SE4': 2802,
            'FI': 37737,  # NGDP
            'DK2': 2216,
            'NO1': 1000,
            'NO2': 500,
            'NO3': 2500,
            'NO4': 1509,
            'NO5': 750
        }

        pv_capacity = {
            'SE1': 92,
            'SE2': 501,
            'SE3': 7278,
            'SE4': 3174,
            'FI': 6318,  # NGDP
            'DK2': 6260,
            'NO1': 4000,
            'NO2': 500,
            'NO3': 500,
            'NO4': 0,
            'NO5': 100
        }

    elif scenario == 'SF45':

        offshore_capacity = {
            'SE1': 0,
            'SE2': 269,
            'SE3': 312,
            'SE4': 794,
            'FI': 1054,  # NGDP
            'DK2': 1167,
            'NO1': 500,
            'NO2': 500,
            'NO3': 1000,
            'NO4': 0,
            'NO5': 750
        }

        onshore_capacity = {
            'SE1': 4583,
            'SE2': 7812,
            'SE3': 6798,
            'SE4': 1985,
            'FI': 33733,  # NGDP
            'DK2': 1981,
            'NO1': 1000,
            'NO2': 500,
            'NO3': 2500,
            'NO4': 1509,
            'NO5': 750
        }
        pv_capacity = {
            'SE1': 1597,
            'SE2': 3049,
            'SE3': 17299,
            'SE4': 7159,
            'FI': 16649,  # NGDP
            'DK2': 16496,
            'NO1': 8000,
            'NO2': 500,
            'NO3': 500,
            'NO4': 0,
            'NO5': 100
        }

    elif scenario == 'FM45':

        offshore_capacity = {
            'SE1': 0,
            'SE2': 725,
            'SE3': 2900,
            'SE4': 3625,
            'FI': 5557,  # NGDP
            'DK2': 6153,
            'NO1': 500,
            'NO2': 500,
            'NO3': 1000,
            'NO4': 0,
            'NO5': 750
        }

        onshore_capacity = {
            'SE1': 5963,
            'SE2': 7978,
            'SE3': 8060,
            'SE4': 2247,
            'FI': 38620,  # NGDP
            'DK2': 2268,
            'NO1': 1000,
            'NO2': 500,
            'NO3': 2500,
            'NO4': 1509,
            'NO5': 750
        }

        pv_capacity = {
            'SE1': 74,
            'SE2': 402,
            'SE3': 5835,
            'SE4': 2545,
            'FI': 5065,  # NGDP
            'DK2': 5019,
            'NO1': 8000,
            'NO2': 500,
            'NO3': 500,
            'NO4': 0,
            'NO5': 100
        }


    # MAF PECD 2019 data is read. Perhaps could use CorRes time series instead. But likely not too large difference
    gen_dict = {}

    # with open('C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Simuleringsdata\\ERAA_22\\PECD-MAF2019-Nordic.pickle','rb') as handle:
    #     maf_data = pkl.load(handle)
    maf_data = pd.read_pickle('C:\\Users\\hnordstr\\OneDrive - KTH\\box_files\\KTH\\Simuleringsdata\\ERAA_22\\PECD-MAF2019-Nordic.pickle')

    # Create one dataframe per MAF area
    maf_dict = {}
    for a in maf_areas:
        maf_dict[a] = pd.DataFrame(columns=['WindOffshore', 'WindOnshore', 'SolarPV'])
        # Offshore wind
        maf_dict[a]['WindOffshore'] = maf_data['WindOffshore'][a][f'{year}'][:days*24].astype(float).tolist()
        maf_dict[a]['WindOnshore'] = maf_data['WindOnshore'][a][f'{year}'][:days * 24].astype(float).tolist()
        maf_dict[a]['SolarPV'] = maf_data['SolarPV'][a][f'{year}'][:days * 24].astype(float).tolist()
        # print(f'Min offsh {a}: {maf_dict[a]["WindOffshore"].min()}')
        # print(f'Min onsh {a}: {maf_dict[a]["WindOnshore"].min()}')
        # print(f'Min PV {a}: {maf_dict[a]["SolarPV"].min()}')


    for a in areas:
        gen_dict[a] = pd.DataFrame(columns=['WindOffshore', 'WindOnshore', 'SolarPV', 'RESTot', 'LMARES'])
        offshore_share = offshore_capacity[a] / (offshore_capacity[a] + onshore_capacity[a] + pv_capacity[a])
        onshore_share = onshore_capacity[a] / (offshore_capacity[a] + onshore_capacity[a] + pv_capacity[a])
        pv_share = pv_capacity[a] / (offshore_capacity[a] + onshore_capacity[a] + pv_capacity[a])
        capacity_factor = pd.DataFrame(columns=['WindOffshore', 'WindOnshore', 'SolarPV'])

        ### Determining capacity factor per hour for each category - between 0 and 1

        if a == 'SE1':
            capacity_factor['WindOffshore'] = maf_dict['SE01']['WindOffshore']
            capacity_factor['WindOnshore'] = maf_dict['SE01']['WindOnshore']
            capacity_factor['SolarPV'] = maf_dict['SE01']['SolarPV']
        elif a == 'SE2':
            capacity_factor['WindOffshore'] = maf_dict['SE02']['WindOffshore']
            capacity_factor['WindOnshore'] = maf_dict['SE02']['WindOnshore']
            capacity_factor['SolarPV'] = maf_dict['SE02']['SolarPV']
        elif a == 'SE3':
            capacity_factor['WindOffshore'] = maf_dict['SE03']['WindOffshore']
            capacity_factor['WindOnshore'] = maf_dict['SE03']['WindOnshore']
            capacity_factor['SolarPV'] = maf_dict['SE03']['SolarPV']
        elif a == 'SE4':
            capacity_factor['WindOffshore'] = maf_dict['SE04']['WindOffshore']
            capacity_factor['WindOnshore'] = maf_dict['SE04']['WindOnshore']
            capacity_factor['SolarPV'] = maf_dict['SE04']['SolarPV']
        elif a == 'FI':
            capacity_factor['WindOffshore'] = maf_dict['FI00']['WindOffshore']
            capacity_factor['WindOnshore'] = maf_dict['FI00']['WindOnshore']
            capacity_factor['SolarPV'] = maf_dict['FI00']['SolarPV']
        elif a == 'DK2':
            capacity_factor['WindOffshore'] = maf_dict['DKE1']['WindOffshore']
            capacity_factor['WindOnshore'] = maf_dict['DKE1']['WindOnshore']
            capacity_factor['SolarPV'] = maf_dict['DKE1']['SolarPV']
        elif a == 'NO1':
            capacity_factor['WindOffshore'] = (0.0458123185083295 * maf_dict['NOM1']['WindOffshore'] + 0.124666699093952 * maf_dict['NOS0']['WindOffshore']) / (0.0458123185083295 + 0.124666699093952)
            capacity_factor['WindOnshore'] = (0.0458123185083295 * maf_dict['NOM1']['WindOnshore'] + 0.124666699093952 * maf_dict['NOS0']['WindOnshore']) / (0.0458123185083295 + 0.124666699093952)
            capacity_factor['SolarPV'] = maf_dict['NOM1']['SolarPV']
        elif a == 'NO2':
            capacity_factor['WindOffshore'] = maf_dict['NOS0']['WindOffshore']
            capacity_factor['WindOnshore'] = maf_dict['NOS0']['WindOnshore']
            capacity_factor['SolarPV'] = maf_dict['NOM1']['SolarPV']
        elif a == 'NO3':
            capacity_factor['WindOffshore'] = (0.0303895164207903 * maf_dict['NON1']['WindOffshore'] + 0.753075806205105 * maf_dict['NOM1']['WindOffshore'] + 0.0109663183681411 * maf_dict['NOS0']['WindOffshore']) / (0.0303895164207903 + 0.753075806205105 + 0.0109663183681411)
            capacity_factor['WindOnshore'] = (0.0303895164207903 * maf_dict['NON1']['WindOnshore'] + 0.753075806205105 * maf_dict['NOM1']['WindOnshore'] + 0.0109663183681411 * maf_dict['NOS0']['WindOnshore']) / (0.0303895164207903 + 0.753075806205105 + 0.0109663183681411)
            capacity_factor['SolarPV'] = (0.0303895164207903 * maf_dict['NON1']['SolarPV'] + 0.753075806205105 * maf_dict['NOM1']['SolarPV']) / (0.0303895164207903 + 0.753075806205105)
        elif a == 'NO4':
            capacity_factor['WindOffshore'] = maf_dict['NON1']['WindOffshore']
            capacity_factor['WindOnshore'] = maf_dict['NON1']['WindOnshore']
            capacity_factor['SolarPV'] = maf_dict['NON1']['SolarPV']
        elif a == 'NO5':
            capacity_factor['WindOffshore'] = (0.201111875286566 * maf_dict['NOM1']['WindOffshore'] + 0.443269201006425 * maf_dict['NOS0']['WindOffshore']) / (0.201111875286566 + 0.443269201006425)
            capacity_factor['WindOnshore'] = (0.201111875286566 * maf_dict['NOM1']['WindOnshore'] + 0.443269201006425 * maf_dict['NOS0']['WindOnshore']) / (0.201111875286566 + 0.443269201006425)
            capacity_factor['SolarPV'] = maf_dict['NOM1']['SolarPV']

        gen_dict[a]['LMARES'] = lma[a].tolist()

        pv = []
        offshore = []
        onshore = []

        ### Computing generation as LMA_RES * (CF_type * share_type) /(CF_PV * share_PV + CF_Offsh * share_offsh + CF_Onsh * share_Onsh)
        ### Renewable energy will every hour be same as in LMA. Could improve by ensuring zero sun in night time

        for i in range(len(gen_dict[a])):
            if capacity_factor['SolarPV'][i] * pv_share == 0 and capacity_factor['WindOnshore'][i] * onshore_share == 0 \
                and capacity_factor['WindOffshore'][i] * offshore_share == 0:
                # Few occurrences with CF for all categories being zero (i.e div by zero with calc method). Here LMA energy is split by wind shares.
                pv.append(0)
                onshore.append(gen_dict[a]['LMARES'][i] * onshore_share / (onshore_share + offshore_share))
                offshore.append(gen_dict[a]['LMARES'][i] * offshore_share / (onshore_share + offshore_share))
            elif pv_share > onshore_share + offshore_share and capacity_factor['SolarPV'][i] > 0:# and capacity_factor['SolarPV'][i] < 0.3 :
                #pv.append(capacity_factor['SolarPV'][i] * pv_capacity[a])
                if capacity_factor['WindOnshore'][i] * onshore_capacity[a] + capacity_factor['WindOffshore'][i] * offshore_capacity[a] > gen_dict[a]['LMARES'][i]:
                    onshore.append(min(capacity_factor['WindOnshore'][i] * onshore_capacity[a], gen_dict[a]['LMARES'][i]))
                    offshore.append(0)
                    pv.append((gen_dict[a]['LMARES'][i] - (onshore[i] + offshore[i])))
                else:
                    onshore.append(capacity_factor['WindOnshore'][i] * onshore_capacity[a])
                    offshore.append(capacity_factor['WindOffshore'][i] * offshore_capacity[a])
                    pv.append((gen_dict[a]['LMARES'][i] - (onshore[i] + offshore[i])))
            else:
                pv.append(gen_dict[a]['LMARES'][i] * pv_share * capacity_factor['SolarPV'][i] / (pv_share * capacity_factor['SolarPV'][i] + onshore_share * capacity_factor['WindOnshore'][i] + offshore_share * capacity_factor['WindOffshore'][i]))
                onshore.append(gen_dict[a]['LMARES'][i] * onshore_share * capacity_factor['WindOnshore'][i] / (pv_share * capacity_factor['SolarPV'][i] + onshore_share * capacity_factor['WindOnshore'][i] + offshore_share * capacity_factor['WindOffshore'][i]))
                offshore.append(gen_dict[a]['LMARES'][i] * offshore_share * capacity_factor['WindOffshore'][i] / (pv_share * capacity_factor['SolarPV'][i] + onshore_share * capacity_factor['WindOnshore'][i] + offshore_share * capacity_factor['WindOffshore'][i]))
        gen_dict[a]['WindOnshore'] = onshore
        gen_dict[a]['WindOffshore'] = offshore
        gen_dict[a]['SolarPV'] = pv
        gen_dict[a]['RESTot'] = gen_dict[a]['SolarPV'] + gen_dict[a]['WindOffshore'] + gen_dict[a]['WindOnshore']
    return gen_dict


# Saving files to pickle
# with open(f'{lma_path}{scenario}\\RES_{scenario}.pickle', 'wb') as handle:
#     pkl.dump(gen_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)