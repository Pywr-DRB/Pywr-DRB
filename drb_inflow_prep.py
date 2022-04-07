### prepare inflow data for NYC reservoirs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

names = ['cannonsville', 'pepacton', 'neversink', 'lackawaxen']
filenames = [f'input_data/{name}_discharge.txt' for name in names]
headers = [30, 29, 30, 29]
capacities = {'cannonsville': 96000, 'pepacton': 140000, 'neversink': 35000, 'lackawaxen': 0}

reservoirs = {}
for filename, name, header in zip(filenames, names, headers):
    res = pd.read_csv(filename, sep='\t', skiprows=[i for i in range(header+1)] +[header+2])
    res['datetime'] = pd.to_datetime(res['datetime'])
    res = res.iloc[:, [2, 4]]
    res.columns = ['datetime', 'discharge']

    ### coerce to numeric, missing values will become nan
    discharge = res['discharge']
    discharge = discharge.apply(pd.to_numeric, errors='coerce')

    ### now replace nan with most recent value
    for i in range(res.shape[0]):
        if np.isnan(discharge[i]):
            discharge[i] = discharge[i-1]
    res['discharge'] = discharge

    ### units are cfs -> get mean cfs, then convert to GPD
    res_daily = res.resample('D', on='datetime').mean() * 0.64631688311

    ### now fill in missing days with previous
    for i in range(res_daily.shape[0]):
        if np.isnan(res_daily.iloc[i, 0]):
            res_daily.iloc[i, 0] = res_daily.iloc[i-1, 0]

    ### keep data 10/1/1990 to 9/30/2021
    res_daily = res_daily[res_daily.index > pd.to_datetime('1990-09-30')]
    res_daily = res_daily[res_daily.index < pd.to_datetime('2021-10-01')]

    ### for now, for testing, actuallly use same flow each day, and increase to acct for demand
    # res_daily['discharge'] = 650.
    res_daily['discharge'] = res_daily['discharge'].mean() + capacities[name] / sum(capacities.values()) * (800)

    plt.plot(res_daily['discharge'])
    plt.title(name)
    plt.show()

    reservoirs[name] = res_daily

### write to csv
for r, name in enumerate(reservoirs):
    if r == 0:
        df = reservoirs[name]
        df.columns = [name]
    else:
        df[name] = reservoirs[name]
df.to_csv('input_data/inflows_clean.csv')
