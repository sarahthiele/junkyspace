import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from fromaaron.asat_sma import period
import rebound.units as u
from functions import *
Mearthkg = u.masses_SI['mearth']
Msunkg = u.masses_SI['solarmass']
Mearth = Mearthkg / Msunkg
G = u.G_SI

nsbm1 = pd.read_hdf('sat2021/sims/russia/NSBM/data_0.003_1000_0.15_10.0.hdf', key='data')#.iloc[:-1]
nsbm2 = pd.read_hdf('sat2021/sims/russia/NSBM/data_0.1_1000_0.15_10.0.hdf', key='data')#.iloc[:-1]
r1 = pd.read_hdf('sat2021/sims/russia/rayleigh/data_0.04_1000_0.15_10.0.hdf', key='data')#.iloc[:-1]
r2 = pd.read_hdf('sat2021/sims/russia/rayleigh/data_0.05_1000_0.15_10.0.hdf', key='data')#.iloc[:-1]
rlist = [nsbm1, nsbm2, r1, r2]

tday = twopi/365.25
fig, ax = plt.subplots(figsize=(8,8))
labels = [r'NSBM, L$_{ c,\rm{min}}$=3 mm', r'NSBM, L$_{ c,\rm{min}}$=10 cm', 
         'Rayleigh, A/M0.04','Rayleigh, A/M0.05']
ls = ['-', '--', '-.', '-.']
colors = ['xkcd:sky blue', 'xkcd:dark blue', 'xkcd:royal blue',
         'xkcd:periwinkle']
for i, df in enumerate(rlist):
    time = df
    time = time.loc[(time.t_deorbit>=5*tday)&(time.t_deorbit<1e9)]
    NFOLLOW = len(time)
    time = time.loc[time.t_deorbit<1e6]
    tplot = time.groupby('t_deorbit').first().index.values
    cumsum = np.cumsum(time.groupby('t_deorbit').size().values)
    plt.plot(tplot/twopi, cumsum/NFOLLOW, lw=2, alpha=1.0, 
            label=labels[i], ls=ls[i], color=colors[i])

tR = pd.read_hdf('russia_deorbit.hdf', key='data')
tR = tR.loc[(tR.t_deorbit>=5*tday)]
NFOLLOW = 1511
tplot = tR.groupby('t_deorbit').first().index.values
cumsum = np.cumsum(tR.groupby('t_deorbit').size().values)
plt.plot(tplot, cumsum/NFOLLOW, lw=2, color='r', label='Russia ASAT test catalogued, 3 mos.')

ax.tick_params(labelsize=16)
plt.xlabel('Time (yrs)', fontsize=20)
plt.ylabel('Fraction of Deorbited Fragments', fontsize=20)
plt.legend(fontsize=18)
plt.savefig('russia_checks.pdf')
