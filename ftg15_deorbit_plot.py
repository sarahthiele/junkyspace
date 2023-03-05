import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tools.asat_sma import period
import rebound.units as u
from functions import *
Mearthkg = u.masses_SI['mearth']
Msunkg = u.masses_SI['solarmass']
Mearth = Mearthkg / Msunkg
G = u.G_SI

df = pd.read_hdf('sat2019/sims/FTG15/NSBM/data_0.003_5000_0.5_0.0192.hdf', key='data')
rlist = [df]
tday = twopi/365.25
fig, ax = plt.subplots(figsize=(8,8))
labels = [r'NSBM, L$_{ c,\rm{min}}$=3 mm', r'NSBM, L$_{ c,\rm{min}}$=10 cm', 
         'Rayleigh, A/M0.04','Rayleigh, A/M0.05']
ls = ['-', '--', '-.', '-.']
colors = ['xkcd:sky blue', 'xkcd:dark blue', 'xkcd:royal blue',
         'xkcd:periwinkle']
for i, df in enumerate(rlist):
    time = df
    time = time.loc[(time.t_deorbit>0)&(time.t_deorbit<1e9)]
    NFOLLOW = len(time)
    time = time.loc[time.t_deorbit<1e6]
    tplot = time.groupby('t_deorbit').first().index.values
    cumsum = np.cumsum(time.groupby('t_deorbit').size().values)
    plt.plot(tplot/twopi, cumsum/NFOLLOW, lw=2, alpha=1.0, 
            label=labels[i], ls=ls[i], color=colors[i])
    plt.scatter(tplot/twopi, cumsum/NFOLLOW, s=5)


ax.tick_params(labelsize=16)
plt.xlabel('Time (yrs)', fontsize=20)
plt.ylabel('Fraction of Deorbited Fragments', fontsize=20)
plt.legend(fontsize=18)
plt.savefig('ftg15_deorbit.png')
