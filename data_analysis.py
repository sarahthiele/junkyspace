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

nsbm1 = pd.read_hdf('satall/sims/russia/NSBM/data_0.1_1000_0.15_10.0.hdf', key='data')
r1 = pd.read_hdf('satall/sims/russia/rayleigh/data_0.05_1000_0.15_10.0.hdf', key='data')
r2 = pd.read_hdf('satall/sims/russia/rayleigh/data_0.04_1000_0.15_10.0.hdf', key='data')
nsbm2 = pd.read_hdf('satall/sims/russia/NSBM/data_0.003_1000_0.15_10.0.hdf', key='data')
scales = [2876, 2876, 2876, 672936, 672936, 672936]
rlist = [nsbm1, r1, r2, nsbm2, r1, r2]
for i, df in enumerate(rlist):
    colprob = (1-np.exp(-np.sum(df.colprob.values)*scales[i]/len(df)))*100
    print('scaled: ', colprob)
    colprob = (1-np.exp(-np.sum(df.colprob.values)))*100
    print('unscaled: ', colprob, '\n')

print('2021:')
nsbm1 = pd.read_hdf('sat2021/sims/russia/NSBM/data_0.1_1000_0.15_10.0.hdf', key='data')#.iloc[:-1]
r1 = pd.read_hdf('sat2021/sims/russia/rayleigh/data_0.05_1000_0.15_10.0.hdf', key='data')#.iloc[:-1]
r2 = pd.read_hdf('sat2021/sims/russia/rayleigh/data_0.04_1000_0.15_10.0.hdf', key='data')#.iloc[:-1]
nsbm2 = pd.read_hdf('sat2021/sims/russia/NSBM/data_0.003_1000_0.15_10.0.hdf', key='data')#.iloc[:-1]
scales = [2876, 2876, 2876, 672936, 672936, 672936]
rlist = [nsbm1, r1, r2, nsbm2, r1, r2]
for i, df in enumerate(rlist):
    colprob = (1-np.exp(-np.sum(df.colprob.values)*scales[i]/len(df)))*100
    print('scaled: ', colprob)
    colprob = (1-np.exp(-np.sum(df.colprob.values)))*100
    print('unscaled: ', colprob, '\n')

