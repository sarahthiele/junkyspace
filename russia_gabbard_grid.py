import pandas as pd
import numpy as np
import matplotlib.pyplot as  plt
from matplotlib.gridspec import GridSpec

from functions import *
import fromaaron.KeplerTools as KT

datinit = np.genfromtxt('satcat_Feb10.csv', skip_header=1, dtype=str, delimiter=',')
datinit = pd.DataFrame(datinit)
headers = np.genfromtxt('satcat_Feb10.csv', dtype=str, delimiter=',', max_rows=1)
datinit.columns = headers

decayed = datinit.loc[datinit.DECAY!='""']
datremain = datinit.drop(decayed.index.values)
dat = datremain[['OBJECT_TYPE', 'DECAY', 'PERIOD', 'INCLINATION',
                 'APOGEE', 'PERIGEE', 'RCS_SIZE', 'LAUNCH_YEAR']]

excepts = []
for i in range(len(dat)):
    try:
        dat.iloc[i].PERIOD = float(dat.iloc[i].PERIOD[1:-1])
        dat.iloc[i].INCLINATION = float(dat.iloc[i].INCLINATION[1:-1])
        dat.iloc[i].APOGEE = float(dat.iloc[i].APOGEE[1:-1])
        dat.iloc[i].PERIGEE = float(dat.iloc[i].PERIGEE[1:-1])
    except:
        excepts.append(i)
excepts = np.array(excepts)
indices = dat.iloc[excepts].index.values
dat = dat.drop(indices)
dat = dat.drop(dat.loc[dat.PERIOD==0.].index.values)

dat['SMA'] = ((dat.PERIGEE + dat.APOGEE) / 2 + REkm) * 1000
dat['e'] = (dat.APOGEE - dat.PERIGEE) / (dat.APOGEE + dat.PERIGEE + 2*REkm)
dat['n'] = 2 * np.pi / (dat.PERIOD.values * 60)

dat = dat.drop(dat.loc[dat.PERIGEE.values > 2500].index.values)


fig = plt.figure(figsize=(24, 12))
gs = GridSpec(nrows=2, ncols=4)

r1 = pd.read_hdf('sat2021/sims/russia/NSBM/data_0.1_1000_0.15_10.0.hdf', key='data')
r1mid = pd.read_hdf('sat2021/sims/russia/NSBM/gabbard_0.25_LEO_0.1_1000_0.15_10.0.hdf', key='data')
r1mid['ecc'] = r1mid.e.values
dlist = [r1, r1mid]
labels = ['NSBM\ninit.', 'NSBM\n3 mos.']
for i, d in enumerate(dlist):
    data = d.loc[d.ecc<1.0]
    SMA = data.SMA.values
    eccs = data.ecc.values
    porb = period(data.SMA.values, G*Mearthkg)
    flag = (SMA*(1-eccs)/1000-REkm>200)
    ax = fig.add_subplot(gs[1, i])
    ax.scatter(porb[flag]/3600, (SMA*(1-eccs)/1000-REkm)[flag], s=5, color='r', label='Perigee (q)')
    ax.scatter(porb[flag]/3600, (SMA*(1+eccs)/1000-REkm)[flag], s=5, color='b', label='Apogee (Q)')
    ax.scatter(94.3/60, 480, s=5, facecolor='xkcd:bright cyan', marker='*',
                  edgecolor='k', label='Cosmos 1408')
    ax.scatter(94.3/60, 480, s=650, facecolor="xkcd:bright cyan", marker='*', edgecolor='k')
    ax.axhline(417., ls='--', lw=2, color='xkcd:dark grey', label='ISS & Tiangong', zorder=0)
    ax.set_ylim(0, 2600)
    ax.set_xlim(1.45, 2.01) 
    ax.tick_params(labelsize=32)
    ax.set_xlabel(r'P$_{\rm{orb}}$ (hr)', fontsize=36)
    ax.text(0.065, 0.75, labels[i], transform=ax.transAxes, fontsize=32)
    if i == 0:
        ax.set_ylabel('Altitude (km)', fontsize=36)

r1 = pd.read_hdf('sat2021/sims/russia/rayleigh/data_0.05_1000_0.15_10.0.hdf', key='data')
r1mid = pd.read_hdf('sat2021/sims/russia/rayleigh/gabbard_0.25_LEO_0.05_1000_0.15_10.0.hdf', key='data')
r1mid['ecc'] = r1mid.e.values
dlist = [r1, r1mid]
labels = ['Rayleigh\ninit.', 'Rayleigh\n3 mos.']

for i, d in enumerate(dlist):
    data = d.loc[d.ecc<1.0]
    SMA = data.SMA.values
    eccs = data.ecc.values
    porb = period(data.SMA.values, G*Mearthkg)
    ax = fig.add_subplot(gs[0,i])
    flag = (SMA*(1-eccs)/1000-REkm>200)
    ax.scatter(porb[flag]/3600, (SMA*(1-eccs)/1000-REkm)[flag], s=5, color='r', label='Perigee (q)')
    ax.scatter(porb[flag]/3600, (SMA*(1+eccs)/1000-REkm)[flag], s=5, color='b', label='Apogee (Q)')
    ax.scatter(94.3/60, 480, s=5, facecolor='xkcd:bright cyan', marker='*',
                  edgecolor='k', label='Cosmos 1408')
    ax.scatter(94.3/60, 480, s=650, facecolor="xkcd:bright cyan", marker='*', edgecolor='k')
    ax.axhline(417., ls='--', lw=2, color='xkcd:dark grey', label='ISS & Tiangong', zorder=0)
    ax.set_ylim(0, 2600)
    ax.set_xlim(1.45, 2.01) 
    ax.tick_params(labelsize=32)
    ax.set_xlabel(r'P$_{\rm{orb}}$ (hr)', fontsize=36)
    ax.text(0.065, 0.75, labels[i], transform=ax.transAxes, fontsize=32)
    if i == 0:
        ax.set_ylabel('Altitude (km)', fontsize=36)

plt.subplots_adjust(hspace=0.3)
fig.tight_layout(w_pad=0.32)

ax0 = fig.add_subplot(gs[:,2:])
ax0.scatter(dat.PERIOD.values/60, dat.PERIGEE.values, s=10, color='r',label='Perigee (q)')
ax0.scatter(dat.PERIOD.values/60, dat.APOGEE.values, s=10, color='b', label='Apogee (Q)')
ax0.text(0.035, 0.845, 'Cosmos-1408\ndebris,\n3 mos.', transform=ax0.transAxes, fontsize=32)
ax0.scatter(94.3/60, 480, s=30, facecolor='xkcd:bright cyan', marker='*',
              edgecolor='k', label='Russia ASAT')
ax0.scatter(94.3/60, 480, s=900, facecolor="xkcd:bright cyan", marker='*', edgecolor='k')
ax0.axhline(417., ls='--', lw=2, color='xkcd:dark grey', label='ISS & Tiangong', zorder=0)
ax0.set_ylim(0, 2600)
ax0.set_xlim(1.45, 2.01) 
ax0.tick_params(labelsize=32)
ax0.set_xlabel(r'P$_{\rm{orb}}$ (hr)', fontsize=36)
ax0.legend(loc='upper right', frameon=True, fontsize=36, markerscale=5)


plt.savefig('gabbard_grid_Russia.pdf')
plt.show()
