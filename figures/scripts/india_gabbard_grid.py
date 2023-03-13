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

r1 = pd.read_hdf('sat12000/sims/rayleigh/data_0.05_1000_0.13_3.0.hdf', key='data')
nsbm1 = pd.read_hdf('sat12000/sims/NSBM/data_0.1_1000_0.13_3.0.hdf', key='data')
dlist = [nsbm1, r1]
md = np.genfromtxt('microsatR_gabbard.txt', usecols=[1,3,5,7])
md = pd.DataFrame(md, columns=['a','q','Q','P'])

fig, ax = plt.subplots(1,3,figsize=(25,8))
for i, d in enumerate(dlist):
    data = d.loc[d.ecc<1.0]
    SMA = data.SMA.values
    eccs = data.ecc.values
    porb = period(data.SMA.values, G*Mearthkg)
    flag = (SMA*(1-eccs)/1000-REkm>200)
    ax[i].scatter(porb[flag]/3600, (SMA*(1-eccs)/1000-REkm)[flag], s=5, color='r', label='Perigee (q)')
    ax[i].scatter(porb[flag]/3600, (SMA*(1+eccs)/1000-REkm)[flag], s=5, color='b', label='Apogee (Q)')

ax[2].scatter(md.P.values/60, md.q.values-REkm, s=5, color='r',label='Perigee (q)')
ax[2].scatter(md.P.values/60, md.Q.values-REkm, s=5, color='b', label='Apogee (Q)')

for i in range(3):
    ax[i].scatter(89.9/60, 283, s=10, facecolor='xkcd:bright cyan', marker='*',
                  edgecolor='k', label='India ASAT')
    ax[i].scatter(89.9/60, 283, s=400, facecolor="xkcd:bright cyan", marker='*', edgecolor='k')
    ax[i].axhline(417., ls='--', lw=2, color='xkcd:dark grey', label='ISS & Tiangong', zorder=0)
    ax[i].set_ylim(0, 2550)
    ax[i].set_xlim(1.45, 1.875) 
    ax[i].tick_params(labelsize=26)
    ax[i].set_xlabel(r'P$_{\rm{orb}}$ (hr)', fontsize=30)
ax[1].text(0.045, 0.9, 'Rayleigh', 
                     transform=ax[1].transAxes, fontsize=28)
ax[0].text(0.045, 0.47, 'NSBM\n'+r'$(\geq10$ cm)', 
                     transform=ax[0].transAxes, fontsize=28)
ax[2].text(0.05, 0.9, 'Microsat-R debris', 
                     transform=ax[2].transAxes, fontsize=28)

ax[0].legend(loc='upper left', frameon=True, fontsize=24, markerscale=6)
ax[0].set_ylabel('Altitude (km)', fontsize=30)
plt.subplots_adjust(wspace=0.2)
#plt.savefig('gabbard_grid_India.pdf')
plt.savefig('gabbard_grid_India.png')
#plt.show()
