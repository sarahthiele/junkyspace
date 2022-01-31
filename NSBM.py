import sys
sys.path.insert(1, '/store/users/sthiele/home/ASATtest/')
from NSBM_functions import *
import pandas as pd
import argparse

path = 'path/NSBM'

SEED=314

np.random.seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument("--numsample", default=1000, type=int)
parser.add_argument("--Lcmin", default=0.01, type=float)
parser.add_argument("--satdistro", default='satsnow', type=str)
args = parser.parse_args()

numsample = int(args.numsample)
altref = 300
deorbit_R = 200.

#grid params for interpolation
satdistro = str(args.satdistro)
if satdistro == 'satsnow':
    NTHETA= 180
    NALT=1200
    altCoMin=(300.+REkm)/aukm
    altCoMax=(1500+REkm)/aukm
    
elif satdistro == 'sats2019':
    NTHETA= 180
    NALT=1700
    altCoMin=(300.+REkm)/aukm
    altCoMax=(2000+REkm)/aukm
    
dAltCo = (altCoMax-altCoMin)/NALT
dThetaCo = np.pi/NTHETA

mtarget = 740
mkill = 10
vkill = 3.4e3
Q = 288.7 + REkm
q = 267.4 + REkm
r = 283 + REkm
a = (Q + q) * 1000 / 2
vr = np.sqrt(G * (Mearthkg + mtarget) * (2/(r*1000) - 1/a))
inc = 96.6 * np.pi / 180
omega=17 * np.pi / 180
vtarget, rtarget = get_target_params(mtarget, vr, r, Q, inc, omega)

nbins = 100
Lc_min = float(args.Lcmin)
Lc_max = 1.0
KEkill = 130e6

N_tot, L_mids, nums, mfrags, vfrags_all, vfrags, vfrags_total, eccs, SMA, AMfrags_all, AMfrags = vel_dis_NBM(mtarget,
                                                                                                             mkill, 
                                                                                                             vkill, 
                                                                                                             vtarget,
                                                                                                             rtarget,
                                                                                                             nbins, 
                                                                                                             Lc_min,
                                                                                                             Lc_max,
                                                                                                             KEkill,
                                                                                                             numsample,
                                                                                                            makev=True)


fig, ax = plt.subplots(1, 4, figsize=(24, 6))
ax[0].plot(L_mids, nums, color='k', lw=3)
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_ylabel(r'N$_{\rm{Frags}}$ per Bin', fontsize=20)
ax[0].text(0.7, 0.9, r'N$_{\rm{TOT}}$='+str(N_tot), fontsize=18, horizontalalignment='center', 
           transform=ax[0].transAxes)

bins = np.logspace(np.log10(np.linalg.norm(vfrags_all, 
                                           axis=1).min()), 
                   np.log10(np.linalg.norm(vfrags_all, axis=1)).max(), 
                   50)
ax[1].hist(np.linalg.norm(vfrags_all, axis=1), bins=bins, 
           histtype='step', color='k', lw=3)
ax[1].set_xlabel(r'$\Delta v$ (m/s))', fontsize=20)
ax[1].set_ylabel('Counts', fontsize=20)
ax[1].set_xscale('log')

ax[2].hist(np.log10(AMfrags_all), bins=100, histtype='step', 
           color='k', lw=3)
ax[2].set_xlabel('Log$_{10}$(A/M ratio)', fontsize=20)
ax[2].set_ylabel('Counts', fontsize=20)

ax[3].hist(np.log10(mfrags), bins=100, histtype='step', 
           color='k', lw=3)
ax[3].set_xlabel('Log$_{10}$(Mass/kg)', fontsize=20)
ax[3].set_ylabel('Counts', fontsize=20)

for i in range(4):
    ax[i].tick_params(labelsize=18)
    
plt.subplots_adjust(wspace=0.35)
ax[0].text(0.5, 1.075, 
           'Mtarget={:.3} ton, Vtarget={:.3} km/s, Mkill={} kg, Kill Energy={:.2} GJ, Intercept Height={} km'.format(mtarget*1./1000, np.round(mag(vtarget), 1)/1e3, mkill, KEkill/1e9, r-REkm), fontsize=22, transform=ax[0].transAxes)

plt.savefig('{}/init_dis_{}_{}.png'.format(path, Lc_min, numsample), bbox_inches='tight')

vfrags_init_deorbit = vfrags[SMA/1000<=REkm+deorbit_R]
AMfrags_init_deorbit = AMfrags[SMA/1000<=REkm+deorbit_R]
SMA_init_deorbit = SMA[SMA/1000<=REkm+deorbit_R]
eccs_init_deorbit = eccs[SMA/1000<=REkm+deorbit_R]
init_deorbit = len(vfrags_init_deorbit)

keep = (SMA*(1-eccs)/1000>REkm+deorbit_R)&(np.linalg.norm(vfrags_total, axis=1)<11186)
vfrags = vfrags[keep]
AMfrags = AMfrags[keep]
vfrags_total = vfrags_total[keep] / to_m_per_s
eccs = eccs[keep]
SMA = SMA[keep]

NFOLLOW = init_deorbit + len(vfrags)

porb = period(SMA, G*Mearthkg)
fig, ax = plt.subplots(figsize=(10, 8))
plt.scatter(porb/3600, SMA*(1-eccs)/1000-REkm, s=5, color='r', label='peri')
plt.scatter(porb/3600, SMA*(1+eccs)/1000-REkm, s=5, color='b', label='apo')
plt.axhline(283, ls='-.', lw=1.5, color='xkcd:light blue', label='Microsat-R')
plt.axhline(400., ls='--', lw=1.5, color='k', label='ISS')
plt.xlabel(r'P$_{\rm{orb}}$ (hr)', fontsize=16)
plt.ylabel('Altitude (km)', fontsize=16)
plt.legend(fontsize=14)
ax.tick_params(labelsize=14)
plt.savefig('{}/init_gabbard_kept_{}_{}.png'.format(path, Lc_min, numsample), bbox_inches='tight')

flag = SMA*(1+eccs)/1000-REkm <= 20000
fig, ax = plt.subplots(figsize=(10, 8))
plt.scatter(porb[flag]/3600, (SMA*(1-eccs)/1000-REkm)[flag], s=5, color='r', label='peri')
plt.scatter(porb[flag]/3600, (SMA*(1+eccs)/1000-REkm)[flag], s=5, color='b', label='apo')
plt.axhline(283, ls='-.', lw=1.5, color='xkcd:light blue', label='Microsat-R')
plt.axhline(400., ls='--', lw=1.5, color='k', label='ISS')
plt.xlabel(r'P$_{\rm{orb}}$ (hr)', fontsize=16)
plt.ylabel('Altitude (km)', fontsize=16)
plt.legend(fontsize=14)
ax.tick_params(labelsize=14)
plt.savefig('{}/init_gabbard_kept_alt2e4_{}_{}.png'.format(path, Lc_min, numsample), bbox_inches='tight')

fig, ax = plt.subplots(1, 2, figsize=(13, 6))

ax[0].hist(np.linalg.norm(vfrags, axis=1), bins=bins, 
           histtype='step', color='k', lw=3)
ax[0].set_xlabel(r'$\Delta v$ (m/s))', fontsize=20)
ax[0].set_ylabel('Counts', fontsize=20)
ax[0].set_xscale('log')

ax[1].hist(np.log10(AMfrags), bins=50, histtype='step', 
           color='k', lw=3)
ax[1].set_xlabel('Log$_{10}$(A/M ratio)', fontsize=20)
ax[1].set_ylabel('Counts', fontsize=20)

for i in range(2):
    ax[i].tick_params(labelsize=18)
    
plt.subplots_adjust(wspace=0.35)
ax[0].text(0.2, 1.075, 
           'Mtarget={:.3} ton, Vtarget={:.3} km/s, Mkill={} kg,\nKill Energy={:.2} GJ, Intercept Height={} km'.format(mtarget*1./1000, np.round(mag(vtarget), 1)/1e3, mkill, KEkill/1e9, r-REkm), fontsize=20, transform=ax[0].transAxes)

plt.savefig('{}/sampled_dis_{}_{}.png'.format(path, Lc_min, numsample), bbox_inches='tight')


posfrags = rtarget + np.random.uniform(-10, 10, (len(keep[keep]),3))
x = posfrags[:,0] / aum
y = posfrags[:,1] / aum
z = posfrags[:,2] / aum

altref = 300
sim = rebound.Simulation()
sim.integrator ="WHFAST"
#sim.integrator ="ias15"
sim.dt = 1e-7  #  force small time step to we sample the satellite density distribution

sim.add(m=Mearth, hash="Earth", r=Ratm)

for i in range(len(vfrags)):
    sim.add(m=0., vx=vfrags_total[i, 0], vy=vfrags_total[i,1], vz=vfrags_total[i,2],
           x=x[i], y=y[i], z=z[i], hash='{}'.format(i+1))

satparams = [NALT, NTHETA, altref, dAltCo]

halfhr = twopi / (365.25 * 24) / 2
tstart = halfhr
tend = twopi
dt = halfhr
chunk = 20

i = 0
ilast = len(sim.particles)-chunk
if ilast < 0:
    ilast = 0
hash = []
nums = np.linspace(1, len(sim.particles)-1, len(sim.particles)-1)
deorbit_times = np.zeros(init_deorbit)
colprobs = np.zeros(init_deorbit)
colprobperyears = np.zeros(init_deorbit)
chunk_i = 1
while ilast >= 0:
    simchunk = sim.copy()
    for j in range(i-1):
        simchunk.remove(index=1)
    for j in range(ilast):
        simchunk.remove(index=len(simchunk.particles)-1)
    for p in simchunk.particles:hash.append(p.hash.value)
    if i == 0:
        num = nums[0:i-1+chunk]
        AMfrag = AMfrags[0:i-1+chunk]
    else:
        num = nums[i-1:i-1+chunk]
        AMfrag = AMfrags[i-1:i-1+chunk]
    if len(sim.particles) < chunk:
        num = nums
        AMfrag = AMfrags
    simafter, deorbit_time, colprob, colprobperyear = integrate_colprob(simchunk, AMfrag, num, 
                                                                        dt=dt, deorbit_R=deorbit_R, 
                                                                        chunk_i=chunk_i, satparams=satparams)
    deorbit_times = np.append(deorbit_times, deorbit_time)
    colprobs = np.append(colprobs, colprob)
    colprobperyears = np.append(colprobperyears, colprobperyear)
    chunk_i += 1
    print(i, ilast)
    print(len(simchunk.particles))
    if ilast == 0.0:
        break
    i += chunk
    ilast -= chunk
    if ilast < 0:
        ilast = 0
hash = np.array(hash)

timedf = pd.DataFrame(deorbit_times[deorbit_times < 1e6].T, columns=['time'])
cumsum = np.cumsum(timedf.groupby('time').size().values)

fig, ax = plt.subplots(figsize=(10,8))
plt.plot(timedf.groupby('time').first().index.values/twopi, cumsum/NFOLLOW, lw=1.5, color='r')
plt.xlabel('Time (yrs)', fontsize=16)
plt.ylabel('Fraction of Fragments Deorbited', fontsize=16)
ax.tick_params(labelsize=14)
plt.title('{:.4}% deorbited after 2 years'.format(cumsum[-1]/NFOLLOW*100), fontsize=16)
plt.savefig('{}/deorbit_times_{}_{}.png'.format(path, Lc_min, numsample), bbox_inches='tight')

vkicks = np.append(np.linalg.norm(vfrags_init_deorbit, axis=1), np.linalg.norm(vfrags, axis=1))
AM = np.append(AMfrags_init_deorbit, AMfrags)
sma = np.append(SMA_init_deorbit, SMA)
e = np.append(eccs_init_deorbit, eccs)
data = np.array([deorbit_times, colprobs, colprobperyears, vkicks, AM, sma, e])
datadf = pd.DataFrame(data.T, columns=['t_deorbit', 'colprob', 'colprobperyear', 'vkick', 'AM', 'SMA', 'ecc'])
datadf = datadf.sort_values('t_deorbit')
datadf.to_hdf('{}/data_{}_{}.hdf'.format(path, Lc_min, numsample), key='data')

fig, ax = plt.subplots(figsize=(10,8))
flaglo = datadf.t_deorbit.values < 1e6
flaghi = datadf.t_deorbit.values == 1e6
plt.scatter(np.log10(datadf.AM.values[flaghi]), datadf.vkick.values[flaghi]/1000,
            c='r', label=r't$_{\rm{deorbit}} \geq$ 2yrs')
plt.scatter(np.log10(datadf.AM.values[flaglo]), datadf.vkick.values[flaglo]/1000,
            c=datadf.t_deorbit.values[flaglo]/twopi, label=r't$_{\rm{deorbit}}$ < 2yrs')
cb = plt.colorbar()
cb.ax.set_ylabel('Deorbit Time (yrs)', fontsize=16)
cb.ax.tick_params(labelsize=14)
plt.xlabel('Log$_{10}$(A/M ratio)', fontsize=16)
plt.ylabel('Velocity Kick (km/s)', fontsize=16)
plt.legend(loc='best', fontsize=14)
ax.tick_params(labelsize=14)
plt.savefig('{}/AM_vkicks_times_{}_{}.png'.format(path, Lc_min, numsample), bbox_inches='tight')




