#########################################################################################################################
# implementation of the NSBM in the context of an ASAT test
# authored by Sarah Thiele
#########################################################################################################################

import sys
sys.path.insert(1, '/store/users/sthiele/home/junkyspace/')
from NSBM_functions import *
import pandas as pd
import argparse
import os

SEED=314

np.random.seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument("--numsample", default=1000, type=int)
parser.add_argument("--Lcmin", default=0.01, type=float)
parser.add_argument("--satdistro", default='satsnow', type=str)
parser.add_argument("--KEkill", default=130e6, type=float)
parser.add_argument("--path", default='satall', type=str)
parser.add_argument("--event", default='India', type=str)
parser.add_argument("--chunk", default=20, type=int)
parser.add_argument("--maxtime", default=2., type=float)
parser.add_argument("--plottime", default=0.0, type=float)
args = parser.parse_args()

path = str(args.path)
path = '/store/users/sthiele/home/junkyspace/' + str(args.path)
data_path = path + '/sims/NSBM'
if str(args.event) == 'Russia':
    data_path = path + '/sims/russia/NSBM'
# Print the current working directory
print("Current working directory: {0}".format(os.getcwd()))
# Change the current working directory
os.chdir(path)
# Print the current working directory
print("Current working directory: {0}".format(os.getcwd()))
    
event = str(args.event)
numsample = int(args.numsample)
maxtime = float(args.maxtime)
altref = 300
deorbit_R = 200.

#grid params for interpolation
satdistro = str(args.satdistro)
if satdistro == 'satsnow':
    NTHETA = 180
    NALT =1200
    altCoMin =(300.+REkm)/aukm
    altCoMax=(1500+REkm)/aukm
    
elif satdistro == 'sats2019':
    NTHETA= 180
    NALT=1700
    altCoMin=(300.+REkm)/aukm
    altCoMax=(2000+REkm)/aukm
    
dAltCo = (altCoMax-altCoMin)/NALT
dThetaCo = np.pi/NTHETA

#########################################################################################################################
# target parameters

if event == 'India':
    mtarget = 740
    mkill = 10
    vkill = 3.4e3
    Q = 288.7 + REkm
    q = 267.4 + REkm
    r = 283 + REkm
    a = (Q + q) * 1000 / 2
    vr = np.sqrt(G * (Mearthkg + mtarget) * (2/(r*1000) - 1/a))
    inc = 96.6 * np.pi / 180
    omega = 17 * np.pi / 180
elif event == 'Russia':
    mtarget = 2200
    mkill = 10
    vkill = 3.2e3
    Q = 497.5 + REkm
    q = 472.0 + REkm
    r = 490.0 + REkm
    a = (Q + q) * 1000 / 2
    vr = np.sqrt(G * (Mearthkg + mtarget) * (2/(r*1000) - 1/a))
    inc = 82.6 * np.pi / 180
    omega= 0. * np.pi / 180
else:
    print('invalid ASAT event. Choose from ["India", "Russia"].')

vtarget, rtarget = get_target_params(mtarget, vr, r, Q, inc, omega)

#########################################################################################################################
# fragments parameters

nbins = 100
Lc_min = float(args.Lcmin)
Lc_max = 1.0
KEkill = float(args.KEkill)

N_tot, L_mids, nums, mfrags, vfrags_all, vfrags, vfrags_total, eccs, SMA, AMfrags_all, AMfrags, Lcvals_all, Lcvals = vel_dis_NBM(mtarget,
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

if numsample == 100:
    numsample = N_tot

#########################################################################################################################
# plot initial parameter distributions

fig, ax = plt.subplots(1, 4, figsize=(24, 6))
ax[0].plot(L_mids, nums, color='k', lw=3, drawstyle='steps-mid')
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_ylabel(r'N$_{\rm{Frags}}$ per Bin', fontsize=20)
ax[0].set_xlabel(r'L$_{\rm{C}}$ (m)', fontsize=20)
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

plt.savefig('{}/init_dis_{}_{}_{}_{}.png'.format(data_path, Lc_min, numsample, KEkill/1e9, maxtime), bbox_inches='tight')

#########################################################################################################################
# make cuts in SMA

init_deorbit = (SMA*(1-eccs)/1000 <= REkm + deorbit_R)&(eccs<1.0)
ejected = eccs >= 1.0

data1 = np.array([np.linalg.norm(vfrags[init_deorbit], axis=1), 
                  AMfrags[init_deorbit], SMA[init_deorbit], eccs[init_deorbit], Lcvals[init_deorbit]])
df1 = pd.DataFrame(data1.T, columns=['vkick', 'AM', 'SMA', 'ecc', 'Lc'])
df1['t_deorbit'] = 0.

data2 = np.array([np.linalg.norm(vfrags[ejected], axis=1), AMfrags[ejected], 
                  SMA[ejected], eccs[ejected], Lcvals[ejected]])
df2 = pd.DataFrame(data2.T, columns=['vkick', 'AM', 'SMA', 'ecc', 'Lc'])
df2['t_deorbit'] = 1e9

df = df1.append(df2)
df['colprob'] = 0.
df['colprobperyear'] = 0.

init_deorbit = len(vfrags[init_deorbit])

keep = (SMA*(1-eccs)/1000>REkm+deorbit_R)&(eccs<1.0)
vfrags = vfrags[keep]
AMfrags = AMfrags[keep]
vfrags_total = vfrags_total[keep] / to_m_per_s
eccs = eccs[keep]
SMA = SMA[keep]
Lcvals = Lcvals[keep]
porb = period(SMA, G*Mearthkg)

NFOLLOW = init_deorbit + len(vfrags)

#########################################################################################################################
# LEO Gabbard plot & sampled distribution plot

flag = SMA*(1+eccs)/1000-REkm <= 2000
fig, ax = plt.subplots(figsize=(10, 8))
plt.scatter(porb[flag]/3600, (SMA*(1-eccs)/1000-REkm)[flag], s=5, color='r', label='peri')
plt.scatter(porb[flag]/3600, (SMA*(1+eccs)/1000-REkm)[flag], s=5, color='b', label='apo')
plt.axhline(283, ls='-.', lw=1.5, color='xkcd:light blue', label='Microsat-R')
plt.axhline(400., ls='--', lw=1.5, color='k', label='ISS')
plt.xlabel(r'P$_{\rm{orb}}$ (hr)', fontsize=16)
plt.ylabel('Altitude (km)', fontsize=16)
plt.legend(fontsize=14, markerscale=3)
ax.tick_params(labelsize=14)
plt.savefig('{}/init_gabbard_kept_LEO_{}_{}_{}_{}.png'.format(data_path, Lc_min, numsample, KEkill/1e9, maxtime), bbox_inches='tight')

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

plt.savefig('{}/kept_init_dis_{}_{}_{}_{}.png'.format(data_path, Lc_min, numsample, KEkill/1e9, maxtime), bbox_inches='tight')


#########################################################################################################################
# add position variations of order 10 metres to all fragments
# and then initialize REBOUND simulation

posfrags = rtarget + np.random.uniform(-10, 10, (len(keep[keep]),3))
x = posfrags[:,0] / aum
y = posfrags[:,1] / aum
z = posfrags[:,2] / aum

altref = 300
sim = rebound.Simulation()
sim.integrator ="WHFAST"
sim.dt = 1e-7

sim.add(m=Mearth, hash="Earth", r=Ratm)

for i in range(len(vfrags)):
    sim.add(m=0., vx=vfrags_total[i, 0], vy=vfrags_total[i,1], vz=vfrags_total[i,2],
           x=x[i], y=y[i], z=z[i], hash='{}'.format(i+1))

#########################################################################################################################
# integrate in chunks

plottime = float(args.plottime)
if plottime != 0.0:
    plotpath = data_path + '/gabbard_{}_LEO_{}_{}_{}_{}.hdf'.format(plottime, Lc_min, numsample,
                                                                    KEkill/1e9, maxtime)
    df0 = pd.DataFrame(columns=['SMA', 'e', 'porb'])
    df0.to_hdf(plotpath, key='data', format='t', append=True)
else:
    plottime = None
    plotpath = None
    
satparams = [NALT, NTHETA, altref, dAltCo]

halfhr = twopi / (365.25 * 24) / 2
dt = halfhr
chunk = int(args.chunk)

i = 0
ilast = len(sim.particles)-chunk
if ilast < 0:
    ilast = 0
nums = np.linspace(1, len(sim.particles)-1, len(sim.particles)-1)
deorbit_times = np.array([])
colprobs = np.array([])
colprobperyears = np.array([])
chunk_i = 1
counter = 0
nancatches = 0
while ilast >= 0:
    simchunk = sim.copy()
    for j in range(i-1):
        simchunk.remove(index=1)
    for j in range(ilast):
        simchunk.remove(index=len(simchunk.particles)-1)
    print('length of chunk: ', len(simchunk.particles)-1)
    counter += len(simchunk.particles)-1
    if i == 0:
        num = nums[0:chunk-1]
        AMfrag = AMfrags[0:chunk-1]
    else:
        num = nums[i-1:i-1+chunk]
        AMfrag = AMfrags[i-1:i-1+chunk]
    if len(sim.particles) < chunk:
        num = nums
        AMfrag = AMfrags
    simafter, deorbit_time, colprob, colprobperyear, nancatch = integrate_colprob(simchunk, AMfrag, num, 
                                                                        dt=dt, deorbit_R=deorbit_R, 
                                                                        chunk_i=chunk_i, satparams=satparams,
                                                                                 maxtime=maxtime, event=event,
                                                                                 plottime=plottime, plotpath=plotpath)
    deorbit_times = np.append(deorbit_times, deorbit_time)
    colprobs = np.append(colprobs, colprob)
    colprobperyears = np.append(colprobperyears, colprobperyear)
    chunk_i += 1
    nancatches += nancatch
    print(i, ilast)
    if ilast == 0.0:
        break
    i += chunk
    ilast -= chunk
    if ilast < 0:
        ilast = 0

if counter != int(nums[-1]):
    print('uh oh!')
    
#########################################################################################################################
# plot final distributions and save data

data3 = np.array([np.linalg.norm(vfrags, axis=1), AMfrags, SMA, eccs, Lcvals, deorbit_times, colprobs, colprobperyears])
df3 = pd.DataFrame(data3.T, columns=['vkick', 'AM', 'SMA', 'ecc', 'Lc','t_deorbit', 'colprob', 'colprobperyear'])

datadf = df.append(df3)
datadf = datadf.sort_values('t_deorbit')
datadf.to_hdf('{}/data_{}_{}_{}_{}.hdf'.format(data_path, Lc_min, numsample, KEkill/1e9, maxtime), 
              key='data')
dataother = pd.DataFrame(np.array([nancatch]), columns=['nancatch'])
dataother.to_hdf('{}/data_{}_{}_{}_{}.hdf'.format(data_path, Lc_min, numsample, KEkill/1e9, maxtime), 
                 key='nancatch')

timedf = pd.DataFrame(datadf.loc[datadf.t_deorbit < 1e6].t_deorbit.values.T, columns=['time'])
cumsum = np.cumsum(timedf.groupby('time').size().values)

fig, ax = plt.subplots(figsize=(10,8))
plt.plot(timedf.groupby('time').first().index.values/twopi, cumsum/NFOLLOW, lw=1.5, color='r')
plt.xlabel('Time (yrs)', fontsize=16)
plt.ylabel('Fraction of Fragments Deorbited', fontsize=16)
ax.tick_params(labelsize=14)
plt.title('{:.4}% deorbited after {} years'.format(cumsum[-1]/NFOLLOW*100, maxtime), fontsize=16)
plt.savefig('{}/deorbit_times_{}_{}_{}_{}.png'.format(data_path, Lc_min, numsample, KEkill/1e9, maxtime), 
            bbox_inches='tight')

flaglo = datadf.t_deorbit.values < 1e6
flaghi = datadf.t_deorbit.values == 1e6
flageject = datadf.t_deorbit.values == 1e9

fig, ax = plt.subplots(figsize=(10,8))

plt.scatter(np.log10(datadf.AM.values[flageject]), datadf.vkick.values[flageject]/1000,
            c='xkcd:cyan', label=r'ejected fragments')
plt.scatter(np.log10(datadf.AM.values[flaghi]), datadf.vkick.values[flaghi]/1000,
            c='r', label=r't$_{\rm{deorbit}} \geq$'+' {}yrs'.format(maxtime))
plt.scatter(np.log10(datadf.AM.values[flaglo]), datadf.vkick.values[flaglo]/1000,
            c=datadf.t_deorbit.values[flaglo]/twopi, label=r't$_{\rm{deorbit}}$ < '+'{}yrs'.format(maxtime))
cb = plt.colorbar()
cb.ax.set_ylabel('Deorbit Time (yrs)', fontsize=16)
cb.ax.tick_params(labelsize=14)
plt.xlabel('Log$_{10}$(A/M ratio)', fontsize=16)
plt.ylabel('Velocity Kick (km/s)', fontsize=16)
plt.legend(loc='best', fontsize=14)
ax.tick_params(labelsize=14)
plt.savefig('{}/AM_vkicks_times_{}_{}_{}_{}.png'.format(data_path, Lc_min, numsample, KEkill/1e9, maxtime),
            bbox_inches='tight')








