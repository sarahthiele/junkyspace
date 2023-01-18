import rebound
import reboundx 
import rebound.units as u
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.insert(1, '/store/users/sthiele/home/junkyspace/')
from fromaaron.asat_sma import *

SEED=314

np.random.seed(SEED)

twopi = 2 * np.pi
aum = u.lengths_SI['au']
aukm= aum / 1000
RE_eq = 6378.135 / aukm
REkm = 6378.
Ratm = (REkm + 200) / aukm
J2 = 1.0827e-3
J4 = -1.620e-6
Mearthkg = u.masses_SI['mearth']
Msunkg = u.masses_SI['solarmass']
Mearth = Mearthkg / Msunkg
G = u.G_SI
g0 = 9.81  # m / s^2
vconv = np.sqrt(6.67e-11*1.989e30/1.496e11)
to_m_per_s = aum / (3600 * 24 * 365.25) * twopi # multiply by this factor to convert rebound speed units to m/s
tmax = twopi

def cartesian_to_spherical(x, y, z):
    '''
    Convert into spherical coordinates. x,y,z must be in au.
    
    Returns r and colatitude in km.
    '''
    x = x * aukm
    y = y * aukm
    z = z * aukm
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / r)
    return r, theta

def get_target_params(mtarget, vr, r, Q, inc, omega):
    '''
    get the target velocity and position to be combined with
    fragment velocities/positions. The output is in m/s and m
    '''
    a = (2/(r*1e3) - vr**2/(G*Mearthkg))**(-1)/1000
    e = Q/a - 1
    f = np.arccos((a/r*(1-e**2)-1)/e)
    if f < np.pi:
        df = np.pi - f
        f = np.pi + df

    sim = rebound.Simulation()
    sim.integrator ="ias15"
    sim.dt = 1e-7

    sim.add(m=Mearth, hash='Earth', r=Ratm)
    sim.add(m=mtarget/Msunkg, a=a/aukm, f=f, e=e, inc=-inc, Omega=0., omega=omega)

    ps = sim.particles
    x0 = ps[1].x
    y0 = ps[1].y 
    z0 = ps[1].z
    vx0 = ps[1].vx * to_m_per_s
    vy0 = ps[1].vy * to_m_per_s
    vz0 = ps[1].vz * to_m_per_s
    vvec = np.array([vx0, vy0, vz0])
    unitvec = vvec / mag(vvec)
    print(np.array([x0, y0, z0]) * aukm)
    print(mag(np.array([x0, y0, z0]))*aukm-REkm)
    print(vx0, vy0, vz0)
    print(mag(vvec))
    vtarget = vvec
    rtarget = np.array([x0, y0, z0]) * aum
    return vtarget, rtarget

def integrate(sim1, bfrags, hashes, tstart, tend, dt, deorbit_R, chunk_i):
    times = np.arange(tstart, tend, dt)
    NTIME = len(times)
    
    ps = sim1.particles
    
    rebx = reboundx.Extras(sim1)
    gh = rebx.load_force("gravitational_harmonics")
    rebx.add_force(gh)
    ps["Earth"].params["J2"] = J2
    ps["Earth"].params["J4"] = J4 
    ps["Earth"].params["R_eq"] = RE_eq

    # add gas drag
    gd = rebx.load_force("gas_drag")
    rebx.add_force(gd)
    gd.params["code_to_yr"]= 1. / twopi
    gd.params["density_mks_to_code"] = aum**3 / Msunkg
    gd.params["dist_to_m"] = aum
    gd.params["alt_ref_m"] = REkm * 1000
    for i in range(len(ps)-1):
        ps[i+1].params["bstar"] = bfrags[i]
    
    ndebris = len(ps) - 1
    deorbit_total = 0
    deorbit_times = np.ones(len(ps) - 1)
    ps = sim1.particles
    Np = len(ps)-1
    e = np.zeros((NTIME, Np))
    inc = np.zeros((NTIME, Np))
    alt = np.zeros((NTIME, Np))
    porb = np.zeros((NTIME, Np))
    x = np.zeros((NTIME, Np))
    y = np.zeros((NTIME, Np))
    z = np.zeros((NTIME, Np))
    for itime, time in enumerate(times):
        deorbit_i = 0
        print("\nWorking on time {}, t={}, chunk {}".format(itime+1, time, chunk_i))
        print('Number of debris in orbit: {}'.format(len(deorbit_times[deorbit_times==1])))
        sim1.integrate(time)

        ps = sim1.particles
        if len(ps) == 0:
            break
        i = 1
        for d in range(1,Np+1):
            if deorbit_times[d-1] != 1.:
                e[itime, d-1] = 0.
                inc[itime, d-1] = 0.
                alt[itime, d-1] = 0.
                porb[itime, d-1] = 0.
                x[itime, d-1] = 0.
                y[itime, d-1] = 0.
                z[itime, d-1] = 0.
            else:
                h = '{}'.format(int(hashes[d-1]))
                e[itime, d-1] = ps[h].e
                inc[itime, d-1] = ps[h].inc * 360 / twopi
                alt[itime, d-1] = ps[h].a * aukm
                porb[itime, d-1] = period(ps[h].a * aum, G * Mearthkg) / 60 / 60
                x[itime, d-1] = ps[h].x * aukm
                y[itime, d-1] = ps[h].y * aukm
                z[itime, d-1] = ps[h].z * aukm
                r = np.sqrt(ps[h].x ** 2 + ps[h].y ** 2 + ps[h].z ** 2) * aukm
                if r <= REkm + deorbit_R: # or np.isnan(r) == True:
                    deorbit_times[d-1] = time
                    deorbit_i += 1
                    deorbit_total += 1
                    sim1.remove(hash=h)
        print('{} deorbited this step'.format(deorbit_i))

    print('\n {} deorbited total out of {} fragments'.format(deorbit_total, ndebris))

    return sim1, times, e, inc, alt, porb, x, y, z

def integrate_colprob(simchunk, AMfrags, hashes, dt, deorbit_R, chunk_i, satparams,
                      maxtime, event, plottime=None, plotpath=None):
    if plottime != None:
        plot = True
    else:
        plot = False
        
    NALT, NTHETA, altref, dAltCo = satparams
    ps = simchunk.particles

    rebx = reboundx.Extras(simchunk)
    gh = rebx.load_force("gravitational_harmonics")
    rebx.add_force(gh)
    ps["Earth"].params["J2"] = J2
    ps["Earth"].params["J4"] = J4 
    ps["Earth"].params["R_eq"] = RE_eq

    # New collision probability code
    coprob = rebx.load_operator("collision_prob")
    rebx.add_operator(coprob)
    coprob.params["nAltDensity"] = NALT
    coprob.params["nThetaDensity"] = NTHETA
    coprob.params["altReference"] = (altref + REkm) / aukm
    coprob.params["deltaAlt"] = dAltCo
    coprob.params["deltaTheta"] = np.pi/NTHETA
    coprob.params["density_mks_to_code"] = aukm**3
    coprob.params["REarth"] = REkm/aukm
    coprob.params["readTable"] = 1
    for i in range(1, len(ps)):
        ps[i].params["collProb"] = 0.
        ps[i].params["collArea"] = 10 / aum**2

    # add gas drag
    gd = rebx.load_force("gas_drag")
    rebx.add_force(gd)
    
    # Solar minimum occured in Dec. 2019. Therefore to account for solar cycle:
    if event == 'India':  # India ASAT occured March 2019
        gd.params["solar_phase"]= twopi/2 - (9/12)/22*twopi
    if event == 'Russia':  # Russia ASAT occured Nov. 2022 
        gd.params["solar_phase"] = twopi/2 + (1+11/12)/22*twopi
    if event == 'FTG15':  # FTG-15 occured on May 30 2017
        gd.params['solar_phase'] = 0.77 * np.pi
    gd.params["code_to_yr"]= 1. / twopi
    gd.params["density_mks_to_code"] = aum**3 / Msunkg
    gd.params["dist_to_m"] = aum
    gd.params["alt_ref_m"] = REkm * 1000
    for i in range(1, len(ps)):
        ps[i].params["bcoeff"] = AMfrags[i-1]*2.2
    
    Np = len(ps)-1
    deorbit_total = 0
    deorbit_times = np.ones(Np)
    colprob = np.zeros(Np)
    colprobperyear = np.zeros(Np)
    ps = simchunk.particles
    itime = 1
    nancatch = 0
    time = dt
    print('chunk {}'.format(chunk_i))
    while len(ps) > 1:
        deorbit_i = 0

        simchunk.integrate(time)

        ps = simchunk.particles
        if len(ps) == 1:
            break
            
        for d in range(1,Np+1):
            if deorbit_times[d-1] == 1.:
                h = '{}'.format(int(hashes[d-1]))
                if np.isnan(ps[h].params["collProb"]):
                    nancatch+=1
                    simchunk.remove(i)
                    continue
                else:
                    r = np.sqrt(ps[h].x ** 2 + ps[h].y ** 2 + ps[h].z ** 2) * aukm
                    if r <= REkm + deorbit_R:
                        deorbit_times[d-1] = time
                        colprob[d-1] = ps[h].params["collProb"]
                        colprobperyear[d-1] = ps[h].params["collProb"] * twopi / time
                        deorbit_i += 1
                        deorbit_total += 1
                        simchunk.remove(hash=h)
        itime += 1
        time += dt
        
        if plot == True:
            if time >= twopi * plottime:
                ps = simchunk.particles
                SMA = []
                eccs = []
                porb = []
                Np = len(ps)-1
                for d in range(1,Np+1):
                    SMA.append(ps[d].a*aum)
                    eccs.append(ps[d].e)
                    porb.append(period(ps[d].a*aum, G*Mearthkg))
                SMA = np.array(SMA)
                eccs = np.array(eccs)
                porb = np.array(porb)
                df = pd.DataFrame(np.array([SMA, eccs, porb, np.ones(len(SMA))*chunk_i]).T, columns=['SMA', 'e', 'porb', 'chunk'])
                df.to_hdf(plotpath, key='data', format='t', append=True)
                plot = False
                
        if time >= twopi * maxtime:
            print('timeout')
            ps = simchunk.particles
            for d in range(1,Np+1):
                if deorbit_times[d-1] == 1.:
                    h = '{}'.format(int(hashes[d-1]))
                    deorbit_times[d-1] = 1e6
                    colprob[d-1] = ps[h].params["collProb"]
                    colprobperyear[d-1] = ps[h].params["collProb"] * twopi / time
            break

    print('\n {} deorbited total out of {} fragments, {} nans'.format(deorbit_total, Np, nancatch))
    
    return simchunk, deorbit_times, colprob, colprobperyear, nancatch

def vel_dis_rayleigh(vexpl, vtarget, rtar, numsample, AMval):
    vmags = np.random.rayleigh(vexpl, numsample)
    
    unitvecs = np.random.uniform(-1, 1, (numsample, 3))
    vnormed = (unitvecs.T / np.linalg.norm(unitvecs, axis=1)).T
    vfrags = (vnormed.T * vmags).T
    
    GM = G * Mearthkg
    vfrags_total = vfrags + vtarget
    eccs = []
    SMA = []
    for v in vfrags_total:
        ecc = mag(eccVector(v, GM, rtar))
        a = sma(v, GM, mag(rtar))
        eccs.append(ecc)
        SMA.append(a)
        
    eccs = np.array(eccs)
    SMA = np.array(SMA)
    
    AMfrags = AMval * np.ones(numsample)

    return vfrags, vfrags_total, eccs, SMA, AMfrags
