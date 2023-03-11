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
from tools.asat_sma import *

twopi = 2 * np.pi
aum = u.lengths_SI['au']
aukm= aum / 1000
RE_eq = 6378.135 / aukm
REkm = 6378. # Using ~ equatorial value
Ratm = (REkm + 200) / aukm  # effective size of Earth (not really used at this time)
J2 = 1.0827e-3  # harmonics
J4 = -1.620e-6
Mearthkg = u.masses_SI['mearth']  # earth mass in kg
Msunkg = u.masses_SI['solarmass']  # solar mass in kg
Mearth = Mearthkg / Msunkg  # Earth mass in solar masses 
g0 = 9.81  # Earth gravity in m / s^2
vconv = np.sqrt(6.67e-11*1.989e30/1.496e11)
to_m_per_s = aum / (3600 * 24 * 365.25) * twopi # multiply by this factor to 
                                                # convert rebound speed units to m/s
tmax = twopi  # two pi is one year in computational units !! 

def cartesian_to_spherical(x, y, z):
    '''
    Converts positions from cartesian to spherical coordinates. 
    
    INPUT:
    -----------------------
    x,y,z [arrays]: arrays must all be same size and be in au
    
    OUTPUT:
    -----------------------
    r [array]:      radial distance in km
    theta [array]:  colatitude in rads
    '''
    x = x * aukm
    y = y * aukm
    z = z * aukm
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / r)
    return r, theta

def get_target_params(mtarget, vr, r, Q, inc, omega, verbose=True):
    '''
    Get target velocity and position to be combined with
    fragment velocities/positions for debris cloud. 
    
    INPUT:
    -----------------------
    mtarget [float]:  target mass in kg
    vr [float]:       velocity of TARGET at collison altitude r, in m/s
    r [float]:        altitude of collision in km
    Q [float]:        apogee of target in km
    inc [float]:      orbital inclination of target in rads
    omega [float]:    argument of pericenter of target in rads
    
    OUTPUT:
    -----------------------
    vtarget [array]:  velocity vector of target in m/s
    rtarget [array]:  cartesian position vector of target in m
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
    
    vtarget = np.array([vx0, vy0, vz0])
    rtarget = np.array([x0, y0, z0]) * aum
    
    if verbose:
        print('target mass is: ', mtarget, ' kg')
        print('rtarget is: ', np.array([x0, y0, z0]) * aukm, ' km')
        print('altitude of target is: ', mag(np.array([x0, y0, z0]))*aukm-REkm, 
              ' km')
        print('velocity vector of target is: ', vx0, vy0, vz0, 'm/s')
        print('target speed is: ', mag(vvec), 'm/s')
        
    return vtarget, rtarget

def integrate(simchunk, bfrags, hashes, tstart, tend, dt, deorbit_R, chunk_i):
    '''
    Integrate REBOUND simulation of debris fragments from tstart to tend, removing
    them as they deorbit. This function does not keep track of collision probabilities.
    
    INPUT:
    -----------------------
    simchunk [REBOUND sim]: sim we want to integrate. Might be the entire sim or a
                            sub-set of particles from the simulation
    bfrags [array]:         B_*-coefficients of fragments in sim, where B_* = C_D*A/M
    hashes [array]:         hash numbers of fragments that are in this chunk (since 
                            their location in the simchunk will be different than 
                            their actual hashes) - used to remove deorbited frags
    tstart [float]:         starting time of integration !!in code units!!
    tend [float]:           ending time of integration !!in code units!!
    dt [float]:             timestep !!in code units!!
    deorbit_R [float]:      altitude at which to consider a fragment "deorbited" in km
    chunk_i [int]:          which segment of the larger sim this is
    
    OUTPUT:
    -----------------------
    simchunk [REBOUND sim]: newly integrated sim chunk with deorbited fragments removed
    times [array]:          times that we paused at to track status of fragments
    e [array]:              eccentricities at each time step of all fragments
    inc [array]:            orbital inclination at each time step in degrees
    alt [array]:            semi-major axis at each time step in km
    porb [array]:           orbital period at each time step in hours
    x, y, z [array]:        cartesian coordinates at each time step in km   
    
    Note that e, inc, alt, porb, x, y, z, will be set to zero for t > t_deorbit
    for a given deorbited fragment.
    '''
    
    # get times to pause the integration and save data:
    times = np.arange(tstart, tend, dt)
    NTIME = len(times)
    
    ps = simchunk.particles
    
    # add extra forces
    rebx = reboundx.Extras(simchunk)
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
        ps[i+1].params["bcoeff"] = bfrags[i]  # (could also be 2.2 * A/M)
    
    # set up empty arrays. We'll keep track of deorbiting fragments also.
    ndebris = len(ps) - 1
    deorbit_total = 0
    deorbit_times = np.ones(len(ps) - 1)
    ps = simchunk.particles
    Np = len(ps)-1
    e = np.zeros((NTIME, Np))
    inc = np.zeros((NTIME, Np))
    alt = np.zeros((NTIME, Np))
    porb = np.zeros((NTIME, Np))
    x = np.zeros((NTIME, Np))
    y = np.zeros((NTIME, Np))
    z = np.zeros((NTIME, Np))
    
    # time to integrate!
    for itime, time in enumerate(times):
        deorbit_i = 0
        print("\nWorking on time {}, t={}, chunk {}".format(itime+1, 
                                                            time, chunk_i))
        inorbit = len(deorbit_times[deorbit_times==1])
        print('Number of debris still in orbit: {}'.format(inorbit))
        
        # integrate up to current time step
        simchunk.integrate(time)

        # get debris fragments
        ps = sim.particles
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
                if r <= REkm + deorbit_R:
                    deorbit_times[d-1] = time
                    deorbit_i += 1
                    deorbit_total += 1
                    simchunk.remove(hash=h)
        print('{} deorbited this step'.format(deorbit_i))

    print('\n {} deorbited total out of {} fragments'.format(deorbit_total, ndebris))
    
    intvec = simchunk, times, e, inc, alt, porb, x, y, z
    
    return intvec

def integrate_colprob(simchunk, AMfrags, hashes, dt, deorbit_R, chunk_i, satparams,
                      maxtime, event, plottime=None, plotpath=None):
    '''
    Integrate REBOUND simulation of debris fragments up until maxtime, removing
    them as they deorbit. This function keeps track of collision probabilities
    for given satellite parameters satparams, but doesn't save full orbital data
    over time.
    
    INPUT:
    -----------------------
    simchunk [REBOUND sim]: sim we want to integrate. Might be the entire sim or a
                            sub-set of particles from the simulation
    AMfrags [array]:        for B_*'s of frags in sim (B_* = 2.2*A/M), in m^2/kg
    hashes [array]:         hash numbers of fragments that are in this chunk (since 
                            their location in the simchunk will be different than 
                            their actual hashes) - used to remove deorbited frags
    dt [float]:             timestep !!in code units!!
    deorbit_R [float]:      altitude at which to consider a fragment "deorbited" in km
    chunk_i [int]:          which segment of the larger sim this is
    satparams [list]:       vector containing NALT (no. of altitude bins), NTHETA (no.
                            of colatitude bins), altref (minimum altitude in km), and
                            dAltCo (altitude bin width in km)
    maxtime [float]:        maximum integration time in years
    event [str]:            event name determines which solar phase to initialize at
    plottime:               timestep to save data at, None otherwise
    plotpath:               path + filename to save data to
    
    OUTPUT:
    -----------------------
    simchunk [REBOUND sim]: newly integrated sim with deorbited fragments removed
    deorbit_times [array]:  times that the fragments deorbited at 
    colprob [array]:        collision probability of frags integrated over full lifetime
    nancatch [int]:         something is wrong if nancatch > 0
    '''
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
                df = pd.DataFrame(np.array([SMA, eccs, porb, 
                                            np.ones(len(SMA))*chunk_i]).T, 
                                  columns=['SMA', 'e', 'porb', 'chunk'])
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
            break

    print('\n {} deorbited total out of {} fragments, {} nans'.format(deorbit_total, 
                                                                      Np, nancatch))
    
    intvec = simchunk, deorbit_times, colprob, nancatch
    return intvec

def vel_dis_rayleigh(vexpl, vtarget, rtarget, numsample, AMval):
    '''
    Generates debris cloud params using a Rayleigh velocity distribution.
    
    INPUT:
    -----------------------
    vexpl [float]:  "explosion velocity" in m/s, which sets scale param 
                     for Rayleigh distribution. Typically 250 m/s
    vtarget [array]: target velocity vector, usually computed using
                     get_target_params(), in m/s
    rtarget [array]: target position vector, usually computed using
                     get_target_params(), in m
    numsample [int]: number of fragments to generate. Might be a full
                     NSBM number of fragments (large!) or a smaller
                     sampling
    AMval [float]:   the Rayleigh distribution assumes a constant area-to-
                     mass ratio, assigned to all the fragments. In m^2/kg
    
    OUTPUT:
    -----------------------
    vfrags [array]:        fragment velocity kick vectors in m/s
    vfrags_total [array]:  total fragment velocities in m/s
    eccs [array]:          debris fragment eccentricities
    SMA [array]:           semi-major axes of fragments in m
    AMfrags [array]:       array of length numsample of AMval
    '''
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
