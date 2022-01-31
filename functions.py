import rebound
import reboundx 
import rebound.units as u
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.insert(1, '/store/users/sthiele/home/ASATtest/')
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

def integrate_colprob(simchunk, AMfrags, hashes, dt, deorbit_R, chunk_i, satparams):
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
    print('fragment {}'.format(chunk_i))
    while len(ps) > 1:
        deorbit_i = 0
        #print("\nWorking on time {}, t={}, chunk={},\nNumber of debris in orbit: {}".format(itime, time, 
                                                                                            #chunk_i, 
                                                                                           # len(deorbit_times[deorbit_times==1])))
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
        #print('{} deorbited this step'.format(deorbit_i))
        if time >= twopi * 2:
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

def vel_dis_rayleigh(vexpl, vtarget, rtar, numsample):
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
    
    AMfrags = 0.1 * np.ones(numsample)

    return vfrags, vfrags_total, eccs, SMA, AMfrags


#============================================================================================
# !! OLD FUNCTIONS !!
#============================================================================================

def plot_2d(hdata, vdata, hlabel, vlabel):
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')
    plt.xlim([-8370,8370])
    plt.ylim([-8370,8370])
    plt.xlabel('{} [km]'.format(hlabel))
    plt.ylabel('{} [km]'.format(vlabel))
    ax.add_patch(Circle((0,0),6371,color='black',fill=False))
    plt.scatter(hdata*aukm, vdata*aukm, s=1)
    plt.show()
    return fig, ax
    
def plot_3d(x, y, z):
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection="3d")
    ax.set_aspect('equal')
    ax.set_xlim([-8370,8370])
    ax.set_ylim([-8370,8370])
    ax.set_zlim([-8370,8370])
    ax.set_ylabel('Y [km]')
    ax.set_xlabel('X [km]')
    ax.set_zlabel('Z [km]')
    ax.scatter3D(x*aukm, y*aukm, z*aukm, cmap='k', s=1)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    xs = REkm * np.outer(np.cos(u), np.sin(v))
    ys = REkm * np.outer(np.sin(u), np.sin(v))
    zs = REkm * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(xs, ys, zs, color='None', edgecolor='lightgrey', 
                    linewidth=0.5)
    plt.show()
    return fig, ax

def plot_gabbard(alt, ecc, porb):
    perigee = alt * (1 - ecc)
    apogee = alt * (1 + ecc)
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.scatter(porb, perigee, s=5, color='r', label='debris perigee', 
                zorder=1)
    plt.scatter(porb, apogee, s=5, color='b', label='debris apogee', 
                zorder=2)
    ax.axhline(400, linestyle='--', linewidth=1, color='xkcd:grey', 
               label='ISS', zorder=0)
    ASAT_porb = period(283000, G*Mearthkg) / 3600
    plt.scatter(ASAT_porb, 283, s=20, color='xkcd:sky blue', 
                label='Microsat-R', zorder=3)
    plt.xlabel('Orbital Period (hours)', fontsize=18)
    plt.ylabel('Altitude (km)', fontsize=18)
    plt.legend(fontsize=16)
    ax.tick_params(labelsize=16)
    plt.savefig('GABBARD.png')
    return fig, ax


def params_LEO(num_planes, sats_per_plane, alt, inclination):
    a = alt + REkm
    inc = inclination * np.pi / 180
    M0_init = np.linspace(0, 2*np.pi, sats_per_plane, endpoint=False)
    dM0 = 2 * np.pi / sats_per_plane / num_planes
    Omega = np.linspace(0, 2*np.pi, num_planes, endpoint=False)
    return a, inc, M0_init, dM0, Omega

def add_particles_LEO(num_planes, sats_per_plane, alt, inclination, sim):
    a, inc, M0_init, dM0, Omega = params_LEO(num_planes, sats_per_plane, alt, inclination)
    for i in range(num_planes):
        RAAN = Omega[i]
        M0_offset = dM0 * i
        M0 = M0_init + M0_offset
        for j in range(sats_per_plane):
            sim.add(m=260/Msunkg, a=a/aukm, inc=inc, Omega=RAAN, M=M0[j], e=0.0)
    return a, inc, M0_init, dM0, Omega

def params_VLEO(num_sats, alt, inclination):
    ntot = num_sats
    a = alt + REkm
    inc = inclination * np.pi / 180
    Omega = np.linspace(0, 2*np.pi, num_sats, endpoint=False)
    mashuf = np.zeros(ntot)
    nmix = int(np.sqrt(ntot))
    iref=0
    iskip=0
    for i in range(ntot):
        imix = iref+iskip*nmix
        if imix>ntot-1:
            iref+=1
            iskip=0
            imix=iref*1
        mashuf[i] = Omega[imix]*1
        iskip+=1
    return a, inc, mashuf, Omega

def add_particles_VLEO(num_sats, alt, inclination, sim):
    a, inc, mashuf, Omega = params_VLEO(num_sats, alt, inclination)
    for i in range(num_sats):
        RAAN = Omega[i]
        ma = mashuf[i]
        sim.add(m=260/Msunkg, a=a/aukm, inc=inc, Omega=RAAN, M=mashuf[i], e=0.)
    return a, inc, mashuf, Omega

def node_vol(r, dr, theta, dth):
    '''
    Calculates the volume of the azimuthally-averaged 
    node associated with radius at the midpoint between 
    r and r + dr, and colatitude at the midpoint between
    theta and theta + dth.
    
    r and dr in km, theta and dth in rads
    '''
    R = r + dr / 2
    Th = theta + dth / 2
    vol = twopi * R ** 2 * dr * np.sin(Th) * dth
    return vol

def make_nodes(dth, dr, rmin, rmax):
    '''
    Makes a dictionary of the orbital nodes
    between rmin and rmax in alitude around Earth,
    with a resolution/width of dth x dr.
    
    Inputs:
    rmin, rmax, dr in km
    dth in degrees
    
    Outputs:
    Dictionary with node number, minimum and
    maximum semi-major axis of that node, minimum
    and maximum colatitude, the volume of the node.
    '''
    rmin = rmin + REkm
    rmax = rmax + REkm
    dth = dth * np.pi / 180
    colat = np.arange(0, np.pi, dth)
    nodes = []
    node_num = 1
    r_list = []
    t_list = []
    i = 0
    r = rmin
    while r <= rmax - 2 * dr:
        r = rmin + i * dr
        for theta in colat:
            vol = node_vol(r, dr, theta, dth)
            nodes.append({'NUM':node_num, 'RMIN':r, 'RMAX':r+dr, 'THMIN':theta, 
                          'THMAX':theta+dth, 'VOL':vol})
            node_num += 1
            r_list.append(r + dr / 2)
            t_list.append(theta + dth / 2)
        i += 1
    r_list = np.array(r_list)
    t_list = np.array(t_list)
    return nodes, r_list, t_list

def node_density(nodes, r, t):
    '''
    Calculates number density of 
    each orbital node for a given time
    step.
    '''
    density = np.zeros(len(nodes))
    for i in range(len(nodes)):
        node = nodes[i]
        RMIN = node['RMIN']
        RMAX = node['RMAX']
        THMIN = node['THMIN']
        THMAX = node['THMAX']
        VOL = node['VOL']
        where = np.where((r>=RMIN)&(r<=RMAX)&(t>=THMIN)&(t<=THMAX))
        numsats = np.shape(where)[1]
        if numsats > 0:
            density[i] = numsats / VOL
    return density



def OLDFUNCnode_density(nodes, r, t):
    sat_nodes = np.zeros(len(r))
    density = np.zeros(len(nodes))
    for i in range(len(nodes)):
        node = nodes[i]
        NUM = int(node['NUM'])
        RMIN = node['RMIN']
        RMAX = node['RMAX']
        THMIN = node['THMIN']
        THMAX = node['THMAX']
        VOL = node['VOL']
        where = np.where((r>=RMIN)&(r<=RMAX)&(t>=THMIN)&(t<=THMAX))
        numsats = np.shape(where)[1]
        if numsats > 0:
            sat_nodes[where] = NUM
            nodes[i]['DENS'] = numsats / VOL
            density[i] = numsats / VOL
    sat_nodes = sat_nodes.astype(int)
    return density, sat_nodes
