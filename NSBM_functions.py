import rebound
import reboundx 
import rebound.units as u
import numpy as np
import matplotlib.pyplot as plt
import pdb
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import maxwell as maxwell
import scipy.stats

#import sys
#sys.path.insert(1, '/store/users/sthiele/home/junkyspace/')
from fromaaron.asat_sma import *
from functions import *

SEED=314

np.random.seed(SEED)

twopi = 2 * np.pi
aum = u.lengths_SI['au']
aukm= aum / 1000
RE_eq = 6378.135 / aukm
REkm = 6378.  # not entirely sure whether to use Requator or Raverage here.  Using equatorial value 
Ratm = (REkm + 200) / aukm  # effective size of Earth -- not really used at this time
J2 = 1.0827e-3 # harmonics
J4 = -1.620e-6
Mearthkg = u.masses_SI['mearth']
Msunkg = u.masses_SI['solarmass']
Mearth = Mearthkg / Msunkg
G = u.G_SI
g0 = 9.81  # m / s^2
to_m_per_s = aum / (3600 * 24 * 365.25) * twopi # multiply by this factor to convert rebound speed units to m/s

from scipy.stats import norm as norm

# These equations come from payload parameters in Appendix B and C 
# of https://arc.aiaa.org/doi/pdf/10.2514/1.G004939

def get_AM_ratio(L, nums):
    nums = nums.astype(int)
    def alphaL(lamdaL):
        alphaL1 = np.zeros(len(lamdaL))
        alphaL1[lamdaL <= -1.95] = 0
        alphaL1[(lamdaL>-1.95)&(lamdaL<0.55)] = 0.3 + 0.4 * (lamdaL[(lamdaL>-1.95)&(lamdaL<0.55)] + 1.2)
        alphaL1[lamdaL >= 0.55] = 1
        
        alphaL2 = 1 - alphaL1
        return alphaL1, alphaL2
    def muLvals(lamdaL):
        muL1 = np.zeros(len(lamdaL))
        muL1[lamdaL <= -1.1] = -0.6
        muL1[(lamdaL>-1.1)&(lamdaL<0.)] = -0.6 - 0.318 * (lamdaL[(lamdaL>-1.1)&(lamdaL<0.)] + 1.1)
        muL1[lamdaL >= 0.0] = -0.95  
        
        muL2 = np.zeros(len(lamdaL))
        muL2[lamdaL <= -0.7] = -1.2
        muL2[(lamdaL > -0.7)&(lamdaL < -0.1)] = -1.2 - 1.333 * (lamdaL[(lamdaL > -0.7)&(lamdaL < -0.1)] + 0.7)
        muL2[lamdaL >= -0.1] = -2.0
        return muL1, muL2
    def sigmaL(lamdaL):
        sigmaL1 = np.zeros(len(lamdaL))
        sigmaL1[lamdaL <= -1.3] = 0.1
        sigmaL1[(lamdaL>-1.3)&(lamdaL<-0.3)] = 0.1 + 0.2 * (lamdaL[(lamdaL>-1.3)&(lamdaL<-0.3)] + 1.3)
        sigmaL1[lamdaL >= -0.3] = 0.3
        
        sigmaL2 = np.zeros(len(lamdaL))
        sigmaL2[lamdaL <= -0.5] = 0.5
        sigmaL2[(lamdaL > -0.5)&(lamdaL < -0.3)] = 0.5 - (lamdaL[(lamdaL > -0.5)&(lamdaL < -0.3)] + 0.5)
        sigmaL2[lamdaL >= -0.3] = 0.3 
        return sigmaL1, sigmaL2
    def muSvals(lamdaS):
        mu_S = np.zeros(len(lamdaS))
        mu_S[lamdaS <= -1.75] = -0.3
        mu_S[(lamdaS > -1.75)&(lamdaS < -1.25)] = -0.3 - 1.4 * (lamdaS[(lamdaS > -1.75)&(lamdaS < -1.25)] + 1.75)
        mu_S[lamdaS >= -1.25] = -1.0  
        return mu_S
    def sigmaS(lamdaS):
        sigma_S = np.zeros(len(lamdaS))
        sigma_S[lamdaS <= -3.5] = 0.2
        sigma_S[lamdaS >= -3.5] = 0.2 + 0.1333 * (lamdaS[lamdaS >= -3.5] + 3.5)
        return sigma_S
    def alphaM(lamdaM):
        return (lamdaM - np.log10(0.08)) / (np.log10(0.11) - np.log10(0.08))
    
    nums = nums.astype(int)
    
    lamda = np.log10(L)
    aL1, aL2 = alphaL(lamda)
    muL1, muL2 = muLvals(lamda)
    sigL1, sigL2 = sigmaL(lamda)
    sigS = sigmaS(lamda)
    muS1 = muSvals(lamda)
    
    chis = np.array([])
    
    chi = np.linspace(-3,1, 100000)
    dchi = (chi[1:]-chi[:-1])[0]
    
    # small objects:
    flag = L < 0.08
    numsS = nums[flag]
    mu = muS1[flag]
    sig = sigS[flag]
    for i in range(len(L[flag])):
        num = numsS[i]
        pS = scipy.stats.norm(mu[i], sig[i]).pdf(chi)
        cdfS = np.cumsum(pS*dchi)
        P = np.random.uniform(size=num)
        cdfindices = np.digitize(P, bins=cdfS)  
        cdfindices[cdfindices==len(chi)] = len(chi) - 1
        chis = np.append(chis, chi[cdfindices])
        
    # medium objects:
    flag = (L>=0.08)&(L<=0.11)
    numsM = nums[flag]
    a = alphaM(lamda[flag])
    muS = muS1[flag]
    sig_S = sigS[flag]
    aL_1, aL_2 = aL1[flag], aL2[flag]
    muL_1, muL_2 = muL1[flag], muL2[flag]
    sigL_1, sigL_2 = sigL1[flag], sigL2[flag]
    for i in range(len(L[flag])):
        num = numsM[i]
        pS = scipy.stats.norm(muS[i], sig_S[i]).pdf(chi)
        pL = aL_1[i] * scipy.stats.norm(muL_1[i], sigL_1[i]).pdf(chi) + aL_2[i] * scipy.stats.norm(muL_2[i], sigL_2[i]).pdf(chi)
        pM = a[i] * pL + (1-a[i]) * pS
        cdfM = np.cumsum(pM*dchi)
        P = np.random.uniform(size=num)
        cdfindices = np.digitize(P, bins=cdfM)
        cdfindices[cdfindices==len(chi)] = len(chi) - 1
        chis = np.append(chis, chi[cdfindices])
        
    # large objects:
    flag = L>0.11
    numsL = nums[flag]
    a1, a2 = aL1[flag], aL2[flag]
    mu1, mu2 = muL1[flag], muL2[flag]
    sig1, sig2 = sigL1[flag], sigL2[flag]
    for i in range(len(L[flag])):
        num = numsL[i]
        pL = a1[i] * scipy.stats.norm(mu1[i], sig1[i]).pdf(chi) + a2[i] * scipy.stats.norm(mu2[i], sig2[i]).pdf(chi)
        cdfL = np.cumsum(pL*dchi)
        P = np.random.uniform(size=num)
        cdfindices = np.digitize(P, bins=cdfL)
        cdfindices[cdfindices==len(chi)] = len(chi) - 1
        chis = np.append(chis, chi[cdfindices])
        
    AM = 10 ** chis
    return AM


def get_A_M_vals(Lvals, AM, nums):
    nums = nums.astype(int)
    def b_gamma_of_L(Lc):
        if Lc <= 0.00167:
            b = 0.540424
            gamma = 2.
        else:
            b = 0.556945
            gamma = 2.0047077
        return b, gamma
            
    Avals = np.array([])
    Lcvals = np.array([])
    for i in range(len(nums)):
        num = nums[i]
        Lc = Lvals[i]
        b, gamma = b_gamma_of_L(Lc)
        A = b * Lc ** gamma
        Avals = np.append(Avals, A * np.ones(num))
        Lcvals = np.append(Lcvals, Lc * np.ones(num))
        
    mvals = Avals / AM
    
    return Avals, mvals, Lcvals 

def get_delta_vs(AM):
    chi = np.log10(AM)
    mu = 0.9 * chi + 2.9
    sigma = 0.4 * np.ones(len(chi))
    v = np.random.normal(loc=mu, scale=sigma, size=np.array([1, len(chi)]))
    magdv = 10 ** v[0]
    unitvecs = np.random.uniform(-1, 1, (len(magdv), 3))
    vnormed = (unitvecs.T / np.linalg.norm(unitvecs, axis=1)).T
    vfrags = (vnormed.T * magdv).T
    return vfrags

def vel_dis_NBM(mtarget, mkill, vkill, vtarget, rtar, nbins, Lc_min, Lc_max, 
                KEkill, numsample, makev=True):
    M = mtarget + mkill
    if KEkill / mtarget / 1000 >= 40:
        print('catastrophic collision')
        Me = mkill + mtarget
    else:
        print('cratering collision')
        Me = 2 * KEkill / 1000 ** 2
    print('Ejected Mass initial calc: ', Me)
    def dNum(L0, L1, Me):
        '''
        NASA Standard Breakup Model is used to find the 
        number of debris fragments for a particular bin
        of characteristic lengths Lc.
        '''
        N0 = 0.1 * Me ** (0.75) * L0 ** (-1.71)
        N1 = 0.1 * Me ** (0.75) * L1 ** (-1.71)
        return int(N0 - N1)

    Lc_bins = np.logspace(np.log10(Lc_min), np.log10(Lc_max), nbins+1)       
    L_mids = (Lc_bins[1:] + Lc_bins[:-1]) / 2
    nums = []
    print('first loop')
    for i in range(len(Lc_bins)-1):
        Lc_l = Lc_bins[i]
        Lc_h = Lc_bins[i+1]
        num = int(dNum(Lc_l, Lc_h, Me))
        nums.append(num)
    nums = np.array(nums).astype(int)
    AMfrags = get_AM_ratio(L_mids, nums)
    N_tot = len(AMfrags)
    Afrags, mfrags, Lcvals = get_A_M_vals(L_mids, AMfrags, nums)
    M_tot = np.sum(mfrags)
    print('N_tot:', N_tot)
    print('M_tot: ', M_tot)
    
    print('normalizing')
    nums = nums * Me / M_tot
    nums = nums.astype(int)
    AMfrags = get_AM_ratio(L_mids, nums)
    Afrags, mfrags, Lcvals = get_A_M_vals(L_mids, AMfrags, nums)
    M_tot = np.sum(mfrags)
    N_tot = len(AMfrags)
    print(N_tot==np.sum(nums))
    print('Number of debris: ', N_tot)
    print('Total mass of debris:', M_tot)

    while M_tot > Me:
        print('normalizing')
        nums = nums * Me / M_tot
        nums = nums.astype(int)
        AMfrags = get_AM_ratio(L_mids, nums)
        Afrags, mfrags, Lcvals = get_A_M_vals(L_mids, AMfrags, nums)
        M_tot = np.sum(mfrags)
        N_tot = len(AMfrags)
        print(N_tot==np.sum(nums))
        print('Number of debris: ', N_tot)
        print('Total mass of debris:', M_tot)

    
    if makev == False:
        if numsample == 100:
            return N_tot
        else:
            return numsample
    
    elif makev == True:
        print('making velocities')
        
        if numsample == 100:
            vfrags = get_delta_vs(AMfrags)

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

            return N_tot, L_mids, nums, mfrags, vfrags, vfrags, vfrags_total, eccs, SMA, AMfrags, AMfrags, Lcvals, Lcvals   
        
        elif numsample != 100:
            vfrags_all = get_delta_vs(AMfrags)
            
            indices = np.linspace(0, len(AMfrags)-1, len(AMfrags)).astype(int)
            sample = np.random.choice(indices, numsample, replace=True)
            AMsample = AMfrags[sample]
            Lcsample = Lcvals[sample]
            vfrags = get_delta_vs(AMsample)


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

            return N_tot, L_mids, nums, mfrags, vfrags_all, vfrags, vfrags_total, eccs, SMA, AMfrags, AMsample, Lcvals, Lcsample

def initialize(mtarget, vr, r, Q, vkill, KEkill, inc, nbins, mkill, deorbitR, save):
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
    sim.add(m=mtarget/Msunkg, a=a/aukm, f=f, e=e, inc=-inc)

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

    twopi = 2.0 * np.pi

    ndebris, M_mids, nums, mfrags, vfrags, vfrags_total, eccs, SMA, AMfrags = vel_dis_NSBM(mtarget=mtarget, mkill=mkill, vkill=vkill, 
                                                     vtarget=vtarget, rtar=rtarget, nbins=nbins, 
                                                     Lc_min=1e-2, Lc_max=1., KEkill=KEkill)

    print('Number of fragments: ', ndebris)
    vfrag_total = vfrags_total / to_m_per_s

    peri = SMA * (1-eccs) / 1000
    vfrag_far = vfrags_total[peri > REkm + deorbitR] / to_m_per_s
    ecc_far = eccs[peri > REkm + deorbitR]
    SMA_far = SMA[peri > REkm + deorbitR]
    mfrags_far = mfrags[peri > REkm + deorbitR]
    print('There are {} debris with perigees farther than {} km'.format(len(mfrags_far), deorbitR))
    nfar = len(mfrags_far)
    FRAGS1 = np.array([mfrags, vfrags_total[:,0], vfrags_total[:,1], 
                      vfrags_total[:,2], eccs, SMA]).T
    if save == True:
        fname1 = 'ASATtests/fragfiles/frags_all_{}_{}_{}_{}_{}_{}_{}_{}.txt'.format(KEkill/1e6, mtarget, vr, r-REkm, Q-REkm, np.round(inc,2), nbins, deorbitR)
        np.savetxt(fname1, FRAGS1, delimiter=',')
    
    if nfar > 0:
        if save == True:
            fname2 = 'ASATtests/fragfiles/frags_far_{}_{}_{}_{}_{}_{}_{}_{}.txt'.format(KEkill/1e6, mtarget, vr, r-REkm, Q-REkm, np.round(inc,2), nbins, deorbitR)
            FRAGS2 = np.array([mfrags_far, vfrag_far[:,0], vfrag_far[:,1], vfrag_far[:,2], ecc_far, SMA_far]).T
            np.savetxt(fname2, FRAGS2, delimiter=',')
        bfrags = 0.22 * np.ones(nfar)
        
        sim = rebound.Simulation()
        sim.integrator ="ias15"
        sim.dt = 1e-10 

        sim.add(m=Mearth, hash="Earth", r=Ratm)
        for i in range(nfar):
            sim.add(m=mfrags_far[i]/Msunkg, vx=vfrag_far[i,0], vy=vfrag_far[i,1], 
                        vz=vfrag_far[i,2], x=x0, y=y0, z=z0, hash='{}'.format(i+1))

        ps = sim.particles
        Np = len(ps)-1
        e = np.zeros(Np)
        alt = np.zeros(Np)
        porb = np.zeros(Np)
        x = np.zeros(Np)
        y = np.zeros(Np)
        z = np.zeros(Np)
        a = np.zeros(Np)
        for d in range(1,Np+1):
            e[d-1] = ps[d].e
            alt[d-1] = ps[d].a * aukm
            porb[d-1] = period(ps[d].a * aum, G * Mearthkg) / 60 / 60
            x[d-1] = ps[d].x * aukm
            y[d-1] = ps[d].y * aukm
            z[d-1] = ps[d].z * aukm
            ps[d].x = ps[d].x + np.random.uniform(-10, 10) / aum
            ps[d].y = ps[d].y + np.random.uniform(-10, 10) / aum
            ps[d].z = ps[d].z + np.random.uniform(-10, 10) / aum

        flag = porb < 100000
        if nfar < 150:
            s = 20
            ms = 1
        else:
            s = 1
            ms = 10
        fig, ax = plt.subplots(figsize=(10,8))   
        plt.scatter(porb[flag], ((alt*(1-e))[flag]-REkm), s=s, color='b', label='peri')
        plt.scatter(porb[flag], ((alt*(1+e))[flag]-REkm), s=s, color='r', label='apo')
        plt.xlabel('Orbital Period (hrs)', fontsize=18)
        plt.ylabel('Altitude (km)', fontsize=18)
        plt.legend(fontsize=16, markerscale=ms)
        ax.tick_params(labelsize=16)
        plt.title('{} initial debris'.format(nfar), fontsize=18)
        if save == True:
            plt.savefig('ASATtests/init_gabbard_{}_{}_{}_{}_{}_{}_{}_{}.png'.format(KEkill/1e6, mtarget, vr, r-REkm, Q-REkm, np.round(inc,2), nbins, deorbitR))
        plt.show(block=False)
        
        return sim, FRAGS1, vfrags, vtarget, rtarget, nfar, vfrag_far, mfrags_far, nums, bfrags

    else:
        return 0, FRAGS1, vfrags, vtarget, rtarget, nfar, vfrag_far, mfrags_far, nums, 0
    


def get_delta_vs_OLD(AM):
    '''
    an example of what not to do :)
    '''
    chi = np.log10(AM)
    mu = 0.9 * chi + 2.9
    sigma = 0.4 * np.ones(len(chi))
    v = np.random.normal(loc=mu, scale=sigma, size=np.array([1, len(chi)]))
    magdv = 10 ** v[0]
    unitv = np.random.rand(len(chi), 3)
    vfrags = (unitv.T * magdv).T
    return vfrags

def dNum_OLD(Lc_l, Lc_h, Me):
    '''
    NASA Standard Breakup Model is used to find the 
    number of debris fragments for a particular bin
    of characteristic lengths Lc.
    '''
    dN = -0.171 * Me ** (0.75) * (Lc_h ** (-2.71) - Lc_l ** (-2.71))
    return int(dN)