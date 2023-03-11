#============================================================================
# Implementation of the NSBM
#============================================================================
import rebound
import reboundx 
import rebound.units as u
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats
from scipy.stats import maxwell as maxwell
from scipy.stats import norm as norm

from tools.asat_sma import *
from functions import *

#SEED=314

#np.random.seed(SEED)

# These equations come from payload parameters in Appendix B and C 
# of https://arc.aiaa.org/doi/pdf/10.2514/1.G004939

def get_AM_ratio(L, nums):
    '''
    Get area-to-mass ratio distribution of all debris fragments.
    
    INPUT:
    -----------------------
    L [array]:    midpoint values of characteristic length bins
    nums [array]: array of number of fragments in each length bin 
                  (length must be the same as L)
    
    OUTPUT:
    ------------------------
    AM [array]:   array of length sum(nums) of area-to-mass ratios.
    '''
    nums = nums.astype(int)
    def alphaL(lamdaL):
        alphaL1 = np.zeros(len(lamdaL))
        alphaL1[lamdaL <= -1.95] = 0
        alphaL1[(lamdaL>-1.95)&(lamdaL<0.55)] = 0.3 + 0.4 *\
                                (lamdaL[(lamdaL>-1.95)&(lamdaL<0.55)] + 1.2)
        alphaL1[lamdaL >= 0.55] = 1
        
        alphaL2 = 1 - alphaL1
        return alphaL1, alphaL2
    def muLvals(lamdaL):
        muL1 = np.zeros(len(lamdaL))
        muL1[lamdaL <= -1.1] = -0.6
        muL1[(lamdaL>-1.1)&(lamdaL<0.)] = -0.6 - 0.318 *\
                                (lamdaL[(lamdaL>-1.1)&(lamdaL<0.)] + 1.1)
        muL1[lamdaL >= 0.0] = -0.95  
        
        muL2 = np.zeros(len(lamdaL))
        muL2[lamdaL <= -0.7] = -1.2
        muL2[(lamdaL > -0.7)&(lamdaL < -0.1)] = -1.2 - 1.333 *\
                            (lamdaL[(lamdaL > -0.7)&(lamdaL < -0.1)] + 0.7)
        muL2[lamdaL >= -0.1] = -2.0
        return muL1, muL2
    def sigmaL(lamdaL):
        sigmaL1 = np.zeros(len(lamdaL))
        sigmaL1[lamdaL <= -1.3] = 0.1
        sigmaL1[(lamdaL>-1.3)&(lamdaL<-0.3)] = 0.1 + 0.2 *\
                            (lamdaL[(lamdaL>-1.3)&(lamdaL<-0.3)] + 1.3)
        sigmaL1[lamdaL >= -0.3] = 0.3
        
        sigmaL2 = np.zeros(len(lamdaL))
        sigmaL2[lamdaL <= -0.5] = 0.5
        sigmaL2[(lamdaL > -0.5)&(lamdaL < -0.3)] = 0.5 - \
                            (lamdaL[(lamdaL > -0.5)&(lamdaL < -0.3)] + 0.5)
        sigmaL2[lamdaL >= -0.3] = 0.3 
        return sigmaL1, sigmaL2
    def muSvals(lamdaS):
        mu_S = np.zeros(len(lamdaS))
        mu_S[lamdaS <= -1.75] = -0.3
        mu_S[(lamdaS > -1.75)&(lamdaS < -1.25)] = -0.3 - 1.4 *\
                            (lamdaS[(lamdaS > -1.75)&(lamdaS < -1.25)] + 1.75)
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
        pL = aL_1[i] * scipy.stats.norm(muL_1[i], 
                                        sigL_1[i]).pdf(chi) + \
             aL_2[i] * scipy.stats.norm(muL_2[i], sigL_2[i]).pdf(chi)
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
        pL = a1[i] * scipy.stats.norm(mu1[i], 
                                      sig1[i]).pdf(chi) +\
             a2[i] * scipy.stats.norm(mu2[i], sig2[i]).pdf(chi)
        cdfL = np.cumsum(pL*dchi)
        P = np.random.uniform(size=num)
        cdfindices = np.digitize(P, bins=cdfL)
        cdfindices[cdfindices==len(chi)] = len(chi) - 1
        chis = np.append(chis, chi[cdfindices])
        
    AM = 10 ** chis
    return AM


def get_A_M_vals(Lvals, AM, nums):
    '''
    Get area and mass values of debris fragments.
    
    INPUT:
    -----------------------
    Lvals [array]:  midpoint values of characteristic length bins in m
    AM [array]:     area-to-mass ratios of fragments, length sum(nums) in m^2/kg
    nums [array]:   array of number of fragments in each length bin, same
                    length as Lvals

    OUTPUT:
    -----------------------
    Avals [array]:  areas of fragments in m^2
    mvals [array]:  masses of fragments in kg
    Lcvals [array]: sizes array of fragments (length sum(nums)) in m
    '''
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
    '''
    Get velocity kicks of fragments from the NSBM.
    
    INPUT:
    -----------------------
    AM [array]: area-to-mass ratios of fragments in m^2/kg
    
    OUTPUT:
    -----------------------
    vfrags [(length(AM)x3) array]: velocity kick vectors in m/s
    '''
    chi = np.log10(AM)
    mu = 0.9 * chi + 2.9
    sigma = 0.4 * np.ones(len(chi))
    v = np.random.normal(loc=mu, scale=sigma, size=np.array([1, len(chi)]))
    magdv = 10 ** v[0]
    
    # need to note normal distribution for future work (uniform before)
    unitvecs = np.random.normal(0, 1, (len(magdv), 3))
    vnormed = (unitvecs.T / np.linalg.norm(unitvecs, axis=1)).T
    vfrags = (vnormed.T * magdv).T
    return vfrags

def vel_dis_NBM(mtarget, mkill, vkill, vtarget, rtar, nbins, Lc_min, Lc_max, 
                KEkill, numsample, makev=True):
    '''
    Full implementation of the NASA Standard Breakup Model (NSBM) to
    simulate a debris cloud from an on-orbit collision of a missile
    and a target satellite.
    
    INPUT:
    -----------------------
    mtarget [float]:     target mass in kg
    mkill [float]:       kill vehicle mass in kg
    vtarget [array]:     velocity vector of target in m/s
    rtar [array]:        cartesian position vector of target in m
    nbins [int]:         number of bins for Lc distribution
    Lc_min [float]:      minimum characteristic length in m
    Lc_max [float]:      maximum characteristic length in m
    KEkill [float]:      kill energy of collision in J
    makev [bool]:        simulate/return full velocity distribution
    numsample [int]:     number of debris fragments to simulate...
       
    !! numsample=100 will generate FULL NSBM distribution !!
       

    OUTPUT:
    -----------------------
    If makev=False:
    numsample [int]:      number of fragments created according to NSBM 

    If makev = True:
    N_tot [int]:          total number of fragments (numsample or NSBM number)
    L_mids [array]:       midpoints of characteristic length bins
    nums [array]:         number of fragments in each Lc bin
    mfrags [array]:       masses of fragments
    vfrags_all [array]:   all frag velocity kick vectors generated by NSBM
    vfrags [array]:       kick vectors of sampled frags (= vfrags_all 
                          if numsample=100)
    vfrags_total [array]: total velocity vectors of sampled fragments
    eccs [array]:         eccentricities of sampled fragments
    SMA [array]:          semi-major axes of sampled fragments
    AMfrags [array]:      area-to-mass ratios of all fragments
    AMsample [array]:     AM ratios of sampled frags (=AMfrags if numsample=100)
    Lcvals [array]:       Lc distribution of all frags
    Lcsample [array]:     Lc dis. of sampled frags (=Lcvals if numsample=100)
    '''

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
        
        INPUT: 
        -----------------------
        L0 [float]: bin lower bound in m
        L1 [float]: bin upper bound in m
        Me [float]: ejected mass from collision in kg
        
        OUTPUT:
        -----------------------
        integer number of fragments in bin
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

            velvec = [N_tot, L_mids, nums, mfrags, vfrags, vfrags, 
                      vfrags_total, eccs, SMA, AMfrags, AMfrags, Lcvals, 
                      Lcvals] 
            return velvec
        
        elif numsample != 100:
            vfrags_all = get_delta_vs(AMfrags)
            
            indices = np.linspace(0, len(AMfrags)-1, 
                                  len(AMfrags)).astype(int)
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
            
            velvec = [N_tot, L_mids, nums, mfrags, vfrags_all, vfrags, 
                      vfrags_total, eccs, SMA, AMfrags, AMsample, Lcvals, 
                      Lcsample]
            
            return velvec
