#########################################################################
# Original code by Aaron Boley
# March 2021
#########################################################################

import numpy as np
import matplotlib.pylab as plt

def mag(v):
    '''
    S: magnitude of velocity vector <v_x, v_y, v_z>
    '''
    return np.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])

def period(a,GM):
    '''
    S: Period of an object from Kepler's law
    '''
    return 2*np.pi*np.sqrt(a**3/GM)

def sma(v,GM,r):
    '''
    S: Calculates the semi-major axis of an object
    using the vis-viva equation
    '''
    v2 = (v[0]*v[0]+v[1]*v[1]+v[2]*v[2])
    return GM*r/(2*GM-v2*r)

def eccVector(v,GM,r):
    '''
    S: For Kepler orbits the eccentricity vector is a constant of motion.
    It's a dimensionless vector with direction pointing from apoapsis 
    to periapsis and with magnitude equal to the orbit's scalar eccentricity
    '''
    v = np.array(v)
    r = np.array(r)
    v2 = (v[0]*v[0]+v[1]*v[1]+v[2]*v[2])
    rdotv = r[0]*v[0]+r[1]*v[1]+r[2]*v[2]
    rmag = np.sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2])
    return (v2/GM - 1/rmag)*r - rdotv/GM*v

def smaFragments(alt0=850e3,GM=3.982e14,msat=740.,mkill=10.,vkill=10e3,
                 ndebris=3000,R0=6371e3,killenergy=130e6,etaMOM=0,etaKE=0.33,PLOT=True):
    '''
    Snotes: changed msat from 750 to 740 kg (matches Mission Shakti satellite). vkill=10km/s. 
    On wiki (https://en.wikipedia.org/wiki/Indian_Ballistic_Missile_Defence_Programme), the
    mass of the interceptor is 18.87 tons, or 17118.576 kg. Should I change mkill to this?
    Changed to output mdebris for REBOUND.
    Why is vmag = 6*vexpl?
    The delta-v to the spacecraft after impact is cited here to be -21 m/s: 
    https://www.youtube.com/watch?v=Pzhtc-rFbvM
    This function gives us values for the semi-major axis, eccentriciy, apogee, perrigee,
    and orbital period in minutes of all the fragments from the ASAT test explosion.
    '''
    sma0 = R0 + alt0
    
    vc = np.sqrt(GM/(sma0))
    
    print('Initial satellite speed: {} m/s'.format(vc))

    vinter = vkill-vc
    
    print("Relative speed of impactor to satellite: {} m/s".format(vinter))

    vafter = (msat * vc - mkill * vinter * etaMOM )/(msat)
    #vafter = vc - 21
    
    print("Satellite speed immediately after impact {} m/s ".format(vafter))

    if killenergy is None: ekill = 0.5*mkill*vkill**2
    else: ekill = killenergy

    print("Kill energy is {} MJ".format(ekill/1e6))

    eExpPerMass = ekill/msat  # S: energy of the explosion per mass

    mdebris = msat/ndebris  # S: mass per debris fragment assuming constant mass distribution

    vexpl = np.sqrt(2*eExpPerMass*etaKE)

    print("Fragment velocity relative to satellite is {} m/s ".format(vexpl))

# now set sma of satellite due to momentum transfer

    smaAfter = sma([vafter,0,0],GM,sma0)
    REkm = 6378.
    print("SMA {} m, assuming momentum transfer coupling {}, i.e. alt = {} km".format(smaAfter, 
                                                                                      etaMOM,
                                                                                      smaAfter/1000 - REkm))
    
    vfrag = []
    for i in range(ndebris):
       v = np.array([np.random.uniform(-1,1),np.random.uniform(-1,1),np.random.uniform(-1,1)])
       vnorm = np.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])

       vmag = 6*vexpl
       while vmag > 5*vexpl:
            P = np.random.uniform(0,1)
            vmag = np.abs((vexpl/3)*np.tan(np.pi*(P-0.5)))
       v = v/vnorm*vmag
       vfrag.append(v)

    checkE=0.
    for v in vfrag:
       checkE += 0.5*mdebris*mag(v)**2
    print("Check energy in to energy out {}".format(ekill/checkE))

    vfrag=np.array(vfrag)

    vfrag = vfrag + np.array([vafter,0.,0.])

    smadistro=[]
    for v in vfrag:
       smadistro.append(sma(v,GM,sma0))

    smadistro=np.array(smadistro)
    perMinute = period(smadistro,GM)/60.
    alt = smadistro - R0

    ecc=[]
    rinst = np.array([0.,sma0,0.])
    for i,v in enumerate(vfrag):
        eccV=(eccVector(v,GM,rinst))
        ecc.append(np.sqrt(eccV[0]*eccV[0]+eccV[1]*eccV[1]+eccV[2]*eccV[2]))

    ecc=np.array(ecc)
    apo  = ((1.+ecc)*smadistro-R0)/1e3
    peri = ((1.-ecc)*smadistro-R0)/1e3
    alt = alt/1e3
    flag = peri > 100.

    if PLOT:
       plt.figure()
       #plt.yscale('log')
       #plt.xscale('log')
       plt.scatter(perMinute[flag],peri[flag],marker='.')
       plt.scatter(perMinute[flag],apo[flag],marker='.')
       plt.figure()
       plt.hist(alt[flag],bins=100,range=[0,2000])
       plt.show()
       return "Done"

    return mdebris, smadistro, peri, apo, ecc, perMinute
        
#smaFragments(ndebris=400,vkill=9e3,killenergy=0)
