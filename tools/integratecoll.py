#########################################################################
# Original code by Aaron Boley
# March 2021
#########################################################################

import numpy as np
import matplotlib.pyplot as plt

G=6.67e-11  # mks 
Me=5.97e24  # kg
Re=6378e3 # m
altStop=100e3 # m


denmodel0=[ 8.220E-07, 4.509E-10, 1.858E-12, 2.601E-13, 5.725E-14, 1.549E-14, 4.752E-15, 1.580E-15,
   5.614E-16,
   2.153E-16,
   9.183E-17,
   4.504E-17,
   2.572E-17,
   1.673E-17,
   1.192E-17,
   8.956E-18,
   6.943E-18,
   5.480E-18,
   4.374E-18,
   3.520E-18] # F10.7 67 sfu

zmodel=[50e3,100e3,150e3,200e3,250e3,300e3,350e3,400e3,450e3,500e3,550e3,600e3,650e3,700e3,750e3,800e3,850e3,900e3,950e3,1000e3]

denmodel1=[ 8.459E-07,
  4.653E-10,
  1.955E-12,
  3.217E-13,
  9.051E-14,
  3.133E-14,
  1.215E-14,
  5.057E-15,
  2.207E-15,
  9.986E-16,
  4.668E-16,
  2.260E-16,
  1.143E-16,
  6.108E-17,
  3.495E-17,
  2.160E-17,
  1.444E-17,
  1.035E-17,
  7.838E-18,
  6.181E-18]

denmodel0=np.array(denmodel0)*1000. # model density above is in g/cc
denmodel1=np.array(denmodel1)*1000.

logden0 = np.log10(denmodel0)
logden1 = np.log10(denmodel1)
logz = np.log10(zmodel)

yr = 3600*24*365.25

def MA(TA,ecc):
    '''
    Calculates mean anomaly from the true anomaly and the eccentricity
    of the object, where the true anomaly is the angle from periapsis to the
    objects current location on a Keplerian orbit from the focus of the
    orbit around which the object is orbiting.
    
    Input: true anomaly, eccentricity (arrays or single values)
    output: mean anomaly(ies)
    '''
    return TA -2*ecc*np.sin(TA) + (0.75*ecc**2 + 0.125*ecc**4)*np.sin(2*TA) - 1./3. * ecc**3 *np.sin(3*TA) + 5./32.*ecc**4*np.sin(4*TA)

def rhoatm(alt,t,phase,low=None):
    '''
    Calculates the density of the atmosphere using a given atmospheric
    model, at a particular time, altitude and orbital phase
    '''
    i=int(alt/50e3)-1
    if i > len(zmodel)-2: i=len(zmodel)-2
    if i < 0: i=0

    try:
       logalt = np.log10(alt)
    except:
       logalt = 0.
    d0 = 10.**(  logden0[i]+(logden0[i+1]-logden0[i])/(logz[i+1]-logz[i])*(logalt-logz[i]) )
    d1 = 10.**(  logden1[i]+(logden1[i+1]-logden1[i])/(logz[i+1]-logz[i])*(logalt-logz[i]) )
 
    if low==None:
      w = np.sin(t/yr*2*np.pi/21.63653+phase)**2
      d = d0*(1.-w) + d1*w
    else: 
       if low is True: d=d0
       else: d=d1

    return d


def dadt(t,phase,alt,diameter,rhom,aoverm,CD,low=None):
    '''
    Derivative of altitude due to atmospheric drag force. Orbits are
    assumed circular for now.
    '''
    if aoverm<0:amratio = 3./(2.*rhom*diameter)  # area to mass ratio, assuming a sphere
    else: amratio=aoverm
    rho=rhoatm(alt,t,phase,low=low)
    adot= -np.sqrt(G*Me*(alt+Re))*CD*rho*amratio
    #print("alt={} rho={} amratio={}".format(alt,rho,amratio))
    return adot

def Integrate(alt0,diameter=1.,rhom=2000.,aoverm=-1,CD=2.2,dt=100.,phase=0.,MAXDT=None,
              PLOT=False,tfinal=None,cross_section=10.,lowActivity=None):
    '''
    This function follows the path of a fragment over time in its orbit, assuming
    a circular orbit, as it decays due to drag forces. It assumes satellite shells
    using a Gaussian width estimate of a given sigma value, and samples the fragment's
    collision probability every 10 metres of its decay. It then sums over the stepwise
    probability to find a cumulative probability. The probability itself is essentially
    the inverse of an optical depth faced by the fragment as it travels along its path.
    '''
    time=[]
    alt_array=[]

    t=0.
    alt=alt0*1
    time.append(t)
    alt_array.append(alt)

    def satDenModel(alt,sig=1000):
         ''' sig in m
         
         this function estimates each satellite shell using Gaussian radial widths
         with a sigma value "sig". For any given altitude, it infers the number of
         satellites from a particular satellite set within that altitude shell
         from a normal distribution scaled to the number of total sats in each
         set and their altitude. It then divides by the surface area of the
         sphere at the test altitude alt to get a number density for that shell.
         
         Snotes: changed s1 coefficient from 1600 to 1584
         '''
         sig2=sig*sig
         c = 1/(sig*np.sqrt(2*np.pi))
         s1 = 1584*c*np.exp(-0.5*(alt-550.e3)**2/sig2)/(4*np.pi*(Re+550e3)**2)
         s2 = 1600*c*np.exp(-0.5*(alt-1110.e3)**2/sig2)/(4*np.pi*(Re+1110e3)**2)
         s3 = 400*c*np.exp(-0.5*(alt-1130.e3)**2/sig2)/(4*np.pi*(Re+1130e3)**2)
         s4 = 375*c*np.exp(-0.5*(alt-1275.e3)**2/sig2)/(4*np.pi*(Re+1275e3)**2)
         s5 = 450*c*np.exp(-0.5*(alt-1325.e3)**2/sig2)/(4*np.pi*(Re+1325e3)**2)
         s6 = 2547*c*np.exp(-0.5*(alt-345.6e3)**2/sig2)/(4*np.pi*(Re+345.6e3)**2)
         s7 = 2478*c*np.exp(-0.5*(alt-340.8e3)**2/sig2)/(4*np.pi*(Re+340.8e3)**2)
         s8= 2493*c*np.exp(-0.5*(alt-335.9e3)**2/sig2)/(4*np.pi*(Re+335.9e3)**2)

         return (s1+s2+s3+s4+s5+s6+s7+s8)
      

    cumulProb=[]
    cumulProb.append(0.)
    while alt > altStop:
        # loops through the altitudes of the fragment as it decays. At each time step,
        # finds the orbital speed, altitude, and satellite number density at that altitude,
        # and thus a probability for that step. Then repeats this process for the next time
        # step and averages the two, essentially as a corrector step.
        dadt_test= dadt(t,phase,alt,diameter=diameter,rhom=rhom,aoverm=aoverm,CD=CD,low=lowActivity)
        alt_test = alt + dadt_test*dt

        rho_test = satDenModel(alt_test)
        vcirc_test = np.sqrt(G*Me/(alt_test+Re))
        prob_test = rho_test*vcirc_test*cross_section*dt

        dadt_cor=dadt(t,phase,alt_test,diameter,rhom,aoverm=aoverm,CD=CD,low=lowActivity)
        alt_cor= alt + dadt_cor*dt
 
        rho_cor = satDenModel(alt_cor)
        vcirc_cor = np.sqrt(G*Me/(alt_cor+Re))
        prob_cor = rho_cor*vcirc_cor*cross_section*dt

        cumulProb.append(0.5*(prob_test+prob_cor))
       
        alt = 0.5 * (alt_test+alt_cor)
        dadt_avg = 0.5*(dadt_test+dadt_cor)
        t+=dt

        #if FIXEDSTEP==False: dt = -alt/(100*dadt_avg)
        #dt = -alt/(1000*dadt_avg)
        dt = -10/(dadt_avg)  # time set to decrease by 10 metre!
        if MAXDT is not None: dt=min(dt,MAXDT) 
        #print(dt/3.155e7)


        time.append(t)
        alt_array.append(alt)

        if tfinal is not None:
              if t >= tfinal*yr:break

        
    cumulProb=np.array(cumulProb)
    sumProb = np.zeros(len(cumulProb))
    sumProb[0]=cumulProb[0]*1 

    for i in range(1,len(sumProb)): sumProb[i]=sumProb[i-1]+cumulProb[i]

    alt_array=np.array(alt_array)/1000.
    time=np.array(time)/(24*3600*365.25)

    if PLOT==True:
        print("Time {} d for de-orbit".format(time[-1]))

        plt.plot(time,alt_array)
        plt.ylabel('Radius above Earth (km)')
        plt.xlabel('Years')
        return plt.show()
    return time,alt_array,sumProb

def IntegratePeriApo(alt0,ecc=0.,diameter=1.,rhom=2000.,aoverm=-1,CD=2.2,
                     dt=1.,phase=0.,MAXDT=None,PLOT=False,tfinal=None,wfrac=0.524):
    '''
    Integrates the orbital elements of an object, taking into account non-circular orbits.
    Returns the average altitude, the low/high altitude (at either extreme of the orbit taking
    eccentricity into account), and the eccentricity for a series of times.
    '''
    time=[]
    alt_array=[]
    ecc_array=[]
    altl_array=[]
    alth_array=[]

    t=0.
    alt=alt0*1
    time.append(t)
    alt_array.append(alt)

    altl = (alt+Re)*(1-ecc)-Re
    if alt < altStop: 
         ecc = 0.
         alt=altStop*1
    elif altl < altStop:
         ecc = 1.-(altStop+Re)/(alt+Re)
    altl = (alt+Re)*(1-ecc)-Re
    #altl0=(alt+Re)*(1-ecc)-Re

    while alt > altStop:
        alth = (alt+Re)*(1+ecc)-Re
        altl = (alt+Re)*(1-ecc)-Re
        #if alt < altl0: altl=alt*1
        #else: altl=altl0
        w = MA(wfrac,ecc)/np.pi
        #print(alt,altl,ecc,w)
        dadt_test= ((w)*dadt(t,phase,altl,diameter=diameter,rhom=rhom,aoverm=aoverm,CD=CD)+(1.-w)*dadt(t,phase,alth,diameter=diameter,rhom=rhom,aoverm=aoverm,CD=CD))
        alt_test = alt + dadt_test*dt
        if alt_test < altStop: 
             alt=altStop
             break       

        ecc_test = max(1.-(altl+Re)/(alt_test+Re),0.)
        altl = (alt_test+Re)*(1.-ecc_test)-Re
        alth = (alt_test+Re)*(1.+ecc_test)-Re
        w = MA(wfrac,ecc_test)/np.pi
        dadt_cor=((w)*dadt(t,phase,altl,diameter,rhom,aoverm=aoverm,CD=CD)+(1.-w)*dadt(t,phase,alth,diameter,rhom,aoverm=aoverm,CD=CD))
        alt_cor= alt + dadt_cor*dt
        ecc_cor = max(1.-(altl+Re)/(alt_cor+Re),0.)
        
        alt = 0.5 * (alt_test+alt_cor)
        dadt_avg = 0.5*(dadt_test+dadt_cor)
        ecc = 0.5 * (ecc_test + ecc_cor)
        t+=dt

        #if FIXEDSTEP==False: dt = -alt/(100*dadt_avg)
        dt = -alt/(100*dadt_avg)
        if MAXDT is not None: dt=min(dt,MAXDT) 
        #print(dt/3.155e7)


        time.append(t)
        alt_array.append(alt)
        ecc_array.append(ecc)
        altl = (alt+Re)*(1.-ecc)-Re
        alth = (alt+Re)*(1.+ecc)-Re
        altl_array.append(altl)
        alth_array.append(alth)

        if tfinal is not None:
              if t >= tfinal*yr:break

        

    alt_array=np.array(alt_array)/1000.
    time=np.array(time)/(24*3600*365.25)
    ecc_array=np.array(ecc_array)
    altl_array=np.array(altl_array)/1000.
    alth_array=np.array(alth_array)/1000.

    if PLOT==True:
        print("Time {} d for de-orbit".format(time[-1]))

        plt.plot(time,alt_array)
        plt.ylabel('Radius above Earth (km)')
        plt.xlabel('Years')
        return plt.show()
    return time,alt_array,ecc_array,altl_array,alth_array


# you can enter a density and diameter or an aoverm (area to mass ratio). MKS.  If you want to use density and diameter, set aoverm=-1 or not include.
#Integrate(alt0=800e3, aoverm=0.01,dt=100,MAXDT=3.155e7,PLOT=True)
