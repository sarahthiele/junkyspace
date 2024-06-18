# authored by Aaron Boley

import sys
sys.path.insert(1, '/store/users/sthiele/home/ASATtest/')
from tools.asat_sma import *
from tools.KeplerTools import *
from functions import *
from NSBM_functions import *
import pandas as pd

MAKE = False
vconv = np.sqrt(6.67e-11*1.989e30/1.496e11)
Nout = 1
ndebris=300
torbit =  period((1500+REkm)*1000, G*Mearthkg)
sec_comp_conv = twopi / 365.25 / 24 / 3600
tstart = 3e-6
tend = torbit / 2 * sec_comp_conv
NTIME = 200
altref = 300

#grid params for interpolation
NTHETA= 180
NALT=1200
altCoMin=(300.+REkm)/aukm
altCoMax=(1500+REkm)/aukm
dAltCo=(altCoMax-altCoMin)/NALT
dThetaCo = np.pi/NTHETA


ICs=[]

ICs.append({'NPLANES':7178,'SATPP':1,'INC':30,'ALT':328}) 
ICs.append({'NPLANES':7178,'SATPP':1,'INC':40,'ALT':334}) 
ICs.append({'NPLANES':7178,'SATPP':1,'INC':53,'ALT':345}) 
ICs.append({'NPLANES':40,'SATPP':50,'INC':96.9,'ALT':360}) 
ICs.append({'NPLANES':1998,'SATPP':1,'INC':75,'ALT':373})
ICs.append({'NPLANES':4000,'SATPP':1,'INC':53,'ALT':499})
ICs.append({'NPLANES':12,'SATPP':12,'INC':148,'ALT':604})
ICs.append({'NPLANES':18,'SATPP':18,'INC':115.7,'ALT':614})

# Starlink 12000
ICs.append({'NPLANES':2547,'SATPP':1,'INC':53,'ALT':345.6})
ICs.append({'NPLANES':2478,'SATPP':1,'INC':48,'ALT':340.8})
ICs.append({'NPLANES':2493,'SATPP':1,'INC':42,'ALT':335.9})
#ICs.append({'NPLANES':32,'SATPP':50,'INC':53,'ALT':550})
ICs.append({'NPLANES':72,'SATPP':22,'INC':53,'ALT':550})
ICs.append({'NPLANES':72,'SATPP':22,'INC':53.2,'ALT':540})
ICs.append({'NPLANES':36,'SATPP':20,'INC':70,'ALT':570})
ICs.append({'NPLANES':6,'SATPP':58,'INC':97.6,'ALT':560})
ICs.append({'NPLANES':4,'SATPP':43,'INC':97.6,'ALT':560.1})

# OneWeb
ICs.append({'NPLANES':18,'SATPP':40,'INC':87.9,'ALT':1200})
ICs.append({'NPLANES':36,'SATPP':49,'INC':87.9,'ALT':1201})
ICs.append({'NPLANES':32,'SATPP':72,'INC':40,'ALT':1202})
ICs.append({'NPLANES':32,'SATPP':72,'INC':55,'ALT':1203})


ICs.append({'NPLANES':16,'SATPP':30,'INC':85,'ALT':590})
ICs.append({'NPLANES':40,'SATPP':50,'INC':50,'ALT':600})
ICs.append({'NPLANES':60,'SATPP':60,'INC':55,'ALT':508})
ICs.append({'NPLANES':48,'SATPP':36,'INC':30,'ALT':1145})
ICs.append({'NPLANES':48,'SATPP':36,'INC':40,'ALT':1145})
ICs.append({'NPLANES':48,'SATPP':36,'INC':50,'ALT':1145})
ICs.append({'NPLANES':48,'SATPP':36,'INC':60,'ALT':1145})

ICs.append({'NPLANES':34,'SATPP':34,'INC':51.9,'ALT':630} )
ICs.append({'NPLANES':36,'SATPP':36,'INC':42,'ALT':610 })
ICs.append({'NPLANES':28,'SATPP':28,'INC':33,'ALT':590 })

sim = rebound.Simulation()
sim.integrator ="WHFAST"
#sim.integrator ="ias15"
sim.dt = 3e-6
elow=3e-4

sim.add(m=Mearth,hash="Earth",r=Ratm)


for ic in ICs:
    nplanes=ic['NPLANES']
    nsat=ic['SATPP']
    ntot=nplanes*nsat
    sma = (ic['ALT']+REkm)/aukm

    if nsat==1:
        dplane = twopi/ntot
        ma = np.linspace(0,twopi,ntot,endpoint=False)
        colat = np.pi*0.5-ic['INC']*np.pi/180
        angOpt = np.sqrt( (np.cos(colat)-np.cos(colat+np.pi*0.5))*twopi/ntot)

        mashuf=np.zeros(ntot)
  
        nmix=int(np.sqrt(ntot))
  
        iref=0
        iskip=0
        for i in range(ntot):
            imix = iref+iskip*nmix
            if imix>ntot-1:
                iref+=1
                iskip=0
                imix=iref*1
            mashuf[i] = ma[imix]*1
            iskip+=1
        isum=0
        for ilocal in range(nsat):
            for iplane in range(nplanes):
                Omega = iplane*dplane
                omega = twopi*np.random.uniform()
                sim.add(m=0.,a=sma,inc=ic['INC']*twopi/360.,Omega=Omega,M=mashuf[isum]-omega,e=elow,omega=omega)
                isum+=1
    else: 
        dplane=twopi/nplanes
        mashuf = np.linspace(0,twopi,nsat,endpoint=False)
        print("Simple approach")
        for ilocal in range(nsat):
            for iplane in range(nplanes):
                Omega = iplane*dplane
                omega = twopi*np.random.uniform()
                sim.add(m=0.,a=sma,inc=ic['INC']*twopi/360.,Omega=Omega,
                        M=mashuf[ilocal]+iplane*dplane/nplanes-omega,
                        e=elow,omega=omega)
                

if MAKE == True:
    ps = sim.particles
     
    #rebx = reboundx.Extras(sim)
    #gh = rebx.load_force("gravitational_harmonics")
    #rebx.add_force(gh)
    
    #ps["Earth"].params["J2"] = J2
    #ps["Earth"].params["J4"] =-1.620e-6 
    #ps["Earth"].params["R_eq"] = RE_eq
    
    
    ### set up grid for satellite densities
    
    thetaCo = np.zeros(NTHETA)
    
    for j in range(NTHETA): thetaCo[j]=dThetaCo*(j+0.5)
    
    
    satDensity = np.zeros([NALT,NTHETA])
    altCo=[]
    for i in range(NALT):
        altCo.append(altCoMin + i*dAltCo )
    
    altCo=np.array(altCo)
    
    satPhi=[]
    satTheta=[]
    times = np.linspace(tstart,tend,NTIME)
    print(times)
    
    for iout, time in enumerate(times):
        print("Working on time {}".format(time))
        sim.integrate(time)
    
        print("Analyze particles")
        ps=sim.particles
        Np = len(ps)-1
        x = np.zeros(Np)
        y = np.zeros(Np)
        z = np.zeros(Np)
        for d in range(1,Np+1):
            x[d-1] = ps[d].x
            y[d-1] = ps[d].y
            z[d-1] = ps[d].z
        
        rsphere = np.sqrt(x**2+y**2+z**2)
        thetaSat=np.arccos(z/rsphere)
        phiSat=np.arctan2(y,x)
    
    
        satPhi.append(phiSat)
        satTheta.append(thetaSat)
    
        #ialt = ((rsphere-altCoMin)/dAltCo-0.5).astype(int)
        #itheta = (thetaSat/dThetaCo).astype(int)
        #satDensity[ialt,itheta]+=1./((altCo[ialt])**2*dAltCo*aukm**3*2*np.pi*(np.cos(thetaCo[itheta]-0.5*dThetaCo)-np.cos(thetaCo[itheta]+0.5*dThetaCo))*NTIME)
    
    
        for i in range(Np):
            ialt = int((rsphere[i]-altCoMin)/dAltCo-0.5)
            itheta = int(thetaSat[i]/dThetaCo)
            satDensity[ialt][itheta]+=1./((altCo[ialt])**2*dAltCo*aukm**3*2*np.pi*(np.cos(thetaCo[itheta]-0.5*dThetaCo)-np.cos(thetaCo[itheta]+0.5*dThetaCo))*NTIME)
    
    
    satPhi=np.array(satPhi)*180/np.pi
    satTheta=90-np.array(satTheta)*180/np.pi
    
    phiSat=phiSat*180/np.pi
    thetaSat=90-thetaSat*180/np.pi
    
    
    #import matplotlib.pylab as plt
    #fig=plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(x,y,z,s=1)
    
    #plt.figure()
    #plt.scatter(x,y,s=1)
    
    #plt.figure()
    #plt.scatter(phiSat,thetaSat,s=1)
    
    
    #plt.figure()
    #plt.scatter(a,ecc,s=1)
    #plt.show()
    
    
    
    fout1 = open("smooth2d_python_200.dat","w")
    fout2 = open("smoothdensity2d_python_200.dat","w")
    for i in range(NALT):
        for j in range(NTHETA):
            fout1.write("AltSmooth {} THETA {} DENSITY {}\n".format(altCo[i]*aukm-REkm,90-thetaCo[j]*180/np.pi,satDensity[i][j]))
            fout2.write("{}\n".format(satDensity[i][j]))
    fout1.close()
    fout2.close()
