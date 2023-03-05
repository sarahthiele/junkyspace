import rebound
import reboundx 
import numpy as np
twopi = 2.0*np.pi


# should actually use rebound units routines to be more precise, but this is OK for now. 
aukm = 1.496e8
RE_eq = 6378.135/aukm
REkm = 6378.  # not entirely sure whether to use Requator or Raverage here.  
Ratm = (REkm+200)/aukm
J2=1.0827e-3
Mearthkg = 5.97e24
Msunkg = 1.989e30
Mearth = Mearthkg/Msunkg
vconv = np.sqrt(6.67e-11*1.989e30/1.496e11)
Nout = 1
ndebris=300
torbit = 0.0001*twopi*91/(365.25*24*60)/2 # VLEO orbit timescale (approximately 91 min)
tstart=0
tend=torbit # 0.00000000001*torbit
NTIME=1

#grid params for interpolation
NTHETA= 180  # number of different values in colatitude space
NALT=1200  # number of different values in altitude 
altCoMin=(300.+REkm)/aukm  # minimum grid altitude?
altCoMax=(1500+REkm)/aukm  # maximum grid altitude?
dAltCo=(altCoMax-altCoMin)/NALT  # dR
dThetaCo = np.pi/NTHETA  # dtheta


sim = rebound.Simulation()
sim.integrator ="WHFAST"
#sim.integrator ="ias15"
sim.dt = 3e-6
elow=3e-4

sim.add(m=Mearth,hash="Earth",r=Ratm)  # ask about this. Why Ratm? Does Rebound get rid of planets then
# that get within 200km of Earth?? Which would be nice

ICs=[]

ICs.append({'NPLANES':7178,'SATPP':1,'INC':30,'ALT':328}) 
ICs.append({'NPLANES':7178,'SATPP':1,'INC':40,'ALT':334}) 
ICs.append({'NPLANES':7178,'SATPP':1,'INC':53,'ALT':345}) 
ICs.append({'NPLANES':40,'SATPP':50,'INC':96.9,'ALT':360}) 
ICs.append({'NPLANES':1998,'SATPP':1,'INC':75,'ALT':373})
ICs.append({'NPLANES':4000,'SATPP':1,'INC':53,'ALT':499})
ICs.append({'NPLANES':12,'SATPP':12,'INC':148,'ALT':604})
ICs.append({'NPLANES':18,'SATPP':18,'INC':115.7,'ALT':614})

ICs.append({'NPLANES':2547,'SATPP':1,'INC':53,'ALT':345.6})
ICs.append({'NPLANES':2478,'SATPP':1,'INC':48,'ALT':340.8})
ICs.append({'NPLANES':2493,'SATPP':1,'INC':42,'ALT':335.9})
ICs.append({'NPLANES':32,'SATPP':50,'INC':53,'ALT':550})
ICs.append({'NPLANES':72,'SATPP':22,'INC':53.2,'ALT':540})
ICs.append({'NPLANES':36,'SATPP':20,'INC':70,'ALT':570})
ICs.append({'NPLANES':6,'SATPP':58,'INC':97.6,'ALT':560})
ICs.append({'NPLANES':4,'SATPP':43,'INC':97.6,'ALT':560.1})
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


for ic in ICs:
   nplanes=ic['NPLANES']
   nsat=ic['SATPP']
   ntot=nplanes*nsat  # total number of satellites in shell
   sma = (ic['ALT']+REkm)/aukm  # altitude of shell

   if nsat==1:  # case for shell in which all satellites occupy their own orbital plane
     print('Hard approach')
     dplane = twopi/ntot  # distribute planes evenly in RAAN
     ma = np.linspace(0,twopi,ntot,endpoint=False)  # distribute evenly in mean anomaly
     colat = np.pi*0.5-ic['INC']*np.pi/180  # colatitude
     angOpt = np.sqrt( (np.cos(colat)-np.cos(colat+np.pi*0.5))*twopi/ntot)  

# optimal spacing of grids in order to not have them overlapping is sqrt(number of 
# satellites). Each satellite has a unique orbital plane in the case that SATPP==1, thus they 
# are each given a unique, evenly space RAAN. However, if they all have the same MA and 
# inclination, they wouldn't spread evenly in a sphere, it would be a line. When we shuffle
# the MA's below, we go around in circles of evenly spaced RAAN's, placing satellites every
# sqrt(N) points.

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

     #np.random.shuffle(ma)
     isum=0
     for ilocal in range(nsat):
        for iplane in range(nplanes):
            Omega = iplane*dplane  # evenly spaced RAAN's
            omega = twopi*np.random.uniform()  # random argument of periapsis
            sim.add(m=0.,a=sma,inc=ic['INC']*twopi/360.,Omega=Omega,M=mashuf[isum]-omega,e=elow,omega=omega)  # add to Rebound sim
            isum+=1
   else: 
     dplane=twopi/nplanes  # spacing of orbital planes
     mashuf = np.linspace(0,twopi,nsat,endpoint=False)  # evenly space MA in each plane
     print("Simple approach")
     for ilocal in range(nsat):
          for iplane in range(nplanes):
              Omega = iplane*dplane
              omega = twopi*np.random.uniform()
              sim.add(m=0.,a=sma,inc=ic['INC']*twopi/360.,Omega=Omega,M=mashuf[ilocal]+iplane*dplane/nplanes-omega,e=elow,omega=omega)
   
ps=sim.particles

rebx = reboundx.Extras(sim)
gh = rebx.load_force("gravitational_harmonics")
rebx.add_force(gh)

ps["Earth"].params["J2"] = J2
ps["Earth"].params["J4"] = -1.620e-6 
ps["Earth"].params["R_eq"] = RE_eq


### set up grid for satellite densities

thetaCo = np.zeros(NTHETA)

for j in range(NTHETA): thetaCo[j]=dThetaCo*(j+0.5)  # colatitude distribution (midpoints)


satDensity = np.zeros([NALT,NTHETA])
altCo=[]
for i in range(NALT):
  altCo.append(altCoMin + i*dAltCo )  # altitude distribution

altCo=np.array(altCo)

satPhi=[]
satTheta=[]
times = np.linspace(tstart,tend,NTIME)
print(times)

for iout, time in enumerate(times):
    print("Working on time {}".format(time))
    sim.integrate(time)
    #output=sim.particles_ascii()
    #fh=open("output.ascii."+repr(iout),"w")
    #fh.write(output)
    #fh.close()

    print("Analyze particles")
    ps=sim.particles
    Np = len(ps)-1
    x = np.zeros(Np)
    y = np.zeros(Np)
    z = np.zeros(Np)
    vx = np.zeros(Np)
    vy = np.zeros(Np)
    vz = np.zeros(Np)
    a = np.zeros(Np)
    ecc = np.zeros(Np)
    inc = np.zeros(Np)
    for d in range(1,Np+1):
       x[d-1] = ps[d].x
       y[d-1] = ps[d].y
       z[d-1] = ps[d].z
       vx[d-1] = ps[d].vx
       vy[d-1] = ps[d].vy
       vz[d-1] = ps[d].vz
       a[d-1] = ps[d].a
       ecc[d-1] = ps[d].e
       inc[d-1] = ps[d].inc*180/np.pi

    rsphere = np.sqrt(x**2+y**2+z**2)  # radial distances
    thetaSat = np.arccos(z/rsphere)  # colatitudes
    phiSat = np.arctan2(y,x)  # azimuthal angle


    satPhi.append(phiSat)
    satTheta.append(thetaSat)

    for i in range(Np):
       ialt = int((rsphere[i]-altCoMin)/dAltCo-0.5)  # altitude index for satellite grid
       itheta = int(thetaSat[i]/dThetaCo)  # colatitude index for satellite grid
       satDensity[ialt][itheta]+=1./((altCo[ialt])**2*dAltCo*aukm**3*2*np.pi*(np.cos(thetaCo[itheta]-0.5*dThetaCo)-np.cos(thetaCo[itheta]+0.5*dThetaCo))*NTIME)  # add one satellite to satellite density


satPhi=np.array(satPhi)*180/np.pi
satTheta=90-np.array(satTheta)*180/np.pi

phiSat=phiSat*180/np.pi
thetaSat=90-thetaSat*180/np.pi


import matplotlib.pylab as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D

from mpl_toolkits.basemap import Basemap

plt.figure()
# setup Lambert Conformal basemap.
m = Basemap(width=12000000,height=9000000,
            rsphere=(6378137.00,6356752.3142),\
            resolution='l',area_thresh=1000.,projection='lcc',\
            lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.)
m.drawcoastlines()
#m.fillcontinents(color='coral',lake_color='aqua')
# draw parallels and meridians.
#m.drawmapboundary(fill_color='aqua')
m.drawparallels(np.arange(-80.,81.,10.))
m.drawmeridians(np.arange(-180.,181.,10.))
ax = plt.gca()
#lon, lat = m(x,y,inverse=True)
poly = m.scatter(phiSat,thetaSat,s=1,latlon=True,zorder=10)
plt.title("Satellite Distribution (Lat-Lon Projection)")
plt.savefig("LambComform.pdf")

plt.figure()
# setup Lambert Conformal basemap.
#m = Basemap(width=12000000,height=9000000,
#            rsphere=(6378137.00,6356752.3142),\
#            resolution='l',area_thresh=1000.,projection='lcc',\
#            lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.)
m = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,\
            llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')
m.drawcoastlines()
#m.fillcontinents(color='coral',lake_color='aqua')
# draw parallels and meridians.
m.drawparallels(np.arange(-80.,81.,20.))
m.drawmeridians(np.arange(-180.,181.,20.))
#m.drawmapboundary(fill_color='aqua')
ax = plt.gca()
#lon, lat = m(x,y,inverse=True)
poly = m.scatter(phiSat,thetaSat,s=1,alpha=0.5,latlon=True,zorder=10)
plt.title("Satellite Distribution (Lat-Lon Projection)")
plt.savefig("mercator.pdf")




fig=plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z,s=1)

plt.figure()
plt.scatter(x,y,s=1)

plt.figure()
plt.scatter(phiSat,thetaSat,s=1)


plt.figure()
plt.scatter(a,ecc,s=1)

fout1 = open("smooth2d.dat","w")
fout2 = open("smoothdensity2d.dat","w")
for i in range(NALT):
   for j in range(NTHETA):
        fout1.write("AltSmooth {} THETA {} DENSITY {}\n".format(altCo[i]*aukm-REkm,90-thetaCo[j]*180/np.pi,satDensity[i][j]))
        fout2.write("{}\n".format(satDensity[i][j]))

plt.show()


