#
# Written hastily by Aaron C. Boley
# Updated: 22 June 2021
#
import rebound as rb
import reboundx 
import numpy as np
import sys
twopi = 2.0*np.pi

SEED=314


# shoudl actually use rebound units routines to be more precise, but this is OK for now. 
RE_eq = 6378.135 # will convert to code
REkm=RE_eq # will keep in km
J2=1.0827e-3 # harmonics
J4=-1.620e-6
Mearthkg = 5.97e24
Ratm = (REkm+200)  # effective size of Earth -- not really used at this time. will convert to code

# some timing considerations
torbit = twopi*91/(365.25*24*60) # VLEO orbit timescale (approximately 91 min) For reference
tstart=0
tend=1*twopi/365 # Integrate for 5 years. For current situation, de-orbits faster than that.  
NTIME=1000  # number of breaths the code takes (C code to Python).  Used mainly to check whether particle is reaching decay limit until I do something better


#units
aum = rb.units.lengths_SI['au']
au=aum*1e2
aukm=aum/1e3 # au in km
Msunkg=rb.units.masses_SI['msun'] # kg
code2sec = rb.units.times_SI['yr2pi']
sec2code = 1./code2sec
vcode2cmps=au*sec2code
Mearth = Mearthkg/Msunkg
vconv = vcode2cmps/1e2 # convert code V to m/s.

print(vconv,vcode2cmps)

RE_eq/=aukm
Ratm /=aukm

# explosion precursor and some debris considerations
INC=96.6
RLIM=0.01
afragment=283. # where the Kaboom happens
EXPLODE=130e6/3 # kaboom in J
PAYLOAD=740 # mass of thing getting kaboomed in kg
REMOVALT=100 # stop following after this altitude in km
Lc0=0.1 # Length of debris in metre
Lc1=1.0 # Length of debris in metre
LDIV=10 # number of L divisions in log space
RHOPAYLOAD=2700. # kg/m^3
BCOEFF=0.22  # CD * A/A/MM

#grid params for interpolation.  Set by satden.dat file
NTHETA=180
NALTDENSITY=1200
deltaAlt=1.0/aukm

# Select integrator. 
sim = rb.Simulation()
sim.integrator ="WHFAST"
#sim.integrator ="ias15"
sim.dt = 1e-7  # MUST force small time step because we want to ensure that we sample the satellite density distribution
               # this gives 0.5 s. That seems to be OK for this problem, as an all of LEO eccentric particle will have
               # a radial velocity maximum of about 1 km/s. The altitude shells are distributed in 0.5 km cell widths. 

sim.add(m=Mearth,hash="Earth",r=Ratm)

sma = (afragment+REkm)/aukm
sim.add(m=0.,a=sma,inc=INC*twopi/360.,Omega=0,M=0,e=0.0)
   
ps=sim.particles

# sanity check
print("Earth: x={}, y={}, z={}".format(ps[0].x,ps[0].y,ps[0].z))
print("First particle: x={}, y={}, z={}".format(ps[1].x,ps[1].y,ps[1].z))


def dNdl(l0,l1,M,BINS):
    BINS=int(BINS)
    dN=np.zeros(BINS)
    dM=np.zeros(BINS)
    edges = 10**(np.log10(l0)+(np.log10(l1)-np.log10(l0))/BINS*np.array(range(0,BINS+1)))
    mass=0
    for i in range(BINS):
      dN[i] = int(  -0.171*M**0.75*(edges[i+1]**(-2.71)-edges[i]**-2.71) )
      dM[i] = RHOPAYLOAD*dN[i]*(edges[i+1]+edges[i])**2*np.pi/8*0.01  # 1 cm thick  <- missing factor of 2?
      mass+=dM[i]
    dN=dN*M/mass
    mass=0
    cum=0
    for i in range(BINS):
      dN[i]=int(dN[i])
      cum+=dN[i]
      dM[i]=RHOPAYLOAD*dN[i]*(edges[i+1]+edges[i])**2*np.pi/8*0.01
      mass+=dM[i]
    return dN,dM,edges,cum,mass

dN,dM,edges,cum,mass=dNdl(Lc0,Lc1,PAYLOAD,LDIV)
Nfrags=int(cum)


print("Cumulative debris number {} and mass {}".format(cum,mass))

vpert = np.zeros(Nfrags)

#
im=-1
iter=0
size=[]
ESMALLFRAGS=EXPLODE*(1-(PAYLOAD-mass)/PAYLOAD)  # not sure about this.  Partition energy to be in fragment mass
for n in dN:
   im+=1
   for i in range(int(n)):
      vpert[iter]=np.sqrt(2*ESMALLFRAGS*n/dM[im]/Nfrags)/vconv
      size.append((edges[im]+edges[im+1])/2)
      iter+=1

size=np.array(size)

np.random.seed(SEED)

dvx0 = np.random.uniform(-1,1,Nfrags)
dvy0 = np.random.uniform(-1,1,Nfrags)
dvz0 = np.random.uniform(-1,1,Nfrags)

norms = np.sqrt(dvx0**2+dvy0**2+dvz0**2)

dvx = dvx0/norms*vpert
dvy = dvy0/norms*vpert
dvz = dvz0/norms*vpert

vfrag=np.array([dvx,dvy,dvz])

NPERT=Nfrags

ps[1].r=size[0]
for i in range(Nfrags-1):
   p = np.random.uniform(-1,1)*1e-12 # Rebound has an issue with objects overlapping.  There is no clear (or good) reason for this
   Omega=3*np.pi/2.
   if size[i+1] > RLIM:
      sim.add(m=0.,a=sma+p,inc=INC*twopi/360.,r=size[i+1]) # size is not really needed, but I thought it might help with the above problem. It does not.
   

ps=sim.particles

print("Total number of particles is {}".format(len(ps)))
for iter,p in enumerate(ps):
   if iter==0: continue
   print(iter)
   p.vx+=vfrag[0][iter-1]
   p.vy+=vfrag[1][iter-1]
   p.vz+=vfrag[2][iter-1]

arra=[]
erra=[]
Period=[]
for p in ps[1:-1]:
  arra.append(p.a)
  erra.append(p.e)
  Period.append(p.P)

arra=np.array(arra)*aukm
peri=arra*(1.-np.array(erra))-REkm
apo=arra*(1.+np.array(erra))-REkm
arra-=REkm

Period=np.array(Period)*365.25*24*60/(2*np.pi)

import matplotlib.pylab as plt
import matplotlib
matplotlib.use('Agg') 
plt.figure()
bins,edges,patches=plt.hist(arra,bins=100,range=[0,2000])
plt.savefig("hist_a_start.png")

plt.figure()
plt.ylabel("Altitude [km]")
plt.xlabel("Period [min]")
plt.ylim([180,2400])
plt.scatter(Period,apo)
plt.scatter(Period,peri)
plt.savefig("gabbard_start.png")



rebx = reboundx.Extras(sim)
gh = rebx.load_force("gravitational_harmonics")
rebx.add_force(gh)
ps["Earth"].params["J2"] = J2
ps["Earth"].params["J4"] =-1.620e-6 
ps["Earth"].params["R_eq"] = RE_eq


# New collision probability code
coprob = rebx.load_operator("collision_prob")
rebx.add_operator(coprob)
coprob.params["nAltDensity"]=NALTDENSITY
coprob.params["nThetaDensity"]=NTHETA
coprob.params["altReference"]=(300+REkm)/aukm
coprob.params["deltaAlt"]=deltaAlt
coprob.params["deltaTheta"]=np.pi/NTHETA
coprob.params["density_mks_to_code"]=aukm**3
coprob.params["REarth"]=REkm/aukm
coprob.params["readTable"]=1

# add gas drag
gd = rebx.load_force("gas_drag")
rebx.add_force(gd)
gd.params["code_to_yr"]=1./twopi
gd.params["density_mks_to_code"]=1.496e11**3/1.989e30
gd.params["dist_to_m"]=1.496e11
gd.params["alt_ref_m"]=6378e3

for i in range(1,len(ps)): 
  b = BCOEFF # 2.2*10.**(np.random.uniform(-2,0,1))
  ps[i].params["bcoeff"]=b*1  # this is the area-to-mass ratio in m^2/kg * CD
  ps[i].params["collProb"]=0.
  ps[i].params["collArea"]=10/aum**2


times = np.linspace(tstart,tend,NTIME)

totcol=0.
removed=0
nancatch=0
OK=1
for iout, time in enumerate(times):
    print("Working on time {}".format(time))
    sim.integrate(time)

    ps=sim.particles

    if sim.N==1:
      print("All Particles Deorbited at time {}".format(time/twopi))
      break 
    

    i=1
    while True:
        if i >= sim.N: break
        if np.isnan(ps[i].params["collProb"]): # I HAVE NO IDEA WHY THIS IS HAPPENING, BUT IT REMAINS SMALL. I AM WORKING ON IT.
           nancatch+=1
           sim.remove(i)
           continue
           
        elif ps[i].a*aukm < REMOVALT+REkm: 
            totcol += ps[i].params["collProb"]
            sim.remove(i)
            removed+=1
            print("Removing 1 particle")
            continue
        i+=1
   

ps=sim.particles
for i in range(1,len(ps)):
    if not np.isnan(ps[i].params["collProb"]):
        print("PARTICLE {} alt {} ecc {} inc {} size {} prob {}".format(i,ps[i].a*aukm-REkm,ps[i].e,ps[i].inc*180/np.pi,ps[i].r,ps[i].params["collProb"]))
        totcol += ps[i].params["collProb"]
    else: nancatch+=1

try:
  print("Number of particles removed is {}".format(removed))
  print("Particle 1's end eccentricity = {} and inc = {}".format(ps[1].e,ps[1].inc*360/twopi))
  print("Here is the collProb for particle 1 = {:3.10e}".format(ps[1].params["collProb"]))
  print("Here is the collProb per year for particle 1 = {:3.10e}".format(ps[1].params["collProb"]*twopi/(times[-1])))
  print("Particle 1's end sma = {} in altitude".format(ps[1].a*aukm-REkm))
except: print("No debris left")
print("Sum of all collision probabilities = {} after time {}".format(totcol,times[-1]/twopi))
print("nancatch Number {}".format(nancatch))

arra=[]
erra=[]
Period=[]
for p in ps[1:-1]:
  arra.append(p.a)
  erra.append(p.e)
  Period.append(p.P)

arra=np.array(arra)*aukm
peri=arra*(1.-np.array(erra))-REkm
apo=arra*(1.+np.array(erra))-REkm
arra-=REkm

Period=np.array(Period)*365.25*24*60/(2*np.pi)

plt.figure()
bins,edges,patches=plt.hist(arra,bins=100,range=[0,2000])
plt.savefig("hist_a_end.png")

plt.figure()
plt.ylabel("Altitude [km]")
plt.xlabel("Period [min]")
plt.ylim([180,2400])
plt.scatter(Period,apo)
plt.scatter(Period,peri)
plt.savefig("gabbard_end.png")


