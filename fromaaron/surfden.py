import numpy as np
import matplotlib.pylab as plt


dalt = 1.
dtheta=1*np.pi/180.
fname="smooth2d.dat"

RE=6378.135
NTHETA=180

fh = open(fname,"r")

alt=[]
theta=[]
den=[]
for line in fh:
   vals=line.rstrip().split()
   alt.append(float(vals[1]))
   theta.append(float(vals[3]))
   den.append(float(vals[5]))

alt=np.array(alt)
theta=np.pi*0.5-np.array(theta)*np.pi/180.
den=np.array(den)

ntot=0.
for i in range(len(alt)):
   ntot += 2*np.pi*(np.cos(theta[i]-dtheta*0.5)-np.cos(theta[i]+dtheta*0.5))*(RE+alt[i])**2*dalt*den[i]

print("Total sats {}".format(ntot))

surf=np.zeros(NTHETA)
Theta2d=np.zeros(NTHETA)
num=np.zeros(NTHETA)
cum=np.zeros(int(NTHETA/2))
NALT=alt.size
numalt=np.zeros(NALT)

for j in range(NTHETA):Theta2d[j]=(np.pi/2-(j+0.5)*dtheta)*180/np.pi


for i in range(len(alt)):
   itheta = int(theta[i]/dtheta)
   surf[itheta]+= den[i]*(RE+alt[i])**2*dalt*(np.pi/180)**2
   num[itheta]+=2*np.pi*(np.cos(theta[i]-dtheta*0.5)-np.cos(theta[i]+dtheta*0.5))*(RE+alt[i])**2*dalt*den[i]
   numalt[i]+=2*np.pi*(np.cos(theta[i]-dtheta*0.5)-np.cos(theta[i]+dtheta*0.5))*(RE+alt[i])**2*dalt*den[i]

cum[0]=num[0]*1
for i in range(1,len(cum)):
    cum[i]=cum[i-1]+num[i]

norm=cum[-1]
print(cum[-1])

cum=cum/norm


plt.figure()
plt.plot(Theta2d,surf)
plt.xlim(-90,90)

plt.figure()
plt.plot(alt,numalt)

plt.figure()
plt.title("Satellites within 1 deg Latitudinal Bands")
plt.xlabel("Latitude")
plt.ylabel("Satellite Counts")
plt.plot(Theta2d,num)
plt.xlim(-90,90)

plt.savefig("SatLatCounts.png")

plt.figure()
plt.plot(Theta2d[0:len(cum)],cum)
plt.xlim([90,0])


plt.show()






