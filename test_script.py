# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Tue Jan 17 10:37:37 2023


# @author: albertsmith
# """

import pyRelaxSim as RS
import numpy as np
import matplotlib.pyplot as plt


expsys=RS.ExpSys(850,['15N','1H'],vr=60000,pwdavg=RS.PowderAvg(q=5))
expsys.set_inter('dipole',delta=22000,i0=0,i1=1)
expsys.set_inter('CS',i=0,ppm=0)

expsys1=expsys.copy()
expsys1.set_inter('dipole',delta=22000,i0=0,i1=1,euler=[0,90*np.pi/180,0])

H0=RS.Hamiltonian(expsys=expsys)
H1=RS.Hamiltonian(expsys=expsys1)

kex=np.array([[-5e6,5e6],[5e6,-5e6]])
L=RS.Liouvillian((H0,H1),kex=kex)
# L=RS.Liouvillian(H0)


#%% Full REDOR sequence

seq1=RS.Sequence(L)
v1=150000
shift=L.taur/6
t=[L.taur/2+shift-1/v1/2,L.taur/2+shift,L.taur-1/v1/2,L.taur]
seq1.add_channel('1H',t=t,v1=[v1,0,v1,0],phase=[0,0,np.pi/4,0])

t=[0,1/v1/2,L.taur/2-shift,L.taur/2+1/v1/2-shift]
seq2=RS.Sequence(L)
seq2.add_channel('1H',t=t,v1=[v1,0,v1,0],phase=[np.pi/4,0,0,0])

t=[L.taur-1/v1/2,L.taur+1/v1/2]
seq3=RS.Sequence(L)
seq3.add_channel('1H',t=t,v1=[v1,0])
seq3.add_channel('15N',t=t,v1=[v1,0])

ax=plt.figure().add_subplot(111)
z=np.linspace(-8,-2,4)
for tc in 10**z:
    L.kex=np.array([[-1/tc,1/tc],[1/tc,-1/tc]])
    U1=seq1.U()
    U2=seq2.U()
    Upi=seq3.U()
    
    rho=RS.Rho(rho0='15Nx',detect='15Nx',L=L)
    U1a=U1**0
    U1b=U2**0
    for k in range(100):
        rho.reset()
        (U1b*Upi*U1a*rho)()
        U1a*=U1
        U1b*=U2
        
    rho.plot(ax=ax)
ax.legend([f'z={z0}' for z0 in z])

#%% R1p relaxation
ax=plt.figure().add_subplot(111)
seq=RS.Sequence(L)
seq.add_channel('15N',t=[0,seq.taur],v1=25)
for tc in np.logspace(-10,-1,9):
    L.kex=np.array([[-1/tc,1/tc],[1/tc,-1/tc]])

    U=seq.U()
    U1=U**100
      
    rho=RS.Rho(rho0='15Nx',detect='15Nx',L=L)
    rho.DetProp(U1,n=1000)
    
    rho.plot(ax=ax)
ax.legend