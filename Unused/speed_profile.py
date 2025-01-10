#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 10:55:24 2024

@author: albertsmith
"""

import sys
sys.path.append('/Users/albertsmith/Documents/GitHub')
import SLEEPY as RS
import numpy as np
import matplotlib.pyplot as plt
from time import time


ex0=RS.ExpSys(850,Nucs=['15N','1H'],vr=60000,pwdavg=RS.PowderAvg('rep10'),n_gamma=50)
ex0.set_inter('dipole',delta=22000,i0=0,i1=1)
ex0.set_inter('CS',i=0,ppm=0)

ex1=ex0.copy()
ex1.set_inter('dipole',delta=22000,i0=0,i1=1,euler=[0,25*np.pi/180,0])

H0=RS.Hamiltonian(expsys=ex0)
H1=RS.Hamiltonian(expsys=ex1)

kex=np.array([[-1e6,1e6],[1e6,-1e6]])
L=RS.Liouvillian((ex0,ex1),kex=kex)
# L=RS.Liouvillian(H0)


#%% Full REDOR sequence

seq1=RS.Sequence(L)
v1=150000
shift=L.taur/6
t=[0,L.taur/2+shift-1/v1/2,L.taur/2+shift,L.taur-1/v1/2,L.taur]
seq1.add_channel('1H',t=t,v1=[0,v1,0,v1,0],phase=[0,0,0,np.pi/4,0])

t=[0,1/v1/2,L.taur/2-shift,L.taur/2+1/v1/2-shift,L.taur]
seq2=RS.Sequence(L)
seq2.add_channel('1H',t=t,v1=[v1,0,v1,0,0],phase=[np.pi/4,0,0,0,0])

t=[0,L.taur-1/v1/2,L.taur+1/v1/2,L.taur*2]
seq3=RS.Sequence(L)
seq3.add_channel('1H',t=t,v1=[0,v1,0,0])
seq3.add_channel('15N',t=t,v1=[0,v1,0,0])

ax=plt.figure().add_subplot(111)
z=np.linspace(-8,-2,1)
for tc in 10**z:
    t0=time()
    L.kex=np.array([[-1/tc,1/tc],[1/tc,-1/tc]])
    L.reset_prop_time()
    U1=seq1.U()
    Upi=seq3.U()
    U2=seq2.U()
    
    
    
    rho=RS.Rho(rho0='15Nx',detect='15Nx',L=L)
    U1a=L.Ueye()
    U1b=L.Ueye()
    for k in range(75):
        rho.reset()
        (U1b*Upi*U1a*rho)()
        print(time()-t0)
        U1a*=U1
        U1b*=U2       
    rho.plot(ax=ax)
    
ax.legend([f'z={z0}' for z0 in z])