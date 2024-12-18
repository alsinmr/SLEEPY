# -*- coding: utf-8 -*-

import sys
sys.path.append('/Users/albertsmith/Documents/GitHub')
import SLEEPY as sl
import numpy as np
from time import time

# sl.Defaults['parallel']=True

ex0=sl.ExpSys(v0H=600,Nucs=['15N','1H'],vr=60000,pwdavg=sl.PowderAvg('bcr20'),n_gamma=30)
# After varying the powder average and n_gamma
# a beta-average and 30 gamma angles were determined to be sufficient
delta=sl.Tools.dipole_coupling(.102,'15N','1H')
phi=35*np.pi/180

ex0.set_inter('dipole',i0=0,i1=1,delta=delta)
ex,kex=sl.Tools.Setup3siteSym(ex0,tc=1e-9,phi=phi)

L=sl.Liouvillian(ex,kex=kex)

v1=150e3 #100 kHz pulse
tp=1/v1/2 #pi/2 pulse length

t=[0,L.taur/2-tp,L.taur/2,L.taur-tp,L.taur]
first=L.Sequence().add_channel('1H',t=t,v1=[0,v1,0,v1],phase=[0,0,0,np.pi/2,0])
t=[0,tp,L.taur/2,L.taur/2+tp,L.taur]
second=L.Sequence().add_channel('1H',t=t,v1=[v1,0,v1,0],phase=[np.pi/2,0,0,0,0])

centerH=L.Sequence().add_channel('1H',t=[0,L.taur/2-tp/2,L.taur/2+tp/2,L.taur],v1=[0,v1,0])

rho=sl.Rho('15Np','15Nx')

rho,f,s,c,Ueye=rho.ReducedSetup(first,second,centerH,L.Ueye())
# rho,f,s,c,Ueye=rho,first,second,centerH,L.Ueye()

Ufirst=f.U()
Usecond=s.U()
Ucenter=c.U()

U1=Ueye
U2=Ueye

t0=time()
for k in range(32):
    rho.reset()
    (U2*Ucenter*U1*rho)()
    U1=Ufirst*U1
    U2=Usecond*U2
print(time()-t0)