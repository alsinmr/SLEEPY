#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 10:37:37 2023


@author: albertsmith
"""

import pyRelaxSim as RS
import numpy as np


expsys=RS.ExpSys(600,['13C','1H'],vr=100000)
expsys.set_inter('dipole',delta=22000,i0=0,i1=1)
expsys.set_inter('CSA',delta=5000,i=0)
expsys.set_inter('CS',i=1,ppm=0)

H0=RS.Hamiltonian(expsys=expsys)


expsys1=expsys.copy()
expsys1.set_inter('dipole',delta=22000,i0=0,i1=1,euler=[0,np.pi/4,0])

H1=RS.Hamiltonian(expsys=expsys1)

kex=np.array([[-1e10,1e10],[1e10,-1e10]])
L=RS.Liouvillian((H0,H1),kex=kex)

L=RS.Liouvillian((H0))


seq=RS.Sequence(L)

t=[seq.taur/2-2.5e-6,seq.taur/2+2.5e-6]
seq.add_channel('1H',t=t,v1=[100000,0])
t=[seq.taur/2-5e-6,seq.taur/2+5e-6]
seq.add_channel('13C',t=t,v1=[50000,0])

# seq.plot()

U=seq.U()

rho=RS.Rho(rho0='13Cx',detect='13Cx',L=L)

rho.DetProp(U,n=101)

# for _ in range(100):
#     rho()
#     U*rho
    

rho.plot()
