#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:54:30 2024

@author: albertsmith
"""

import SLEEPY as sl
import numpy as np
import matplotlib.pyplot as plt

ex=sl.ExpSys(v0H=600,Nucs=['e-','1H'],LF=[False,True],vr=0,T_K=80,pwdavg=sl.PowderAvg(q=2)[5])
ex.set_inter('hyperfine',i0=0,i1=1,Axx=-1000000,Ayy=-1000000,Azz=2000000)

L=ex.Liouvillian()

L.clear_relax()
L.add_relax(Type='T2',i=0,T2=.890e-6)
L.add_relax(Type='T2',i=1,T2=5e-3)
L.add_relax(Type='T1',i=0,T1=1.4e-3)
L.add_relax(Type='T1',i=1,T1=20)
_=L.add_relax(Type='recovery')

seq=L.Sequence()
seq.add_channel(channel='e-',t=[0,1/5000],v1=3e6,voff=600e6)


U=seq.U()


rho=sl.Rho(rho0='ez',detect=['ez','1Hz'])

# Direct propagation
rho.clear()
rho()
for _ in range(20):
    (U*rho)()
    
    
# In eigenbasis (fails)
rho.clear()
rho.DetProp(U,n=20)

ax=rho.plot()
ax.figure.set_size_inches([10,8])