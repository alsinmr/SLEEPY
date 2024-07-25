# -*- coding: utf-8 -*-

import SLEEPY as sl
import numpy as np
from SLEEPY.LFrf import LFrf
import matplotlib.pyplot as plt

ex0=sl.ExpSys(v0H=600,Nucs=['13C','1H'],vr=10000,LF=True,pwdavg=sl.PowderAvg(),n_gamma=30)
ex0.set_inter('dipole',i0=0,i1=1,delta=44000)
ex1=ex0.copy()
ex1.set_inter('dipole',i0=0,i1=1,delta=44000,euler=[0,45*np.pi/180,0])

L=sl.Liouvillian(ex0,ex1)
L.kex=sl.Tools.twoSite_kex(5e-10)
L.add_relax('T2',i=1,T2=.005)

seq=L.Sequence().add_channel('1H',v1=100)

rho=sl.Rho(ex0.Op[0].z+ex0.Op[1].z,['1Hz','13Cz'])

# rho,seq=rho.ReducedSetup(seq)

lfrf=LFrf(seq)

U=lfrf.U()


rho.DetProp(U,n=100000)
rho.plot()


rho1=sl.Rho('1Hz',['1Hz','13Cz'])
seq=L.Sequence()

rho1.DetProp(seq,n=100000)
rho1.plot()

rho2=sl.Rho('13Cz',['1Hz','13Cz'])
rho2.DetProp(seq,n=100000)
rho2.plot()

#%% Test min_steps

fig,ax=plt.subplots(1,2)


for min_steps in [3,4,8]:
    lfrf=LFrf(seq,min_steps=min_steps)
    U=lfrf.U()
    rho=sl.Rho(ex0.Op[0].z+ex0.Op[1].z,['1Hz','13Cz'])

    rho.DetProp(U,n=100000)
    
    rho.plot(ax=ax[0],det_num=0)
    rho.plot(ax=ax[1],det_num=1)

#%% One-spin tests

ex=sl.ExpSys(v0H=600,Nucs='1H',LF=True)

seq=ex.Liouvillian().Sequence(Dt=1e-4).add_channel('1H',v1=100,voff=100)

fig,ax=plt.subplots(1,2)


for min_steps in [2,3,4,8,16,32]:
    lfrf=LFrf(seq,min_steps=min_steps)

    U=lfrf.U()

    rho=sl.Rho('1Hz',['1Hy','1Hz'])
    rho.DetProp(U,n=1001)
    rho.plot(ax=ax[0],det_num=0)
    rho.plot(ax=ax[1],det_num=1)