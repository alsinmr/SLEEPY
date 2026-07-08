#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 14:47:30 2026

@author: albertsmith
"""

import SLEEPY as sl
import numpy as np
import matplotlib.pyplot as plt

vr=60000
ex=sl.ExpSys(500,Nucs=['1H','13C'],LF=True,vr=vr,pwdavg=2,n_gamma=30)
ex.set_inter('dipole',i0=0,i1=1,delta=10000)
L=ex.Liouvillian()
seq=L.Sequence(Dt=1/vr).add_channel('1H',v1=25000).add_channel('13C',v1=35000)

lfrf=sl.LFrf(seq)
U0=lfrf.U0(0)

rho=sl.Rho('13Cz+1Hz',['13Cz','1Hz']).DetProp(U0,n=50000)

ax=rho.plot()
ax.set_ylim(ax.get_ylim())
lbl=ax.get_xlabel()
sc=1e6 if '\mu' in lbl else (1e3 if 'ms' in lbl else 1)
ax.plot(1/seq.v1[0,0]*np.ones(2)*sc,ax.get_ylim(),color='grey',linestyle=':')
ax.plot(1/seq.v1[1,0]*np.ones(2)*sc,ax.get_ylim(),color='grey',linestyle='--')


Ustep=lfrf.Ustep(0)

rho=sl.Rho('13Cz+1Hz',['13Cz','1Hz']).DetProp(Ustep,n=1000)

ax=rho.plot()
ax.set_ylim(ax.get_ylim())
lbl=ax.get_xlabel()
sc=1e6 if '\mu' in lbl else (1e3 if 'ms' in lbl else 1)
ax.plot(1/seq.v1[0,0]*np.ones(2)*sc,ax.get_ylim(),color='grey',linestyle=':')
ax.plot(1/seq.v1[1,0]*np.ones(2)*sc,ax.get_ylim(),color='grey',linestyle='--')

U=lfrf.U()

rho=sl.Rho('13Cz+1Hz',['13Cz','1Hz']).DetProp(U,n=100)

ax=rho.plot()
ax.set_ylim(ax.get_ylim())
lbl=ax.get_xlabel()
sc=1e6 if '\mu' in lbl else (1e3 if 'ms' in lbl else 1)
ax.plot(1/seq.v1[0,0]*np.ones(2)*sc,ax.get_ylim(),color='grey',linestyle=':')
ax.plot(1/seq.v1[1,0]*np.ones(2)*sc,ax.get_ylim(),color='grey',linestyle='--')


rho=sl.Rho('1Hx',['1Hx','13Cx']).DetProp(U,n=100).plot()


ex=sl.ExpSys(500,Nucs=['1H','13C'],LF=False,vr=vr,pwdavg=2,n_gamma=30)
ex.set_inter('dipole',i0=0,i1=1,delta=10000)
L=ex.Liouvillian()
seq=L.Sequence(Dt=1/vr).add_channel('1H',v1=25000).add_channel('13C',v1=35000)

rho=sl.Rho('1Hx',['1Hx','13Cx']).DetProp(seq,n=100).plot()



#%% Tests to see if we can produce a pulse sequence in the lab frame
ex=sl.ExpSys(500,Nucs=['1H','13C'],LF=True,vr=vr,pwdavg=2,n_gamma=30)
ex.set_inter('dipole',i0=0,i1=1,delta=10000)
L=ex.Liouvillian()
t=[0,L.taur,L.taur]
seq=L.Sequence().add_channel('1H',t=t,v1=[25000,0]).add_channel('13C',t=t,v1=[35000,0])

lfrf=sl.LFrf(seq)
lfrf.irradiate_all_spins=True
U=lfrf.U()

rho=sl.Rho('1Hx',['1Hx','13Cx']).DetProp(U,n=100).plot()

ex=sl.ExpSys(500,Nucs=['1H','13C'],LF=False,vr=vr,pwdavg=2,n_gamma=30)
ex.set_inter('dipole',i0=0,i1=1,delta=10000)
L=ex.Liouvillian()
t=[0,L.taur,L.taur]
seq=L.Sequence().add_channel('1H',t=t,v1=[25000,0]).add_channel('13C',t=t,v1=[35000,0])

U=seq.U()

rho=sl.Rho('1Hx',['1Hx','13Cx']).DetProp(U,n=100).plot()

ex=sl.ExpSys(500,Nucs=['1H','13C'],LF=[True,False],vr=vr,pwdavg=2,n_gamma=30)
ex.set_inter('dipole',i0=0,i1=1,delta=10000)
L=ex.Liouvillian()
t=[0,L.taur,L.taur]
seq=L.Sequence().add_channel('1H',t=t,v1=[25000,0]).add_channel('13C',t=t,v1=[35000,0])

lfrf=sl.LFrf(seq)
U=lfrf.U()

rho=sl.Rho('1Hx',['1Hx','13Cx']).DetProp(U,n=100).plot()

#%% Bloch-siegert shift
v1=1e6
ex=sl.ExpSys(1,Nucs='1H',LF=True)
seq=ex.Liouvillian().Sequence(Dt=0.1/v1).add_channel('1H',v1=v1)

lfrf=sl.LFrf(seq,min_steps=16)
U=lfrf.U()

rho=sl.Rho('1Hz','1Hz').DetProp(U,n=1000)
rho.plot()


v1=1e6
ex=sl.ExpSys(1,Nucs='1H',LF=False)
seq=ex.Liouvillian().Sequence(Dt=0.1/v1).add_channel('1H',v1=v1)

U=seq.U()

rho=sl.Rho('1Hz','1Hz').DetProp(U,n=1000)
rho.plot()



v0N=sl.Tools.NucInfo('15N')

v0=5
v1=100e3

fig,ax=plt.subplots(2,3,sharex=True,sharey=True)
for n,a in zip([2,4,8,16,32,64],ax.flatten()):
    
    sl.Tools.NucInfo['15N','gyro']=sl.Tools.NucInfo['1H','gyro']/-10
    ex=sl.ExpSys(v0H,Nucs=['1H','15N'],LF=True)
    seq=ex.Liouvillian().Sequence(Dt=0.1/v1).add_channel('15N',v1=v1)
    
    lfrf=sl.LFrf(seq,min_steps=n)
    lfrf.irradiate_all_spins=True
    U=lfrf.U()
    
    rho=sl.Rho('1Hz+15Nz',['1Hz']).DetProp(U,n=1000)
    rho.plot(ax=a)
    a.text(0,.5,f'# of steps: {n}')


fig,ax=plt.subplots(2,3,sharex=True,sharey=True)
for n,a in zip([2,4,8,16,32,64],ax.flatten()):
    sl.Tools.NucInfo['15N','gyro']=v0N
    ex=sl.ExpSys(v0H,Nucs=['1H','15N'],LF=True)
    seq=ex.Liouvillian().Sequence(Dt=0.1/v1).add_channel('15N',v1=v1)
    
    lfrf=sl.LFrf(seq,min_steps=n)
    lfrf.irradiate_all_spins=True
    U=lfrf.U()
    
    rho=sl.Rho('1Hz+15Nz',['1Hz']).DetProp(U,n=1000)
    rho.plot(ax=a)
    a.text(0,.5,f'# of steps: {n}')
ax[0,0].set_ylim([0.,1])
fig.tight_layout()


