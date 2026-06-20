#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 14:47:30 2026

@author: albertsmith
"""

import SLEEPY as sl
import numpy as np

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