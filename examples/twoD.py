#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 16:27:14 2024

@author: albertsmith
"""

import SLEEPY as sl
import numpy as np

ex0=sl.ExpSys(v0H=600,Nucs='13C').set_inter('CS',i=0,ppm=5)
ex1=ex0.copy().set_inter('CS',i=0,ppm=-5)

L=sl.Liouvillian(ex0,ex1,kex=sl.Tools.twoSite_kex(tc=1,p1=.75))

L.add_relax('T2',i=0,T2=.01)
seq=L.Sequence(Dt=1/(2*10*150))


rho=sl.Rho('13Cx','13Cp').DetProp(seq,n=4096)
rho.plot(FT=True)

v1=100000
tpi2=1/v1/4
t=[0,tpi2,5,5+tpi2]
seq_trX=L.Sequence().add_channel('13C',t=t,v1=[v1,0,v1],phase=[-np.pi/2,0,np.pi/2])
seq_trY=L.Sequence().add_channel('13C',t=t,v1=[v1,0,v1],phase=[0,0,np.pi/2])

twoD=sl.Tools.TwoD_Builder(rho.clear(),seq,seq,seq_trX,seq_trY)

twoD(n_in=32,n_dir=32)
twoD.apod_pars['wdw']='qsine'
twoD.apod_pars['SI']=[1024,1024]


twoD.plot()