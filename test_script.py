#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 10:37:37 2023

@author: albertsmith
"""

import pyRelaxSim as RS
import numpy as np

expsys=RS.ExpSys(600,['13C','1H'],vr=5000)
expsys.set_inter('dipole',delta=22000,i1=0,i2=1)
expsys.set_inter('CSA',delta=5000,i=0)
expsys.set_inter('CS',i=1,ppm=10)

rf=RS.RF({'13C':(20000,0,0)},expsys=expsys)

H0=RS.Hamiltonian(expsys=expsys,rf=rf)

expsys.remove_inter(Type='dipole')

expsys.set_inter('dipole',delta=22000,i1=0,i2=1,euler=[0,0,0])

H1=RS.Hamiltonian(expsys=expsys,rf=rf)

kex=np.array([[-1e6,1e6],[1e6,-1e6]])
L=RS.Liouvillian((H0,H1),kex=kex)