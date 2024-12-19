#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 09:50:06 2024

@author: albertsmith
"""

import SLEEPY as sl
import numpy as np
from SLEEPY.Para import ParallelManager


ex0=sl.ExpSys(v0H=600,Nucs='15N').set_inter('CSA',i=0,delta=113)
ex1=ex0.copy().set_inter('CSA',i=0,delta=113,euler=[0,15*np.pi/180,0])

L=sl.Liouvillian(ex0,ex1,kex=sl.Tools.twoSite_kex(tc=1e-5))

pm=ParallelManager(L,t0=0,Dt=L.taur)

U=pm()