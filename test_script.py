#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 14:28:49 2021

@author: albertsmith
"""

import pyRelaxSim as RS
import numpy as np

pi=np.pi
exp=RS.Hamiltonians.ExperSys(500,['1H','13C'],vr=10000)


exp.set_inter('dipole',0,1,delta=22000)
# exp.set_inter('dipole',1,2,delta=3000,euler=[0,pi*30/180,pi*15/180])
# exp.set_inter('dipole',0,2,delta=5000)
exp.set_inter('J',0,1,15)

pwdavg=RS.Powder.PowderAvg(PwdType='JCP59',q=3)

H=RS.Hamiltonians.Hamiltonian(exp,pwdavg=pwdavg)
