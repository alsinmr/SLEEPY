#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 17:45:32 2023

@author: albertsmith
"""

import SLEEPY as RS
from SLEEPY.Tools import Ham2Super
import numpy as np

ex0=RS.ExpSys(v0H=600,Nucs=['13C','13C'],LF=True)
ex1=ex0.copy()
ex0.set_inter('_larmor',i=0)
ex0.set_inter('_larmor',i=1)
ex0.set_inter('J',i0=0,i1=1,J=100)

ex1.set_inter('dipole',i0=0,i1=1,delta=10000)

H0=ex0.Hamiltonian()[0].Hn(0)
H1=ex1.Hamiltonian()[0].Hn(2)

d,v=np.linalg.eig(H0)

#We can construct H0 out of these, but it's not a complete basis
V=[v[:,k:k+1]@v[:,k:k+1].T*d[k] for k in range(4)]

#So this isn't generally correct
H1V=[np.trace(V0@H1)/np.trace(V0.conj().T@V0)*V0 for V0 in V]  #

commute=lambda a,b:a@b-b@a

#Verify that the H1V commute
for k in range(4):
    print(commute(H1V[k],H0))
    
print(np.sum(H1V,axis=0)-H1)  