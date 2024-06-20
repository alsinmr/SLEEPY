import sys
sys.path.append('/Users/albertsmith/Documents/GitHub/')
import SLEEPY as sl
import numpy as np
from time import time

ex0=sl.ExpSys(v0H=600,Nucs=['13C','1H'],vr=60000)
ex0.set_inter('dipole',i0=0,i1=1,delta=44000)
ex1=ex0.copy()
ex1.set_inter('dipole',i0=0,i1=1,delta=44000,euler=[0,15*np.pi/180,0])

L=sl.Liouvillian((ex0,ex1))
L.kex=sl.Tools.twoSite_kex(tc=1e-5)

seq=L.Sequence().add_channel('13C',v1=25000)

rho=sl.Rho(rho0='13Cx',detect='13Cx')
rho.BlockDiagonal=True

t0=time()
rho.DetProp(seq,n=1000)
print(time()-t0)


L=sl.Liouvillian((ex0,ex1))
L.kex=sl.Tools.twoSite_kex(tc=1e-5)

seq=L.Sequence().add_channel('13C',v1=25000)

rho=sl.Rho(rho0='13Cx',detect='13Cx')
rho.BlockDiagonal=False

t0=time()
rho.DetProp(seq,n=1000)
print(time()-t0)