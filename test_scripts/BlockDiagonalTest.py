import sys
sys.path.append('/Users/albertsmith/Documents/GitHub/')
import SLEEPY as sl
import numpy as np
from time import time
sl.Defaults['parallel']=True

ex0=sl.ExpSys(v0H=600,Nucs=['13C','1H'],vr=60000)
ex0.set_inter('dipole',i0=0,i1=1,delta=44000)
ex1=ex0.copy()
ex1.set_inter('dipole',i0=0,i1=1,delta=44000,euler=[0,15*np.pi/180,0])

L=sl.Liouvillian((ex0,ex1))
L.kex=sl.Tools.twoSite_kex(tc=1e-5)

seq=L.Sequence().add_channel('13C',v1=25000)

rho=sl.Rho(rho0='13Cx',detect='13Cx')
rho.Reduce=False

t0=time()
rho.DetProp(seq,n=1000)
print(time()-t0)


L=sl.Liouvillian((ex0,ex1))
L.kex=sl.Tools.twoSite_kex(tc=1e-5)

seq=L.Sequence().add_channel('13C',v1=25000)

rho=sl.Rho(rho0='13Cx',detect='13Cx')
rho.Reduce=True

t0=time()
rho.DetProp(seq,n=1000)
print(time()-t0)



#%% Try with multiple sequences
L=sl.Liouvillian((ex0,ex1))
L.kex=sl.Tools.twoSite_kex(tc=1e-5)

pi2=L.Sequence().add_channel('13C',v1=62500,t=4e-6,phase=np.pi/2)
lock=L.Sequence().add_channel('13C',v1=25000,voff=10000)

rho=sl.Rho(rho0='13Cz',detect='13Cm')

rho,pi2,lock=rho.ReducedSetup(pi2,lock)

(pi2*rho).DetProp(lock,n=1000)