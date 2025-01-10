# -*- coding: utf-8 -*-



import SLEEPY as sl
import numpy as np
from time import time
from copy import copy
from scipy.linalg import expm

sl.Defaults['parallel']=False

Dt=1e-3

# System setup
ex=sl.ExpSys(v0H=400,Nucs=['13C','1H'],LF=False,vr=0,pwdavg=sl.PowderAvg('zcw376'))
ex.set_inter('dipole',i0=0,i1=1,delta=44000)
ex.set_inter('CSA',i=0,delta=100,euler=[0,np.pi/2,0])

L=ex.Liouvillian()
L.rf.add_field('1H',v1=50000)

rho=sl.Rho('13Cx','13Cp')

# Reference calculation
t0=time()
L.U(Dt=Dt)*rho

current=time()-t0
print(current)

ref_rho=rho[0]


# Faster function (hopefully?)


# def expmv(L,rho,Dt,N=150,s=100):
#     mu=np.trace(L)/len(L)
#     L-=mu*np.eye(L.shape[0])
#     out=copy(rho)
#     Lrho=copy(rho)    
#     for m in range(s):

#         k=1
#         while k<N:
#             Lrho=(Dt/k/s)*L@Lrho
#             if k==1:x0=Lrho
#             out+=Lrho
#             k+=1
#             if np.linalg.norm(Lrho)*10000<np.linalg.norm(x0):break #Termination condition?
#         print(k)
#         out*=np.exp(Dt*mu/s) #Why is this in the loop??
#         Lrho=out
#     # out*=np.exp(Dt*mu) #Instead of out here without s?
    
#     return out

from expmv import expmv
        


rho=sl.Rho('13Cx','13Cp',L=L)    
expmv(L[0].L(0)*Dt,rho[0])
t0=time()

for k,(L0,rho0) in enumerate(zip(L,rho)): # Loop over the powder average (don't try to optimize this)
    rho._rho[k]=expmv(L0.L(0)*Dt,rho0)
    # rho._rho[k]=expm(L0.L(0)*Dt)@rho0
    # break

new=time()-t0
print(new)

print(np.abs(rho[0]-ref_rho).max())
        

# from scipy.sparse.linalg import expm_multiply

# rho=sl.Rho('13Cx','13Cp',L=L)  
# t0=time()
# for k,(L0,rho0) in enumerate(zip(L,rho)): # Loop over the powder average (don't try to optimize this)
#     rho._rho[k]=expm_multiply(L0.L(0)*Dt,rho0,traceA=0)
#     # rho._rho[k]=expm(L0.L(0)*Dt)@rho0
#     # break

# new=time()-t0
# print(new)