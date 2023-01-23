# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Tue Jan 17 10:37:37 2023


# @author: albertsmith
# """

import pyRelaxSim as RS
import numpy as np
import matplotlib.pyplot as plt
from time import time


expsys=RS.ExpSys(850,['15N','1H'],vr=60000,pwdavg=RS.PowderAvg(q=5))
expsys.set_inter('dipole',delta=22000,i0=0,i1=1)
expsys.set_inter('CS',i=0,ppm=0)

expsys1=expsys.copy()
expsys1.set_inter('dipole',delta=22000,i0=0,i1=1,euler=[0,90*np.pi/180,0])

H0=RS.Hamiltonian(expsys=expsys)
H1=RS.Hamiltonian(expsys=expsys1)

kex=np.array([[-5e6,5e6],[5e6,-5e6]])
L=RS.Liouvillian((H0,H1),kex=kex)
# L=RS.Liouvillian(H0)


#%% Full REDOR sequence

seq1=RS.Sequence(L)
v1=150000
shift=L.taur/6
t=[L.taur/2+shift-1/v1/2,L.taur/2+shift,L.taur-1/v1/2,L.taur]
seq1.add_channel('1H',t=t,v1=[v1,0,v1,0],phase=[0,0,np.pi/4,0])

t=[0,1/v1/2,L.taur/2-shift,L.taur/2+1/v1/2-shift]
seq2=RS.Sequence(L)
seq2.add_channel('1H',t=t,v1=[v1,0,v1,0],phase=[np.pi/4,0,0,0])

t=[L.taur-1/v1/2,L.taur+1/v1/2]
seq3=RS.Sequence(L)
seq3.add_channel('1H',t=t,v1=[v1,0])
seq3.add_channel('15N',t=t,v1=[v1,0])

ax=plt.figure().add_subplot(111)
z=np.linspace(-8,-2,4)
for tc in 10**z:
    t0=time()
    L.kex=np.array([[-1/tc,1/tc],[1/tc,-1/tc]])
    U1=seq1.U()
    U2=seq2.U()
    Upi=seq3.U()
    print(time()-t0)
    
    rho=RS.Rho(rho0='15Nx',detect='15Nx',L=L)
    U1a=U1**0
    U1b=U2**0
    for k in range(100):
        rho.reset()
        (U1b*Upi*U1a*rho)()
        U1a*=U1
        U1b*=U2
        
    rho.plot()
ax.legend([f'z={z0}' for z0 in z])

#%% R1p relaxation
expsys=RS.ExpSys(850,Nucs=['15N','1H'],vr=60000,pwdavg=RS.PowderAvg(q=2))
expsys.set_inter('dipole',delta=22000,i0=0,i1=1)
expsys.set_inter('CS',i=0,ppm=0)

expsys1=expsys.copy()
expsys1.set_inter('dipole',delta=22000,i0=0,i1=1,euler=[0,90*np.pi/180,0])

H0=RS.Hamiltonian(expsys=expsys)
H1=RS.Hamiltonian(expsys=expsys1)

kex=np.array([[-5e6,5e6],[5e6,-5e6]])
L=RS.Liouvillian((H0,H1),kex=kex)

ax=plt.figure().add_subplot(111)
seq=RS.Sequence(L)
seq.add_channel('15N',t=[0,seq.taur],v1=25000)
z=np.linspace(-9,-1,9)
rho=RS.Rho(rho0='15Nx',detect='15Nx',L=L)
for tc in 10**z:
    print(tc)
    t0=time()
    L.kex=np.array([[-1/tc,1/tc],[1/tc,-1/tc]])

    U=seq.U()
    print(time()-t0)
    U1=U**100
      
    rho.clear()
    rho.DetProp(U1,n=100)
    
    rho.plot()
ax.legend([f'z={z0}' for z0 in z])


#%% Spinning side bands?
expsys=RS.ExpSys(850,['13C'],vr=7000,pwdavg=RS.PowderAvg(q=5))
expsys.set_inter('CSA',delta=100,eta=0,i=0)

expsys1=expsys.copy()
expsys1.set_inter('CSA',delta=100,i=0,euler=[0,90*np.pi/180,0])

H0=RS.Hamiltonian(expsys)
H1=RS.Hamiltonian(expsys1)

kex=np.array([[-4e4,4e4],[4e4,-4e4]])*50

L=RS.Liouvillian((H0,H1),kex=kex)
# L=RS.Liouvillian(H0)

ax=plt.figure().add_subplot(111)
for kex in [1e6,4e4]:
    L.kex=np.array([[-kex,kex],[kex,-kex]])
    U=list()
    n=50
    for k in range(n):
        U.append(L.U(t0=L.taur/n*k,tf=L.taur/n*(k+1)))
        
    Ur=list()
    for k in range(n):
        Ur.append(U[k])
        for m in range(k+1,k+n+1):
            Ur[-1]=U[m%n]*Ur[-1]
        
        
        
    rho=RS.Rho(rho0='13Cx',detect='13Cp',L=L)
    for k in range(n):
        rho.reset()
        for q in range(k):U[q]*rho
        rho.DetProp(Ur[k],n=1000)
    
    from copy import copy
    
    f=1/(2*rho.t_axis[1])*np.linspace(-1,1,len(rho.t_axis))
    I=copy(rho.I)[0]
    I[0]*=0.5
    # I*=np.exp(-np.linspace(0,1,len(I))*10)    
    S=np.fft.fftshift(np.fft.fft(I))
    ax.plot(f,S.real)
    
#%% Cross polarization

expsys=RS.ExpSys(850,['15N','1H'],vr=60000,pwdavg=RS.PowderAvg(q=3))
expsys.set_inter('dipole',delta=22000,i0=0,i1=1)
expsys.set_inter('CS',i=0,ppm=-5)

expsys1=expsys.copy()
expsys1.set_inter('dipole',delta=22000,i0=0,i1=1,euler=[0,0*np.pi/180,0])
expsys1.set_inter('CS',i=1,ppm=5)

H0=RS.Hamiltonian(expsys)
H1=RS.Hamiltonian(expsys1)

kex=np.array([[-4e4,4e4],[4e4,-4e4]])*50

L=RS.Liouvillian((H0,H1),kex=kex)

axH=plt.figure().add_subplot(111)
axN=plt.figure().add_subplot(111)

z=np.linspace(-7,-2,6)
for tc in 10**z:
    L.kex=[[-1/tc,1/tc],[1/tc,-1/tc]]
    seq=RS.Sequence(L)
    seq.add_channel('1H',v1=80000,t=[0,seq.taur])
    seq.add_channel('15N',v1=20000,t=[0,seq.taur])
    
    U=seq.U()
    
    rho=RS.Rho(rho0='1Hx',detect=['1Hx','15Nx'],L=L)
    rho.DetProp(U,n=int(.002//rho.taur))
    
    rho.plot(det_num=0,ax=axH)
    rho.plot(det_num=1,ax=axN)
    
axH.legend([f'z={z0}' for z0 in z])
axN.legend([f'z={z0}' for z0 in z])

#%% CEST
#%% R1p relaxation
expsys=RS.ExpSys(850,['13C'],vr=60000,pwdavg=RS.PowderAvg(q=1))
expsys.set_inter('CSA',delta=0,i=0)
expsys.set_inter('CS',i=0,ppm=10)

expsys1=expsys.copy()
expsys.set_inter('CS',i=0,ppm=-10)

H0=RS.Hamiltonian(expsys=expsys)
H1=RS.Hamiltonian(expsys=expsys1)

kex=np.array([[-1e2,1e4],[1e2,-1e4]])
L=RS.Liouvillian((H0,H1),kex=kex)

L.add_relax(i=0,T1=5,T2=.1)

ax=plt.figure().add_subplot(111)
seq=RS.Sequence(L)
seq.add_channel('13C',t=[0,seq.taur],v1=1000,voff=5125)


t0=time()

U=seq.U()
print(time()-t0)
U1=U**100
  
rho=RS.Rho(rho0='13Cz',detect='13Cz',L=L)
rho.DetProp(U1,n=100)

rho.plot()
