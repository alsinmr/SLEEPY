# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Tue Jan 17 10:37:37 2023


# @author: albertsmith
# """

import SLEEPY as RS
import SLEEPY as sl
import numpy as np
import matplotlib.pyplot as plt
from time import time


ex0=RS.ExpSys(850,Nucs=['15N','1H'],vr=60000,pwdavg=RS.PowderAvg(q=6),n_gamma=30)
ex0.set_inter('dipole',delta=22000,i0=0,i1=1)
ex0.set_inter('CS',i=0,ppm=0)

ex1=ex0.copy()
ex1.set_inter('dipole',delta=22000,i0=0,i1=1,euler=[0,25*np.pi/180,0])

H0=RS.Hamiltonian(expsys=ex0)
H1=RS.Hamiltonian(expsys=ex1)

kex=np.array([[-1e6,1e6],[1e6,-1e6]])
L=RS.Liouvillian((ex0,ex1),kex=kex)
# L=RS.Liouvillian(ex0)

#%% Full REDOR sequence

seq1=RS.Sequence(L)
v1=150000
shift=L.taur/6
t=[0,L.taur/2+shift-1/v1/2,L.taur/2+shift,L.taur-1/v1/2,L.taur]
seq1.add_channel('1H',t=t,v1=[0,v1,0,v1,0],phase=[0,0,0,np.pi/2,0])

t=[0,1/v1/2,L.taur/2-shift,L.taur/2+1/v1/2-shift,L.taur]
seq2=RS.Sequence(L)
seq2.add_channel('1H',t=t,v1=[v1,0,v1,0,0],phase=[np.pi/2,0,0,0,0])

t=[0,L.taur-1/v1/2,L.taur+1/v1/2,L.taur*2]
seq3=RS.Sequence(L)
seq3.add_channel('1H',t=t,v1=[0,v1,0,0])
seq3.add_channel('15N',t=t,v1=[0,v1,0,0])

# ax=plt.figure().add_subplot(111)
z=np.linspace(-8,-2,1)
for tc in 10**z:
    t0=time()
    # L.kex=np.array([[-1/tc,1/tc],[1/tc,-1/tc]])
    L.reset_prop_time()
    U1=seq1.U()
    Upi=seq3.U()
    U2=seq2.U()
    
    
    
    rho=RS.Rho(rho0='15Nx',detect='15Nx',L=L)
    U1a=L.Ueye()
    U1b=L.Ueye()
    for k in range(75):
        rho.reset()
        (U1b*Upi*U1a*rho)()
        print(time()-t0)
        U1a*=U1
        U1b*=U2       
    rho.plot(ax=ax)
    
ax.legend([f'z={z0}' for z0 in z])

#%% R1p relaxation
expsys=RS.ExpSys(850,Nucs=['15N','1H'],vr=60000,pwdavg=RS.PowderAvg('rep30'),n_gamma=30)
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
z=np.linspace(-9,-1,3)
rho=RS.Rho(rho0='15Nx',detect='15Nx',L=L)
t0=time()
for tc in 10**z:
    print(tc)
    
    L.kex=np.array([[-1/tc,1/tc],[1/tc,-1/tc]])

    U=seq.U()
    
    U1=U**100
      
    rho.clear()
    rho.DetProp(U1,n=100)
    print(time()-t0)
    rho.plot(ax=ax)
ax.legend([f'z={z0}' for z0 in z])


#%% Spinning side bands?
expsys=RS.ExpSys(850,Nucs=['13C'],vr=7000,pwdavg=RS.PowderAvg(q=5))
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

expsys=RS.ExpSys(850,Nucs=['15N','1H'],vr=60000,pwdavg=RS.PowderAvg(q=3))
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
expsys=RS.ExpSys(850,Nucs=['13C'],vr=60000,pwdavg=RS.PowderAvg(q=1))
expsys.set_inter('CSA',delta=0,i=0)
expsys.set_inter('CS',i=0,ppm=10)

expsys1=expsys.copy()
expsys.set_inter('CS',i=0,ppm=-10)

H0=RS.Hamiltonian(expsys=expsys)
H1=RS.Hamiltonian(expsys=expsys1)

kex=np.array([[-1e2,1e4],[1e2,-1e4]])
L=RS.Liouvillian((H0,H1),kex=kex)

L.add_relax(Type='T1',i=0,T1=1)
L.add_relax(Type='T2',i=0,T2=.1)

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

#%% Decoupling

import sys
sys.path.append('/Users/albertsmith/Documents/GitHub.nosync/')
import SLEEPY as sl
import numpy as np
import matplotlib.pyplot as plt

sl.Defaults['parallel']=False
sl.Defaults['cache']=True
ex=sl.ExpSys(850,Nucs=['15N','1H'],vr=5000,pwdavg=sl.PowderAvg(q=3),n_gamma=30)
ex.set_inter('dipole',delta=22000,i0=0,i1=1)
L=sl.Liouvillian(ex)

seq=L.Sequence()

v1=60000
dt=1/v1/4
seq.add_channel('1H',t=[0,dt,3*dt,6*dt],v1=v1,phase=[0,np.pi,0,0],voff=30000)
# seq.plot()

rho=sl.Rho('15Nx', '15Np')
rho.clear()
rho.DetProp(seq=seq,n=4096)
rho.plot(FT=True)

voff0=np.linspace(-30000,30000,7)
waltz=L.Sequence()
cw=L.Sequence()
Iwaltz=[]
Icw=[]
rhoHz=sl.Rho(rho0='1Hz',detect='1Hz')
for voff in voff0:
    waltz.add_channel('1H',t=[0,dt,3*dt,6*dt],v1=v1,phase=[0,np.pi,0,0],voff=voff)
    cw.add_channel('1H',t=[0,6*dt],v1=v1,voff=voff)
    rhoHz.clear()
    (waltz.U()*rhoHz)()
    Iwaltz.append(rhoHz.I[0][0].real)
    rhoHz.clear()
    (cw.U()*rhoHz)()
    Icw.append(rhoHz.I[0][0].real)
    
ax=plt.subplots()[1]
ax.plot(voff0,Iwaltz)
ax.plot(voff0,Icw)


#%% Quick Test
import SLEEPY as sl
sl.Defaults['cache']=True
sl.Defaults['parallel']=False
ex=sl.ExpSys(850,Nucs=['15N','1H'],vr=5000,pwdavg=sl.PowderAvg(q=1),n_gamma=48)
ex.set_inter('dipole',delta=0,i0=0,i1=1)
L=sl.Liouvillian(ex)
v1=60000
dt=1/v1/4
seq=L.Sequence()
seq.add_channel('1H',t=[0,dt,3*dt,6*dt],v1=v1,phase=[0,np.pi,0,0],voff=0)
# seq.add_channel('1H',t=[0,3*dt,6*dt],v1=plw1,phase=0,voff=0)


# L.reset_prop_time()
U=seq.U()
print(U.t0)
rhoHz=sl.Rho(rho0='1Hz',detect='1Hz')




# L.rf.add_field(1,v1=60000)
# U0=L.U(Dt=L.taur/16)
# U1=L.U(Dt=L.taur/16)

# U=U1*U0

(U*rhoHz)()
print(rhoHz.I)

