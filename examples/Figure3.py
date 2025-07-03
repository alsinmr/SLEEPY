#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 17:20:53 2025

@author: albertsmith

The following code runs a python script to perform the 9 simulations found in
Figure 3 of the paper "SLEEPY: Simple simulation of relaxation and dynamics 
in nuclearmagnetic resonanc"

(a total of XX seconds on my laptop).

Before running, SLEEPY needs to be installed via pip

pip install sleepy-nmr

Run the script from a terminal by typing
python3 Figure3.py

Results will be stored in the Figure3Figs folder 
(created relative to the run location)
"""


import SLEEPY as sl
import numpy as np
import os
from time import time
import matplotlib.pyplot as plt
plt.style.use(['default'])
plt.rcParams['font.size']=14

t0=time()
directory='Figure3Figs'
if not(os.path.exists(directory)):
    os.mkdir(directory)

    
t00=time()
    
#%% Part A: EXSY
t0=time()
from matplotlib.colors import LinearSegmentedColormap

ex0=sl.ExpSys(Nucs='13C',v0H=600)    #We need a description of the experiment for both states (ex0, ex1)
ex1=ex0.copy()
ex0.set_inter(Type='CS',i=0,ppm=-5)   #Add the chemical shifts
_=ex1.set_inter(Type='CS',i=0,ppm=5)

L=sl.Liouvillian(ex0,ex1)           #Builds the two different Hamiltonians and exports them to Liouville space

tc=.001     #Correlation time (1 s)
p1=0.66  #Population of state 1
p2=1-p1  #Population of state 2

L.kex=sl.Tools.twoSite_kex(tc=tc,p1=p1)

_=L.add_relax(Type='T2',i=0,T2=.01)

rho=sl.Rho(rho0='S0x',detect='S0p')
# L.Udelta('13C',np.pi/2,np.pi/2)*rho

Dt=1/(2*10*150)  #Delay for a spectrum about twice as broad as the chemical shift difference
seq=L.Sequence(Dt=Dt)  #Sequence for indirect and direct dimensions
seq_trX=L.Sequence()  #X-component of transfer
seq_trY=L.Sequence()  #Y-component of transfer

v1=50000     #pi/2 pulse field strength
tpi2=1/v1/4  #pi/2 pulse length
dly=5
t=[0,tpi2,dly,dly+tpi2] #pi/2 pulse, 1 second delay, pi/2 pulse
seq_trX.add_channel('13C',t=t,v1=[v1,0,v1],phase=[-np.pi/2,0,np.pi/2]) #Convert x to z, delay, convert z to x
seq_trY.add_channel('13C',t=t,v1=[v1,0,v1],phase=[0,0,np.pi/2]) #Convert y to z, delay, convert z to x

twoD=sl.Tools.TwoD_Builder(rho,seq_dir=seq,seq_in=seq,seq_trX=seq_trX,seq_trY=seq_trY)
twoD(32,32)

ax=twoD.plot()
ax.set_xlabel(r'$\delta$($^{13}$C) / kHz')
_=ax.set_ylabel(r'$\delta$($^{13}$C) / kHz')
ax.set_ylabel('')
ax.figure.set_size_inches([7,7])
ax.figure.tight_layout()

ax.figure.savefig(os.path.join(directory,'exsy.png'),transparent=True)

print(f'EXSY time: {time()-t0:.0f} s')

#%% Part B: CEST
t0=time()

sl.Defaults['verbose']=False
ex0=sl.ExpSys(v0H=600,Nucs='13C',T_K=298) #We need a description of the experiment for both states (ex0, ex1)
ex1=ex0.copy()
ex0.set_inter(Type='CS',i=0,ppm=-7)
_=ex1.set_inter(Type='CS',i=0,ppm=7)

L=sl.Liouvillian((ex0,ex1))  #Builds the two different Hamiltonians and exports them to Liouville space

tc=1e-3     #Correlation time
p1=0.95  #Population of state 1

L.kex=sl.Tools.twoSite_kex(tc=tc,p1=p1)    #Add exchange to the Liouvillian

L.add_relax(Type='T1',i=0,T1=1.5)   #Add T1 relaxation to the system
L.add_relax(Type='T2',i=0,T2=.05)             #Add T2 relaxation to the system
_=L.add_relax(Type='recovery') #This brings the spins back into thermal equilibrium

rho=sl.Rho(rho0='13Cz',detect='13Cp')  #Initial density matrix

# Make a sequence for saturation
seq=L.Sequence()    #Saturation and pi/2 pulse
t=[0,0.5,0.5+2.5e-6] #Preparation sequence (500 ms saturation, 100 kHz pi-pulse)

# Make a sequence for detection
Dt=1/(4*10*150)  #Broad enough to capture 10 ppm
evol=L.Sequence(Dt=Dt) #Evolution sequence

voff0=np.linspace(-20,20,500)*ex0.v0[0]/1e6     #5 ppm*150 MHz / 1e6 =750 Hz
spec=list()
for voff in voff0:
    seq.add_channel('13C',t=t,v1=[25,100e3],
                    voff=[voff,0],phase=[0,np.pi/2])
    rho.clear()
    (seq*rho).DetProp(evol,n=1024)
    spec.append(rho.FT[0].real)
    
spec=np.array(spec)   #Convert the list of spectra to a numpy array
I=spec[:,400:620].sum(1)  #Integrate over the main peak

ax=plt.subplots()[1]
ax.plot(voff0*1e6/ex0.v0[0],I,color='purple')
ax.set_xlabel(r'$\nu_{off}$ / ppm')
ax.set_ylabel('I / a.u.')
ax.invert_xaxis()

ax.figure.tight_layout()

ax.figure.savefig(os.path.join(directory,'CEST.png'),transparent=True)

ax=rho.plot(FT=True,color='darkcyan',axis='ppm')
fig=ax.figure
ax.set_yticklabels([])
ax.set_ylabel('')
fig.set_size_inches([3,2])
fig.savefig(os.path.join(directory,'CEST_spec.png'),transparent=True)

print(f'CEST time: {time()-t0:.0f} s')

#%% Part C: Solution-state T1 and NOE
# By default, we get a powder average when including anisotropic terms.
# so we set it explicitly to a 1-element powder average
t0=time()
ex0=sl.ExpSys(v0H=400,Nucs=['15N','1H'],LF=True)
delta=sl.Tools.dipole_coupling(.102,'1H','15N')
ex0.set_inter('dipole',i0=0,i1=1,delta=delta)

# Set up 4-site motion
L=sl.Tools.SetupTumbling(ex0,tc=1e-9,q=2) #q=1 gives just the tetrahedral orientations

seq=L.Sequence(Dt=.1)

L.add_relax('DynamicThermal')
rho=sl.Rho(rho0='zero',detect=['15Nz','1Hz'])
rho.DetProp(seq,n=100)
ax=rho.plot(axis='s')
ax.set_xlim(ax.get_xlim())
ax.plot(ax.get_xlim(),ex0.Peq[0]*np.ones(2),linestyle=':',color='black')
_=ax.plot(ax.get_xlim(),ex0.Peq[1]*np.ones(2),linestyle='--',color='grey')

ax.figure.savefig(os.path.join(directory,'T1_NOE.png'),transparent=True)

print(f'T1/NOE time: {time()-t0:.0f} s')

#%% Part D: REDOR
t0=time()
ex0=sl.ExpSys(v0H=600,Nucs=['15N','1H'],vr=60000,pwdavg=sl.PowderAvg('bcr20'),n_gamma=30)
# After varying the powder average and n_gamma
# a beta-average and 30 gamma angles were determined to be sufficient
delta=sl.Tools.dipole_coupling(.102,'15N','1H')
phi=35*np.pi/180

ex0.set_inter('dipole',i0=0,i1=1,delta=delta)
L=sl.Tools.Setup3siteSym(ex0,tc=1e-9,phi=phi)

v1=120e3 #100 kHz pulse
tp=1/v1/2 #pi/2 pulse length

t=[0,L.taur/2-tp,L.taur/2,L.taur-tp,L.taur]
first=L.Sequence().add_channel('1H',t=t,v1=[0,v1,0,v1],phase=[0,0,0,np.pi/2,0])
t=[0,tp,L.taur/2,L.taur/2+tp,L.taur]
second=L.Sequence().add_channel('1H',t=t,v1=[v1,0,v1,0],phase=[np.pi/2,0,0,0,0])

centerH=L.Sequence().add_channel('1H',t=[0,L.taur/2-tp/2,L.taur/2+tp/2,L.taur],v1=[0,v1,0])

rho=sl.Rho('15Np','15Nx')

rho,f,s,c,Ueye=rho.ReducedSetup(first,second,centerH,L.Ueye())

Ufirst=f.U()
Usecond=s.U()
Ucenter=c.U()

rho_list=[]
legend=[]
t0=time()
for tc in np.logspace(-6,-3,8):
    L.kex=sl.Tools.nSite_sym(n=3,tc=tc)

    rho_list.append(rho.copy_reduced())

    Ufirst=f.U()
    Usecond=s.U()
    Ucenter=c.U()

    U1=Ueye
    U2=Ueye

    for k in range(24):
        rho_list[-1].reset()
        (U2*Ucenter*U1*rho_list[-1])()
        
        U1=Ufirst*U1
        U2=Usecond*U2

    legend.append(fr'$\log_{{10}}(\tau_c)$ = {np.log10(tc):.1f}')
    print(f'REDOR: log10(tc /s) = {np.log10(tc):.1f}, {time()-t0:.0f} seconds elapsed')

fig,ax=plt.subplots(2,4,figsize=[9,4.5])
ax=ax.flatten()
for a,l,r in zip(ax,legend,rho_list):
    r.plot(ax=a)
    a.set_title(l)
    if not(a.is_first_col()):
        a.set_ylabel('')
        a.set_yticklabels([])
    if not(a.is_last_row()):
        a.set_xlabel('')
        a.set_xticklabels([])
fig.tight_layout()

fig.savefig(os.path.join(directory,'REDOR.png'),transparent=True)


#%% Part E: R1p (reorientation and chemical exchange)
t0=time()
delta=sl.Tools.dipole_coupling(.102,'1H','15N')
ex0=sl.ExpSys(v0H=600,Nucs=['15N','1H'],vr=60000,LF=False)
ex0.set_inter('dipole',i0=0,i1=1,delta=delta)
# The 15N CSA is oriented about 23 degrees away from the H–N bond
ex0.set_inter('CSA',i=0,delta=113,euler_d=[0,23,0])
phi=25 #25 degree hop
ex1=ex0.copy()
ex1.set_inter('dipole',i0=0,i1=1,delta=delta,euler_d=[0,phi,0])
# We want the same hop amplitude on both interactions
_=ex1.set_inter('CSA',i=0,delta=113,euler_d=[[0,23,0],[0,phi,0]])

ex0.set_inter('CS',i=0,ppm=-5)
ex1.set_inter('CS',i=0,ppm=5)

L=sl.Liouvillian(ex0,ex1,kex=sl.Tools.twoSite_kex(tc=1e-4))
seq=L.Sequence()
rho=sl.Rho('15Nx','15Nx')

R1p=[]
v10=np.linspace(1,51,51)*1e3
for v1 in v10:
    seq.add_channel('15N',v1=v1)
    R1p.append(rho.extract_decay_rates(seq,mode='pwdavg'))
    
ax=plt.subplots(figsize=[5,4])[1]
ax.plot(v10/1e3,R1p)
ax.set_xlabel(r'$\nu_1$ / kHz')
_=ax.set_ylabel(r'$R_{1\rho}$ / s$^{-1}$')

ax.figure.savefig(os.path.join(directory,'R1p.png'),transparent=True)
print(f'T1p time: {time()-t0:.0f} s')

#%% Part F: solid-state T1 histograme
t0=time()

delta=sl.Tools.dipole_coupling(.102,'15N','1H')
hop=40*np.pi/180
ex0=sl.ExpSys(v0H=600,Nucs=['15N','1H'],vr=0,LF=True,pwdavg=7)
ex0.set_inter('CSA',i=0,delta=113,euler=[0,23*np.pi/180,0])
ex0.set_inter('dipole',i0=0,i1=1,delta=delta)
ex1=ex0.copy()
ex1.set_inter('CSA',i=0,delta=113,euler=[[0,23*np.pi/180,0],[0,hop,0]])
ex1.set_inter('dipole',i0=0,i1=1,delta=delta,euler=[0,hop,0])

L=sl.Liouvillian(ex0,ex1)
L.kex=sl.Tools.twoSite_kex(1e-9)
_=L.add_relax('DynamicThermal')

seq=L.Sequence(Dt=1e-2)
U=seq.U()
rho=sl.Rho('1Hz','15Nz')

_=rho.DetProp(U,n=10000)

ax=rho.plot(axis='s')
_=ax.plot(ax.get_xlim(),np.ones(2)*ex0.Peq[0],color='black',linestyle=':')

ax.figure.savefig(os.path.join(directory,'T1static.png'),transparent=True)

plt.rcParams['font.size']=14

def histogram(R1,A,bins=None,ax=None):
    if bins is None:
        bins=np.linspace(np.log(R1.min()),np.log(R1.max()),25)
        
    db=bins[1]-bins[0]
    
    i=np.digitize(np.log(R1),bins)-1
    h=np.zeros(len(bins))
    for k in range(i.max()+1):
        h[k]=A[k==i].sum()
        
    if ax is None:
        ax=plt.subplots()[1]
        alpha=1
        color=plt.get_cmap('tab10')(2)
    else:
        alpha=.5    
        color='red'
    ax.bar(bins,h,width=(bins[2]-bins[1])*.9,alpha=alpha,color=color)
    ax.set_xticks(bins[np.arange(5)*5])
    ax.set_xticklabels([f'{10**bins[k:k+2].mean()*1e3:6.2f}' for k in range(0,25,5)])
    ax.set_xlabel(r'$R_1$ / s$^{-1}$')
    ax.set_ylabel('Intensity')
    
    return bins,ax

rho.reset()

# Calculate a weighted histogram
R1_static,A_static=rho.extract_decay_rates(U,mode='wt_rates')
bins,ax=histogram(R1_static,A_static)

R1_avg_static=(R1_static*A_static).sum()/A_static.sum()

ax.text(bins[0]-.2,1.e-7,fr'$\langle R_1\rangle_{{static}}$: {R1_avg_static:.2f} $s^{{-1}}$')
ax.figure.tight_layout()
ax.figure.savefig(os.path.join(directory,'T1hist.png'),transparent=True)

print(f'T1 histogram time: {time()-t0:.0f} s')

#%% Part G: Overhauser effect
t0=time()

ex0=sl.ExpSys(v0H=212,Nucs=['1H','e'],LF=True,vr=5000,n_gamma=30,pwdavg=sl.PowderAvg(q=2)[10],T_K=80)
ex0.set_inter('g',i=1,gxx=2.0027,gyy=2.0031,gzz=2.0034)
Adip=[-1e6,-1e6,2e6]
Aiso0=.75e6
ex0.set_inter('hyperfine',i0=0,i1=1,Axx=Adip[0]+Aiso0,Ayy=Adip[1]+Aiso0,Azz=Adip[2]+Aiso0)
ex1=ex0.copy()
Aiso1=.25e6
ex1.set_inter('hyperfine',i0=0,i1=1,Axx=Adip[0]+Aiso1,Ayy=Adip[1]+Aiso1,Azz=Adip[2]+Aiso1,euler=[0,np.pi/8,0])

L=sl.Liouvillian(ex0,ex1)
L.kex=sl.Tools.twoSite_kex(tc=1e-12)

ex0.pwdavg=sl.PowderAvg(2)[10]
L.add_relax('DynamicThermal')

gavg=(2.0027+2.0031+2.0034)/3
ge=sl.Constants['ge']

# Note that we have to be careful to get the electron on-resonance
seq=L.Sequence().add_channel('e',v1=5e6,voff=ex0.v0[1]*(gavg-ge)/ge)  #5 MHz irradiating field
U=sl.LFrf(seq).U()

rho=sl.Rho('Thermal',['ez','1Hz'])
# This extracts the final polarization
# normalize to thermal polarization
e=(U**np.inf*rho)().I[1,0].real/ex0.Peq[0]

# Clear, run the buildup
rho.clear()
rho.DetProp(U,n=20000)
ax=rho.plot(axis='s')

ax.figure.set_size_inches([5,4])
ax.figure.savefig(os.path.join(directory,'OE.png'),transparent=True)

print(f'Overhauser effect time: {time()-t0:.0f} s')

#%% Part H: Pseudocontact Shift orientational dependence
t0=time()
delta=sl.Tools.dipole_coupling(1,'e-','13C')    #10 Angstroms from electron
# Sweep euler angles
sl.Defaults['verbose']=False
SZ=[15,30]
beta=np.linspace(0,np.pi/2,SZ[0])
gamma=np.linspace(0,np.pi*2,SZ[1])
N=beta.size*gamma.size

beta,gamma=np.meshgrid(beta,gamma)
shift=[]
for k,(beta0,gamma0) in enumerate(zip(beta.reshape(np.prod(SZ)),gamma.reshape(np.prod(SZ)))):
    ex0=sl.ExpSys(v0H=600,Nucs=['13C','e-'],LF=True,vr=0,T_K=100,pwdavg=sl.PowderAvg('alpha0beta0'))  #Electron-nuclear system
    ex0.set_inter('hyperfine',i0=0,i1=1,Axx=-delta/2,Ayy=-delta/2,Azz=delta)
    ex0.set_inter('g',i=1,gxx=1,gyy=1,gzz=4,euler=[0,beta0,gamma0])

    L=sl.Tools.SetupTumbling(ex0,tc=1e-9,q=2) #1 ns tumbling

    L.add_relax(Type='T1',i=1,T1=1e-11,OS=True,Thermal=True)
    L.add_relax(Type='T2',i=1,T2=1e-11,OS=True)

    seq=L.Sequence(Dt=1e-4)

    rho=sl.Rho('Thermal','13Cp')  #Generate initial state, detection operator
    Upi2=L.Udelta('13C',np.pi/2,np.pi/2)
    (Upi2*rho).DetProp(seq,n=800) #Propagate the system
    rho.downmix()

    i=np.argmax(rho.FT[0])
    shift.append(rho.v_axis[i])
        
shift=np.array(shift).reshape([SZ[1],SZ[0]])

from copy import copy
x=shift*np.sin(beta)*np.cos(gamma)
y=shift*np.sin(beta)*np.sin(gamma)
z=shift*np.cos(beta)

ax=plt.figure(figsize=[8,8]).add_subplot(1,1,1,projection='3d')
i=shift>0

x0,y0,z0=copy(x),copy(y),copy(z)
for q in [x0,y0,z0]:q[i]=0
ax.plot_surface(x0,y0,z0,color='#801010')
ax.plot_surface(-x0,-y0,-z0,color='#801010')

x0,y0,z0=copy(x),copy(y),copy(z)
for q in [x0,y0,z0]:q[~i]=0
ax.plot_surface(x0,y0,z0,color="#0093AF")
ax.plot_surface(-x0,-y0,-z0,color="#0093AF")


# ax.plot_surface(x[~i],y[~i],z[~i])
for q in ['x','y','z']:
    getattr(ax,f'set_{q}lim')([-400,400])
    getattr(ax,f'set_{q}label')(r'$\delta_{PCS} / Hz$')
    getattr(ax,f'set_{q}ticks')([-400,0,400])
ax.figure.set_size_inches([4,4])
ax.figure.tight_layout()
ax.figure.savefig(os.path.join(directory,'PCS_orient_depend.png'),transparent=True)

print(f'PCS liquids time: {time()-t0:.0f} s')

#%% Part I: Pseudocontact shift in solids (rank-4 tensor)
t0=time()
delta=sl.Tools.dipole_coupling(1,'e-','13C')    #10 Angstroms from electron
ex=sl.ExpSys(v0H=600,Nucs=['13C','e-'],vr=6000,LF=True,T_K=200,pwdavg=6,n_gamma=30)  #Electron-nuclear system
ex.set_inter('hyperfine',i0=0,i1=1,Axx=-delta/2,Ayy=-delta/2,Azz=delta)
ex.set_inter('g',i=1,gxx=1,gyy=1,gzz=4)
    
L=sl.Liouvillian(ex)        #Generate a Liouvillian

L.add_relax('T1',i=1,T1=2e-8,OS=True,Thermal=True)  
L.add_relax('T2',i=1,T2=2e-8,OS=True)

seq=L.Sequence() #Generate an empty sequence

rho200=sl.Rho('13Cx','13Cp')  #Generate initial state, detection operator
_=rho200.DetProp(seq,n=8000,n_per_seq=1) #Propagate the system

rho200.downmix()
ax=rho200.plot(FT=True,color='maroon') #Plot the results into the same axis
_=ax.set_xlim([1,-1])
ax.set_yticklabels('')
ax.set_xticks([-1,-.5,0,.5,1])
ax.figure.set_size_inches([6,4])
ax.figure.savefig(os.path.join(directory,'PCS_solids.png'),transparent=True)

print(f'PCS solids time: {time()-t0:.0f} s')


print(f'Total time: {time()-t00:.0f} s')