#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 16:46:51 2025

@author: albertsmith

The following code runs a python script to perform 6 relatively fast 
simulations with SLEEPY (a total of 42 seconds on my laptop).

Before running, SLEEPY needs to be installed via pip

pip install sleepy-nmr

Run the script from a terminal by typing
python3 ShortExamples.py

Results will be stored in the ShortExamoplesFigs folder 
(created relative to the run location)
"""

import SLEEPY as sl
import numpy as np
import os
from time import time

t0=time()
if not(os.path.exists('ShortExampleFigs')):
    os.mkdir('ShortExampleFigs')


#%% Example 1: 1D Spectrum in Exchange
ex0=sl.ExpSys(v0H=600,Nucs='13C')
ex1=ex0.copy()
ex0.set_inter('CS',i=0,ppm=5)
ex1.set_inter('CS',i=0,ppm=-5)

L=sl.Liouvillian(ex0,ex1,kex=sl.Tools.twoSite_kex(tc=1e-3))
seq=L.Sequence(Dt=1/3000)  #1/(2*10 ppm*150 MHz)

rho=sl.Rho('13Cx','13Cp')
rho.DetProp(seq,n=4096)
ax=rho.plot(FT=True,axis='ppm')
ax.figure.savefig(os.path.join('ShortExampleFigs','oneDex.png'))

#%% Example 2:  T1 relaxation in solid-state NMR
ex0=sl.ExpSys(v0H=600,Nucs=['13C','1H'],vr=10000,LF=True)  #T1 occurs only due to terms in the lab frame
ex1=ex0.copy()
ex0.set_inter('dipole',i0=0,i1=1,delta=44000)
ex1.set_inter('dipole',i0=0,i1=1,delta=44000,euler=[0,30*np.pi/180,0])

L=sl.Liouvillian(ex0,ex1,kex=sl.Tools.twoSite_kex(tc=1e-9))
seq=L.Sequence() #Defaults to 1 rotor period

rho=sl.Rho('13Cz','13Cz')
rho.DetProp(seq,n=10000*10) #10 seconds
ax=rho.plot(axis='s')
ax.figure.savefig(os.path.join('ShortExampleFigs','T1solid.png'))

#%% Example 3: T1rho relaxation
ex0=sl.ExpSys(v0H=600,Nucs=['13C','1H'],vr=10000)
ex1=ex0.copy()
ex0.set_inter('dipole',i0=0,i1=1,delta=44000)
ex1.set_inter('dipole',i0=0,i1=1,delta=44000,euler_d=[0,30,0])

L=sl.Liouvillian(ex0,ex1,kex=sl.Tools.twoSite_kex(tc=1e-7))
seq=L.Sequence().add_channel('13C',v1=25000) #Defaults to 1 rotor period

rho=sl.Rho('13Cx','13Cx')
rho.DetProp(seq,n=1500) #100 ms
ax=rho.plot()
ax.figure.savefig(os.path.join('ShortExampleFigs','T1rho.png'))

#%% Example 4: Chemical Exchange Saturation Transfer
ex0=sl.ExpSys(v0H=600,Nucs='13C')
ex1=ex0.copy()
ex0.set_inter('CS',i=0,Hz=750)
ex1.set_inter('CS',i=0,Hz=-750)

L=sl.Liouvillian(ex0,ex1,kex=sl.Tools.twoSite_kex(tc=1e-1,p1=.95))  #5% population 2
L.add_relax('T1',i=0,T1=1)
L.add_relax('T2',i=0,T2=.1)
L.add_relax('recovery')
seq=L.Sequence(Dt=.5)  #1/(2*10 ppm*150 MHz)

rho=sl.Rho('13Cz','13Cz')
voff0=np.linspace(-1500,1500,101)

for voff in voff0:
    rho.reset()
    seq.add_channel('13C',v1=50,voff=voff)
    (seq*rho)()
ax=rho.plot()
ax.set_xticks(np.linspace(0,101,11))
ax.set_xticklabels(voff0[np.linspace(0,100,11).astype(int)]/1000)
_=ax.set_xlabel(r'$\nu_{off}$ / kHz')
ax.figure.savefig(os.path.join('ShortExampleFigs','CEST.png'))

#%% Example 5: Contact shift
ax=None
ex=sl.ExpSys(v0H=600,Nucs=['13C','e']).set_inter('hyperfine',i0=0,i1=1,Axx=1e5,Ayy=1e5,Azz=1e5)
for T in [50,100,200,400]:
    ex.T_K=T
    L=ex.Liouvillian()
    L.add_relax('T1',i=1,T1=1e-9)
    L.add_relax('T2',i=1,T2=1e-10)
    L.add_relax('recovery')

    seq=L.Sequence(Dt=5e-5)

    rho=sl.Rho('13Cx','13Cp')
    ax=rho.DetProp(seq,n=2048).plot(FT=True,ax=ax)
ax.figure.savefig(os.path.join('ShortExampleFigs','ContactShift.png'))
    
#%% Example 6: Spinning sidebands
ax=sl.Rho('31Px','31Pp').DetProp(
    sl.ExpSys(B0=6.9009,Nucs='31P',vr=2060).set_inter('CSA',i=0,delta=-104,eta=.56).\
    Liouvillian().Sequence(),
    n=4096,n_per_seq=20).plot(FT=True,axis='kHz',apodize=True)
    
ax.figure.savefig(os.path.join('ShortExampleFigs','SpinningSidebands.png'))

print(f'Total time: {time()-t0:.0f} s')