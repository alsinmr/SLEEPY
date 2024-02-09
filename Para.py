#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 11:10:03 2024

@author: albertsmith
"""

import numpy as np
from scipy.linalg import expm
import multiprocessing as mp
from . import Defaults
from copy import copy


"""
New attempt at parallel processing including shared memory and a cache for
propagators
"""


class ParallelManager():
    def __init__(self,L,t0,Dt):
        self.L=L
        
        
        
        
        if L.static:
            self.pars=(Dt,)
        else:
            tf=t0+Dt
            dt=L.dt
            n0=int(t0//dt)
            nf=int(tf//dt)
            
            tm1=t0-n0*dt
            tp1=tf-nf*dt
            if tm1<=0:tm1=dt
            n_gamma=L.pwdavg.n_gamma
            self.pars=(n0,nf,tm1,tp1,dt,n_gamma)
        
        self._calc_index=None
        self._U=None
        self._sm0=None
        self._sm1=None
        
        self._index=-1
        
    
    def __getitem__(self,i):
        out=copy(self)
        out._index=i%len(self)
        out.L=self.L[out._index]
        return out
        
    def __len__(self):
        return len(self.L)
    
    def __iter__(self):
        self._index=-1
        return self
    
    def __next__(self):
        self._index+=1
        if self._index==len(self):
            self._index=-1
            raise StopIteration
        else:
            return self[self._index]
            
    @property
    def Ln(self):
        return [self.L.Ln(k) for k in range(-2,3)]
    
    
    @property
    def pwdavg(self):
        return self.L.pwdavg
    
    @property
    def PropCache(self):
        return self.L._PropCache
    
    @property
    def index(self):
        return [self.PropCache.index(n) for n in range(self.pwdavg.n_gamma)]
    
    @property
    def step_index(self):
        return [self.PropCache.step_index(n) for n in range(self.pwdavg.n_gamma)]
    
    
    @property
    def sm0(self):
        return self.PropCache.sm0
    
    @property
    def sm1(self):
        return self.PropCache.sm1
    
    @property
    def setup(self):
        if self.L.static:
            return [(pm.L.L(0),*self.pars) for pm in self]
        return [(pm.Ln,self.L.Lrf,*self.pars,self.sm0,self.sm1,self.index,self.step_index,self.PropCache.SZ) for pm in self]
    
    @property
    def cpu_count(self):
        if isinstance(Defaults['ncores'],int):return Defaults['ncores']
        return mp.cpu_count()
    
    def __call__(self):
        
        if self.L.static:
            fun=prop_static
        else:
            fun=prop
        
        X=self.setup
        if Defaults['parallel']:
            with mp.Pool(processes=self.cpu_count) as pool:
                U=pool.map(fun,X)
                
        else:
            U=[fun(X0) for X0 in X]
        return U
    

#%% Parallel functions
def prop(X):
    Ln0,Lrf,n0,nf,tm1,tp1,dt,n_gamma,sm0,sm1,index,step_index,SZ=X
    
    # Setup if using the shared cache
    if sm0 is not None:
        ci=np.ndarray(SZ[:2],dtype=bool,buffer=sm0.buf)
        Ucache=np.ndarray(SZ,dtype=Defaults['ctype'],buffer=sm1.buf)
    else:
        ci=None
        
    #Initial propagator
    if tm1:
        ph=np.exp(1j*2*np.pi*n0/n_gamma)
        L=np.sum([Ln0[m+2]*(ph**(-m)) for m in range(-2,3)],axis=0)+Lrf
        U=expm(L*tm1)
    else:
        U=np.eye(Ln0[2].shape[0],dtype=Ln0[2].dtype)
        
    for n,i,si in zip(range(n0+1,nf),index,step_index):
        if ci is None or not(ci[i,si]): #Not cached
            ph=np.exp(1j*2*np.pi*n/n_gamma)
            L=np.sum([Ln0[m+2]*(ph**(-m)) for m in range(-2,3)],axis=0)+Lrf
            U0=expm(L*dt)
            if ci is not None:
                Ucache[i,si]=U0
                ci[i,si]=True
        else:
            U0=Ucache[i,si]
        U=U0@U
            
    if tp1>1e-10: #Last propagator
        ph=np.exp(1j*2*np.pi*nf/n_gamma)
        L=np.sum([Ln0[m+2]*(ph**(-m)) for m in range(-2,3)],axis=0)+Lrf
        U=expm(L*dt)@U
    
    return U

def prop_static(X):
    L,Dt=X
    return expm(L*Dt)
        
        