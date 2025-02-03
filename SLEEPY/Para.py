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
def StepCalculator(t0,Dt,dt):
    tf=t0+Dt
    # n0=int(np.round(t0/dt,2))
    # nf=int(np.round(tf/dt,2))
    
    n0=int(t0//dt)
    nf=int(tf//dt)
    
    tm1=dt-(t0-n0*dt)
    tp1=tf-nf*dt
    if nf==n0:
        tm1=Dt
        tp1=0
        
    return n0,nf,tm1,tp1

class ParallelManager():
    def __init__(self,L,t0,Dt):
        self.L=L
        self.LrelaxOS=L.LrelaxOS if L.LrelaxOS.active else None
        
        self.cache=Defaults['cache']
        
        if L.static:
            self.pars=(Dt,)
        else:
            
            dt=L.dt
            n_gamma=L.expsys.n_gamma
            n0,nf,tm1,tp1=StepCalculator(t0=t0,Dt=Dt,dt=dt)
            
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
        if self.LrelaxOS is not None:
            # This is a bit tricky. We have to be careful to get an independent copy of LrelaxOS
            # Otherwise, it seems like we end up with the same instance for all orientations
            out.LrelaxOS=copy(out.L.LrelaxOS)
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
    def parallel(self):
        return Defaults['parallel'] if self.LrelaxOS is None else False
    
    
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
        if self.parallel and self.cache:
            return self.PropCache.sm0
        elif self.cache:
            return self.PropCache.calc_index
    
    @property
    def sm1(self):
        if self.parallel and self.cache:
            return self.PropCache.sm1
        elif self.cache:
            return self.PropCache.U
    
    @property
    def setup(self):
        if self.L.static:
            return [(pm.L.L(0),*self.pars) for pm in self]
        
        if self.parallel or True:
            out=[(pm.Ln,self.L.Lrf,pm.LrelaxOS,*self.pars,self.sm0,self.sm1,self.index,self.step_index,self.PropCache.SZ) for pm in self]
        else:
            out=((pm.Ln,self.L.Lrf,pm.LrelaxOS,*self.pars,self.sm0,self.sm1,self.index,self.step_index,self.PropCache.SZ) for pm in self)
        
        # TODO why is the next line necessary?
        # There's some failure to update reduced in LrelaxOS without it
        # self.L[0].LrelaxOS.reduced
        # print(self[0].L.reduced)
        return out

    
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
    Ln0,Lrf,LrelaxOS,n0,nf,tm1,tp1,dt,n_gamma,sm0,sm1,index,step_index,SZ=X
    
    if LrelaxOS is None:LrelaxOS=lambda n:0
    
    # Setup if using the shared cache
    if sm0 is None:
        ci=None
    elif hasattr(sm0,'buf'):
        ci=np.ndarray(SZ[:2],dtype=bool,buffer=sm0.buf)
        Ucache=np.ndarray(SZ,dtype=Defaults['ctype'],buffer=sm1.buf)
    else:
        ci=sm0
        Ucache=sm1
        
    #Initial propagator
    # count0,count1=0,0
    if ci is not None and tm1==dt and ci[index[n0%n_gamma],step_index[n0%n_gamma]]:
        U=Ucache[index[n0%n_gamma],step_index[n0%n_gamma]]
        # count0+=1
    else:
        ph=np.exp(1j*2*np.pi*n0/n_gamma)
        L=np.sum([Ln0[m+2]*(ph**(-m)) for m in range(-2,3)],axis=0)+Lrf+LrelaxOS(n0)
        U=expm(L*tm1)
        if tm1==dt and ci is not None:
            # count1+=1
            Ucache[index[n0%n_gamma],step_index[n0%n_gamma]]=U
            ci[index[n0%n_gamma],step_index[n0%n_gamma]]=True

    for n in range(n0+1,nf):
        i,si=index[n%n_gamma],step_index[n%n_gamma]
        if ci is None or not(ci[i,si]): #Not cached
            ph=np.exp(1j*2*np.pi*n/n_gamma)
            L=np.sum([Ln0[m+2]*(ph**(-m)) for m in range(-2,3)],axis=0)+Lrf+LrelaxOS(n)
            U0=expm(L*dt)
            if ci is not None:
                # count1+=1
                Ucache[i,si]=U0
                ci[i,si]=True
        else:
            # count0+=1
            U0=Ucache[i,si]
        U=U0@U
    # print(count0,count1)
            
    if tp1>1e-10: #Last propagator
        ph=np.exp(1j*2*np.pi*nf/n_gamma)
        L=np.sum([Ln0[m+2]*(ph**(-m)) for m in range(-2,3)],axis=0)+Lrf+LrelaxOS(nf)
        U=expm(L*tp1)@U
    
    return U

def prop_static(X):
    L,Dt=X
    return expm(L*Dt)
        
        