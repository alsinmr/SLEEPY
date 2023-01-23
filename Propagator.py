#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 21:29:29 2023

@author: albertsmith
"""

import numpy as np
from fractions import Fraction
import warnings
from pyRelaxSim import Defaults


dtype=Defaults['dtype']
tol=1e-6

class Propagator():
    def __init__(self,U,t0,tf,taur,L,isotropic):
        self.U=U
        self.pwdavg=False if hasattr(U,'shape') else True
        self.t0=t0
        self.tf=tf
        self.taur=taur
        self.L=L
        self._index=-1
        self._isotropic=isotropic
        
        
    @property
    def isotropic(self):
        return self._isotropic
        
    @property
    def Dt(self):
        return self.tf-self.t0
    
    @property
    def expsys(self):
        return self.L.expsys
    
    @property
    def shape(self):
        return self.L.shape
    
    def __mul__(self,U):
        if str(U.__class__)!=str(self.__class__):
            return NotImplemented
        
        if not(self.isotropic) and np.abs((self.t0-U.tf)%self.taur)>tol and np.abs((U.tf-self.t0)%self.taur)>tol:
            warnings.warn(f'\nFirst propagator ends at {U.tf%self.taur} but second propagator starts at {self.t0%U.taur}')
            # print(f'Warning: First propagator ends at {U.tf%self.taur} but second propagator starts at {self.t0%U.taur}')
        
        if self.pwdavg:
            assert U.pwdavg,"Both propagators should have a powder average or bot not"
        else:
            assert not(U.pwdavg),"Both propagators should have a powder average or bot not"
        

        Uout=[U1@U2 for U1,U2 in zip(self,U)]
        
        if not(self.pwdavg):Uout=Uout[0]
        return Propagator(Uout,t0=U.t0,tf=U.tf+self.Dt,taur=self.taur,L=self.L,isotropic=self.isotropic)
    
    def __or__(self,U):
        return self.__mul__(U)
    
    @property
    def rotor_fraction(self):
        """
        Determines how we may divide the propagator into the rotor cycle. For
        example, if the propagator has a length of 0.4 rotor cycles, then 
        every 2 rotor cycles, corresponding to 5 propagator steps, we may then
        recycle the propagator. Then, this function would return a 2 and a 5

        Returns
        -------
        tuple

        """
        out=Fraction(self.Dt/self.taur).limit_denominator(100000)
        return out.numerator,out.denominator
        
    
    def __pow__(self,n):
        """
        Raise the operator to a given power
        
        Probably, we should consider when to use eigenvalues and when to do
        a direct calculation. Currently always uses eigenvalues, which always
        takes about the same amount of time regardless of n. For a 32x32 matrix,
        n=100 is roughly equally as fast for both operations.  
        
        But, this is the rule for a 32x32 L matrix. Scaling is supposedly
        O(n^3) for matrix multiplication
        O(n^2) for eigenvalue decomposition.

        Parameters
        ----------
        n : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        if not(self.isotropic) and self.Dt%self.taur>tol:
            warnings.warn('Power of a propagator should only be used if the propagator is exactly one rotor period')

        if not(self.isotropic) and not(isinstance(n,int)):
            warnings.warn('Warning: non-integer powers may not accurately reflect state of propagator in the middle of a rotor period')

    

        Uout=list()
        for U in self:
            d,v=np.linalg.eig(U)
            i=np.abs(d)>1
            d[i]/=np.abs(d[i]) #Avoid growth from numerical error
            D=d**n
            Uout.append(v@np.diag(D)@np.linalg.pinv(v))
            
        if not(self.pwdavg):Uout=Uout[0]
        return Propagator(Uout,t0=self.t0,tf=self.t0+self.Dt*n,taur=self.taur,L=self.L,isotropic=self.isotropic)
            
    
    
            
    
    def __getitem__(self,i:int):
        """
        Return the ith propagator

        Parameters
        ----------
        i : int
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        if not(self.pwdavg):return self.U
        
        return self.U[i%len(self)]
    
    def __len__(self):
        if not(self.pwdavg):return 1
        return len(self.U)
    
    def __next__(self):
        self._index+=1
        if self._index==len(self):
            self._index=-1           
            raise StopIteration
        else:
            return self[self._index]
    
    def __iter__(self):
        def fun():
            for k in range(len(self)):
                yield self[k]
        return fun()
        
                
            
        