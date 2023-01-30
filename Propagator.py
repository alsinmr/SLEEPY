#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 21:29:29 2023

@author: albertsmith
"""

import numpy as np
from fractions import Fraction
import warnings
from copy import copy

tol=1e-10

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
        self._eig=None

        
    
    @property
    def calculated(self):
        if isinstance(self.U,dict):return False
        return True
    
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
    
    def eig(self):
        """
        Calculates eigenvalues/eigenvectors of all stored propagators. Stored for
        later usage (self._eig). Subsequent operations will be performed in
        the eigenbasis where possible. Note that we will also ensure that no
        eigenvalues have an absolute value greater than 1. For systems that 
        deviate due to numerical error, this approach may stabilize the system.

        Returns
        -------
        None.

        """
        if self._eig is None:
            self._eig=list()
            for U in self:
                d,v=np.linalg.eig(U)
                dabs=np.abs(d)
                i=dabs>1
                d[i]/=dabs[i]
                self._eig.append((d,v))
        
    
    def calcU(self):
        """
        We have the option when generating U, to not actually calculate its value
        until an operation requiring U is performed. This is potentially useful
        since operations rho*U do not actually require ever calculating U
        explicitely. This function calculates and stores U.
        
        Note that this function runs when self.U is a dictionary instead of
        a list, with keys t, v1, phase, and voff. Once run, U is replaced by
        the propagator matrices for each element of the powder average.

        Returns
        -------
        None.

        """
        
        if not(self.calculated):
            dct=self.U
            t=dct['t']
            
            L=self.L
            U=L.Ueye(t[0])
            
            
            ini_fields=copy(L.fields)
    
            U=L.Ueye(t[0])
            for m,(ta,tb) in enumerate(zip(t[:-1],t[1:])):
                for k,(v1,phase,voff) in enumerate(zip(dct['v1'],dct['phase'],dct['voff'])):
                    L.fields[k]=(v1[m],phase[m],voff[m])
                U0=self.L.U(t0=ta,Dt=tb-ta,calc_now=True)
                U=U0*U
                
            L.fields.update(ini_fields)
            
            self.U=U.U
        
        
    
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

    
        self.eig()

        Uout=list()
        _eig=list()
        for d,v in self._eig:
            D=d**n
            Uout.append(v@np.diag(D)@np.linalg.pinv(v))
            _eig.append((D,v))
        
        # for U in self:
        #     d,v=np.linalg.eig(U)
        #     i=np.abs(d)>1
        #     d[i]/=np.abs(d[i]) #Avoid growth from numerical error
        #     D=d**n
        #     Uout.append(v@np.diag(D)@np.linalg.pinv(v))
            
        if not(self.pwdavg):Uout=Uout[0]
        
        out=Propagator(Uout,t0=self.t0,tf=self.t0+self.Dt*n,taur=self.taur,L=self.L,isotropic=self.isotropic)
        out._eig=_eig
        return out            
    
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
        self.calcU()
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
        self._index=-1
        return self
    
    def __repr__(self):
        out=f'Propagator with length of {self.Dt*1e6:.3f} microseconds (t0={self.t0*1e6:.3f},tf={self.tf*1e6:.3f})\n'
        out+='Constructed from the following Liouvillian:\n\t'
        out+=self.L.__repr__().replace('\n','\n\t')
        return out
                
            
        