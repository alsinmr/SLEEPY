#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 21:29:29 2023

@author: albertsmith
"""

from Tools import BlockDiagonal
import numpy as np

dtype=np.complex64

class Propagator():
    def __init__(self,U,t0,tf,taur):
        self.U=U
        self.pwdavg=False if hasattr(U,'shape') else True
        self.t0=t0
        self.tf=tf
        self.taur=taur
        self._index=-1
        
        
    @property
    def Dt(self):
        return self.tf-self.t0
    
    def __mul__(self,U):
        if self.t0%self.taur!=U.tf%U.taur:
            print(f'Warning: First propagator ends at {U.tf%self.taur} but second propagator starts at {self.t0%U.taur}')
        
        if self.pwdavg:
            assert U.pwdavg,"Both propagators should have a powder average or bot not"
        else:
            assert not(U.pwdavg),"Both propagators should have a powder average or bot not"
        

        Uout=[U1@U2 for U1,U2 in zip(self,U)]
        # Uout=list()
        # for U1,U2 in zip(self,U):
        #     # Blks=BlockDiagonal(np.logical_or(np.abs(U1)>1e-6,np.abs(U2)>1e-6))
        #     # # Blks=[np.ones(U1.shape[0],dtype=bool)]
        #     # U12=np.zeros(U1.shape,dtype=dtype)
        #     # for blk in Blks:
        #     #     temp=U1[blk][:,blk]@U2[blk][:,blk]
        #     #     for k,m in enumerate(np.argwhere(blk)[:,0]):
        #     #         U12[blk,m]=temp[:,k]
        #     # Uout.append(U12)
        #     Uout.append(U1@U2)
        
        if not(self.pwdavg):Uout=Uout[0]
        return Propagator(Uout,t0=U.t0,tf=U.tf+self.Dt,taur=self.taur)
    
    def __pow__(self,n):
        """
        Raise the operator to a given power

        Parameters
        ----------
        n : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        if self.t0%self.taur!=self.tf%self.taur:
            print('Power of a propagator should only be used if the propagator is exactly one rotor period')

        if not(isinstance(n,int)):
            print('Warning: non-integer powers may not accurately reflect state of propagator in the middle of a rotor period')

        Uout=list()
        for U in self:
            d,v=np.linalg.eig(U)
            i=np.abs(d)>1
            d[i]/=np.abs(d[i]) #Avoid growth from numerical error
            D=d**n
            Uout.append(v@np.diag(D)@np.linalg.pinv(v))
            
        if not(self.pwdavg):Uout=Uout[0]
        return Propagator(Uout,t0=self.t0,tf=self.t0+self.Dt*n,taur=self.taur)
            
            
            
    
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
        
                
            
        