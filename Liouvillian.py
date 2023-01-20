#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 15:06:02 2023

@author: albertsmith
"""

import numpy as np
from copy import copy
from scipy.linalg import expm
from Propagator import Propagator

dtype=np.complex64

class Liouvillian():
    def __init__(self,H:list,kex=None):
        """
        Creates the full Liouvillian, and provides some functions for 
        propagation of the Liouvillian

        Parameters
        ----------
        H : list
            List of Hamiltonians.
        kex : np.array, optional
            Exchange Matrix. The default is None.

        Returns
        -------
        None.

        """
        
        if hasattr(H,'shape'):H=[H]
        self.H=H
        
        for H in self.H:
            assert H.pwdavg==self.pwdavg,"All Hamiltonians must have the same powder average"
            if H.rf is not self.rf:
                H.rf=self.rf

        
        self.kex=kex
        self.sub=False
        
        self._Lex=None
        self._index=-1
        self.Lrelax=None
    

    @property
    def pwdavg(self):
        return self.H[0].pwdavg
    
    @property
    def expsys(self):
        """
        Returns the expsys of the first stored Hamiltonian

        Returns
        -------
        ExpSys
            Description of the experimental conditions and system

        """
        return self.H[0].expsys
    
    @property
    def taur(self):
        """
        Length of a rotor period

        Returns
        -------
        None.

        """
        if self.H[0].isotropic:return None
        return 1/self.expsys.vr
    
    @property
    def dt(self):
        """
        Time step for changing the rotor angle

        Returns
        -------
        float

        """
        return self.taur/self.expsys.n_gamma
    
    @property
    def shape(self):
        """
        Shape of the resulting Liouvillian

        Returns
        -------
        tuple

        """
        return np.prod(self.H[0].shape)*len(self.H),np.prod(self.H[0].shape)*len(self.H)
    
    @property
    def rf(self):
        return self.H[0].rf
    
    def __setattr__(self,name,value):
        """
        Resets certain parameters if edits occur

        Parameters
        ----------
        name : str
            Parameter name.
        value : TYPE
            Parameter value.

        Returns
        -------
        None.

        """
        
        if name=='kex':
            self._Lex=None
        
        super().__setattr__(name,value)
    
    @property
    def Lex(self):
        """
        Returns the exchange component of the Liouvillian

        Returns
        -------
        np.array

        """
        
        if self._Lex is None:
            if self.kex is None:
                self.kex=np.eye(len(self.H),dtype=dtype)
                if len(self.H)>1:print('Warning: Exchange matrix was not defined')
            self._Lex=np.kron(self.kex,np.eye(np.prod(self.H[0].shape)))
            
        return self._Lex
    
    def __getitem__(self,i:int):
        """
        Goes to a particular item of the powder average

        Parameters
        ----------
        i : int
            Element of the powder average to go to.

        Returns
        -------
        Liouvillian

        """
        out=copy(self)
        
        out.H=[H0[i] for H0 in self.H]
        out.sub=True
        return out
    
    def Ln_H(self,n:int):
        """
        Returns the nth rotating component of the Liouvillian resulting from
        the Hamiltonians. That is, contributions from exchange and relaxation
        matrices are not included.
        
        Only works if we are at a particular index of the Liouvillian
        L[0].Ln_H(0)
        Other

        Parameters
        ----------
        n : int
            Index of the rotating component (-2,-1,0,1,2).

        Returns
        -------
        np.array

        """
        assert self.sub,"Calling Ln_H requires indexing to a specific element of the powder average"
        out=np.zeros(self.shape,dtype=dtype)
        q=np.prod(self.H[0].shape)
        for k,H0 in enumerate(self.H):
            out[k*q:(k+1)*q][:,k*q:(k+1)*q]=H0.Ln(n)
        out*=-1j*2*np.pi
        return out
    
    def add_relax(self,M=None):
        """
        Add explicit relaxation to the Liouvillian. This is provided by a matrix,
        M, directly. The matrix itself can be produced with the RelaxationMatrix
        class. Note that the matrix can either have the same shape as the full
        Liouvillian, or the shape for just one Hamiltonian. For example, for
        a two-spin 1/2 system in two-site exchange, M may have size 16x16 or
        32x32. The 32x32 matrix allows different relaxation properties for the
        two sites.

        Parameters
        ----------
        M : np.array, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        q=np.prod(self.H[0].shape)
        if M.shape[0]==q:
            self.Lrelax=np.zeros(self.shape)
            for k,H0 in enumerate(self.H):
                self.Lrelax[k*q:(k+1)*q][:,k*q:(k+1)*q]=M
        elif M.shape[0]==self.shape[0]:
            self.Lrelax=M
        else:
            assert False,f"M needs to have size ({q},{q}) or {self.shape}"
            
    def Ln(self,n:int):
        """
        Returns the nth rotation component of the total Liouvillian. 

        Parameters
        ----------
        n : int
            DESCRIPTION.

        Returns
        -------
        None.

        """
            
        assert self.sub,"Calling Ln requires indexing to a specific element of the powder average"
        
        out=self.Ln_H(n)
        if n==0:  #Only add these terms to the n=0 term
            out+=self.Lex
            if self.Lrelax is not None:
                out+=self.Lrelax
        return out
    
    def U(self,t0:float=0,tf:float=None):
        """
        Calculates the propagator between times t0 and tf. By default, we will
        calculate one rotor period

        Parameters
        ----------
        t0 : float, optional
            Initial time for the propagator. The default is 0.
        tf : float, optional
            Final time for the propagator. The default is None, which will be
            set to taur (length of rotor period)

        Returns
        -------
        None.

        """
        
        # assert self.sub,"Calling L.U requires indexing to a specific element of the powder average"
    
    
        if tf is None:tf=self.taur
        
        if self.sub:
            
            Ln=[self.Ln(n) for n in range(-2,3)]
            
            dt=self.dt
            n0=int(t0//dt)
            nf=int(tf//dt)
            
            n_gamma=self.expsys.n_gamma
            
            tm1=t0-n0*dt
            tp1=tf-nf*dt
            
            if tm1<=0:tm1=dt
            
            ph=np.exp(1j*2*np.pi*n0/n_gamma)
            L=np.sum([Ln0*ph**(-m) for Ln0,m in zip(Ln,range(-2,3))],axis=0)
            U=expm(L*tm1)
                
            for n in range(n0+1,nf):
                ph=np.exp(1j*2*np.pi*n/n_gamma)
                L=np.sum([Ln0*ph**(-m) for Ln0,m in zip(Ln,range(-2,3))],axis=0)
                U=expm(L*dt)@U
            if tp1>1e-10:
                ph=np.exp(1j*2*np.pi*nf/n_gamma)
                L=np.sum([Ln0*ph**(-m) for Ln0,m in zip(Ln,range(-2,3))],axis=0)
                U=expm(L*tp1)@U
            return Propagator(U,t0=t0,tf=tf,taur=self.taur,L=self)
        else:
            U=[L0.U(t0=t0,tf=tf).U for L0 in self]
            return Propagator(U=U,t0=t0,tf=tf,taur=self.taur,L=self)

    

    def Ueig(self):
        """
        Returns eigenvalues and eigenvectors of U for one rotor period. Can
        be used for fast propagation in the eigenbasis

        Returns
        -------
        tuple

        """
        
        # d,v=eig(self.U(),k=self.shape[0]-1)
        d,v=np.linalg.eig(self.U())
        i=np.abs(d)>1
        d[i]/=np.abs(d[i])
        return d,v
    
    def __len__(self):
        if self.pwdavg is None:return 1
        return self.pwdavg.N
    
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
        
        
        