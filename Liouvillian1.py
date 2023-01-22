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
from pyRelaxSim import Defaults
from Tools import Ham2Super

dtype=Defaults['dtype']

class Liouvillian():
    def __init__(self,H:list,kex=None,save_mem:int=0):
        """
        Creates the full Liouvillian, and provides some functions for 
        propagation of the Liouvillian

        Parameters
        ----------
        H : list
            List of Hamiltonians.
        kex : np.array, optional
            Exchange Matrix. The default is None.
        save_mem: Determines whether to optimize for speed or memory usage
            0:  Pre-calculate matrices for all powder elements and time steps in 
                rotor period
            1:  Pre-calculate Ln for all powder elements
            2:  No pre-calculation
        
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
                
        assert save_mem in [0,1,2],"save_mem must be 0, 1, or 2"
        self._save_mem=save_mem
        
        self.kex=kex
        self._kex=copy(kex)
        self.sub=False
        
        self._Lex=None
        self._index=-1
        self._Lrelax=None

        
        self._Ln=None
        self._L=None
        self._Lrf=None
        self._fields=None  #Stored field when self.Lrf is called
        self.setup()
    
    def setup(self):
        """
        Performs initial calculation of the Liouville matrix, depending on
        the save_mem setting

        Returns
        -------
        None.

        """
        if self.save_mem==2:return
        
        if self.save_mem==1:
            self._Ln=[None for _ in range(len(self))]
        elif self.save_mem==0:
            self._L=[None for _ in range(len(self))]

        

    @property
    def save_mem(self):
        return self._save_mem

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
    
    @property
    def fields(self):
        return self.rf.fields
    
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
        
        super().__setattr__(name,value)
    

    
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
            self._Lrelax=np.zeros(self.shape)
            for k,H0 in enumerate(self.H):
                self._Lrelax[k*q:(k+1)*q][:,k*q:(k+1)*q]=M
        elif M.shape[0]==self.shape[0]:
            self._Lrelax=M
        else:
            assert False,f"M needs to have size ({q},{q}) or {self.shape}"
        self.setup()
    
    @property        
    def Lrelax(self):
        if self._Lrelax is None:
            self.shape
            self._Lrelax=np.zeros(self.shape,dtype=dtype)
        return self._Lrelax
    
    @property
    def Lex(self):
        """
        Returns the exchange component of the Liouvillian

        Returns
        -------
        np.array

        """
        
        if not(np.all(self.kex==self._kex)):
            if self._Lex is not None:
                self.setup()
                self._Lex=None
        
        if self._Lex is None:
            if self.kex is None:
                self.kex=np.zeros([len(self.H),len(self.H)],dtype=dtype)
                if len(self.H)>1:print('Warning: Exchange matrix was not defined')
            self._Lex=np.kron(self.kex,np.eye(np.prod(self.H[0].shape)))
            
        return self._Lex
    
    def Ln_H(self,n:int):
        """
        Returns the nth rotating component of the Liouvillian resulting from
        the Hamiltonians. Contributions from exchange, relaxation, and rf
        matrices are not included. 
        
        Only works if we are at a particular index of the Liouvillian
        L[0].Ln_H(0)
        

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
               
    def Ln(self,n:int):
        """
        Returns the nth rotation component of the total Liouvillian. Does not
        include contributions from the rf Hamiltonian

        Parameters
        ----------
        n : int
            DESCRIPTION.

        Returns
        -------
        None.

        """
            
        assert self.sub,"Calling Ln requires indexing to a specific element of the powder average"
        
        if self.save_mem==1:   #Store Ln for all elements of the powder average
            if self._Ln[self._index] is None:
                self._Ln[self._index]=[self.Ln_H(n) for n in range(-2,3)]
                self._Ln[self._index][2]+=self.Lex+self.Lrelax
            return self._Ln[self._index][n]                
        
        if self._Ln is None:   #Store Ln for this element of the powder average
            #Note, we do this also if we store all L matrices (pwd x n_gamma) (so save_mem=0 or 2)
            self._Ln=[self.Ln_H(n) for n in range(-2,3)]
            self._Ln[2]+=self.Lex+self.Lrelax
            
        return self._Ln[n+2]
    
    @property
    def Lrf(self):
        """
        Returns the Liouvillian for the applied RF field. Depends on the current
        settings for L.fields

        Returns
        -------
        np.array

        """
        if self._fields!=self.fields:
            self._Lrf=None

        if self._Lrf is None:
            self._fields=copy(self.fields)
            self._Lrf=np.zeros(self.shape,dtype=dtype)
            n=self.H[0].shape[0]**2
            Lrf=Ham2Super(self.rf())
            for k in range(len(self.H)):
                self._Lrf[k*n:(k+1)*n][:,k*n:(k+1)*n]=Lrf
            
        return self._Lrf
    
    def L(self,step:int):
        """
        Returns the Liouvillian for a given step in the rotor cycle (t=step*L.dt).

        Parameters
        ----------
        step : Index of the step for the rotor cycle ()
            DESCRIPTION.

        Returns
        -------
        None.

        """
        assert self.sub,"Calling Ln requires indexing to a specific element of the powder average"
        
        if self.save_mem==0:
            if self._L[self._index] is None:
                self._L[self._index]=[None for _ in range(self.expsys.n_gamma)]
            if self._L[self._index][step] is None:
                Ln=[self.Ln(n) for n in range(-2,3)]
                ph=np.exp(1j*2*np.pi*step/self.expsys.n_gamma)
                self._L[self._index][step]=\
                  np.sum([Ln0*ph**(-m) for Ln0,m in zip(Ln,range(-2,3))],axis=0)
            return self._L[self._index][step]-1j*2*np.pi*self.Lrf
        
        Ln=[self.Ln(n) for n in range(-2,3)]
        n_gamma=self.expsys.n_gamma
        ph=np.exp(1j*2*np.pi*step/n_gamma)
        return np.sum([Ln0*ph**(-m) for Ln0,m in zip(Ln,range(-2,3))],axis=0)-1j*2*np.pi*self.Lrf
 
    
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
            
            # Ln=[self.Ln(n) for n in range(-2,3)]
            
            dt=self.dt
            n0=int(t0//dt)
            nf=int(tf//dt)
            
            # n_gamma=self.expsys.n_gamma
            
            tm1=t0-n0*dt
            tp1=tf-nf*dt
            
            if tm1<=0:tm1=dt
            
            # ph=np.exp(1j*2*np.pi*n0/n_gamma)
            
            # L=np.sum([Ln0*ph**(-m) for Ln0,m in zip(Ln,range(-2,3))],axis=0)
            L=self.L(n0)
            
            U=expm(L*tm1)
                
            for n in range(n0+1,nf):
                # ph=np.exp(1j*2*np.pi*n/n_gamma)
                # L=np.sum([Ln0*ph**(-m) for Ln0,m in zip(Ln,range(-2,3))],axis=0)
                L=self.L(n)
                U=expm(L*dt)@U
            if tp1>1e-10:
                # ph=np.exp(1j*2*np.pi*nf/n_gamma)
                # L=np.sum([Ln0*ph**(-m) for Ln0,m in zip(Ln,range(-2,3))],axis=0)
                L=self.L(nf)
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
        
        
        