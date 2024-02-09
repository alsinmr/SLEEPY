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
from scipy.linalg import expm
from . import Defaults

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
    def isotropic(self):            #Not sure this is necessary. Why not just get the value out of L?
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
    
    @property
    def static(self):
        return self.L.static
    

    
    def eig(self,back_calc:bool=True):
        """
        Calculates eigenvalues/eigenvectors of all stored propagators. Stored for
        later usage (self._eig). Subsequent operations will be performed in
        the eigenbasis where possible. Note that we will also ensure that no
        eigenvalues have an absolute value greater than 1. For systems that 
        deviate due to numerical error, this approach may stabilize the system.

        Parameters
        ----------
        back_calc : bool, optional
            Back-calculates the stored propagators from the eigenvalues which
            have been corrected such that they do not have magnitude greater than
            one. This may stabilize some calculations.

        Returns
        -------
        self

        """
        if self._eig is None:
            self._eig=list()
            Unew=list()
            for k,U in enumerate(self):
                d,v=np.linalg.eig(U)
                dabs=np.abs(d)
                i=dabs>1
                d[i]/=dabs[i]
                if self.L.Peq:
                    i=np.argmax(d)
                    v[:,i]=self.L.rho_eq(pwdindex=k)
                    v[:,i]/=np.sqrt((v[:,i].conj()*v[:,i]).sum())
                    d[i]=1.
                self._eig.append((d,v))
                if back_calc:
                    Unew.append(v@np.diag(d)@np.linalg.pinv(v))
            if back_calc:self.U=Unew
        return self
        
    
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
        return self
         
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
        
    def __mul__(self,U):
        if str(U.__class__)!=str(self.__class__):
            return NotImplemented
        
        
        if not(self.static) and np.abs((self.t0-U.tf)%self.taur)>tol and np.abs((U.tf-self.t0)%self.taur)>tol:
            warnings.warn(f'\nFirst propagator ends at {U.tf%self.taur} but second propagator starts at {self.t0%U.taur}')
            # print(f'Warning: First propagator ends at {U.tf%self.taur} but second propagator starts at {self.t0%U.taur}')
        
        if self.pwdavg:
            assert U.pwdavg,"Both propagators should have a powder average or bot not"
        else:
            assert not(U.pwdavg),"Both propagators should have a powder average or bot not"

        Uout=[U1@U2 for U1,U2 in zip(self,U)]
        
        if not(self.pwdavg):Uout=Uout[0]
        return Propagator(Uout,t0=U.t0,tf=U.tf+self.Dt,taur=self.taur,L=self.L,isotropic=self.isotropic)
    
    # def __rmul__(self,U):
    #     if U==1:
    #         return self
        
    
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

        if not(self.static) and self.Dt%self.taur>tol:
            warnings.warn('Power of a propagator should only be used if the propagator is exactly one rotor period')

        if not(self.static) and not(isinstance(n,int)):
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
                
    
import multiprocessing as mp
if 'shared_memory' in dir(mp):
    from multiprocessing.shared_memory import SharedMemory
    SM=True
else:
    SM=False

class PropCache():
    def __init__(self,L):
        """
        Stores propagators that may be later recycled.

        Parameters
        ----------
        L : TYPE
            DESCRIPTION.
        active : bool, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        None.

        """
        self.L=L
        
        self.reset()
        
        
    def reset(self):
        self.fields=[]
        
        self._U=[]
        self._calc_index=[]
        self._sm0=[]
        self._sm1=[]
        self.cache=Defaults['cache']
        return self
        
        
    @property
    def pwdavg(self):
        return self.L.pwdavg
    
    
    #%% Return the applied-field specific information
    @property
    def field(self):
        return tuple(x for x in self.L.rf.fields.values())
    
    @property
    def field_index(self):
        if self.field not in self.fields:
            self.add_field()
        return self.fields.index(self.field)
    
    @property
    def U(self):
        return self._U[self.field_index]
    @U.setter
    def U(self,U):
        self._U.append(U)
    
    @property
    def calc_index(self):
        if not(self.cache):return None
        return self._calc_index[self.field_index]
    @calc_index.setter
    def calc_index(self,x):
        self._calc_index.append(x)
    
    @property
    def sm0(self):
        if not(self.cache):return None
        return self._sm0[self.field_index]
    @sm0.setter
    def sm0(self,sm0):
        self._sm0.append(sm0)
    
    @property
    def sm1(self):
        if not(self.cache):return None
        return self._sm1[self.field_index]
    @sm1.setter
    def sm1(self,sm1):
        self._sm1.append(sm1)
    
    
    @property
    def nbytes(self):
        nb=np.zeros(1,dtype=Defaults['ctype']).nbytes
        return np.prod(self.SZ)*nb
        
    #%% Sizes/indices
    @property
    def SZ(self):
        return (self.pwdavg.N,(self.pwdavg.n_gamma if self.pwdavg._gamma_incl else 1),*self.L.shape)
    
    def index(self,n):
        if self.pwdavg._gamma_incl:
            return self.L._index
        return (self.L._index+n*self.pwdavg.n_alpha)%self.pwdavg.N
    
    def step_index(self,n):
        if self.pwdavg._gamma_incl:
            return n
        return 0
 
    #%% Add field + data management
    def add_field(self):
        if not(self.cache):return
        if self.field not in self.fields:
            self.fields.append(self.field)
            if Defaults['parallel'] and SM:
                self.sm0=SharedMemory(create=True,size=np.prod(self.SZ[:2]))
                self.sm1=SharedMemory(create=True,size=self.nbytes)
                self.calc_index=np.ndarray(shape=self.SZ[:2],dtype=bool,buffer=self.sm0.buf)
                self.U=np.ndarray(shape=self.SZ,dtype=Defaults['ctype'],buffer=self.sm1.buf)
            else:
                self.sm0=None
                self.sm1=None
                self.calc_index=np.zeros(self.SZ[:2],dtype=bool)
                self.U=np.zeros(self.SZ,dtype=Defaults['ctype'])
            self._calc_index.append(np.zeros(self.SZ[:2],dtype=bool))
        return self
    
    def __del__(self,*args):
        for sm in [*self._sm0,*self._sm1]:
            if sm is None:continue
            sm.unlink()
        
    
    #%% Return propagators
    def __getitem__(self,n:int):
        return self.get_prop(n%len(self))
    
    def __len__(self):
        return self.pwdavg.n_gamma
    
    def get_prop(self,n:int):
        if not(self.cache):return expm(self.L.L(n)*self.L.dt)
        U=self.U
        i0,i1=self.index(n),self.step_index(n)
        if not(self.calc_index[i0,i1]):
            U[i0,i1]=expm(self.L.L(n)*self.L.dt)
            self.calc_index[i0,i1]=True
            
        return U[i0,i1]

            
    
        
    
            
        