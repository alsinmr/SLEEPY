#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 22:32:33 2023

@author: albertsmith
"""

from copy import copy
import numpy as np
import warnings
import matplotlib.pyplot as plt

dtype=np.dtype

class Rho():
    def __init__(self,rho0,detect,L):
        """
        Creates an object that contains both the initial density matrix and
        the detector matrix. One may then apply propagators to the density
        matrix and detect the magnetization.

        Strings for specifying operators:
                
        S0x, S1y, S2alpha:
            Specifies a spin by index (S0, S1, etc.), followed by the operator
            type (x,y,z,p,m,alpha,beta)
            
        13Cx, 1Hy, 15Np:
            Specifies the channel followed by the operator type (sum of all nuclei of that type)
            
        Custom operators may be produced by adding together the matrices found
        in expsys.Op
        

        Parameters
        ----------
        rho0 : Spinop or, str, optional
            Initial density matrix, specify by string or the operator itself. 
            Operators may be found in expsys.Op.
        detect : Detection matrix or list of matrices, specify by string or the
            operator itself. Operators may be found in expsys.Op. Multiple 
            detection matrices may be specified by providing a list of operators.

        Returns
        -------
        None.

        """
        
        
        self.rho0=rho0
        self.rho=copy(rho0)
        if not(isinstance(detect,list)):detect=[detect]  #Make sure a list
        self.detect=detect
        self._L=L
        self._Setup()
        
        

        
    @property
    def L(self):
        return self._L
    
    @property
    def n_det(self):
        return len(self.detect)
    
    @property
    def expsys(self):
        return self.L.expsys
    
    @property
    def pwdavg(self):
        return self.L.pwdavg
        
    @property
    def Op(self):
        return self.expsys.Op
    
    @property
    def taur(self):
        return self.expsys.taur
        
    @property
    def t(self):
        """
        Current time (sum of length of all applied propagators)

        Returns
        -------
        float
            time.

        """
        return self._t
    
    @property
    def t_axis(self):
        """
        Time axis corresponding to when detection was performed.

        Returns
        -------
        array
            Array of all times at which the detection was performed.

        """
        return np.sort(self._taxis)
    
    @property
    def Ipwd(self):
        """
        Npwd x Nd x Nt matrix of detected amplitudes, where Npwd is the number
        of angles in the powder average, Nd is the number of detection matrices
        that have been defined, and Nt is the number of time points stored.

        Returns
        -------
        None.

        """
        if len(self.t_axis):
            i=np.argsort(self._taxis)
            return np.array(self._Ipwd).T[i].T
    
    @property
    def I(self):
        """
        Nd x Nt matrix of detected amplitude (powder average applied), where Nd
        is the number of detection matrices that have been defined and Nt is
        the number of time points stored

        Returns
        -------
        None.

        """
        return (self.Ipwd.T*self.pwdavg.weight).sum(-1).T
    
    def _Setup(self):
        """
        At initialization, we do not require Rho to know the spin-system yet. 
        However, for most functions, this is in fact required. Therefore, at
        the first operation with a propagator, we will run _setup to finalize
        the Rho setup.

        Returns
        -------
        None.

        """
        
        self._Ipwd=[[list() for _ in range(self.n_det)] for _ in range(self.pwdavg.N)]
        self._taxis=list()
        
        self._rho0=self.Op2vec(self.strOp2vec(self.rho0))
        self._detect=[self.Op2vec(self.strOp2vec(det),detect=True) for det in self.detect]
        self.reset()
        
    def prop(self,U):
        """
        Propagates the density matrix by the provided propagator

        Parameters
        ----------
        U : Propagator
            Propagator object.

        Returns
        -------
        None.

        """
            
        if U.L is not self.L:
            warnings.warn('Propagating using a system with a different Liouvillian than the initial Liouvillian')
            
        if self.t%self.taur!=U.t0%U.taur:
            warnings.warn('The initial time of the propagator is not equal to the current time of the density matrix')
            
        self._rho=[U0@rho for U0,rho in zip(U,self._rho)]
        self._t+=U.Dt
        return self
        
    def __rmul__(self,U):
        """
        Runs rho.prop(U) and returns self

        Parameters
        ----------
        U : Propagator
            Propagator object.

        Returns
        -------
        self

        """
        return self.prop(U)
    
    def __ror__(self,U):
        return self.prop(U)
    
    def Detect(self):
        """
        Evaluates the density matrices at the current time point and stores
        the result

        Returns
        -------
        None.

        """
        self._taxis.append(self.t)
        for k,rho in enumerate(self._rho):
            for m,det in enumerate(self._detect):
                self._Ipwd[k][m].append((rho*det).sum())
        return self
    
        
    def __call__(self):
        return self.Detect()
    
    def __getitem__(self,i:int):
        """
        Return the density operator for the ith element of the powder average

        Parameters
        ----------
        i : int
            Index for the density operator.

        Returns
        -------
        None.

        """
        i%=len(self)
        return self._rho[i]
    
    def __len__(self):
        return self.L.__len__()
    
    def DetProp(self,U,n:int=1):
        """
        Executes a series of propagation/detection steps. Detection occurs first,
        followed by propagation for n steps. If n>100, then we will use
        eigenvalue decomposition for the propagation

        Parameters
        ----------
        U : Propagator
            Propagator applied. Should be an integer number of rotor periods
        n : int, optional
            Number of steps. The default is 1.

        Returns
        -------
        self

        """
        if n>=100:
            self()
            for k,(U0,rho) in enumerate(zip(U,self)):
                d,v=np.linalg.eig(U0)
                rho0=np.linalg.pinv(v)@rho
                dp=np.cumprod(np.repeat([d],n,axis=0),axis=0)
                rho_d=dp*rho0
                self._rho[k]=v@rho_d[-1]
                for m,det in enumerate(self._detect):
                    det_d=det@v
                    self._Ipwd[k][m].extend((det_d*rho_d[:-1]).sum(-1))
            
            self._taxis.extend([self.t+k*U.Dt for k in range(n-1)])
            self._t+=n*U.Dt
                
        else:
            for _ in range(n):
                U*self()
        return self
    
    def strOp2vec(self,OpName:str):
        """
        Converts the string for an operator into a matrix

        Strings for specifying operators:
                
        S0x, S1y, S2alpha:
            Specifies a spin by index (S0, S1, etc.), followed by the operator
            type (x,y,z,p,m,alpha,beta)
            
        13Cx, 1Hy, 15Np:
            Specifies the channel followed by the operator type (sum of all nuclei of that type)

        Parameters
        ----------
        OpName : str
            Name of the desired operator.

        Returns
        -------
        Operator matrix

        """
        
        if not(isinstance(OpName,str)):return OpName #Just return if already a matrix
        
        if OpName[0]=='S':
            i=int(OpName[1])   #At the moment, I'm assuming this program won't work with 11 spins...
            return getattr(self.Op[0],OpName[2:])
        
        if OpName[-1] in ['x','y','z','p','m']:
            a=OpName[-1]
            Nuc=OpName[:-1]
        elif 'alpha' in OpName:
            a='alpha'
            Nuc=OpName[:-5]
        elif 'beta' in OpName:
            a='beta'
            Nuc=OpName[:-4]
        Op=np.zeros(self.Op.Mult.prod()*np.ones(2,dtype=int),dtype=dtype)
        i=self.expsys.Nucs==Nuc
        if not(np.any(i)):
            warnings.warn('Nucleus is not in the spin system or was not recognized')
        for i0 in np.argwhere(i)[:,0]:
            Op+=getattr(self.Op[i0],a)
        return Op
    
    def Op2vec(self,Op,detect:bool=False):
        """
        Converts a matrix operator for one Hamiltonian into a vector for the
        full Liouville space. Required for initial density matrix and
        detection

        Parameters
        ----------
        Op : np.array
            Square matrix for rho or detection
        detect : bool, optional
            Set to true for the detection vectors, where we need to take the
            conjugate of the matrix. The default is False.

        Returns
        -------
        Operator vector
        
        """
        nHam=len(self.L.H)
        if detect:
            Op=Op.T.conj()
            Op/=np.trace(Op.T.conj()@Op)
        return np.tile(Op.reshape(Op.size),nHam)
            
    
    def reset(self):
        """
        Resets the density matrices back to rho0

        Returns
        -------
        None.

        """
        self._rho=[self._rho0 for _ in range(self.pwdavg.N)]
        self._t=0
    
    def clear(self):
        """
        Clears variables in order to start over propagation. 
        
        Note that if you want to set the system back to the initial rho0 value,
        but want to retain the amplitudes and times recorded, run rho.reset()
        instead of rho.clear()

        Parameters
        ----------
        clear_all : bool, optional
            Completely reset rho. The default is False.

        Returns
        -------
        None.

        """
        
        self._t=0
        
        self._Ipwd=[[list() for _ in range(self.n_det)] for _ in range(self.pwdavg.N)]
        self._taxis=list()
        self._rho=list() #Storage for numerical rho
        
    def plot(self,fig=None):
        """
        Plots the amplitudes as a function of time

        Parameters
        ----------
        fig : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        if fig is None:fig=plt.figure()
        
        ax=[fig.add_subplot(1,self.n_det,k+1) for k in range(self.n_det)]
        
        for a,I in zip(ax,self.I):
            a.plot(self.t_axis*1e3,I.real)
            a.plot(self.t_axis*1e3,I.imag)
            a.set_ylabel('I [a.u.]')
        ax[-1].set_xlabel('t / ms')
            
        