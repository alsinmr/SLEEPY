#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 10:54:04 2021

@author: albertsmith
"""

import numpy as np
from types import MethodType
from pyRelaxSim.Tools import NucInfo

#%% Functions to generate spin operators

def so_m(S=None):
    "Calculates Jm for a single spin"
    if S is None:S=1/2
    M=np.round(2*S+1).astype(int)    
    jm=np.zeros(M**2)
    jm[M::M+1]=np.sqrt(S*(S+1)-np.arange(-S+1,S+1)*np.arange(-S,S))
    return jm.reshape(M,M)

def so_p(S=None):
    "Calculates Jp for a single spin"
    if S is None:S=1/2
    M=np.round(2*S+1).astype(int)    
    jp=np.zeros(M**2)
    jp[1::M+1]=np.sqrt(S*(S+1)-np.arange(-S+1,S+1)*np.arange(-S,S))
    return jp.reshape(M,M)
        
def so_x(S=None):
    "Calculates Jx for a single spin"
    return 0.5*(so_m(S)+so_p(S))

def so_y(S=None):
    "Calculates Jx for a single spin"
    return 0.5*1j*(so_m(S)-so_p(S))

def so_z(S=None):
    "Calculates Jz for a single spin"
    if S is None:S=1/2
    M=np.round(2*S+1).astype(int)
    jz=np.zeros(M**2)
    jz[::M+1]=np.arange(S,-S-1,-1)
    return jz.reshape(M,M)

def so_alpha(S=None):
    "Calculates the alpha state for a single spin"
    Sz=so_z(S)
    return np.eye(Sz.shape[0])/2+Sz

def so_beta(S=None):
    "Calculates the beta state for a single spin"
    Sz=so_z(S)
    return np.eye(Sz.shape[0])/2-Sz
    
def Op(obj,Type,n):
    """
    Returns spin operator, specified by Type (see so_* at beginning of file),
    for spin with index n
    """
    assert 'so_'+Type in globals().keys(),'Unknown spin operator type'
    
    ops=getattr(obj,'__'+Type)
    if ops[n] is not None:return ops[n]
    op0=globals()['so_'+Type](obj.S[n])
    
    ops[n]=np.kron(np.kron(np.eye(obj.Mult[:n].prod()),op0),np.eye(obj.Mult[n+1:].prod()))
    
    setattr(obj,'__'+Type,ops)
    
    return ops[n]

def Op_fun(Type):
    def fun(obj,n):
        return Op(obj,Type,n)
    return fun

def getOp(obj,Type):
    @property
    def fun():
        return getattr(obj._parent,Type)(obj._n)
    return fun

#%% Class to store the spin system
class SpinOp:
    """
    Here we create a class designed for generating spin systems, similar to the 
    previous n_spin_system in matlab
    """
    def __init__(self,S=None,N=None):
        """
        Initializes the class. Provide either a list of the spins in the system,
        or set n, the number of spins in the system (all spin 1/2)
        """
        
        if N is not None:
            self.__S=np.ones(N)*1/2
        elif hasattr(S,'__len__'):
            self.__S=np.array(S)
        else:
            self.__S=np.ones(1)*S
    
        self.__Mult=(2*self.__S+1).astype(int)
        self.__N=len(self.S)
        self.__index=-1
    
        for k in globals().keys():
            if 'so_' in k:
                Type=k[3:]
                setattr(self,'__'+Type,[None for _ in range(self.__N)])
                setattr(self,Type,MethodType(Op_fun(Type),self))
    def __getitem__(self,n):
        "Returns the single spins of the spin system"
        return OneSpin(self,n)
    def __next__(self):
        self.__index+=1
        if self.__index==self.__N:
            self.__index==-1
            raise StopIteration
        else:
            return OneSpin(self,self.__index)
    def __iter__(self):
        return self
        
    @property
    def S(self):
        return self.__S.copy()    
    @property
    def N(self):
        return self.__N
    @property
    def Mult(self):
        return self.__Mult
        
class OneSpin:
    """Object for containing spin matrices just for one spin (but of a larger
    overall spin system)
    """
    def __init__(self,spinops,n):
        self._parent=spinops
        self._n=n
        
"Here, we add all the required properties to OneSpin"        
for Type in dir(SpinOp(N=1)):
    if 'so_'+Type in globals().keys():
        setattr(OneSpin,Type,property(lambda self,t=Type:getattr(self._parent,t)(self._n)))


#%% Class to store the information about the spin system

        
class SpinSys():
    """
    Stores various information about the spin system. Initialize with a list of
    all nuclei in the spin system.
    """
    def __init__(self,*Nucs):
        self.Nucs=np.atleast_1d(Nucs).squeeze()
        self.N=len(self.Nucs)
        self.S=np.array([NucInfo(nuc,'spin') for nuc in self.Nucs])
        self.gamma=np.array([NucInfo(nuc,'gyro') for nuc in self.Nucs])
        self.Op=SpinOp(self.S)
        self.inter=dict()
        
        for k in globals().keys():
            if 'int_' in k:
                code=globals()[k].__code__
                args=code.co_varnames[:code.co_argcount]  
                self.inter[k[4:]]={a:np.zeros([self.N,self.N]) for a in args[2:]} \
                    if ('S2' in args) else {a:np.zeros(self.N) for a in args[1:]}
                
        
    def set_inter(self,Type,n1,n2=None,**kwargs):
        "Set the parameters for a given interaction"
        for Type,value in kwargs.items():
            self.inter['Type'][n1,n2]=value

#%%

#%% Class to store Hamiltonians
class Hamiltonian():
    """
    Stores a Hamiltonian, and returns its value for a particular orientation
    """
    def __init__(self,H,channels=None):
        """
        Initializes the Hamiltonian. requirements are A, a matrix containing
        the rotating components of the Hamiltonizn (Nx3), and Op, the spin
        operator corresponding to the interaction.
        """
        
        self.channels=channels
        self.H=H
        
    def __call__(self,q,gamma=0,v1=None,offset=None):
        """
        Returns the Hamiltonian for the qth element of the powder average, 
        including an additional gamma rotation
        """
        
        
        return 
#%% Calculate rotating components of interactions 
def int_dipolar(S1,S2,delta,*euler):
    """
    Calculates the dipolar interaction
    """
    pass

def int_J(S1,S2,J):
    """
    Calucates the J interaction
    """