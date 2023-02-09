#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 10:38:34 2023

@author: albertsmith
"""

import numpy as np
from . import Operators as Op0
from . import Defaults


dtype=np.complex64

class SpinOp:
    def __init__(self,S:list=None,N:int=None):
        """
        Generates and contains the spin operators for a spin system of arbitrary
        size. Provide the number of spins (N) if all spin-1/2. Otherwise, list
        the desired spin-system. Note the the spin system is locked after
        initial generation.

        Parameters
        ----------
        S : list, optional
            DESCRIPTION. The default is None.
        N : int, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        
        
        assert S is not None or N is not None,'Either S or N must be defined to initiate SpinOp'
        if S is not None:
            if not(hasattr(S,'__len__')):S=[S]
            self.Mult=(np.array(S)*2+1).astype(int)
        elif N is not None:
            self.Mult=(np.ones(N)*2).astype(int)
            
        self._OneSpin=[OneSpin(self.S,n) for n in range(len(self))]
        
        self._index=-1
        
        self._initialized=True
            
    def __setattr__(self,name,value):
        if hasattr(self,'_initialized') and self._initialized and \
            name not in ['_initialized','_index']:
            print('SpinOp cannot be edited after initialization!')
        else:
            super().__setattr__(name,value)

    @property
    def S(self):
        return (self.Mult-1)/2

    @property
    def N(self):
        return len(self.S)
    
    def __len__(self):
        return self.N
    
    def __getitem__(self,i):
        return self._OneSpin[i%len(self)]
    
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
    
from copy import copy    
class OneSpin():
    def __init__(self,S,n):
        for k in dir(Op0):
            if 'so_' in k:
                Type=k[3:]
                Mult=(S*2+1).astype(int)
                op0=getattr(Op0,'so_'+Type)(S[n])
                op=np.kron(np.kron(np.eye(Mult[:n].prod(),dtype=Defaults['ctype']),op0),
                            np.eye(Mult[n+1:].prod(),dtype=Defaults['ctype']))
                setattr(self,Type,op)
        self.T=SphericalTensor(self)
    def __getattribute__(self, name):
        return copy(super().__getattribute__(name))  #Ensure we don't modify the original object
    
class SphericalTensor():
    def __init__(self,Op0,S0:float=None,Op1=None):
        """
        Initialize spherical tensors for one or two spins. Note that we only
        calculate up to rank 2.

        Parameters
        ----------
        Op0 : OneSpin
            One-spin spin operator .
        Op1 : OneSpin, optional
            One-spin operator for a second spin. Use for calculating the tensor
            product. The default is None.

        Returns
        -------
        None.

        """
        
        self._Op0=Op0
        self._Op1=Op1
        self._S0=S0
        
        if Op1 is None:
            # M=np.round(2*S0+1).astype(int)
            self._T=[None for _ in range(2)]
            self._T[0]=[Op0.eye]
            self._T[1]=[-1/np.sqrt(2)*Op0.p,Op0.z,1/np.sqrt(2)*Op0.m]
            return
        else:
            self._T=[None for _ in range(3)]
            self._T[0]=[-1/np.sqrt(3)*(Op0.x@Op1.x+Op0.y@Op1.y+Op0.z*Op1.z)]
            
            self._T[1]=[-1/2*(Op0.m@Op1.z-Op0.z@Op1.m),
                        -1/(2*np.sqrt(2))*(Op0.p@Op1.m-Op0.m@Op1.p),
                        -1/2*(Op0.p@Op1.z-Op0.z@Op1.p)]
            
            self._T[2]=[1/2*Op0.m@Op1.m,
                        -1/2*(Op0.p@Op1.z+Op0.z@Op1.p),
                        1/np.sqrt(6)*(2*Op0.z@Op1.z-(Op0.x@Op1.x+Op0.y@Op1.y)),
                        1/2*(Op0.m@Op1.z+Op0.z@Op1.m),
                        1/2*Op0.p@Op1.p]
            
        
    def __getitem__(self,index):
        assert isinstance(index,tuple) and len(index)==2,"Spherical tensors should be accessed with a 2-element tuple"
        rank,comp=index
        assert rank<len(self._T),f"This spherical tensor object only contains objects up to rank {len(self._T)-1}"
        assert np.abs(comp)<=rank,f"|comp| cannot be greater than rank ({rank})"
        
        return self._T[rank][comp+rank]
        
    def __mul__(self,T):
        """
        Returns the Tensor product for two tensors (up to rank-2 components)

        Parameters
        ----------
        T : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        assert str(T.__class__).rsplit('.',maxsplit=1)[1].split("'")[0]=='SphericalTensor',"Tensor product only defined between two spherical tensors"
        return SphericalTensor(Op0=self._Op0,Op1=T._Op0)
        
        
        