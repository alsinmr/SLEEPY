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
    def __getattribute__(self, name):
        return copy(super().__getattribute__(name))  #Ensure we don't modify the original object
        