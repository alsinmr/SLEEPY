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
                op=(np.kron(np.kron(np.eye(Mult[:n].prod(),dtype=Defaults['ctype']),op0),
                            np.eye(Mult[n+1:].prod(),dtype=Defaults['ctype'])))
                setattr(self,Type,op)
        self.T=SphericalTensor(self)
    def __getattribute__(self, name):
        if name=='T':
            return super().__getattribute__(name)
        return copy(super().__getattribute__(name))  #Ensure we don't modify the original object

#Doesn't work below. Not sure why
# class OpMat(np.ndarray):
#     def __new__(cls, x):
#         return super().__new__(cls, shape=x.shape,dtype=x.dtype)
#     def __init__(self,x):
#         self[:]=x
#     def __mul__(self,x):
#         if x.ndim==2 and x.shape[0]==x.shape[1] and x.shape[0]==self.shape[0]:
#             return self@x
#         else:
#             return self*x
    
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
        
        self._T=None
        
        # if Op1 is None:
        #     # M=np.round(2*S0+1).astype(int)
        #     self._T=[None for _ in range(2)]
        #     self._T[0]=[Op0.eye]
        #     self._T[1]=[-1/np.sqrt(2)*Op0.p,Op0.z,1/np.sqrt(2)*Op0.m]
        #     return
        # else:
        #     self._T=[None for _ in range(3)]
        #     self._T[0]=[-1/np.sqrt(3)*(Op0.x@Op1.x+Op0.y@Op1.y+Op0.z*Op1.z)]
            
        #     self._T[1]=[-1/2*(Op0.m@Op1.z-Op0.z@Op1.m),
        #                 -1/(2*np.sqrt(2))*(Op0.p@Op1.m-Op0.m@Op1.p),
        #                 -1/2*(Op0.p@Op1.z-Op0.z@Op1.p)]
            
        #     self._T[2]=[1/2*Op0.m@Op1.m,
        #                 -1/2*(Op0.p@Op1.z+Op0.z@Op1.p),
        #                 1/np.sqrt(6)*(2*Op0.z@Op1.z-(Op0.x@Op1.x+Op0.y@Op1.y)),
        #                 1/2*(Op0.m@Op1.z+Op0.z@Op1.m),
        #                 1/2*Op0.p@Op1.p]
            
    def set_mode(self,mode:str=None):
        """
        Sets the type of rank-2 sphereical tensors to return. Options are as 
        follows:
            
            If Op1 is not defined (1 spin):
            '1spin': Rank-1, 1 spin tensors (default)
            'B0_LF': Interaction between field and spin. 
            
            If Op1 is defined (2-spin)
            'LF_LF': Full rank-2 tensors in the lab frame (default)
            'LF_RF': First spin in lab frame, second spin in rotating frame
            'RF_LF': First spin in rotating frame, second spin in lab frame
            'het'  : Both spins in rotating frame. Heteronuclear coupling
            'homo' : Both spins in rotating frame. Homonuclear coupling

        Parameters
        ----------
        mode : str, optional
            DESCRIPTION. The default is 'LF_LF'.

        Returns
        -------
        None.

        """
        
        if self._Op1 is None:
            Op=self._Op0
            if mode is None:mode='1spin'
            assert mode in ['1spin','B0_LF'],'1-spin modes are 1spin and B0_LF'
            if mode=='1spin':
                self._T=[None for _ in range(2)]
                # print('checkpoint')
                self._T[0]=[Op.eye]
                self._T[1]=[-1/np.sqrt(2)*Op.p,Op.z,1/np.sqrt(2)*Op.m]
            elif mode=='B0_LF':
                zero=np.zeros(Op.eye.shape)
                self._T=[None for _ in range(3)]
                self._T[0]=[-1/np.sqrt(3)*Op.z]
                self._T[1]=[-1/2*Op.m,zero,-1/2*Op.p] #Not really convinced about sign here
                self._T[2]=[zero,1/2*Op.m,np.sqrt(2/3)*Op.z,-1/2*Op.p,zero]
        else:
            if mode is None:mode='LF_LF'
            assert mode in ['LF_LF','LF_RF','RF_LF','het','homo'],'2-spin modes are LF_LF,LF_RF,RF_LF,het, and homo'
            Op0,Op1=self._Op0,self._Op1
            
            self._T=[None for _ in range(3)]
            if mode=='LF_LF':
                self._T[0]=[-1/np.sqrt(3)*(Op0.x@Op1.x+Op0.y@Op1.y+Op0.z*Op1.z)]
                
                self._T[1]=[-1/2*(Op0.m@Op1.z-Op0.z@Op1.m),
                            -1/(2*np.sqrt(2))*(Op0.p@Op1.m-Op0.m@Op1.p),
                            -1/2*(Op0.p@Op1.z-Op0.z@Op1.p)]
                
                self._T[2]=[1/2*Op0.m@Op1.m,
                            -1/2*(Op0.p@Op1.z+Op0.z@Op1.p),
                            1/np.sqrt(6)*(2*Op0.z@Op1.z-(Op0.x@Op1.x+Op0.y@Op1.y)),
                            1/2*(Op0.m@Op1.z+Op0.z@Op1.m),
                            1/2*Op0.p@Op1.p]
            elif mode=='LF_RF':
                zero=np.zeros(Op0.eye.shape)
                self._T[0]=[-1/np.sqrt(3)*(Op0.z*Op1.z)]
                
                self._T[1]=[-1/2*(Op0.m@Op1.z),
                            zero,
                            -1/2*(Op0.p@Op1.z)]
                
                self._T[2]=[zero,
                            -1/2*(Op0.p@Op1.z),
                            1/np.sqrt(6)*(2*Op0.z@Op1.z),
                            1/2*(Op0.m@Op1.z),
                            zero]
            elif mode=='RF_LF':
                zero=np.zeros(Op0.eye.shape)
                self._T[0]=[-1/np.sqrt(3)*(Op0.z*Op1.z)]
                
                self._T[1]=[-1/2*(-Op0.z@Op1.m),
                            zero,
                            -1/2*(-Op0.z@Op1.p)]
                
                self._T[2]=[zero,
                            -1/2*(Op0.z@Op1.p),
                            1/np.sqrt(6)*(2*Op0.z@Op1.z),
                            1/2*(Op0.z@Op1.m),
                            zero]
            elif mode=='het':
                zero=np.zeros(Op0.eye.shape)
                self._T[0]=[-1/np.sqrt(3)*(Op0.z*Op1.z)]
                
                self._T[1]=[zero,
                            -1/(2*np.sqrt(2))*(Op0.p@Op1.m-Op0.m@Op1.p),
                            zero]
                
                self._T[2]=[zero,
                            zero,
                            1/np.sqrt(6)*(2*Op0.z@Op1.z),
                            zero,
                            zero]
            elif mode=='homo':
                zero=np.zeros(Op0.eye.shape)
                self._T[0]=[-1/np.sqrt(3)*(Op0.x@Op1.x+Op0.y@Op1.y+Op0.z*Op1.z)]
                
                self._T[1]=[zero,
                            -1/(2*np.sqrt(2))*(Op0.p@Op1.m-Op0.m@Op1.p),
                            zero]
                
                self._T[2]=[zero,
                            zero,
                            1/np.sqrt(6)*(2*Op0.z@Op1.z-(Op0.x@Op1.x+Op0.y@Op1.y)),
                            zero,
                            zero]
            else:
                assert 0,'Unknown mode'
                
                
                    
                
            
            
        
    def __getitem__(self,index):
        if self._T is None:self.set_mode()
        if isinstance(index,int):
            return self._T[index]
        assert isinstance(index,tuple) and len(index)==2,"Spherical tensors should be accessed with one element rank index or a 2-element tuple"
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
        
        
        