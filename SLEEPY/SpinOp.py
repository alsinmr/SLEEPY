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
        
        self._shape=self.Mult.prod(),self.Mult.prod()
        self._state_index=None
        
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
    
    @property
    def shape(self):
        return self._shape
    
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
    
    @property
    def Llabels(self):
        """
        State labels (Ialpha*Sz, for example). Provided as a list

        Returns
        -------
        None.

        """
        if not(np.all(self.Mult==2)):
            return None
        
        labels=[]
        N=len(self.Mult)
        for index in self.state_index:
            labels.append('')
            for s,i in enumerate(index):
                labels[-1]=labels[-1]+[rf'S$_{s}^\alpha$',rf'S$_{s}^+$',rf'S$_{s}^-$',rf'S$_{s}^\beta$'][i]
                
        return labels
            
    @property
    def Hlabels(self):
        """
        State labels (Ialpha*Sz, for example). Provided as a list

        Returns
        -------
        None.

        """
        if not(np.all(self.Mult==2)):
            return None
        
        labels=[]
        N=len(self.Mult)
        for k in range(np.prod(self.Mult)):
            labels.append('')
            for q in range(N):
                mult=self.Mult[N-q-1]
                i=np.mod(k,mult)
                k-=i
                k//=mult
                if np.mod(mult,2):
                    start=(mult-1)//2
                    labels[-1]=f'{start-i}'+labels[-1]
                else:
                    start=mult//2
                    if start-2*i<0:
                        labels[-1]=rf'$-\dfrac{{{np.abs(start-2*i)}}}{{2}}$,'+labels[-1]
                    else:
                        labels[-1]=rf'$\dfrac{{{start-2*i}}}{{2}}$,'+labels[-1]
            labels[-1]=labels[-1][:-1]
                
        return labels
    
    @property
    def state_index(self):
        """
        Returns the index (0:mult) for each spin state as a function of the
        overall state        
        e.g.
        spin state (1/2):
            0:Ialpha
            1:I+
            2:I-
            3:Ibeta

        Returns
        -------
        np.array

        """
        mult=self.Mult**2
        N=len(mult)
        i=np.arange(mult.prod())
        
        index=np.zeros([mult.prod(),N],dtype=int)
        for i0 in i:
            mat=np.ones([1,1],dtype=bool)
            i00=i0
            index0=[]
            for k in range(N):
                index0.append(np.mod(i00,mult[k]))
                i00=i00-index0[-1]
                i00//=mult[k]
                mat0=np.zeros(mult[k],dtype=bool)
                mat0[index0[-1]]=1
                
                mat=np.kron(mat,mat0.reshape([self.Mult[k],self.Mult[k]]))
                
            index[mat.reshape(mult.prod())]=index0
        super().__setattr__('_state_index', index)
        
            
        return self._state_index
            
            
    
        
        
    
from copy import copy    
class OneSpin():
    def __init__(self,S,n):
        self._M=(2*S+1).astype(int)
        self._n=n
        
        for k in dir(Op0):
            if 'so_' in k:
                Type=k[3:]
                Mult=(S*2+1).astype(int)
                op0=getattr(Op0,'so_'+Type)(S[n])
                op=(np.kron(np.kron(np.eye(Mult[:n].prod(),dtype=Defaults['ctype']),op0),
                            np.eye(Mult[n+1:].prod(),dtype=Defaults['ctype'])))
                setattr(self,Type,op)
        self.T=SphericalTensor(self)
        self._co=None
        
    @property
    def S(self):
        return (self._M[self._n]-1)/2
    @property
    def M(self):
        return self._M[self._n]
    
    def __getattribute__(self, name):
        if name=='T':
            return super().__getattribute__(name)
        return copy(super().__getattribute__(name))  #Ensure we don't modify the original object
    
    @property
    def coherence_order(self):
        if self._co is None:
            co=np.zeros([self.M,self.M],dtype=int)
            for k in range(-int(2*self.S),int(2*self.S+1)):
                co+=k*np.diag(np.ones(self.M-np.abs(k),dtype=int),k=k)
            co=(np.kron(np.kron(np.eye(self._M[:self._n].prod(),dtype=int),co),
                        np.eye(self._M[self._n+1:].prod(),dtype=int)))
            self._co=co.reshape(co.size)
        return self._co
            
    

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
                if Op.S>0.5:
                    self._T.append([1/2*Op.m@Op.m,
                                -1/2*(Op.p@Op.z+Op.z@Op.p),
                                1/np.sqrt(6)*(2*Op.z@Op.z-(Op.x@Op.x+Op.y@Op.y)),
                                1/2*(Op.m@Op.z+Op.z@Op.m),
                                1/2*Op.p@Op.p])
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
        
        
        