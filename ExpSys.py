#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 11:22:59 2023

@author: albertsmith
"""

import numpy as np
from Tools import NucInfo
from SpinOperator import SpinOp
import HamTypes


class ExpSys():
    """
    Stores various information about the spin system. Initialize with a list of
    all nuclei in the spin system.
    """
    def __init__(self,v0H,Nucs,vr=10000,rotor_angle=np.arccos(np.sqrt(1/3)),n_gamma=100):
        self.v0H=v0H
        self.B0=self.v0H*1e6/NucInfo('1H')
        self.Nucs=np.atleast_1d(Nucs).squeeze()
        self.N=len(self.Nucs)
        self.S=np.array([NucInfo(nuc,'spin') for nuc in self.Nucs])
        self.gamma=np.array([NucInfo(nuc,'gyro') for nuc in self.Nucs])
        self.Op=SpinOp(self.S)
        self._index=-1
        self.vr=vr
        self.rotor_angle=rotor_angle
        self.n_gamma=n_gamma
        self.inter=[]
        
        self.inter_types=dict()
                    
        for k in dir(HamTypes):
            fun=getattr(HamTypes,k)
            if hasattr(fun,'__code__') and fun.__code__.co_varnames[0]=='es':
                self.inter_types[k]=fun.__code__.co_varnames[1:fun.__code__.co_argcount]
                setattr(self,k,[])
    
    
    @property
    def v0(self):
        return self.B0*self.gamma
    
                
    def set_inter(self,Type,**kwargs):
        """
        Adds an interaction to the total Hamiltonian. We list the required arguments
        for each type of interaction
        
        Spin-Field:
            Isotropic: i1 (spin index) and value (chemical shift in ppm)
            Anisotropic: i1 (spin index) and anisotropy (delta, in ppm). Asymmetry
                and Euler angles also optional
        Spin-Spin:
            Isotropic: i1,i2 (spin indices) and value (J coupling gin Hz)
            Anisotropic: i1,i2 (spin indices) and anisotropy (delta, in Hz. Asymmetry
                and Euler angles also optional
        """
        
        assert Type in self.inter_types.keys(),"Unknown interaction type"
        
        getattr(self,Type).append(kwargs)
        self.inter.append({'Type':Type,**kwargs})
        

    def get_abs_index(self,n=None,Type=None,i1=None,i2=None):
        """
        Get the absolute index of a given interaction
        """
        
        if Type is None:
            assert n<self.__Ninter,"n must be less than the number of defined interactions ({0})".format(self.__Ninter)
            return n
        else:
            assert Type in self.inter_types.keys(),'Interaction {0} is not defined'.format(Type)
            n0=0
            for t in self.inter_types.keys():
                if Type==t:
                    break
                else:
                    n0+=len(getattr(self,t))
            if n is None:
                if i2 is not None:i1,i2=np.sort([i1,i2])
                for k,m in enumerate(getattr(self,Type)):
                    if ('i2' in m.keys() and i1==m['i1'] and i2==m['i2']) or i1==m['i1']:
                        return n0+k
                else:
                    assert False,"Interaction {0} with given indices not defined"
            else:
                assert n0+n<self.__Ninter,""
                return n0+n
                    
    
    def __getitem__(self,n):
        """
        Returns parameters for the nth interaction. Indexing sweeps through 
        each interaction type sequentially
        """
        
        return self.inter[n%len(self)]
            
    def __len__(self):
        return len(self.inter)
    
    def __next__(self):
        self._index+=1
        if self._index==len(self):
            self._index==-1
            raise StopIteration
        else:
            return self[self._index]
    def __iter__(self):
        self._index=-1
        return self
    
    def remove_inter(self,i=None,**kwargs):
        """
        Removes interaction index "i"
        
        --or--
        
        Removes all interactions for which the arguments given in kwargs match
        the values stored for the interactions. For example,
        
        expsys.remove_inter(Type='dipole') 
        
        will remove all dipole interactions.
        
        expsys.remove_inter(Type='dipole',i1=0) 
        
        will remove all dipole interactions where the i1 index is 0, and
        
        expsys.remove_inter(Type='dipole',i1=0,i2=1)
        
        will remove the specific dipole interation between spins 1 and 2.
        Note that if i1 and i2 are switches for the stored interaction, then
        this will not be removed.
        
        
        If remove_inter is called without arguments, then all interactions will
        be removed

        Parameters
        ----------
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
   
        if i is not None:
            self.inter.pop(i)
            return
        
        index=np.ones(len(self),dtype=bool)
        for k,v in kwargs.items():
            index&=[k in i and i[k]==v for i in self.inter]
            
        for i in np.argwhere(index)[0][::-1]:
            self.inter.pop(i)
            
        
            