#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 11:22:59 2023

@author: albertsmith
"""

import numpy as np
from Tools import NucInfo
from SpinOp import SpinOp
from PowderAvg import PowderAvg
import HamTypes
from copy import deepcopy as DC
from copy import copy
from Hamiltonian import RF

class ExpSys():
    """
    Stores various information about the spin system. Initialize with a list of
    all nuclei in the spin system.
    """
    _iso_powder=PowderAvg('alpha0beta0')
    def __init__(self,v0H,Nucs,vr=10000,rotor_angle=np.arccos(np.sqrt(1/3)),n_gamma=100,pwdavg=PowderAvg()):
        
        self.v0H=v0H
        self.B0=self.v0H*1e6/NucInfo('1H')
        self.Nucs=np.atleast_1d(Nucs)
        self.N=len(self.Nucs)
        self.S=np.array([NucInfo(nuc,'spin') for nuc in self.Nucs])
        self.gamma=np.array([NucInfo(nuc,'gyro') for nuc in self.Nucs])
        self.Op=SpinOp(self.S)
        self._index=-1
        self.vr=vr
        self.rotor_angle=rotor_angle
        self.n_gamma=n_gamma
        self.pwdavg=pwdavg
        self.inter=[]
        self._rf=RF(expsys=self)
        
        self.inter_types=dict()
                    
        for k in dir(HamTypes):
            fun=getattr(HamTypes,k)
            if hasattr(fun,'__code__') and fun.__code__.co_varnames[0]=='es':
                self.inter_types[k]=fun.__code__.co_varnames[1:fun.__code__.co_argcount]
                setattr(self,k,[])
    
    
    @property
    def v0(self):
        return self.B0*self.gamma
    
    @property
    def taur(self):
        return 1/self.vr
    
    @property
    def nspins(self):
        return len(self.Op)
    
    
    def __copy__(self):
        return self.copy()
                
    def copy(self,deepcopy:bool=False):
        """
        Return a copy of the ExpSys object. This copy method will use a 
        shallow copy on parameters expect the interactions, which will be 
        deep-copied. This is the ideal behavior for creating a Liouvillian, where
        exchange will leave the field, spin system, gamma, etc. fixed, but 
        will change the interactions.
        
        Setting deepcopy to True will perform a deep copy of all attributes


        Parameters
        ----------
        deepcopy : bool, optional
            Return a deep copy. The default is False.

        Returns
        -------
        ExpSys
            Copy of the Expsys.

        """
        
        if deepcopy:return DC(self)
        
        cls = self.__class__
        out = cls.__new__(cls)
        out.__dict__.update(self.__dict__)
        out.inter=[copy(i) for i in self.inter]
        return out
                
    def set_inter(self,Type,**kwargs):
        """
        Adds an interaction to the total Hamiltonian. 
        """
        
        self.remove_inter(Type=Type,**kwargs)
        
        assert 'i' in kwargs or ('i0' in kwargs and 'i1' in kwargs),"Either i must be provided or both i0 and i1 must be provided"
        
        if 'i0' in kwargs and 'i1' in kwargs:
            i0,i1=kwargs['i0'],kwargs['i1']
            if i0>i1:kwargs['i0'],kwargs['i1']=i1,i0
            assert i0<self.nspins,'i0 must be less than expsys.nspins'
            assert i1<self.nspins,'i1 must be less than expsys.nspins'
            assert i0!=i1,'i0 and i1 cannot be equal'
        else:
            assert kwargs['i']<self.nspins,'i must be less than expsys.nspins'
        
        
        assert Type in self.inter_types.keys(),"Unknown interaction type"
        
        getattr(self,Type).append(kwargs)
        self.inter.append({'Type':Type,**kwargs})
        
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
    
    def remove_inter(self,i=None,Type=None,i0=None,i1=None,**kwargs):
        """
        Removes interaction index "i"
        
        --or--
        
        Removes all interactions by type or by type+index.
        
        
        expsys.remove_inter(i=0)   #Removes the first interaction
        expsys.remove_inter(Type='dipole')  #Removes all dipole couplings
        expsys.remove_inter(Type='dipole',i0=0,i1=1) #Removes dipole coupling between spin 0 and 1
        expsys.remove_inter(Type='CS',i=0)  #Removes CS on spin 0 (note that i is used differently here)
        
        

        Parameters
        ----------
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        if Type is None:
            self.inter.pop(i)
            return
           
        if i0 is not None and i1 is not None and i0>i1:  #Make i0<i1
            i0,i1=i1,i0
        
        index=list()
        if i0 is not None and i1 is not None:
            for inter in self:
                if 'i0' in inter and 'i1' in inter and inter['Type']==Type \
                    and inter['i0']==i0 and inter['i1']==i1:
                    index.append(True)
                else:
                    index.append(False)
        elif i is not None:
            for inter in self:
                if 'i' in inter and inter['Type']==Type and inter['i']==i:
                    index.append(True)
                else:
                    index.append(False)
        else:
            for inter in self:
                if inter['Type']==Type:
                    index.append(True)
                else:
                    index.append(False)
                    
        if np.any(index):
            for i in np.argwhere(index)[0][::-1]:
                self.inter.pop(i)
            
        
            