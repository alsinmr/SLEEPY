#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 11:49:00 2023

@author: albertsmith
"""

import HamTypes
from copy import copy
from PowderAvg import PowderAvg
from Tools import Ham2Super
import numpy as np

dtype=np.complex64

class Hamiltonian():
    def __init__(self,expsys,pwdavg=PowderAvg(),rf=None):
        self._expsys=expsys
        self.pwdavg=pwdavg
        self.rf=rf
        if rf is not None:self.rf.expsys=expsys

        
        
        #Attach Hamiltonians for each interaction
        self.Hinter=list()
        isotropic=True
        for i in self.expsys:
            dct=i.copy()
            Ham=getattr(HamTypes,dct.pop('Type'))(expsys,**dct)
            isotropic&=Ham.isotropic
            self.Hinter.append(Ham)
        if isotropic:
            self.pwdavg=None #Delete the powder average if unused
        for Ham in self.Hinter:Ham.pwdavg=self.pwdavg #Share the powder average
            
            
        self.sub=False
        self._index=-1
        
        self._initialized=True
    
    @property
    def isotropic(self):
        return self.pwdavg is None
    
    @property
    def expsys(self):
        return self._expsys
    
    def __setattr__(self,name,value):
        if hasattr(self,'_initialized') and self._initialized and \
            name not in ['_initialized','_index','sub']:
            print('Hamiltonian cannot be edited after initialization!')
        else:
            super().__setattr__(name,value)
            
        
    def __getitem__(self,i:int):
        """
        Returns the ith element of the powder average, yielding terms Hn for the
        total Hamiltonian

        Parameters
        ----------
        i : int
            Element of the powder average.

        Returns
        -------
        None.

        """
        i%=len(self)
        out=copy(self)
        for k,H0 in enumerate(self.Hinter):
            out.Hinter[k]=H0[i]
        out._index=i
        out.sub=True
        return out
    
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
    
    def Hn(self,n:int):
        """
        Returns the nth rotating component of the total Hamiltonian

        Parameters
        ----------
        n : int
            Component (-2,-1,0,1,2)

        Returns
        -------
        np.array

        """
        assert self.sub or self.pwdavg is None,'Calling Hn requires indexing to a specific element of the powder average'
        
        
        out=None
        for Hinter in self.Hinter:
            if out is None:
                out=Hinter.Hn(n)
            else:
                out+=Hinter.Hn(n)
                
        if n==0 and self.rf is not None:
            out+=self.rf()
                
        return out
    
    @property
    def shape(self):
        """
        Shape of the Hamiltonian to be returned

        Returns
        -------
        tuple

        """
        return self.expsys.Op.Mult.prod(),self.expsys.Op.Mult.prod()
        
    
    def Ln(self,n:int):
        """
        Returns the nth rotation component of the Liouvillian

        Parameters
        ----------
        n : int
            Component (-2,-1,0,1,2)

        Returns
        -------
        np.array

        """
        
        return Ham2Super(self.Hn(n))
    
    
class RF():
    def __init__(self,fields,expsys=None):
        """
        Generates an RF Hamiltonian for a given expsys. Expsys can be provided
        after initialization, noting that RF will not be callable until it is
        provided.
        
        Fields is a dictionary and allows us to either define fields by channel
        ('1H','13C', etc)or by index (0,1,2) to apply to a specific spin. The
        channel or index is a dictionary key. The latter approach is 
        unphysical for a real experiment, but allows one to consider, for example,
        selective behavior without actually designing a selective pulse.
        
        fields={'13C':[50000,0,5000],'1H':[10000,0,0]}
        Applied fields on both 13C and 1H, with 50 and 10 kHz fields effectively
        Phases are 0 on both channels (radians), and offsets are 5 and 0 kHz,
        respectively.
        
        Alternatively:
        fields={0:[50000,0,5000],1:[10000,0,0]}
        If we just have the two spins (13C,1H), this would produce the same
        result as above. However, this could also allow us to apply different
        fields to the same type of spin.
        
        Note that one may just provide the field strength if desired, and the
        phase/offset will be set to zero
        
        

        Parameters
        ----------
        fields : dict
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        self.fields=fields
        self.expsys=expsys
        
    def __call__(self):
        """
        Returns the Hamiltonian for the RF fields (non-rotating component)

        Returns
        -------
        None.

        """
        assert self.expsys is not None,"expsys must be defined before RF can be called"
        
        n=self.expsys.Op.Mult.prod()
        out=np.zeros([n,n],dtype=dtype)
        
        for name,value in self.fields.items():
            if not(hasattr(value,'__len__')):value=[value,0,0]  #Phase/offset default to zero
            if isinstance(name,str):
                for x,S in zip(self.expsys.Nucs==name,self.expsys.Op):
                    if x:
                        out+=(np.cos(value[1])*S.x+np.sin(value[1])*S.y)*value[0]+value[2]*S.z
            else:
                S=self.expsys.Op[name]
                out+=(np.cos(value[1])*S.x+np.sin(value[1])*S.y)*value[0]+value[2]*S.z
        return out
                        
                
        
        
        
        
    