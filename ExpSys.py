#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 11:22:59 2023

@author: albertsmith
"""

import numpy as np
from .Tools import NucInfo
from .SpinOp import SpinOp
from .PowderAvg import PowderAvg
from . import HamTypes as HamTypes
# import pyRelaxSim.HamTypes as HamTypes
from copy import deepcopy as DC
from copy import copy
from .Hamiltonian import RF

class ExpSys():
    """
    Stores various information about the spin system. Initialize with a list of
    all nuclei in the spin system.
    """
    _iso_powder=PowderAvg('alpha0beta0')
    def __init__(self,v0H=None,B0=None,Nucs=[],vr=10000,T_K=298,rotor_angle=np.arccos(np.sqrt(1/3)),n_gamma=100,pwdavg=PowderAvg(),LF:list=None):
        
        assert B0 is not None or v0H is not None,"B0 or v0H must be specified"
        self.B0=B0 if B0 is not None else v0H*1e6/NucInfo('1H')
        self.Nucs=np.atleast_1d(Nucs)
        S=np.array([NucInfo(nuc,'spin') for nuc in self.Nucs])
        self.gamma=np.array([NucInfo(nuc,'gyro') for nuc in self.Nucs])
        self.Op=SpinOp(S)
        if LF is None or LF is False:
            self.LF=[False for _ in range(len(self.Op))]  #Calculate Hamiltonians in the lab frame
        elif hasattr(LF,'__len__'):
            assert len(LF)==len(self.Op),'LF (Lab frame) must be a list of logicals with same length as number of spins'
            self.LF=LF
        else:
            assert LF is True or LF is False,'LF must be a list of logicals or a single boolean'
            self.LF=[LF for _ in range(len(self.Op))]
        self._index=-1
        self.vr=vr
        self.T_K=T_K
        self.rotor_angle=rotor_angle
        self.n_gamma=n_gamma
        self.pwdavg=pwdavg
        self.inter=[]
        self._rf=RF(expsys=self)
        self._tprop=0
        
        self.inter_types=dict()
                    
        for k in dir(HamTypes):
            fun=getattr(HamTypes,k)
            if hasattr(fun,'__code__') and fun.__code__.co_varnames[0]=='es':
                self.inter_types[k]=fun.__code__.co_varnames[1:fun.__code__.co_argcount]
                setattr(self,k,[])
        
    @property
    def v0H(self):
        return self.B0*NucInfo('1H')
    
    @property
    def v0(self):
        return self.B0*self.gamma
    
    @property
    def S(self):
        return self.Op.S
    
    @property
    def taur(self):
        return 1/self.vr
    
    @property
    def nspins(self):
        return len(self.Op)
    
    def __copy__(self):
        return self.copy()
    
    @property
    def Peq(self):
        """
        Polarization of the individual spins

        Returns
        -------
        None.

        """

        return np.tanh(self.gamma*6.62607015e-34*self.B0/(2*1.380649e-23*self.T_K))
    
    def reset_prop_time(self,t:float=0):
        """
        Resets the current time for propagators to t
        
        (L.expsys._tprop=t)

        Parameters
        ----------
        t : float, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        None.

        """
        self._tprop=t
            
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
                
    def __repr__(self): 
        out=f'{len(self.Nucs)}-spin system ('+','.join([f'{Nuc}' for Nuc in self.Nucs])+')\n'
        out+=f'B0 = {self.B0:.3f} T ({self.v0H/1e6:.3f} MHz 1H frequency)\n'
        out+=f'rotor angle = {self.rotor_angle*180/np.pi:.3f} degrees\n'
        out+=f'rotor frequency = {self.vr/1e3} kHz\n'
        out+=f'Temperature = {self.T_K} K\n'
        out+=self.pwdavg.__repr__().rsplit('\n',2)[0].replace('\nType:\t',': ')
        # out+=f'Powder average with {self.pwdavg.N} angles, {self.n_gamma} steps per rotor period\n'
        out+='\nInteractions:\n'
        
        def ef(euler):
            if hasattr(euler[0],'__iter__'):
                return ','.join([ef(e) for e in euler])
            else:
                return '['+','.join([f'{a*180/np.pi:.2f}' for a in euler])+']'
        
        for i in self.inter:
            dct=copy(i)
            if 'i' in dct:
                out+=f'\t{dct.pop("Type")} on spin {dct.pop("i")} with arguments: ('+\
                    ','.join([f'{key}={ef(value) if key=="euler" else value}' for key,value in dct.items()])+')\n'
            else:
                out+=f'\t{dct.pop("Type")} between spins {dct.pop("i0")},{dct.pop("i1")} with arguments:\n\t\t('+\
                    ','.join([f'{key}={ef(value) if key=="euler" else value}' for key,value in dct.items()])+')\n'
        out+='\n'+super().__repr__()    
        return out
        
        
        
            