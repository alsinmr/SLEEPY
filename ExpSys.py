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
from .Hamiltonian import RF,Hamiltonian
from .Liouvillian import Liouvillian
import matplotlib.pyplot as plt


inter_types=dict()
for k in dir(HamTypes):
    fun=getattr(HamTypes,k)
    if hasattr(fun,'__code__') and fun.__code__.co_varnames[0]=='es':
        inter_types[k]=fun.__code__.co_varnames[1:fun.__code__.co_argcount]

class ExpSys():
    """
    Stores various information about the spin system. Initialize with a list of
    all nuclei in the spin system.
    """
    _iso_powder=PowderAvg('alpha0beta0')
    inter_types=inter_types
    
    def __init__(self,v0H=None,B0=None,Nucs=[],vr=10000,T_K=298,rotor_angle:float=None,
                 n_gamma=100,pwdavg=PowderAvg(),LF:list=None,gamma_encoded:bool=None):
        
        self._ex0=None
        
        if rotor_angle is None:
            rotor_angle=0 if vr==0 else np.arccos(np.sqrt(1/3))
        if vr==0:n_gamma=1
        
        assert B0 is not None or v0H is not None,"B0 or v0H must be specified"
        self.B0=B0 if B0 is not None else v0H*1e6/NucInfo('1H')
        
        self.Nucs=np.atleast_1d(Nucs).astype('<U5')
        Nucs=np.atleast_1d(Nucs)
        self.Nucs[np.array([nuc[0]=='e' for nuc in self.Nucs])]='e-'
        S=np.array([ElectronSpin(nuc) if nuc[0]=='e' else NucInfo(nuc,'spin') for nuc in Nucs])
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
        
        if isinstance(pwdavg,str):
            pwdavg=PowderAvg(pwdavg,gamma_encoded=gamma_encoded)
        elif isinstance(pwdavg,int):
            pwdavg=PowderAvg(q=pwdavg,gamma_encoded=gamma_encoded)
        self.pwdavg=pwdavg
        self.n_gamma=n_gamma
        self.inter=[]
        self._rf=RF(expsys=self)
        self._tprop=0
        
        
        # self.inter_types=dict()
                    
        # for k in dir(HamTypes):
        #     fun=getattr(HamTypes,k)
        #     if hasattr(fun,'__code__') and fun.__code__.co_varnames[0]=='es':
        #         self.inter_types[k]=fun.__code__.co_varnames[1:fun.__code__.co_argcount]
                # setattr(self,k,[])
    
    @property
    def pwdavg(self):
        if self._ex0 is not None:
            return self._ex0.pwdavg
        return self._pwdavg
    
    @pwdavg.setter
    def pwdavg(self,pwdavg):
        self._pwdavg=pwdavg
    
    @property
    def n_gamma(self):
        return self._n_gamma
    
    @n_gamma.setter
    def n_gamma(self,n_gamma):
        self._n_gamma=n_gamma
        
        self.pwdavg.n_gamma=n_gamma
    
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
        if self.vr==0:return
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
    
    @property
    def current_time(self):
        return self._tprop
    
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
        out._ex0=self
        return out
    
    def Hamiltonian(self):
        """
        Creates a Hamiltonian from the ExpSys object.

        Returns
        -------
        Hamiltonian.

        """
        return Hamiltonian(self)
        
    def Liouvillian(self):
        """
        Creates a Liouvillian from the ExpSys Object

        Returns
        -------
        Liouvillian
            Liouvillian for this ExpSys object

        """
        return Liouvillian(self)
                
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
        
        # getattr(self,Type).append(kwargs)
        self.inter.append({'Type':Type,**kwargs})
        
        return self
    
    def plot_inter(self,i:int,n:int=0,ax=None):
        """
        Creates 3D scatter plots representing the size of the z-component of the
        interaction as a function of orientation. Helpful in visualizing
        tensors, and also in determining how well a powder average may deal
        with a given interaction.
        
        Provide the index of the interaction, an optionally a 3D axis onto
        which the tensor will be plotted.
        

        Parameters
        ----------
        i : int
            Index of the desired interaction.
        n : int
            Component (-2,-1,0,1,2) of the interaction to view. Default is 0
        ax : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        axis

        """
        
        H=self.Hamiltonian()
        H[0]  #Index once to finish initialization
        
        if ax is None:ax=plt.figure().add_subplot(1,1,1,projection='3d')
        
        if H.Hinter[i].rotInter is None:
            A=H.Hinter[i].avg
            if n:A=0
            alpha=self.pwdavg.alpha
            beta=self.pwdavg.beta
            if not(self.pwdavg._gamma_incl):
                i=self.pwdavg.gamma==0
                alpha=alpha[i]
                beta=beta[i]
            
            x=np.cos(alpha)*np.sin(beta)*A
            y=np.sin(alpha)*np.sin(beta)*A
            z=np.cos(beta)*A
            
            ax.scatter3D(x,y,z,linewidth=0.2,antialiased=True,color='red' if A>0 else 'blue')
            ax.set_box_aspect((1,1,1))
            ax.set_xlabel('Hz')
            ax.set_ylabel('Hz')
            ax.set_zlabel('Hz')
            ax.set_title(f'n={n}')
            return ax
            
        H.Hinter[i].rotInter.plot(avg=H.Hinter[i].avg,n=n,ax=ax)
        ax.set_title(f'n={n}')
        return ax
        
        
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
            self._index=-1
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
                    ','.join([f'{key}={ef(value)}' if key=='euler' else f'{key}={value:.2f}' for key,value in dct.items()])+')\n'
            else:
                out+=f'\t{dct.pop("Type")} between spins {dct.pop("i0")},{dct.pop("i1")} with arguments:\n\t\t('+\
                    ','.join([f'{key}={ef(value)}' if key=="euler" else f'{key}={value:.2f}' for key,value in dct.items()])+')\n'
        out+='\n'+super().__repr__()    
        return out
    
#Here we add some functionality to ExpSys dynamically
        
def list_inter_pars(self,k):
    out=[]
    for inter in self:
        if inter['Type']==k:
            out.append(inter)
    return out
    
for k in dir(HamTypes):
    fun=getattr(HamTypes,k)
    if hasattr(fun,'__code__') and fun.__code__.co_varnames[0]=='es':
        setattr(ExpSys,k,property(lambda self,k=k:list_inter_pars(self,k)))
        
        
def ElectronSpin(string):
    """
    We do not have a mechanism to easily change the spin of a nucleus (obviously),
    but for an electron, this is not so uncommon. We allow the specification 
    within the nucleus string such as:
        
        e1 : spin 1 electron (or e-1)
        e-5/2 : spin 5/2 electron (or e5/2)
        e1.5 : spin 3/2 electron (or e-1.5)

    Parameters
    ----------
    string : TYPE
        DESCRIPTION.

    Returns
    -------
    float
        Electron spin

    """
    string=string[1:] #Remove the e
    if not(string):return 0.5  #If the string is empty at this point, then default to spin 1/2
    if string[0]=='-':string=string[1:] #Remove the - if included
    
    if not(string):return 0.5  #If the string is empty at this point, then default to spin 1/2
    if string[-2:]=='.5':return float(string)  #1/2 integer spin, specified with a decimal
    
    if string[-2:]=='/2':   #Specified with a fraction
        return int(string[:-2])/2
    
    return float(string)
        
            