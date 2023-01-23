#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 11:49:36 2023

@author: albertsmith
"""

import numpy as np
from pyRelaxSim.PowderAvg import RotInter
from copy import copy

dtype=np.complex64

class Ham1inter():
    def __init__(self,M=None,H=None,isotropic=False,delta=0,eta=0,euler=[0,0,0],
                 rotor_angle=np.arccos(np.sqrt(1/3)),info={}):
        self.M=M
        self.H=H
        self.isotropic=isotropic
        self.delta=delta
        self.eta=eta
        self.euler=euler
        self.pwdavg=None
        self.rotInter=None
        self.info=info
        self.rotor_angle=rotor_angle
        
        self.A=None
        
    def __getitem__(self,i:int):
        """
        Get the ith element of the powder average

        Parameters
        ----------
        i : int
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if not(self.isotropic):
            assert self.pwdavg is not None,'pwdavg must be assigned before extracting components of anisotropic Hamiltonians'
            if self.rotInter is None:
                self.rotInter=RotInter(self.pwdavg,delta=self.delta,eta=self.eta,euler=self.euler,rotor_angle=self.rotor_angle)
            out=copy(self)
            out.A=self.rotInter.Azz[i]
            return out
        return self
            
            
    def Hn(self,n=0,t=None):
        """
        Returns components of the Hamiltonian for a given orientation of the 
        powder average. Only works if the orientation has been set by indexing,
        i.e., use
        
        H[k].Hn(0)
        
        to extract these terms.
        
        Parameters
        ----------
        n : int
            Component rotating at n times the rotor frequency (-2,-1,0,1,2)

        Returns
        -------
        np.array
            Hamiltonian for the nth rotation component
        """

        assert n in [-2,-1,0,1,2],'n must be in [-2,-1,0,1,2]'

        if self.isotropic:
            if n==0:
                return self.H
            else:
                return np.zeros(self.H.shape,dtype=dtype)
            
        if self.A is None:
            return None

        if self.H is None:
            return self.M*self.A[n+2]
        else:
            return self.H[n+2]

def dipole(es,i0:int,i1:int,delta:float,eta:float=0,euler=[0,0,0]):
    """
    Dipole Hamiltonian

    Parameters
    ----------
    es : exp_sys
        Experimental system object.
    i0 : int
        index of the first spin.
    i1 : int
        index of the second spin.
    delta : float
        anisotropy of the dipole coupling
    eta   : float
        asymmetry of the dipole coupling (usually 0). Default is 0
    euler : list
        3 elements giving the euler angles for the dipole coupling.
        Default is [0,0,0]

    Returns
    -------
    Ham1inter

    """
    
    S,I=es.Op[i0],es.Op[i1]
    if es.Nucs[i0]==es.Nucs[i1]:
        M=np.sqrt(2/3)*(S.z*I.z-0.5*(S.x@I.x+S.y@I.y))     #Be careful. S.z*I.z is ok, but S.x*I.x is not (diag vs. non-diag)
    else:
        M=np.sqrt(2/3)*S.z*I.z
        
    info={'Type':'dipole','i0':i0,'i1':i1,'delta':delta,'eta':eta,'euler':euler}
    
    return Ham1inter(M=M,isotropic=False,delta=delta,eta=eta,euler=euler,rotor_angle=es.rotor_angle,info=info)

def J(es,i0:int,i1:int,J:float):
    """
    J-coupling Hamiltonian

    Parameters
    ----------
    es : exp_sys
        Experimental system object.
    i0 : int
        index of the first spin.
    i1 : int
        index of the second spin.
    J : float
        Size of the J-coupled Hamiltonian.

    Returns
    -------
    None.

    """
    S,I=es.Op[i0],es.Op[i1]
    if es.Nucs[i0]==es.Nucs[i1]:
        H=J*(S.x@I.x+S.y@I.y+S.z*I.z)
        
    info={'Type':'J','i0':i0,'i1':i1,'J':J}
    
    return Ham1inter(H=H,isotropic=True,info=info)

def CS(es,i:int,ppm:float):
    """
    Isotropic chemical shift

    Parameters
    ----------
    es : exp_sys
        Experimental system object.
    i : int
        index of the spin.
    ppm : float
        Chemical shift offset in ppm.

    Returns
    -------
    None.

    """
    
    S=es.Op[i]
    H=ppm*es.v0[i]/1e6*S.z
    
    info={'Type':'CS','i':i,'ppm':ppm}
    return Ham1inter(H=H,isotropic=True,info=info)
    
def CSA(es,i:int,delta:float,eta:float=0,euler=[0,0,0]):
    """
    Chemical shift anisotropy

    Parameters
    ----------
    es : exp_sys
        Experimental system object.
    i : int
        index of the spin.
    delta : float
        anisotropy of the CSA (ppm)
    eta   : float
        asymmetry of the CSA. Default is 0
    euler : list
        3 elements giving the euler angles for the CSA (or a list of 3 element euler angles)
        Default is [0,0,0]

    Returns
    -------
    None.

    """
    
    S=es.Op[i]
    M=S.z
    delta=delta*es.v0[i]/1e6
    
    info={'Type':'CSA','i':i,'delta':delta,'eta':eta,'euler':euler}
    
    return Ham1inter(M=M,isotropic=False,delta=delta,eta=eta,euler=euler,rotor_angle=es.rotor_angle,info=info)
                    