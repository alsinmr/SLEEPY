#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 11:49:36 2023

@author: albertsmith
"""

import numpy as np
import warnings
from .PowderAvg import RotInter
from copy import copy
from . import Defaults

class Ham1inter():
    def __init__(self,M=None,H=None,isotropic=False,delta=0,eta=0,iso=0,euler=[0,0,0],
                 rotor_angle=np.arccos(np.sqrt(1/3)),info={}):
        
        self.M=M
        self.H=H
        self.isotropic=isotropic
        self.delta=delta
        self.eta=eta
        self.iso=iso
        self.euler=euler
        self.pwdavg=None
        self.rotInter=None
        self.info=info
        self.rotor_angle=rotor_angle
        
        self.A=None
        
    @property
    def _ctype(self):
        return Defaults['ctype']
        
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
                return np.zeros(self.H.shape,dtype=self._ctype)
            
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
    Ham1inter

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
    Ham1inter

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
    Ham1inter

    """
    
    S=es.Op[i]
    M=np.sqrt(2/3)*S.z    #Is the multiplication factor correct?
    delta=delta*es.v0[i]/1e6
    
    info={'Type':'CSA','i':i,'delta':delta,'eta':eta,'euler':euler}
    
    return Ham1inter(M=M,isotropic=False,delta=delta,eta=eta,euler=euler,rotor_angle=es.rotor_angle,info=info)
 

def hyperfine(es,i0:int,i1:int,Axx:float=0,Ayy:float=0,Azz:float=0,euler=[0,0,0]):
    """
    Hyperfine between electron and nucleus. Note that in this implementation,
    we are only including the secular term. This will allow transverse relaxation
    due to both electron T1 and hyperfine tensor reorientation, and also the
    pseudocontact shift. However, DNP will not be possible except for buildup
    in the transverse plane (no SE, CE, NOVEL, etc.)

    Parameters
    ----------
    es : exp_sys
        Experimental system object.
    i0 : int
        index of the first spin.
    i1 : int
        index of the second spin.
    Axx : float
        Axx component of the hyperfine.
    Ayy : float
        Ayy component of the hyperfine.
    Azz : flat
        Azz component of the hyperfine.
    euler : TYPE, optional
        DESCRIPTION. The default is [0,0,0].

    Returns
    -------
    Ham1inter

    """
    if es.Nucs[i0][0]!='e' and es.Nucs[i1][0]!='e':
        warnings.warn(f'Hyperfine coupling between two nuclei ({es.Nucs[i0]},{es.Nucs[i1]})')
    
    avg=(Axx+Ayy+Azz)/3
    Ayy,Axx,Azz=np.sort([Axx-avg,Ayy-avg,Azz-avg])
    
    iso=avg*np.sqrt(3/2)  #Cancel out the sqrt(2/3) in M
    delta=Azz
    eta=(Ayy-Axx)/delta if delta else 0
    info={'Type':'Hyperfine','i0':i0,'i1':i1,'Axx':Axx+avg,'Ayy':Ayy+avg,'Azz':Azz+avg,'euler':euler}

    S,I=es.Op[i0],es.Op[i1]
    M=np.sqrt(2/3)*S.z*I.z
    if delta:                        
        return Ham1inter(M=M,isotropic=False,delta=delta,eta=eta,iso=iso,euler=euler,rotor_angle=es.rotor_angle,info=info)
    else:
        return Ham1inter(H=M*iso,isotropic=True,info=info)

def quadrupole(es,i:int,delta:float=0,eta:float=0,euler=[0,0,0]):
    """
    Quadrupole coupling defined by its anisotropy (delta) and asymmetry (eta). 
    
    (Maybe we would rather have the input be Cqcc?)

    Parameters
    ----------
    es : exp_sys
        Experimental system object.
    i : int
        index of the spin.
    delta : float
        anisotropy of the quadrupole coupling. Default is 0
    eta   : float
        asymmetry of the quadrupole coupling (usually 0). Default is 0
    euler : list
        3 elements giving the euler angles for the quadrupole coupling.
        Default is [0,0,0]

    Returns
    -------
    None.

    """
    
    S=es.Op[i]

    I=es.S[i]

    M=1/2*(3*S.z@S.z-I*(I+1)*S.eye)     
    
    info={'Type':'quadrupole','i':i,'delta':delta,'eta':eta,'euler':euler}
    print('Quadrupole Hamiltonian does not include 2nd order terms')
    return Ham1inter(M=M,isotropic=False,delta=delta,eta=eta,iso=0,euler=euler,
                      rotor_angle=es.rotor_angle,info=info)



                   