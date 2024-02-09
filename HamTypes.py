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
from .Tools import NucInfo

class Ham1inter():
    def __init__(self,M=None,H=None,T=None,isotropic=False,delta=0,eta=0,euler=[0,0,0],
                 rotor_angle=np.arccos(np.sqrt(1/3)),info={}):
        
        self.M=M
        self.T=T
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
            if self.T is None:
                out.A=self.rotInter.Azz[i]
            else:
                out.A=self.rotInter.Afull[i]
            return out
        return self
    
    def __repr__(self):
        dct=copy(self.info)
        out='Hamiltonian for a single interaction\n'
        out+=f'Type: {dct.pop("Type")} '
        out+=f'on spin {dct.pop("i")}\n' if 'i' in dct else f'between spins {dct.pop("i0")} and {dct.pop("i1")}\n'
        
        def ef(euler):
            if hasattr(euler[0],'__iter__'):
                return ','.join([ef(e) for e in euler])
            else:
                return '['+','.join([f'{a*180/np.pi:.2f}' for a in euler])+']'
        
        if len(dct):
            out+='Arguments:\n\t'+\
                '\n\t'.join([f'{key}={ef(value) if key=="euler" else value}' for key,value in dct.items()])+'\n'
        out+='\n'+super().__repr__()    
        return out
            
            
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
        
        if self.T is None:
            out=self.M*self.A[n+2]
        else:
            out=np.sum([A*T*(-1)**q for T,A,q in zip(self.T[2],self.A[n+2],range(-2,3))],axis=0)
        
        if self.H is not None and n==0:
            out+=self.H

        return out

def _larmor(es,i:int):
    """
    Larmor frequency Hamiltonian (for Lab-frame simulation)

    Parameters
    ----------
    es : exp_sys
        Experimental system object.
    i : int
        index of the spin.

    Returns
    -------
    None.

    """
    info={'Type':'larmor','i':i}
    S=es.Op[i]
    return Ham1inter(H=es.v0[i]*S.z,isotropic=True,info=info)

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
    info={'Type':'dipole','i0':i0,'i1':i1,'delta':delta,'eta':eta,'euler':euler}
    
    if es.LF[i0] or es.LF[i1]:  #Lab frame calculation
        T=es.Op[i0].T*es.Op[i1].T
        
        if es.LF[i0] and es.LF[i1]:
            T.set_mode('LF_LF')
        elif es.LF[i0]:
            T.set_mode('LF_RF')
        else:
            T.set_mode('RF_LF')
        
        return Ham1inter(T=T,isotropic=False,delta=delta,eta=eta,euler=euler,rotor_angle=es.rotor_angle,info=info)

    else:
        S,I=es.Op[i0],es.Op[i1]
        if es.Nucs[i0]==es.Nucs[i1]:
            M=np.sqrt(2/3)*(S.z*I.z-0.5*(S.x@I.x+S.y@I.y))     #Be careful. S.z*I.z is ok, but S.x*I.x is not (diag vs. non-diag)
        else:
            M=np.sqrt(2/3)*S.z*I.z
        
    
    
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
    else:
        H=J*S.z*I.z
        
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
    
    
    delta=Azz
    eta=(Ayy-Axx)/delta if delta else 0
    info={'Type':'Hyperfine','i0':i0,'i1':i1,'Axx':Axx+avg,'Ayy':Ayy+avg,'Azz':Azz+avg,'euler':euler}

    if es.LF[i0] or es.LF[i1]:  #Lab frame calculation
        T=es.Op[i0].T*es.Op[i1].T
        
        if es.LF[i0] and es.LF[i1]:
            T.set_mode('LF_LF')
        elif es.LF[i0]:
            T.set_mode('LF_RF')
        else:
            T.set_mode('RF_LF')
        
        H=-np.sqrt(3)*avg*T[0,0]   #Rank-0 contribution
        
        if delta:
            return Ham1inter(H=H,T=T,isotropic=False,delta=delta,eta=eta,euler=euler,rotor_angle=es.rotor_angle,info=info)
        else:
            return Ham1inter(H=H,isotropic=True,info=info)
    else:  #Rotating frame calculation
        S,I=es.Op[i0],es.Op[i1]
        M=np.sqrt(2/3)*S.z*I.z
        H=avg*S.z*I.z
        if delta:                        
            return Ham1inter(M=M,H=H,isotropic=False,delta=delta,eta=eta,euler=euler,rotor_angle=es.rotor_angle,info=info)
        else:
            return Ham1inter(H=H,isotropic=True,info=info)

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

def g(es,i:int,gxx:float=0,gyy:float=0,gzz:float=0,euler=[0,0,0]):
    """
    electron g-tensor Hamiltonian. Note that the g-tensor values should be 
    typically positive.
    

    Parameters
    ----------
    es : exp_sys
        Experimental system object.
    i : int
        index of the spin.
    gxx : float, optional
        xx-component of the electron g-tensor. The default is 0.
    gyy : float, optional
        yy-component of the electron g-tensor. The default is 0.
    gzz : float, optional
        xx-component of the electron g-tensor. The default is 0.
    euler : TYPE, optional
        3 elements giving the euler angles for the g-tensor
        The default is [0,0,0].

    Returns
    -------
    None.

    """
    
    if es.Nucs[i][0]!='e':
        warnings.warn('g-tensor is being applied to a nucleus')
    
    avg=(gxx+gyy+gzz)/3
    if avg<0:
        warnings.warn('Expected a positive g-tensor')
        
    gyy,gxx,gzz=np.sort([gxx-avg,gyy-avg,gzz-avg])
    
    mub=-9.2740100783e-24/6.62607015e-34  #Bohr magneton in Hz. Take positive g-values by convention
    
    avg=mub*avg-NucInfo('e-')            #Values in Hz. Note that we take this in the rotating frame
    delta=gzz*mub
    eta=(gyy-gxx)/delta if delta else 0
    info={'Type':'g','i':i,'gxx':gxx+avg,'gyy':gyy+avg,'gzz':gzz+avg,'euler':euler}
    
    if es.LF[i]:  #Lab frame calculation
        T=es.Op[i].T
        T.set_mode('B0_LF')
        H=-np.sqrt(3)*avg*T[0,0]   #Rank-0 contribution
        if delta:
            return Ham1inter(H=H,T=T,isotropic=False,delta=delta,eta=eta,euler=euler,rotor_angle=es.rotor_angle,info=info)
        else:
            return Ham1inter(H=H,isotropic=True,info=info)
    else:  #Rotating frame calculation
        S=es.Op[i]
        M=np.sqrt(2/3)*S.z
        H=avg*S.z
        if delta:                        
            return Ham1inter(M=M,H=H,isotropic=False,delta=delta,eta=eta,euler=euler,rotor_angle=es.rotor_angle,info=info)
        else:
            return Ham1inter(H=H,isotropic=True,info=info)




                   