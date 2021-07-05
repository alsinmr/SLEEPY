#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 10:14:04 2021

@author: albertsmith
"""

import numpy as np
import os
from pyRelaxSim.Tools import D2,d2



class PowderAvg():
    def __init__(self,PwdType=None,**kwargs):
        """
        Initializes the powder average. May be initialized with a powder average
        type (see set_powder_type)
        """
        
        self._pwdpath=os.path.join(os.path.dirname(os.path.realpath(__file__)),'PowderFiles')
        self._inter=list()
        if PwdType is not None:self.set_powder_type(PwdType,**kwargs)
        
                
    def set_powder_type(self,PwdType,**kwargs):
        """
        Set the powder type. Provide the type of powder average 
        (either a filename in PowderFiles, or a function in this file, don't 
        include .txt or pwd_ in PwdType). Duplicate names in PowderFiles and stored as 
        functions will result in the stored file to take precedence.
        """
        pwdfile=os.path.join(self._pwdpath,PwdType+'.txt')
        if os.path.exists(pwdfile):
            with open(pwdfile,'r') as f:
                alpha,beta,weight=list(),list(),list()
                for line in f:
                    a,b,w=[float(x) for x in line.split(' ')]
                    alpha.append(a)
                    beta.append(b)
                    weight.append(w)
            self.N=len(alpha)
            self.alpha,self.beta,self.gamma=np.array(alpha),np.array(beta),np.zeros(self.N)
            self.weight=np.array(weight)
        elif 'pwd_'+PwdType in globals().keys():
            out=globals()['pwd_'+PwdType](**kwargs)
            self.N=len(out[0])
            if len(out)==4:
                self.alpha,self.beta,self.gamma,self.weight=np.array(out)
            elif len(out)==3:
                self.alpha,self.beta=np.array(out[:2])
                self.gamma=np.zeros(self.N)
                self.weight=np.array(out[2])
        else:
            print('Unknown powder average, try one of the following:')
            for f in np.sort(os.listdir(self._pwdpath)):
                if '.txt' in f:
                    print(f.split('.')[0])
            for fname in globals().keys():
                if 'pwd_' in fname:
                    fun=globals()[fname]
                    print(fname[4:]+' with args:',fun.__code__.co_varnames[:fun.__code__.co_argcount])
        
    def new_inter(self,Type=None,index=None,iso=0,delta=0,eta=0,euler=[0,0,0]):
        if not(hasattr(self,Type)):
            setattr(self,Type,list())
        
        self._inter.append({'Type':Type,'index':index,'delta':delta,'eta':eta,'euler':euler,\
                            'RotObj':RotInter(iso=iso,delta=delta,eta=eta,euler=euler,pwdavg=self)})
    
        at=getattr(self,Type)
        at.append({'index':index,'delta':delta,'eta':eta,'euler':euler,\
                   'RotObj':self._inter[-1]['RotObj']})
                    
class RotInter():
    """
    Stores the information for one interaction (isotropic value, delta, eta, and
    euler angles). Also rotates the interaction, either for a specific set of
    Euler angles, or for all angles in a powder average (powder average should
    be stored in the object).
    
    Frames are 
        PAS: Principle axis system, without any rotations. Diagonal in the Cartesian system
        
    """
    def __init__(self,iso=0,delta=0,eta=0,euler=[0,0,0],pwdavg=None):
        """
        Initialize the interaction with its isotropic value, delta, eta, and euler
        angles. Note, euler angles should be of the form [alpha,beta,gamma], 
        however, one may use a list of multiple sets of  Euler angles, in which
        case the angles will be executed in sequence (may be useful for setting
        multiple interactions)
        """
        self.setPAS(iso,delta,eta)
        self.PAS2MOL(euler)
        self.pwdavg=pwdavg
        
    def setPAS(self,iso=0,delta=0,eta=0):
        """
        Stores the isotropic value of an interaction, and converts the delta and
        eta values of the interaction into spherical tensor components in the 
        principle axis.
        
        Note that this class only stores the isotropic value, and does not 
        include it in any computations.
        """
        self.iso=iso
        self.delta=delta
        self.eta=eta
        self.PAS=np.array([-0.5*delta*eta,0,np.sqrt(3/2)*delta,0,-0.5*delta*eta])
        
        return self.PAS
        
    def PAS2MOL(self,euler=[0,0,0]):
        """
        Takes a list of Euler angles (give as a three element list: [alpha, beta,gamma]), 
        and applies these Euler angles to the tensor in the PAS. Multiple 
        sets of Euler angles may be applied, just include more than one list.
        """
        
        if not(hasattr(euler[0],'__len__')):euler=[euler]
        A=self.PAS.copy()
        for alpha,beta,gamma in euler:
            A=np.array([(A*D2(alpha,beta,gamma,mp=None,m=m)).sum() for m in range(-2,3)])
        self.MOL=A
        
        return self.MOL
        
    def MOL2LAB_Azz(self,pwdavg=None,rotor_angle=np.arccos(np.sqrt(1/3))):
        """
        Applies the powder average to the tensor stored in the molecular frame.
        One may provide the powder average, although usually this will already
        be stored in this object.
        
        This returns the n=-2,-1,0,1,2 rotating components of the term A0 in the
        lab frame, after scaling by 2/np.sqrt(6) (that is, we return the Azz 
        component). Note, the off-diagonal terms of the tensors are omitted in 
        this case (the A-2,A-1,A1,A2 componets). 

        Output of this term sould be multiplied by T_{2,0}*sqrt(3/2),
        which we give for a few interactions:
            Heteronuclear dipole:   Iz*Sz
            Homonuclear dipole:     (Iz*Sz-1/2*(Ix*Sx+Iy*Sy))
            CSA:                    Iz
            
        Note that the isotropic part of the interaction is added to the n=0
        (non-rotating) component
        """
        
        if pwdavg is not None:self.pwdavg=pwdavg
        assert self.pwdavg is not None,"If pwdavg is not previously defined, then it must bet provided"
        self.rotor_angle=rotor_angle
        A=self.MOL.copy()
        alpha,beta,gamma=[getattr(self.pwdavg,x) for x in ['alpha','beta','gamma']]
        
        A=np.array([(D2(alpha,beta,gamma,mp=None,m=m)*A).sum(1) for m in range(-2,3)]).T
        A=2/np.sqrt(6)*d2(rotor_angle,mp=None,m=0)*A
        
        self.Azz=A
        return A
    
    def MOL2LAB_A(self,pwdavg=None):
        """
        Applies the powder average to the tensor stored in the molecular frame.
        One may provide the powder average, although usually this will already
        be stored in this object.
        
        This returns the lab components of the interaction (no rotor angle 
        considered)

        Output of this term should be multiplied by terms T_{2,m}, for example,
        for a dipole, this is
        
        T(I)_(1,-1)=1/sqrt(2)*I-
        T(I)_(1,0)=Iz
        T(I)_(1,1)=-1/sqrt(2)*I+
        
        T_(2,-2)=T(I)_(1,-1)*T(I)_(1,-1)
        T_(2,-1)=1/sqrt(2)*(T(I)_(1,-1)*T(S)_(1,0)+T(S)_(1,-1)*T(I)_(1,0))
        T_(2,0)=1/sqrt(6)*(2*T(I)_(1,0)*T(S)_(1,0)+T(I)_(1,1)*T(S)_(1,-1)+T(I)_(1,-1)*T(S)_(1,1))
        T_(2,1)=1/sqrt(2)*(T(I)_(1,1)*T(S)_(1,0)+T(S)_(1,1)*T(I)_(1,0))
        T_(2,2)=T(I)_(1,1)*T(I)_(1,1)

        Note that the isotropic component is scaled and added to   
        """
        
        if pwdavg is not None:self.pwdavg=pwdavg
        assert self.pwdavg is not None,"If pwdavg is not previously defined, then it must bet provided"
        self.rotor_angle=rotor_angle
        A=self.MOL.copy()
        alpha,beta,gamma=[getattr(self.pwdavg,x) for x in ['alpha','beta','gamma']]
        
        A=np.array([(D2(alpha,beta,gamma,mp=None,m=m)*A).sum(1) for m in range(-2,3)]).T
        
        self.A=A
        return A
    
    def MOL2LAB_Afull(self,pwdavg=None,rotor_angle=np.arccos(np.sqrt(1/3))):
        """
        Applies the powder average to the tensor stored in the molecular frame.
        One may provide the powder average, although usually this will already
        be stored in this object.
        
        This returns all components of the tensor in the lab frame, additionally
        considering spinning

        Output of this term should be multiplied by terms T_{2,m}, for example,
        for a dipole, this is
        
        T(I)_(1,-1)=1/sqrt(2)*I-
        T(I)_(1,0)=Iz
        T(I)_(1,1)=-1/sqrt(2)*I+
        
        T_(2,-2)=T(I)_(1,-1)*T(I)_(1,-1)
        T_(2,-1)=1/sqrt(2)*(T(I)_(1,-1)*T(S)_(1,0)+T(S)_(1,-1)*T(I)_(1,0))
        T_(2,0)=1/sqrt(6)*(2*T(I)_(1,0)*T(S)_(1,0)+T(I)_(1,1)*T(S)_(1,-1)+T(I)_(1,-1)*T(S)_(1,1))
        T_(2,1)=1/sqrt(2)*(T(I)_(1,1)*T(S)_(1,0)+T(S)_(1,1)*T(I)_(1,0))
        T_(2,2)=T(I)_(1,1)*T(I)_(1,1)
        
        Output shape is Nx5x5, where N is the number of elements in the powder
        average, the next dimension corresponds to which spherical component,
        and the final axis is the rotating component for the rotor.
        """
        
        if pwdavg is not None:self.pwdavg=pwdavg
        assert self.pwdavg is not None,"If pwdavg is not previously defined, then it must bet provided"
        self.rotor_angle=rotor_angle
        A=self.MOL.copy()
        alpha,beta,gamma=[getattr(self.pwdavg,x) for x in ['alpha','beta','gamma']]
        
        "Now the rotor frame"
        A=np.array([(D2(alpha,beta,gamma,mp=None,m=m)*A).sum(1) for m in range(-2,3)])
        
        A=np.array([(d2(rotor_angle,mp=None,m=0)*A.T).T for m in range(-2,3)]).T
        
        self.Afull=A
        return A
    
    
#%% Functions for powder averaging    
def pwd_JCP59(q=3):
    """
    Generates a powder average with quality quality q (1 to 12). 
    According to JCP 59 (8) 3992 (1973) (copied from Matthias Ernst's Gamma
    scripts).
    """
    
    q+=-1   #We use q as an index, switch to python indexing
    
    value1=[2,50,100,144,200,300,538,1154,3000,5000,7000,10000];
    value2=[1,7,27,11,29,37,55,107,637,1197,1083,1759];
    value3=[1,11,41,53,79,61,229,271,933,1715,1787,3763];

    count=np.arange(1,value1[q])

    alpha=2*np.pi*np.mod(value2[q]*count,value1[q])/value1[q];
    beta=np.pi*count/value1[q];
    gamma=2*np.pi*np.mod(value3[q]*count,value1[q])/value1[q];

    weight=np.sin(beta);
    weight*=1/weight.sum();

    return alpha,beta,gamma,weight