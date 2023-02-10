#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 10:14:04 2021

@author: albertsmith
"""

import numpy as np
import os
from copy import copy
from .Tools import D2,d2
from . import PwdAvgFuns


class PowderAvg():
    def __init__(self,PwdType='JCP59',**kwargs):
        """
        Initializes the powder average. May be initialized with a powder average
        type (see set_powder_type)
        """
        
        self._pwdpath=os.path.join(os.path.dirname(os.path.realpath(__file__)),'PowderFiles')
        self._inter=list()
        
        self.PwdType=None
        self.set_powder_type(PwdType,**kwargs)
        
        self.__Types=list()
        
        self.__index=-1
        
                
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
            self.PwdType=PwdType
        elif hasattr(PwdAvgFuns,'pwd_'+PwdType):
            out=getattr(PwdAvgFuns,'pwd_'+PwdType)(**kwargs)
            self.N=len(out[0])
            if len(out)==4:
                self.alpha,self.beta,self.gamma,self.weight=np.array(out)
            elif len(out)==3:
                self.alpha,self.beta=np.array(out[:2])
                self.gamma=np.zeros(self.N)
                self.weight=np.array(out[2])
            self.PwdType=PwdType
        else:
            print('Unknown powder average, try one of the following:')
            for f in np.sort(os.listdir(self._pwdpath)):
                if '.txt' in f:
                    print(f.split('.')[0])
            for fname in dir(PwdAvgFuns):
                if 'pwd_' in fname:
                    fun=getattr(PwdAvgFuns,fname)
                    print(fname[4:]+' with args: '+','.join(fun.__code__.co_varnames[:fun.__code__.co_argcount]))
    
    @property
    def list_powder_types(self):
        pwd=list()
        for file in os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),'PowderFiles')):
            if len(file)>4 and file[-4:]=='.txt':
                pwd.append(file[:-4])
        for fname in dir(PwdAvgFuns):
            if 'pwd_' in fname:
                pwd.append(fname[4:])
        return pwd
        
    
    def __eq__(self,pwdavg):
        if pwdavg is self:return True
        for key in ['N','alpha','beta','gamma']:
            if not(np.all(getattr(pwdavg,key)==getattr(self,key))):return False
        return True
    
    def __getitem__(self,i):
        """
        Returns the ith element of the powder average as a new powder average

        Parameters
        ----------
        def __getitem__[i] : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        out=copy(self)
        out.alpha=self.alpha[i:i+1]
        out.beta=self.beta[i:i+1]
        out.gamma=self.gamma[i:i+1]
        out.N=1
        out.weight=np.ones([1])
        return out
    
    def __repr__(self):
        out='Powder Average\n'
        if self.PwdType is None:
            out+='[undefined type]'
        else:
            plural='s' if self.N>1 else ''
            out+=f'Type:\t{self.PwdType} with {self.N} angle{plural}\n'
            if self.gamma.max()==0:
                out+='Gamma not included\n'
        out+='\n'+super().__repr__()
        return out
            
        
    
                    
class RotInter():
    """
    Stores the information for one interaction (isotropic value, delta, eta, and
    euler angles). Also rotates the interaction, either for a specific set of
    Euler angles, or for all angles in a powder average (powder average should
    be stored in the object).
    
    Frames are 
        PAS: Principle axis system, without any rotations. Diagonal in the Cartesian system
        
    """
    def __init__(self,pwdavg,delta=0,eta=0,euler=[0,0,0],rotor_angle=np.arccos(np.sqrt(1/3))):
        """
        Initialize the interaction with its isotropic value, delta, eta, and euler
        angles. Note, euler angles should be of the form [alpha,beta,gamma], 
        however, one may use a list of multiple sets of  Euler angles, in which
        case the angles will be executed in sequence (may be useful for setting
        multiple interactions)
        """
        self.delta=delta
        self.eta=eta
        self.euler=euler
        self.setPAS()
        self.PAS2MOL()
        self.pwdavg=pwdavg
        self.rotor_angle=rotor_angle

        
    def setPAS(self):
        """
        Stores the isotropic value of an interaction, and converts the delta and
        eta values of the interaction into spherical tensor components in the 
        principle axis.
        
        Note that this class only stores the isotropic value, and does not 
        include it in any computations.
        """
        delta=self.delta
        eta=self.eta
        self.__PAS=np.array([-0.5*delta*eta,0,np.sqrt(3/2)*delta,0,-0.5*delta*eta])
        
    @property
    def PAS(self):
        return self.__PAS.copy()
        
    def PAS2MOL(self):
        """
        Takes a list of Euler angles (give as a three element list: [alpha, beta,gamma]), 
        and applies these Euler angles to the tensor in the PAS. Multiple 
        sets of Euler angles may be applied, just include more than one list.
        """
        
        euler=self.euler
        if not(hasattr(euler[0],'__len__')):euler=[euler]
        A=self.PAS.copy()
        for alpha,beta,gamma in euler:
            A=np.array([(A*D2(alpha,beta,gamma,mp=None,m=m)).sum() for m in range(-2,3)])
        self.__MOL=A
    
    @property
    def MOL(self):
        return self.__MOL.copy()
        
    def MOL2LAB_Azz(self):
        """
        Applies the powder average to the tensor stored in the molecular frame.
        One may provide the powder average, although usually this will already
        be stored in this object.
        
        This returns the n=-2,-1,0,1,2 rotating components of the term A0 in the
        lab frame, after scaling by 2/np.sqrt(6) (that is, we return the Azz 
        component). Note, the off-diagonal terms of the tensors are omitted in 
        this case (the A-2,A-1,A1,A2 components). 

        Output of this term sould be multiplied by T_{2,0},
        which we give for a few interactions:
            Heteronuclear dipole:   sqrt(2/3)*Iz*Sz
            Homonuclear dipole:     sqrt(2/3)*(Iz*Sz-1/2*(Ix*Sx+Iy*Sy))
            CSA:                    sqrt(2/3)*Iz
            
        """
        # pwdavg=self.pwdavg
        rotor_angle=self.rotor_angle
        A=self.MOL.copy()
        alpha,beta,gamma=self.pwdavg.alpha,self.pwdavg.beta,self.pwdavg.gamma
        
        A=np.array([(D2(alpha,beta,gamma,mp=None,m=m)*A).sum(1) for m in range(-2,3)]).T
        A=d2(rotor_angle,mp=None,m=0)*A
        
        self.__Azz=A
    
    @property
    def Azz(self):
        if not(hasattr(self,'.__Azz')):self.MOL2LAB_Azz()
        return self.__Azz.copy()
    
    def MOL2LAB_A(self):
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
        
        # self.pwdavg=pwdavg
        # rotor_angle=self.rotor_angle
        A=self.MOL
        alpha,beta,gamma=[getattr(self.pwdavg,x) for x in ['alpha','beta','gamma']]
        
        A=np.array([(D2(alpha,beta,gamma,mp=None,m=m)*A).sum(1) for m in range(-2,3)]).T
        
        self.__A=A
    
    @property
    def A(self):
        if not(hasattr(self,'.__A')):self.MOL2LAB_A()
        return self.__A.copy()
    
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
        
        pwdavg=self.pwdavg
        rotor_angle=self.rotor_angle
        A=self.MOL
        alpha,beta,gamma=[getattr(pwdavg,x) for x in ['alpha','beta','gamma']]
        
        "Now the rotor frame"
        A=np.array([(D2(alpha,beta,gamma,mp=None,m=m)*A).sum(1) for m in range(-2,3)])
        
        A=np.array([(d2(rotor_angle,mp=None,m=m)*A.T).T for m in range(-2,3)]).T
        
        self.__Afull=A
    
    @property
    def Afull(self):
        if not(hasattr(self,'.__Afull')):self.MOL2LAB_Afull()
        return self.__Afull.copy()
    
