#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 10:14:04 2021

@author: albertsmith
"""

import numpy as np
import os
from copy import copy
import matplotlib.pyplot as plt
from .plot_tools import use_zoom
from .Tools import D2,d2,Djmmp
from . import PwdAvgFuns
from scipy.optimize import lsq_linear



Pwd=list()
for file in os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),'PowderFiles')):
    if len(file)>4 and file[-4:]=='.txt':
        Pwd.append(file[:-4])
for fname in dir(PwdAvgFuns):
    if 'pwd_' in fname:
        Pwd.append(fname[4:])
Pwd.sort()

class PowderAvg():
    list_pwd_types=copy(Pwd)
    def __init__(self,PwdType:str='JCP59',n_gamma:int=100,gamma_encoded:bool=None,**kwargs):
        """
        Initializes the powder average. May be initialized with a powder average
        type (see set_powder_type)
        """
        
        if isinstance(PwdType,int):
            kwargs['q']=PwdType
            PwdType='JCP59'

        if gamma_encoded is None:
            if isinstance(PwdType,str) and len(PwdType)>=5 and PwdType[:5]=='alpha':
                gamma_encoded=True
            else:
                gamma_encoded=False
            
        
        self._alpha=None
        self._beta=None
        self._gamma=None
        self._weight=None
        self.n_gamma=n_gamma
        self.n_alpha=0
        self._gamma_incl=gamma_encoded 
        self._gamma_encoded=gamma_encoded
        self._M=None
        self._sym=None
        
        self._pwdpath=os.path.join(os.path.dirname(os.path.realpath(__file__)),'PowderFiles')
        self._inter=list()
        
        self.PwdType=None
        self.set_powder_type(PwdType,**kwargs)
        
        self.__Types=list()
        
        self.__index=-1
        

    @property
    def gamma_encoded(self):
        return self._gamma_encoded
    
    @property
    def alpha(self):
        if self._alpha is None:return
        if self._gamma_incl:return self._alpha
        return np.tile(self._alpha,self.n_gamma)
    
    @alpha.setter
    def alpha(self,alpha):
        self._alpha=alpha
        self.n_alpha=len(alpha)
        
    @property
    def beta(self):
        if self._beta is None:return
        if self._gamma_incl:return self._beta
        return np.tile(self._beta,self.n_gamma)
    
    @beta.setter
    def beta(self,beta):
        self._beta=beta
        
    @property
    def gamma(self):
        if self._gamma is None and self._gamma_incl:return np.zeros(self.n_alpha)
        if self._gamma_incl:return self._gamma
        if self._alpha is None:return None
        return np.repeat(np.arange(self.n_gamma)*2*np.pi/self.n_gamma,len(self._alpha))
    
    @gamma.setter
    def gamma(self,gamma):
        if gamma is not None:
            self._gamma=gamma
            self._gamma_incl=True
            
    @property
    def weight(self):
        if self._gamma_incl:return self._weight
        return np.tile(self._weight,self.n_gamma)/self.n_gamma
    
    @weight.setter
    def weight(self,weight):
        self._weight=weight
    
    @property
    def N(self):
        return self.n_alpha if self._gamma_incl else self.n_alpha*self.n_gamma
        
                
    def set_powder_type(self,PwdType,gamma_encoded=None,**kwargs):
        """
        Set the powder type. Provide the type of powder average 
        (either a filename in PowderFiles, or a function in this file, don't 
        include .txt or pwd_ in PwdType). If duplicate names are found in
        the powder files and the functions, then the files will be used
        """
        if isinstance(PwdType,self.__class__):
            self.__dict__=PwdType.__dict__
            return self
        
        self._gamma=None
        if gamma_encoded is not None:self._gamma_encoded=gamma_encoded
        self._gamma_incl=self.gamma_encoded
        pwdfile=os.path.join(self._pwdpath,PwdType+'.txt')
        if os.path.exists(pwdfile):
            with open(pwdfile,'r') as f:
                alpha,beta,weight=list(),list(),list()
                for line in f:
                    if len(line.strip().split(' '))==3:
                        a,b,w=[float(x) for x in line.strip().split(' ')]
                        alpha.append(a*np.pi/180)
                        beta.append(b*np.pi/180)
                        weight.append(w)
            self.alpha,self.beta=np.array(alpha),np.array(beta)
            self.weight=np.array(weight)
            if 'bcr' in pwdfile:
                weight=np.sin(beta)
                self.weight=weight/weight.sum()
            
            self.PwdType=PwdType
        elif hasattr(PwdAvgFuns,'pwd_'+PwdType):
            out=getattr(PwdAvgFuns,'pwd_'+PwdType)(**kwargs)
            if len(out)==4:
                self.alpha,self.beta,self.gamma,self.weight=np.array(out)
            elif len(out)==3:
                self.alpha,self.beta=np.array(out[:2])
                # self.gamma=np.zeros(self.N)
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
        # self.n_alpha=len(self._alpha)
        return self
    
    @use_zoom
    def plot(self,ax=None,beta_gamma:bool=False,color='darkcyan',s=3):
        """
        Plots the powder average. By default, plots the alpha and beta angles.
        Changing beta_gamma to True will plot beta and gamma instead

        Parameters
        ----------
        ax : TYPE, optional
            Axis to plot the powder average onto. The default is None.
        mode : TYPE, optional
            DESCRIPTION. The default is 'alphabeta'.
        color : Type, optional
            color to plot with
        s : float, optional
            scatter size

        Returns
        -------
        ax

        """
        if ax is None:ax=plt.figure().add_subplot(111,projection='3d')
        
        beta=self.beta
        theta=self.gamma if beta_gamma else self.alpha
        x=np.sin(beta)*np.cos(theta)
        y=np.sin(beta)*np.sin(theta)
        z=np.cos(beta)
        
        ax.scatter3D(x,y,z,s=s,color=color)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_box_aspect([1,1,1])
        return ax
    
    # def sampling_moment_matrix(self,Lmax:int=None,sym:str='Ci'):
    #     """
    #     Calculates a matrix that, when multiplied by the powder average weighting,
    #     returns the sampling moments.
        
    #     Matthias Eden, Malcolm H. Levitt. "Computational of Orientational Averages in
    #     Solid-State NMR by Gaussian Spherical Quadrature". J. Magn. Reson. 132, 220-239
    #     (1998)
        
        

    #     Parameters
    #     ----------
    #     Lmax : int, optional
    #         Highest rank tensor to calculate. The default is None, which will
    #         generate

    #     Returns
    #     -------
    #     np.array

    #     """
    #     assert sym in ['C0','Ci','D2h'],'Symmetry (sym) must be C0, Ci, or D2h'
        
    #     if self._sym!=sym:
    #         self._M=None
    #         self._sym=sym
        
    #     if Lmax is None:
    #         Lmax=1000
    #         auto=True
    #     else:
    #         auto=False
            
    #     if self._M is None:self._M=[]
    #     M=self._M
        
    #     Neqs=self.sampling_moment_N(Lmax=Lmax,sym=sym)
    #     if Lmax is not None and sym!='C0':Lmax=(Lmax//2)*2

    #     N=self.N if self._gamma_incl else self.N//self.n_gamma
        
    #     start=Lmax+1 if Neqs[-1]<=len(M) else [0,*Neqs.tolist()].index(len(M))
    #     stop=np.argmax(Neqs>=N) if auto else Lmax
        

    #     a,b,g=self.alpha[:N],self.beta[:N],self.gamma[:N]
        
    #     if auto:Lmax=stop
    #     if sym=='C0':
    #         for L in range(start,stop+1):
    #             M0=[]
    #             for m in range(-L,L+1):
    #                 for mp in range(-L,L+1):
    #                     M0.append(Djmmp(a,b,g,m=m,mp=mp,j=L))
    #             M.extend(M0)
    #     elif sym=='Ci':
    #         if start%2:start+=1
    #         for L in range(start,stop+1,2):
    #             M0=[]
    #             for m in range(0,L+1):
    #                 M0.append(Djmmp(a,b,g,m=0,mp=m,j=L).real)
    #             for m in range(1,L+1):
    #                 M0.append(Djmmp(a,b,g,m=0,mp=m,j=L).imag)
    #             M.extend(M0)
    #     elif sym=='D2h':
    #         if start%2:start+=1
    #         for L in range(start,stop+1,2):
    #             M0=[]
    #             for m in range(0,L+1,2):
    #                 M0.append(Djmmp(a,b,g,m=0,mp=m,j=L).real)
    #             M.extend(M0) 

        
    #     return np.array(M[:Neqs[Lmax]])
    
    # def sampling_moment_N(self,Lmax:int=1000,sym:str='Ci'):
    #     """
    #     Returns the number of simultaneous equations for symmetry groups C0,
    #     Ci, and D2h. Returns a vector corresponding to values of L from 0 up to
    #     Lmax (default Lmax=100, thus returning a vector with 101 elements)

    #     Parameters
    #     ----------
    #     Lmax : int, optional
    #         Maximum tensor rank to include. The default is 100.
    #     sym : str, optional
    #         Symmetry group ('C0','Ci', or 'D2h'). The default is 'Ci'.

    #     Returns
    #     -------
    #     np.array

    #     """
    #     assert sym in ['C0','Ci','D2h'],'Symmetry (sym) must be C0, Ci, or D2h'
        
    #     if sym=='C0':
    #         return ((2*np.arange(Lmax+1)+1)**2).cumsum()
    #     if sym=='Ci':
    #         L=np.arange(Lmax+1)
    #         L=np.floor(L/2)*2
    #         return (1/2*L*(L+3)+1).astype(int)
    #     if sym=='D2h':
    #         L=np.arange(Lmax+1)
    #         L=np.floor(L/2)*2
    #         return (1/8*(L+4)*(L+2)).astype(int)
    
    # def sampling_moment(self,Lmax:int=50,sym='Ci',weight=None):
    #     """
    #     Calculates the sampling moments as a function of L

    #     Parameters
    #     ----------
    #     Lmax : int, optional
    #         DESCRIPTION. The default is 50.
    #     sym : TYPE, optional
    #         DESCRIPTION. The default is 'Ci'.
    #     weight : TYPE, optional
    #         DESCRIPTION. The default is None.

    #     Returns
    #     -------
    #     None.

    #     """
    #     N=self.N if self._gamma_incl else self.N//self.n_gamma
    #     if weight is None:
    #         weight=self.weight[:N]
    #         weight/=weight.sum()
    #     M=self.sampling_moment_matrix(Lmax=Lmax,sym=sym)
        
    #     target=np.zeros(M.shape[0])
    #     target[0]=1
    #     var=(M@weight-target)**2
        
    #     error=[]
    #     Neqs=np.concatenate(([0],self.sampling_moment_N(Lmax=Lmax,sym=sym)))
    #     for k in range(len(Neqs)-1):
    #         error.append(np.sqrt(var[Neqs[k]:Neqs[k+1]].sum()/(2*k+1)))
    #     return error
    
    # def plot_sampling_moment(self,Lmax:int=50,sym='Ci',weight=None,ax=None):
    #     """
    #     Plots the sampling moment as a function of L

    #     Parameters
    #     ----------
    #     Lmax : int, optional
    #         DESCRIPTION. The default is 50.
    #     sym : TYPE, optional
    #         DESCRIPTION. The default is 'Ci'.
    #     weight : TYPE, optional
    #         DESCRIPTION. The default is None.

    #     Returns
    #     -------
    #     axis

    #     """
    #     error=self.sampling_moment(Lmax=Lmax,sym=sym,weight=weight)
    #     if ax is None:ax=plt.subplots()[1]
    #     ax.plot(np.arange(Lmax+1),error)
    #     ax.set_xlabel(r'$L$')
    #     ax.set_ylabel(r"$\sigma^S_{Lqq'}$")
    #     return ax
        
    # def SHREWDopt(self,Lmax:int=None,sym:str='Ci',update_wt:bool=True):
    #     """
    #     Generates a weight-optimized optimized powder average according to the 
    #     SHREWD scheme. 
        
    #     Matthias Eden, Malcolm H. Levitt. "Computational of Orientational Averages in
    #     Solid-State NMR by Gaussian Spherical Quadrature". J. Magn. Reson. 132, 220-239
    #     (1998)
        
    #     Input is Lmax, the largest tensor rank to be modeled with the powder 
    #     average, and a powder average to be optimized with new weights. 

    #     Parameters
    #     ----------
    #     Lmax : int, optional
    #         DESCRIPTION. The default is None.
    #     sym : str, optional
    #         DESCRIPTION. The default is 'Ci'.

    #     Returns
    #     -------
    #     None.

    #     """
    #     M=self.sampling_moment_matrix(Lmax=Lmax,sym=sym)
    #     print(M.shape)
    #     target=np.zeros(M.shape[0])
    #     target[0]=1
        
    #     w=lsq_linear(M,target,bounds=(0,1))['x']
    #     w/=w.sum()
        
    #     if update_wt:
    #         if self._gamma_incl:
    #             self.weight=w
    #         else:
    #             self.weight=np.tile(w,self.n_gamma)/self.n_gamma
    #     return w
    
    def __eq__(self,pwdavg):
        if pwdavg is self:return True
        for key in ['N','alpha','beta','gamma']:
            if not(np.all(getattr(pwdavg,key)==getattr(self,key))):return False
        return True
    
    def __len__(self):
        return self.N
    
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
        
        if isinstance(i,slice):
            out=copy(self)
            out._gamma_incl=True
            out.n_gamma=1
            out.alpha=self.alpha[i]
            out.beta=self.beta[i]
            out.gamma=self.gamma[i]
            # out.N=1
            out.weight=self.weight[i]
            out.weight/=out.weight.sum()
            
            return out
        out=copy(self)
        out._gamma_incl=True
        out.n_gamma=1
        j=i+1
        i%=len(self)
        j%=len(self)
        if j<=i:j+=len(self)
        out.alpha=self.alpha[i:j]
        out.beta=self.beta[i:j]
        out.gamma=self.gamma[i:j]
        # out.N=1
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
    def __init__(self,ex,delta=0,eta=0,euler=[0,0,0],rotor_angle=np.arccos(np.sqrt(1/3))):
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
        self.expsys=ex
        self.rotor_angle=rotor_angle
        
        self._Azz=None
        self._A=None
        self._Afull=None
        

    @property
    def pwdavg(self):
        return self.expsys.pwdavg
        
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
        self._PAS=np.array([-0.5*delta*eta,0,np.sqrt(3/2)*delta,0,-0.5*delta*eta])
        
    @property
    def PAS(self):
        return self._PAS.copy()
        
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
        self._MOL=A
    
    @property
    def MOL(self):
        return self._MOL.copy()
        
    def MOL2LAB_Azz(self):
        """
        Applies the powder average to the tensor stored in the molecular frame.
        
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
        
        self._Azz=A
    
    @property
    def Azz(self):
        if self._Azz is None or len(self._Azz)!=self.pwdavg.N:self.MOL2LAB_Azz()
        return self._Azz.copy()
    
    def MOL2LAB_A(self):
        """
        Applies the powder average to the tensor stored in the molecular frame.
        
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
        
        A=self.MOL
        alpha,beta,gamma=[getattr(self.pwdavg,x) for x in ['alpha','beta','gamma']]
        
        A=np.array([(D2(alpha,beta,gamma,mp=None,m=m)*A).sum(1) for m in range(-2,3)]).T
        
        self._A=A
    
    @property
    def A(self):
        if self._A is None or len(self._A)!=self.pwdavg.N:self.MOL2LAB_A()
        return self._A.copy()
    
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
        
        self._Afull=A
    
    @property
    def Afull(self):
        if self._Afull is None or len(self._Afull)!=self.pwdavg.N:self.MOL2LAB_Afull()
        return self._Afull.copy()
    
    def plot(self,n:int=0,avg=0,ax=None):
        """
        Creates a 3D plot of the tensor stored in RotInter. By default, shows
        the magnitude of the n=0 term, although -2, -1, 0, 1, or 2 may be selected
        One may also add an isotropic term, avg. This is not available from 
        RotInter, but is available to the Hamiltonian.
        
        Note if n!=0, then avg is ignored.

        Returns
        -------
        axis

        """
        if ax is None:ax=plt.figure().add_subplot(1,1,1,projection='3d')
        alpha=self.pwdavg.alpha
        beta=self.pwdavg.beta
        A=self.A[:,n+2]*np.sqrt(2/3) if n else self.A[:,2].real*np.sqrt(2/3)

        if not(self.pwdavg._gamma_incl):
            i=self.pwdavg.gamma==0
            alpha=alpha[i]
            beta=beta[i]
            A=A[i]
            
        if n:
            l=np.abs(A)
        else:
            l=A.real+avg
        
        sc=1
        lbl='Hz'
        if np.abs(A).max()>2e3:
            sc=1e-3
            lbl='kHz'
        if np.abs(A).max()>2e6:
            sc=1e-6  
            lbl='MHz'
        
        
        x=l*np.cos(alpha)*np.sin(beta)*sc
        y=l*np.sin(alpha)*np.sin(beta)*sc
        z=l*np.cos(beta)*sc
        
        cmap=plt.get_cmap('hsv')
        if n:
            ax.scatter3D(x,y,z,c=cmap(np.arctan2(A.imag,A.real)/(2*np.pi)+0.5))
        else:
            i=(A+avg)>=0
            if np.any(i):
                ax.scatter3D(x[i],y[i],z[i],linewidth=0.2,antialiased=True,color=cmap(0))
            i=(A+avg)<=0
            if np.any(i):
                ax.scatter3D(x[i],y[i],z[i],linewidth=0.2,antialiased=True,color=cmap(0.5))
        
        lim=max([ax.get_xlim()[1],ax.get_ylim()[1],ax.get_zlim()[1]])
        ax.set_xlim([-lim,lim])
        ax.set_ylim([-lim,lim])
        ax.set_zlim([-lim,lim])
        
        
        ax.set_box_aspect((1,1,1))
        ax.set_xlabel(lbl)
        ax.set_ylabel(lbl)
        ax.set_zlabel(lbl)
        
        return ax
        
    
