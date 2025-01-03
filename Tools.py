#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:44:13 2019

@author: albertsmith
"""

import os
import numpy as np
from copy import copy
import re
from .Info import Info
from .vft import Spher2pars
from . import Defaults
from . import Constants
import warnings
import matplotlib.pyplot as plt

#%% Some useful tools (Gyromagnetic ratios, spins, dipole couplings)
class NucInfo(Info):
    """ Returns the gyromagnetic ratio for a given nucleus. Usually, should be 
    called with the nucleus and mass number, although will default first to 
    spin 1/2 nuclei if mass not specified, and second to the most abundant 
    nucleus. A second argument, info, can be specified to request the 
    gyromagnetic ratio ('gyro'), the spin ('spin'), the abundance ('abund'), or 
    if the function has been called without the mass number, one can return the 
    default mass number ('mass'). If called without any arguments, a pandas 
    object is returned containing all nuclear info ('nuc','mass','spin','gyro',
    'abund')
    """
    def __init__(self):
        h=Constants['h']
        muen=Constants['muen']
        super().__init__()
        dir_path = os.path.dirname(os.path.realpath(__file__))    
        with open(dir_path+'/GyroRatio.txt','r') as f:
            for line in f:
                line=line.strip().split()
                self.new_exper(Nuc=line[3],mass=float(line[1]),spin=float(line[5]),\
                               gyro=float(line[6])*muen/h,abundance=float(line[7])/100)
        self.new_exper(Nuc='e-',mass=0,spin=1/2,gyro=Constants['ge']*Constants['mub'],abundance=1)
    
    def __call__(self,Nuc=None,info='gyro'):
        if Nuc is None:
            return self
        
        if Nuc=='D':Nuc='2H'
        if Nuc=='e':Nuc='e-'
 
        #Separate the mass number from the nucleus type       
        mass=re.findall(r'\d+',Nuc)
        if not mass==[]:
            mass=int(mass[0])
        
            
        if Nuc!='e-':
            Nuc=re.findall(r'[A-Z]',Nuc.upper())
            
            #Make first letter capital
            # Nuc=Nuc.capitalize()
            if np.size(Nuc)>1:
                Nuc=Nuc[0].upper()+Nuc[1].lower()
            else:
                Nuc=Nuc[0]
                        
        ftd=self[self['Nuc']==Nuc]  #Filtered by nucleus input
        
        "Now select which nucleus to return"
        if not mass==[]:    #Use the given mass number
           ftd=ftd[ftd['mass']==mass]
        elif any(ftd['spin']==0.5): #Ambiguous input, take spin 1/2 nucleus if exists
            ftd=ftd[ftd['spin']==0.5] #Ambiguous input, take most abundant nucleus
        elif any(ftd['spin']>0):
            ftd=ftd[ftd['spin']>0]
            
        ftd=ftd[np.argmax(ftd['abundance'])]
        
        if info is None or info=='all':
            return ftd
        else:
            assert info in self.keys,"info must be 'gyro','mass','spin','abundance', or 'Nuc'"
            return ftd[info]
    
    def __repr__(self):
        self['gyro',-1]=Constants['ge']*Constants['mub']
        out=''
        for k in self.keys:out+='{:7s}'.format(k)+'\t'
        out=out[:-1]
        fstring=['{:7s}','{:<7.0f}','{:<7.1f}','{:<3.4f}','{:<4.3f}']
        for nucs in self:
            out+='\n'
            for k,(v,fs) in enumerate(zip(nucs.values(),fstring)):
                out+=fs.format(v*(1e-6 if k==3 else 1))+'\t'
        return out
    
    def __getitem__(self,index):
        if isinstance(index,str):
            if index=='D':   #Deuterium
                return self[1]
            if index in ['e','e-']: #Electron
                self['gyro',-1]=Constants['ge']*Constants['mub']
                return self[-1]
            mass=re.findall(r'\d+',index)
            Nuc=re.findall(r'[A-Z]',index.upper())
            if len(mass) and len(Nuc):
                Nuc=Nuc[0].capitalize()
                i=np.logical_and(self['Nuc']==Nuc,self['mass']==int(mass[0]))
                if np.any(i):
                    return super().__getitem__(i)
        return super().__getitem__(index)
        
        
        
NucInfo=NucInfo()


def dipole_coupling(r,Nuc1,Nuc2):
    """ Returns the dipole coupling between two nuclei ('Nuc1','Nuc2') 
    separated by a distance 'r' (in nm). Result in Hz (gives full anisotropy,
    not b12, that is 2x larger than b12)
    """
    
    gamma1=NucInfo(Nuc1)
    gamma2=NucInfo(Nuc2)
    
    h=6.6260693e-34 #Plancks constant in J s
    mue0 = 12.56637e-7  #Permeability of vacuum [T^2m^3/J]
    
    return h*2*mue0/(4*np.pi*(r/1e9)**3)*gamma1*gamma2


def d2(c=0,s=None,m=None,mp=0):
    """
    Calculates components of the d2 matrix. By default only calculates the components
    starting at mp=0 and returns five components, from -2,-1,0,1,2. One may also
    edit the starting component and select a specific final component 
    (mp=None returns all components, whereas mp may be specified between -2 and 2)
    
    d2_m_mp=d2(m,mp,c,s)  #c and s are the cosine and sine of the desired beta angle
    
        or
        
    d2_m_mp=d2(m,mp,beta) #Give the angle directly
    
    Setting mp to None will return all values for mp in a 2D array
    
    (Note that m is the final index)
    """
    
    if s is None:
        c,s=np.cos(c),np.sin(c)
    
    """
    Here we define each of the components as functions. We'll collect these into
    an array, and then call them out with the m and mp indices
    """
    "First, for m=-2"
    
    if m is None or mp is None:
        assert m is not None or mp is not None,"m or mp must be specified"
        if m is None:
            if mp==-2:
                index=range(0,5)
            elif mp==-1:
                index=range(5,10)
            elif mp==0:
                index=range(10,15)
            elif mp==1:
                index=range(15,20)
            elif mp==2:
                index=range(20,25)
        elif mp is None:
            if m==-2:
                index=range(0,25,5)
            elif m==-1:
                index=range(1,25,5)
            elif m==0:
                index=range(2,25,5)
            elif m==1:
                index=range(3,25,5)
            elif m==2:
                index=range(4,25,5)
    else:
        index=[(mp+2)*5+(m+2)]
    
    out=list()    
    for i in index:
        #mp=-2
        if i==0:x=0.25*(1+c)**2
        if i==1:x=0.5*(1+c)*s
        if i==2:x=np.sqrt(3/8)*s**2
        if i==3:x=0.5*(1-c)*s
        if i==4:x=0.25*(1-c)**2
        #mp=-1
        if i==5:x=-0.5*(1+c)*s
        if i==6:x=c**2-0.5*(1-c)
        if i==7:x=np.sqrt(3/8)*2*c*s
        if i==8:x=0.5*(1+c)-c**2
        if i==9:x=0.5*(1-c)*s
        #mp=0
        if i==10:x=np.sqrt(3/8)*s**2
        if i==11:x=-np.sqrt(3/8)*2*s*c
        if i==12:x=0.5*(3*c**2-1)
        if i==13:x=np.sqrt(3/8)*2*s*c
        if i==14:x=np.sqrt(3/8)*s**2
        #mp=1
        if i==15:x=-0.5*(1-c)*s
        if i==16:x=0.5*(1+c)-c**2
        if i==17:x=-np.sqrt(3/8)*2*s*c
        if i==18:x=c**2-0.5*(1-c)
        if i==19:x=0.5*(1+c)*s
        #mp=2
        if i==20:x=0.25*(1-c)**2
        if i==21:x=-0.5*(1-c)*s
        if i==22:x=np.sqrt(3/8)*s**2
        if i==23:x=-0.5*(1+c)*s
        if i==24:x=0.25*(1+c)**2
        out.append(x)
        
    if m is None or mp is None:
        return np.array(out)
    else:
        return out[0]
        
        
def D2(ca=0,sa=0,cb=0,sb=None,cg=None,sg=None,m=None,mp=0):
    """
    Calculates components of the D2 matrix. By default only calculates the components
    starting at m=0 and returns five components, from -2,-1,0,1,2. One may also
    edit the starting component and select a specific final component 
    (mp=None returns all components, whereas mp may be specified between -2 and 2)
    
    d2_m_mp=d2(ca,sa,cb,sb,cg,sg,m,mp)  #c and s are the cosine and sine of the alpha,beta,gamma angle
    
        or
        
    d2_m_mp=d2(m,mp,beta) #Give the angle directly
    
    Setting mp to None will return all values for mp in a 2D array
    
    (Note that m is the final index)
    """
    
    if sb is None: #Convert to sines and cosines
        ca,sa,cb,sb,cg,sg=np.cos(ca),np.sin(ca),np.cos(sa),np.sin(sa),np.cos(cb),np.sin(cb)

    if m is None:
        p0=(ca-1j*sa*np.sign(mp))**np.abs(mp)
        p=cg-1j*sg
        phase=np.array([p**m for m in range(-2,3)])*p0
    elif mp is None:
        p0=(cg-1j*sg*np.sign(m))**np.abs(m)
        p=ca-1j*sa
        phase=np.array([p**mp for mp in range(-2,3)])*p0
    else:
        phase=((ca-1j*sa*np.sign(mp))**np.abs(mp))*((cg-1j*sg*np.sign(m))**np.abs(m))
        
    return (d2(cb,sb,m,mp)*phase).T

def Ham2Super(H):
    """
    Calculates
    kron(H,eye(H.shape))-kron(eye(H.shape),H.T), while avoiding actually
    calculating the full Kronecker product

    Parameters
    ----------
    H : np.array
        Hamiltonian.

    Returns
    -------
    None.

    """
    n=H.shape[0]
    Lp=np.zeros([n**2,n**2],dtype=H.dtype)
    Lm=np.zeros([n**2,n**2],dtype=H.dtype)
    for k in range(n):
        Lp[k::n][:,k::n]=H
        Lm[k*n:(k+1)*n][:,k*n:(k+1)*n]=H.T
    return Lp-Lm

def LeftSuper(X):
    """
    Returns the super operator in Liouville spacefor square matrix X that 
    yields left multiplication in the Hilbert space
    
    For an nxn matrix X and a matrix or vector m,
    (X*m).reshape(n) = LeftSuper(X)*m.reshape(n)

    Parameters
    ----------
    X : np.array
        Square matrix (nxn)

    Returns
    -------
    Super : np.array
        Superoperator (n**2xn**2)

    """
    n=X.shape[0]
    Super=np.zeros([n**2,n**2],dtype=X.dtype)
    for k in range(n):
        Super[k::n][:,k::n]=X
    return Super

def RightSuper(X):
    """
    Returns the super operator in Liouville spacefor square matrix X that 
    yields right multiplication in the Hilbert space
    
    For an nxn matrix X and a matrix or vector m,
    (m*X).reshape(n) = RightSuper(X)*m.reshape(n)

    Parameters
    ----------
    X : np.array
        Square matrix (nxn)

    Returns
    -------
    Super : np.array
        Superoperator (n**2xn**2)

    """
    n=X.shape[0]
    Super=np.zeros([n**2,n**2],dtype=X.dtype)
    for k in range(n):
        Super[k*n:(k+1)*n][:,k*n:(k+1)*n]=X.T
    return Super
    
    
def BlockDiagonal(M):
    """
    Determines connectivity of a matrix, allowing us to represent a large
    matrix as several smaller matrices. Speeds up matrix exponential, matrix
    multiplication

    Parameters
    ----------
    M : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    assert M.shape[0]==M.shape[1],'Matrix should be square for BlockDiagonal calculation'
    
    X=M.astype(bool)+M.astype(bool).T
    
    Blks=list()
    unchecked=np.ones(X.shape[0],dtype=bool)

    while np.any(unchecked):
        bl=np.zeros(X.shape[0],dtype=bool)
        bl[np.argwhere(unchecked)[0,0]]=True
        while np.any(np.logical_and(bl,unchecked)):
            i=np.argwhere(np.logical_and(unchecked,bl))[0,0]
            unchecked[i]=False
            m=np.logical_and(X[i],unchecked)
            bl[m]=True
        Blks.append(bl)
    
    return Blks
            

def twoSite_kex(tc:float,p1:float=0.5):
    """
    Returns a matrix for two-site exchange with correlation time tc and 
    population of the first state p1 (p2=1-p1)

    Parameters
    ----------
    tc : float
        Correlation time
    p1 : float, optional
        Population of state 1. The default is 0.5
        

    Returns
    -------
    np.array
        kex matrix

    """
    
    return 1/(2*tc)*(np.array([[-1,1],[1,-1]])+(2*p1-1)*np.array([[1,1],[-1,-1]]))

def nSite_sym(n:int,tc:float):
    """
    Returns a matrix for n-sites, where all sites are directly connected with
    all other sites with the same rate constant. This type of exchange always
    has exactly one unique correlation time.

    Parameters
    ----------
    n : int
        Number of sites in exchange.
    tc : float
        Correlation time of the exchange.

    Returns
    -------
    np.array
        kex matrix

    """
    return 1/(n*tc)*(np.ones([n,n])-np.eye(n)*n)

def fourSite_sym(tc:float):
    """
    Exchange matrix which can be used to mimic isotropic tumbling, replaced
    with a four site isotropic hopping. Should be coupled with a tetrahedral
    hopping geometry for isotropic behavior

    Parameters
    ----------
    tc : float
        correlation time.

    Returns
    -------
    None.

    """
    return 1/(4*tc)*(np.ones([4,4])-np.eye(4)*4)
    

def twoSite_S_eta(theta:float,p:float=0.5):
    """
    Calculates S and eta for a two-site hop. Provide one or more opening angles
    (theta) and optionally the population of the first site (p)

    Parameters
    ----------
    theta : float (or list of floats)
        Opening angle or angles.
    p     : float, optional
        Population of the first site. The default is 0.5
        
        

    Returns
    -------
    Tuple
    S and eta for the two site hop

    """
    
    oneD=hasattr(theta,'__len__')
    theta=np.atleast_1d(theta)
    
    A=(p*np.array([0,0,np.sqrt(3/2),0,0])+(1-p)*np.sqrt(3/2)*d2(theta).T).T
    
    S,eta=Spher2pars(A)[:2]
            
    if oneD:return S,eta
    
    return S[0],eta[0]

def SetupTumbling(expsys,tc:float,q:int=3):
    """
    Takes an expsys and adds a simulated tumbling to it, returning a list
    of expsys's and an exchange matrix to use to set up a Liouvillian
    
    e.g.
    L=sl.Liouvillian(ex_list,kex=kex)

    Parameters
    ----------
    expsys : TYPE
        SLEEPY expsys
    tc : float
        Desired correlation time of tumbling.

    Returns
    -------
    ex_list : list
        list of expsys
    kex : np.array
        Exchange matrix
    """
    
    
    kex,euler=tumbling(tc,q)
    
    H=expsys.Hamiltonian()[0].Hinter
    
    ex_list=[]
    for k,euler0 in enumerate(euler):
        ex_list.append(expsys.copy())
        for H0 in H:
            if not(H0.isotropic):
                kwargs=copy(H0.info)
                if hasattr(kwargs['euler'][0],'__len__'):
                    kwargs['euler'].append(euler0)
                else:
                    kwargs['euler']=[kwargs['euler'],euler0]
                
                ex_list[-1].set_inter(**kwargs)
                
    return ex_list,kex

def SetupTetraHop(expsys,tc:float,n:int=4):
    """
    Sets up a system undergoing tetrahedral hopping with from 2-4 sites

    Parameters
    ----------
    expsys : TYPE
        SLEEPY expsys
    tc : float
        Desired correlation time of tumbling.
    n  : number of tetrahedral sites in exchange (2-4)

    Returns
    -------
    ex_list : list
        list of expsys
    kex : np.array
        Exchange matrix
    """
            
    ex_list,kex=SetupTumbling(expsys, tc,q=1)
    if n==4:return ex_list,kex
    return ex_list[4-n:],nSite_sym(n=n, tc=tc)

def Setup3siteSym(expsys,tc:float,phi:float=np.arccos(-1/3)):
    """
    Sets up a system undergoing 3-site symmetric hopping

    Parameters
    ----------
    expsys : TYPE
        SLEEPY expsys
    tc : float
        Desired correlation time of tumbling.
    phi : Desired opening angle of hopping

    Returns
    -------
    ex_list : list
        list of expsys
    kex : np.array
        Exchange matrix
    """
    ex_list,kex=SetupTetraHop(expsys,tc=tc,n=3)
    
    for ex in ex_list:
        for inter in ex.inter:
            if 'euler' in inter:
                inter['euler'][-1][1]=phi
                
    return ex_list,kex

def tumbling(tc:float,q:int=3):
    """
    Constructs an exchange matrix for isotropic tumbling based on one of the
    "rep" powder averages. q (index from 0 to 10, 0 is just a tetrahedral hop).
    Returns the exchange matrix and the corresponding list of euler angles.
    
    Currently set up only for averaging symmetrix (eta=0) tensors
    

    Parameters
    ----------
    pwdavg : TYPE
        DESCRIPTION.
    tc : float
        DESCRIPTION.

    Returns
    -------
    kex : np.array
        Exchange matrix
        
    euler : np.array
        Euler angles corresponding to exchange matrix

    """
    
    assert q>=0 and q<=11,"q must be an integer between 0 and 11"
    

    n0=[3,4,10,20,30,66,100,144,168,256,320,678,2000]
    nc0=[2,3,5,6,6,6,6,6,6,6,6,6,6]

    n=n0[q]
    nc=nc0[q]

    if q==0:
        beta=[0,np.pi/2,np.pi/2]
        gamma=[0,0,np.pi/2]
    elif q==1:
        tetra=np.arccos(-1/3)
        beta=np.array([0,tetra,tetra,tetra])
        gamma=np.array([0,0,2*np.pi/3,4*np.pi/3])
    else:
        pwdpath=os.path.join(os.path.dirname(os.path.realpath(__file__)),'PowderFiles')
        pwdfile=os.path.join(pwdpath,f'rep{n}.txt')
        
        with open(pwdfile,'r') as f:
            alpha,beta=list(),list()
            for line in f:
                if len(line.strip().split(' '))==3:
                    a,b,w=[float(x) for x in line.strip().split(' ')]
                    alpha.append(a*np.pi/180)
                    beta.append(b*np.pi/180)
        
        
        gamma,beta=np.array(alpha),np.array(beta)
        
    euler=np.concatenate([[np.zeros(n)],[beta],[gamma]],axis=0).T
        
    x,y,z=np.sin(beta)*np.cos(gamma),np.sin(beta)*np.sin(gamma),np.cos(beta)
    
    kex=np.zeros([len(gamma),len(gamma)],dtype=Defaults['rtype'])

    for k in range(n):
        c=x[k]*x+y[k]*y+z[k]*z
        i=np.argsort(c)[-nc-1:-1]
        for i0 in i:
            kex[k,i0]=(1/np.arccos(c[i0]))**2
    
    kex=(kex+kex.T)/2
            
    kex-=np.diag(kex.sum(0))
    
    _,_,tc0,A=kex2A(kex,euler)
    
    kavg=((1/tc0)*A).sum()
    
    kex/=kavg*tc
    
    return kex,euler
    
    
def kex2A(kex,euler):
    """
    Calculates the order parameter, correlation times, and amplitudes resulting
    from an exchange matrix, and the corresponding n Euler angles.
    
    The corresponding correlation function is then
    
    C(t)=S2 + (1-S2)*sum(Ai*np.exp(-t/tci))

    Parameters
    ----------
    kex : np.array
        nxn exchange matrix (should satisfy mass conservation and detailed balance).
    euler : np.array
        list of euler angles corresponding to each state in the exchange matrix.

    Returns
    -------
    S2 : float
        Order parameter (S^2, not S).
    peq : np.array
        Equilibrium populations
    tc : np.array
        correlation times.
    A : np.array
        Amplitudes.

    """
    
    tci,v=np.linalg.eigh(kex)
    tc=-1/tci[:-1]
    
    peq=v[:,-1]
    peq/=peq.sum()
    
    beta,gamma=euler.T[-2:]
    
    x,y,z=np.sin(beta)*np.cos(gamma),np.sin(beta)*np.sin(gamma),np.cos(beta)
    
    P2=-1/2+3/2*(np.sum([np.atleast_2d(q).T@np.atleast_2d(q) for q in [x,y,z]],0)**2)
    
    S2=((np.atleast_2d(peq).T@np.atleast_2d(peq))*P2).sum()
    
    vi=np.linalg.pinv(v)
    
    A=list()
    for vim,vm in zip(vi,v.T):
        A.append((np.dot(np.atleast_2d(vm).T,np.atleast_2d(vim*peq))*P2).sum())

    A=np.array(A).real
    
    A=A[:-1]/(1-S2)
    
    return S2,peq,tc,A

def commute(A,B):
    """
    Returns the commutator of square matrices A and B

    Parameters
    ----------
    A : np.array
        Square matrix.
    B : np.array
        Square matrix.

    Returns
    -------
    np.array
        [A,B]=A@B-B@A

    """
    return A@B-B@A

def ApodizationFun(t,WDW:str='em',LB:float=None,SSB:float=2,GB:float=15,**kwargs):
    wdw=WDW.lower()
    if LB is None:LB=5/t[-1]/np.pi
    
    if wdw=='em':
        apod=np.exp(-t*LB*np.pi)
    elif wdw=='gm':
        apod=np.exp(-np.pi*LB*t+(np.pi*LB*t**2)/(2*GB*t[-1]))
    elif wdw=='sine':
        if SSB>=2:
            apod=np.sin(np.pi*(1-1/SSB)*t/t[-1]+np.pi/SSB)
        else:
            apod=np.sin(np.pi*t/t[-1])
    elif wdw=='qsine':
        if SSB>=2:
            apod=np.sin(np.pi*(1-1/SSB)*t/t[-1]+np.pi/SSB)**2
        else:
            apod=np.sin(np.pi*t/t[-1])**2
    elif wdw=='sinc':
        apod=np.sin(2*np.pi*SSB*(t/t[-1]-GB))
    elif wdw=='qsinc':
        apod=np.sin(2*np.pi*SSB*(t/t[-1]-GB))**2
    else:
        warnings.warn(f'Unrecognized apodization function: "{wdw}"')
        apod=np.ones(t.shape)
    return apod


class TwoD_Builder():
    """
    Class for building, running, and processing two-dimensional spectra
    """
    
    def __init__(self,rho,seq_in,seq_dir,seq_trX,seq_trY):
        """
        Sets up the two-D sequence
        

        Parameters
        ----------
        rho : Rho
            Density operator, prepared in the desired starting state.
        seq_in : Sequence
            Sequence to run during the indirect dimension
        seq_dir : Sequence
            Sequence to run during the direct dimension
        seq_trX : Sequence
            Sequence to run between direct and indirect dimensions (to capture x-component)
        seq_trY : Sequence
            Sequence to run between direct and indirect dimensions (to capture y-component)

        Returns
        -------
        None.

        """
        rho.clear()
        rho,seq_in,seq_dir,seq_trX,seq_trY=rho.ReducedSetup(seq_in,seq_dir,seq_trX,seq_trY)
        
        if len(rho.detect)>1:warnings.warn('TwoD_Builder will only use the first detection operator')
        
        self.rho=rho
        self._t=self.rho._t
        self._rho=copy(rho._rho)
        self.seq_in=seq_in
        self.seq_dir=seq_dir
        self.seq_trX=seq_trX
        self.seq_trY=seq_trY
        
        self._U=None
        self.Ireal=None
        self.Iimag=None
        self.Sreal=None
        self.Simag=None
    
        self.apod_pars={'WDW':['qsine','qsine'],'LB':[None,None],'SSB':[2,2],'GB':[15,15],'SI':[None,None]}
        
    def __call__(self,n_in:int,n_dir:int):
        """
        Run the twoD sequence

        Parameters
        ----------
        n_in : int
            DESCRIPTION.
        n_dir : int
            DESCRIPTION.

        Returns
        -------
        self

        """
        
        
        self.Sreal=None
        self.Simag=None
        
        self.L.reset_prop_time()
        for k in range(n_in):
            self.rho.reset()
            self.rho._t=self._t
            self.rho._rho=copy(self._rho)
            if self.Uin is not None:
                self.Uin**k*self.rho
            else:
                for _ in range(k):
                    self.seq_in*self.rho
            if self.UtrX is not None:
                self.UtrX*self.rho
            else:                
                self.seq_trX*self.rho
            if self.Udir is not None:
                self.rho.DetProp(self.Udir,n=n_dir)
            else:
                self.rho.DetProp(self.seq_dir,n=n_dir)
            # Ireal.append(rho0.I[0])
        
        self.L.reset_prop_time()
        for k in range(n_in):
            self.rho.reset()
            self.rho._t=self._t
            self.rho._rho=copy(self._rho)
            if self.Uin is not None:
                self.Uin**k*self.rho
            else:
                for _ in range(k):
                    self.seq_in*self.rho
            if self.UtrY is not None:
                self.UtrY*self.rho
            else:                
                self.seq_trY*self.rho
            if self.Udir is not None:
                self.rho.DetProp(self.Udir,n=n_dir)
            else:
                self.rho.DetProp(self.seq_dir,n=n_dir)
            # Iimag.append(rho0.I[0])
            
        I=(np.array(self.rho._Ipwd[0]).T*self.rho.pwdavg.weight).sum(-1)
            
        self.Ireal=I[:n_in*n_dir].reshape([n_in,n_dir]).T
        self.Iimag=I[n_in*n_dir:].reshape([n_in,n_dir]).T
        return self
        
    def proc(self,apodize:bool=True):
        """
        Processes the data in two dimensions, according to the processing 
        parameters found in apod_pars

        Returns
        -------
        None.

        """
        if self.Ireal is None:return
        ap={key:value[0] for key,value in self.apod_pars.items()}
        apod_in=ApodizationFun(self.t_in, **ap)
        ap={key:value[1] for key,value in self.apod_pars.items()}
        apod_dir=ApodizationFun(self.t_dir, **ap)
        RE=copy(self.Ireal)
        IM=copy(self.Iimag)
        
        # Divide first points by two
        RE[:,0]/=2
        RE[0,:]/=2
        IM[:,0]/=2
        IM[0,:]/=2
        
        if apodize:
            RE=RE*apod_dir
            IM=IM*apod_dir
            RE=(RE.T*apod_in).T
            IM=(IM.T*apod_in).T
        
        if self.apod_pars['SI'][0] is None:
            self.apod_pars['SI'][0]=RE.shape[0]*2
            
        if self.apod_pars['SI'][1] is None:
            self.apod_pars['SI'][1]=RE.shape[1]*2
        
        RE=np.fft.fft(RE,n=self.apod_pars['SI'][1],axis=0)
        IM=np.fft.fft(IM,n=self.apod_pars['SI'][1],axis=0)
        
        self.Sreal=np.fft.fftshift(np.fft.fft(RE.real+1j*IM.real,n=self.apod_pars['SI'][0],axis=1),axes=[0,1])
        self.Simag=np.fft.fftshift(np.fft.fft(RE.imag+1j*IM.imag,n=self.apod_pars['SI'][0],axis=1),axes=[0,1])
        
        return self
        
    def plot(self,ax=None):
        if self.Ireal is None:
            warnings.warn('Run TwoD_Builder before plotting')
            return
        if self.Sreal is None and self.Ireal is not None:self.proc()
        if ax is None:ax=plt.figure().add_subplot(1,1,1,projection='3d')
        
        x,y=np.meshgrid(self.v_in,self.v_dir)
        
        ax.plot_surface(x/1e3,y/1e3,self.Sreal.real,cmap='coolwarm',linewidth=0)
        
        ax.set_xlabel(r'$\delta$ / kHz')
        ax.set_ylabel(r'$\delta$ / kHz')
        
        return ax
        
    @property
    def t_in(self):
        if self.Ireal is None:return None
        return np.arange(self.Ireal.shape[0])*self.seq_in.Dt
    
    @property
    def t_dir(self):
        if self.Ireal is None:return None
        return np.arange(self.Ireal.shape[1])*self.seq_dir.Dt
    
    @property
    def v_in(self):
        if self.apod_pars['SI'][0] is None:return None
        v=1/(2*self.seq_in.Dt)*np.linspace(-1,1,self.apod_pars['SI'][0])
        v-=(v[1]-v[0])/2
        return v
    
    @property
    def v_dir(self):
        if self.apod_pars['SI'][1] is None:return None
        v=1/(2*self.seq_dir.Dt)*np.linspace(-1,1,self.apod_pars['SI'][1])
        v-=(v[1]-v[0])/2
        return v
    
    @property
    def L(self):
        return self.rho.L
    
    @property
    def _seq(self):
        return self.seq_in,self.seq_dir,self.seq_trX,self.seq_trY
    
    @property
    def fixedU(self):
        if self.L.static:return [True,True,True,True]
        Dt=self.L.taur
        out=[]
        for seq in self._seq:
            if hasattr(seq,'add_channel'):
                out.append(Dt==seq.Dt)
            else:
                out.append(True) #U provided instead of sequence
        return out
    
    @property
    def U(self):
        if self._U is not None:
            return self._U
        
        self._U=[]
        for k,seq in enumerate(self._seq):
            if self.fixedU[k]:
                if hasattr(seq,'add_channel'):
                    self._U.append(self._seq[k].U())                    
                else:
                    self._U.append(seq) #U provided instead of sequence

            else:
                self._U.append(None)
                
        return self._U
                
    @property
    def Uin(self):
        return self.U[0]
    @property
    def Udir(self):
        return self.U[1]
    @property
    def UtrX(self):
        return self.U[2]
    @property
    def UtrY(self):
        return self.U[3]
    
    
       