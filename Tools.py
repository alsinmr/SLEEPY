#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:44:13 2019

@author: albertsmith
"""

import os
import numpy as np
import re
from .Info import Info
from .vft import Spher2pars

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
        h=6.6260693e-34
        muen=5.05078369931e-27
        super().__init__()
        dir_path = os.path.dirname(os.path.realpath(__file__))    
        with open(dir_path+'/GyroRatio.txt','r') as f:
            for line in f:
                line=line.strip().split()
                self.new_exper(Nuc=line[3],mass=float(line[1]),spin=float(line[5]),\
                               gyro=float(line[6])*muen/h,abundance=float(line[7])/100)
    
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
NucInfo.new_exper(Nuc='e-',mass=0,spin=1/2,gyro=-1.76085963023e11/2/np.pi,abundance=1)

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
    
    
    
       