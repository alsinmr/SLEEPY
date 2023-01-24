#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 13:55:04 2023

@author: albertsmith
"""

import numpy as np
from . import Defaults

rtype=Defaults['rtype']

#%% Relaxation Functions    
def T1(expsys,i:int,T1:float,Peq=False):
    """
    Constructs a T1 matrix for a given spin in the spin-system, and expands it
    to the full size of the Liouville matrix (for 1 Hamiltonian).
    
    Only includes single quantum relaxation. Note for high-spin systems, relaxation
    is multi-exponential. In this case, the transition rate constants are 
    given by  k_{a,b}=1/(2*T1). For the spin-1/2 system, this will yield a rate
    constant of T1

    Parameters
    ----------
    expsys : ExpSys
        Experiment / Spin-system class
    i : int
        Index of the spin.
    T1 : float
        Desired T1 relaxation rate.
    Peq : bool, optional
        Use relaxation toward thermal equilibrium. By default, relaxation simply
        goes to zero. Temperature taken from expsys.

    Returns
    -------
    None.

    """
    
    N=expsys.Op.Mult[i]
    
    # Matrix leading to T1 relaxation
    p=np.zeros([N,N])
    for n in range(N-1):
        p[n,n+1]=1
        p[n+1,n]=1
    p-=np.diag(p.sum(0))
    p*=1/(T1*2)
    
    # Add offset to get desired polarization
    if Peq:
        Peq=expsys.Peq[i]
        
        peq=np.zeros([N,N])
        for n in range(N-1):
            peq[n,n+1]=Peq/4
            peq[n+1,n]=-Peq/4
        peq-=np.diag(peq.sum(0))
        
        p+=peq
    
    P=np.zeros([N**2,N**2])
    
    index=np.arange(0,N**2,N+1)
    for k in range(len(index)-1):
        P[index[k],index[k+1]]=p[k,k+1]
        P[index[k+1],index[k]]=p[k+1,k]
        P[index[k],index[k]]=p[k,k]
    P[index[-1],index[-1]]=p[-1,-1]
    
    # Expand to the size of the full Liouville space
    out=np.kron(np.kron(np.eye(expsys.Op.Mult[:i].prod()**2),P),np.eye(expsys.Op.Mult[i+1:].prod()**2))
    
    return out.astype(rtype)

def T2(expsys,i:int,T2:float):
    """
    Constructs the T2 relaxation matrix for a given spin in the spin-system. For
    spin>1/2, the multi-quantum relaxation will be the same as the single quantum
    relaxation. 
    
    Parameters
    ----------
    expsys : TYPE
        DESCRIPTION.
    i : int
        Index of the spin.
    T2 : float
        Desired relaxation rate constant.

    Returns
    -------
    None.

    """
    
    N=expsys.Op.Mult[i]
    P=np.eye(N**2,dtype=rtype)*(-1/T2)
    index=np.arange(0,N**2,N+1)
    for i0 in index:P[i0,i0]=0
    
    # Expand to the size of the full Liouville space
    out=np.kron(np.kron(np.eye(expsys.Op.Mult[:i].prod()**2),P),np.eye(expsys.Op.Mult[i+1:].prod()**2))
    
    return out