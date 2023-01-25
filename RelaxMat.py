#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 13:55:04 2023

@author: albertsmith
"""

import numpy as np
from . import Defaults
from .Tools import Ham2Super

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
            peq[n,n+1]=-Peq
            peq[n+1,n]=Peq
        peq-=np.diag(peq.sum(0))
        peq*=1/(T1*2)
        
        p+=peq
    
    
    sz=expsys.Op.Mult.prod()
    # diag=np.arange(0,sz**2,sz+1)
    # step=expsys.Op.Mult[i+1:].prod()
    # Len=expsys.Op.Mult[:i].prod()

    # out=np.zeros([sz**2,sz**2],dtype=rtype)
    # for m in range(Len):
    #     diag0=diag[m*Len:(m+1)*Len]
    #     for k in range(step):
    #         for q,(d,d1) in enumerate(zip(diag0[k:sz//Len:step],diag0[k+step:sz//Len:step])):
    #             out[d,d1]=p[q,q+1]
    #         for q,(d,d1) in enumerate(zip(diag0[k+step:sz//Len:step],diag0[k:sz//Len:step])):
    #             out[d,d1]=p[q+1,q]
    #         for q,d in enumerate(diag0[k:sz//Len:step]):
    #             out[d,d]=p[q,q]
    
    Lp=Ham2Super(expsys.Op[i].p)
    Lm=Ham2Super(expsys.Op[i].m)
    M=Lp@Lm+Lm@Lp
    M-=np.diag(np.diag(M))
    index=np.argwhere(M)
    index.sort()
    index=np.unique(index,axis=0)

    out=np.zeros([sz**2,sz**2],dtype=rtype)
    # This is only valid for spin-1/2!!!
    for id0,id1 in index:
        # out[id0,id0]=p[0,0]
        out[id0,id1]=p[0,-1]
        out[id1,id0]=p[-1,0]
        # out[id1,id1]=p[1,1]
    out-=np.diag(out.sum(0))

    
    # P=np.zeros([N**2,N**2])
    
    # index=np.arange(0,N**2,N+1)
    # for k in range(len(index)-1):
    #     P[index[k],index[k+1]]=p[k,k+1]
    #     P[index[k+1],index[k]]=p[k+1,k]
    #     P[index[k],index[k]]=p[k,k]
    # P[index[-1],index[-1]]=p[-1,-1]
    
    # Expand to the size of the full Liouville space
    # out=np.kron(np.kron(np.eye(expsys.Op.Mult[:i].prod()**2),P),np.eye(expsys.Op.Mult[i+1:].prod()**2))
    
    return out

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
    
    # N=expsys.Op.Mult[i]
    # P=np.eye(N**2,dtype=rtype)*(-1/T2)
    # index=np.arange(0,N**2,N+1)
    # for i0 in index:P[i0,i0]=0
    
    # # # Expand to the size of the full Liouville space
    # # out=np.kron(np.kron(np.eye(expsys.Op.Mult[:i].prod()**2),P),np.eye(expsys.Op.Mult[i+1:].prod()**2))
    
    # N=expsys.Op.Mult.prod()
    
    # i=(expsys.Op[i].p+expsys.Op[i].m).astype(bool).reshape(N**2)
    # out0=np.zeros(N**2,dtype=rtype)
    # out0[i]=1/T2
    
    Lz=Ham2Super(expsys.Op[i].z)
    out=(Lz@Lz).astype(bool).astype(rtype)*(-1/T2)
    
    return out