#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 13:55:04 2023

@author: albertsmith
"""

import numpy as np
from scipy.linalg import expm
from . import Defaults
from .Tools import Ham2Super
from copy import copy

rtype=Defaults['rtype']

#%% Relaxation Functions    
def T1(expsys,i:int,T1:float):
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
        Desired T1 relaxation time.

    Returns
    -------
    None.

    """
    
    N=expsys.Op.Mult[i]
    
    # Matrix leading to T1 relaxation
    # p=np.zeros([N,N])
    # for n in range(N-1):
    #     p[n,n+1]=1
    #     p[n+1,n]=1
    # p-=np.diag(p.sum(0))
    # p*=1/(T1*2)
    
    # Add offset to get desired polarization
    # if Peq:
    #     Peq=expsys.Peq[i]
        
    #     peq=np.zeros([N,N])
    #     for n in range(N-1):
    #         peq[n,n+1]=Peq
    #         peq[n+1,n]=-Peq
    #     peq-=np.diag(peq.sum(0))
    #     peq*=1/(T1*2)
        
    #     p+=peq
    
    
    sz=expsys.Op.Mult.prod()

    Lp=Ham2Super(expsys.Op[i].p)
    Lm=Ham2Super(expsys.Op[i].m)
    M=Lp@Lm+Lm@Lp
    M=Lp@Lm
    # return -M.real/(2*T1)/M[0,0].real
    
    
    M-=np.diag(np.diag(M))
    index=np.argwhere(M)
    index.sort()
    index=np.unique(index,axis=0)

    out=np.zeros([sz**2,sz**2],dtype=Defaults['rtype'])
    # I'm not sure how valid this is for spin>1/2
    for id0,id1 in index:
        out[id0,id1]=1/(T1*2)
        out[id1,id0]=1/(T1*2)
        # out[id0,id0]=p[0,0]
        # out[id0,id1]=p[0,-1]
        # out[id1,id0]=p[-1,0]
        # out[id1,id1]=p[1,1]
    out-=np.diag(out.sum(0))

    return out

def SpinDiffusion(expsys,i:int,k:float):
    """
    Constructs a "spin-diffusion" operator, as suggested by Ernst et al.
    
    Ernst, Zimmermann, Meier, Chem. Phys. Lett. 2000, 317, 581

    Parameters
    ----------
    expsys : TYPE
        DESCRIPTION.
    i : int
        Index of the spin.
    k : float
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    Lx=Ham2Super(expsys.Op[i].x)
    Ly=Ham2Super(expsys.Op[i].y)
    Lz=Ham2Super(expsys.Op[i].z)
    
    M=Lx@Lx+Ly@Ly+Lz@Lz
    
    return -k*M

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
        Desired relaxation time constant.

    Returns
    -------
    None.

    """
    
    # N=expsys.Op.Mult[i]
    # P=np.eye(N**2,dtype=Defaults['rtype'])*(-1/T2)
    # index=np.arange(0,N**2,N+1)
    # for i0 in index:P[i0,i0]=0
    
    # # # Expand to the size of the full Liouville space
    # # out=np.kron(np.kron(np.eye(expsys.Op.Mult[:i].prod()**2),P),np.eye(expsys.Op.Mult[i+1:].prod()**2))
    
    # N=expsys.Op.Mult.prod()
    
    # i=(expsys.Op[i].p+expsys.Op[i].m).astype(bool).reshape(N**2)
    # out0=np.zeros(N**2,dtype=Defaults['rtype'])
    # out0[i]=1/T2
    
    Lz=Ham2Super(expsys.Op[i].z)
    out=(Lz@Lz).astype(bool).astype(Defaults['rtype'])*(-1/T2)
    
    return out

# def Thermal(L,step):
#     if L.reduced:
#         block=L.block
#         L=L._L
#     else:
#         block=None
    
#     out=np.zeros(L[0].Ln(0).shape,dtype=Defaults['ctype'])
#     index=np.argwhere(L.Lrelax-np.diag(np.diag(L.Lrelax)))
#     index.sort(-1)
#     index=np.unique(index,axis=0)
        
#     rho_eq=L.rho_eq(pwdindex=L._index,step=step,sub1=False)
#     E=L.Energy2(step)
#     for i0,i1 in index:
#         if rho_eq[i0]==0 or rho_eq[i1]==0 or True:
#             DelE=E[i0]-E[i1]
#             # DelE=(L.Energy[i0]-L.Energy[i1])
#             rat=np.exp(DelE/(1.380649e-23*L.expsys.T_K))
#         else:
#             rat=rho_eq[i0]/rho_eq[i1]
#         Del=L.Lrelax[i0,i1]*(1-rat)/(1+rat)
#         out[i0,i1]=-Del
#         out[i1,i1]+=Del
#         out[i1,i0]=Del
#         out[i0,i0]+=-Del
        
#     if block is None:
#         return out
#     else:
#         return out[block][:,block]
    

def recovery(expsys,L):
    # L.Lex
    # d,v=np.linalg.eig(L.kex)
    # pop=v[:,np.argmax(d)]    #We need to make sure we start at equilibrium
    # pop/=pop.sum()
    # N=expsys.Op.Mult.prod()
    # rho_eq=np.zeros([N**2*len(L.H)],dtype=Defaults['ctype'])
    # for k,H in enumerate(L[0].H):   #Here, we select the first element of powder average (wrong off magic angle?)
    #     H0=H.Hn(0)
    #     for q,b in enumerate(expsys.LF):
    #         if not(b):
    #             H0+=expsys.v0[q]*expsys.Op[q].z  #Don't forget the Larmor terms!
    #     x=(expm(6.62607015e-34*H0/(1.380649e-23*expsys.T_K)))
    #     Z=np.trace(x@x.conj().T)
    #     # print(Z)
    #     x-=np.eye(H0.shape[0])
    #     x/=Z
    #     # x/=expsys.Op.Mult.prod()
        
    #     rho_eq[k*N**2:(k+1)*N**2]=x.reshape(x.size)
    
    
    ## Version 2
    # rho_eq=L.rho_eq()

    # Lrhoeq=(L[0].Ln(0)@rho_eq)
    # out=np.zeros(L[0].Ln(0).shape,dtype=Defaults['ctype'])
    
    # n=np.prod(L[0].H[0].shape)
    # i0=np.arange(0,n,expsys.Op.Mult.prod()+1)
    # i=np.concatenate([i0+k*n for k in range(len(L.H))])
    
    # out[:,i]=-np.atleast_2d(Lrhoeq).T.repeat(i.size,axis=1)
    # L.recovery=-out
    # print('updated')
    # return out
    
    
    # ## Version 3
    out=np.zeros(L[0].Ln(0).shape,dtype=Defaults['ctype'])
    index=np.argwhere(L.Lrelax-np.diag(np.diag(L.Lrelax)))
    index.sort(-1)
    index=np.unique(index,axis=0)
    
    # Ln_H=L[0].Ln_H(0)/1j/(2*np.pi)
    # for LF,v0,Op in zip(expsys.LF,expsys.v0,expsys.Op):
    #     if not(LF):
    #         n=L.H[0].shape[0]**2
    #         for k in range(len(L.H)):
    #             Ln_H[k*n:(k+1)*n][:,k*n:(k+1)*n]+=Ham2Super(v0*Op.z)
        
    rho_eq=L.rho_eq()
    for i0,i1 in index:

        if rho_eq[i0]==0 or rho_eq[i1]==0:
            DelE=(L.Energy[i0]-L.Energy[i1])
            rat=np.exp(DelE/(1.380649e-23*expsys.T_K))
        else:            
            rat=rho_eq[i0]/rho_eq[i1]
        Del=L.Lrelax[i0,i1]*(1-rat)/(1+rat)
        out[i0,i1]=-Del
        out[i1,i1]+=Del
        out[i1,i0]=Del
        out[i0,i0]+=-Del

    L.recovery=out
    return out


#%% Spin-swap/spin exchange
def SpinExchange(expsys,i:list,tc:float):
    """
    Allows exchange among spins, for example, if a water molecule experiences
    a two-site hop. The hop does not change the values in the overall Hamiltonian,
    but changes the spin indexing. We can treat this kind of dynamics without
    rebuilding the entire Liouvillian; instead we just introduce exchange 
    within a single Liouvillian.
    
    One provides a list of the spins in exchange. Usually, this is just two 
    elements, but more is also possible. For example, methyl 3-site hopping
    would have a list such as [1,2,3]. This means that we either have the
    exchange process 1–>2, 2->3, and 3->1, or 1->3, 2->1, 3->2. The process must
    always be cyclic, with equal populations/equal probabilities in both directions.
    
    The correlation time is the inverse of the mean hopping rate constant. For
    two- and three-site exchange, there is only one unique hopping rate, but for
    higher numbers of states, the mean will be calculated

    Parameters
    ----------
    expsys : TYPE
        DESCRIPTION.
    i : list
        List of spins in exchange (given in order of exchange)
    tc : float
        Correlation time of the exchange (inverse of the rate constant)

    Returns
    -------
    None.

    """
    
    
    states=expsys.Op.state_index
    
    
    assert len(i)<=states.shape[1],"Swap index cannot be longer than the number of spins in the system"
    assert np.all([i0<states.shape[1] for i0 in i]),"Indices must be less than the number of spins"
    
    out=np.zeros([states.shape[0],states.shape[0]],dtype=Defaults['rtype'])
    
    for k in range(states.shape[0]):
        if np.all([states[k][i[0]]==s for s in states[k][i[1:]]]):continue
        
        out[k,k]+=-1
        
        state=copy(states[k])
        
        state[i[-1]]=state[i[0]]
        for a,b in zip(i[:-1],i[1:]):
            state[a]=states[k][b]
        
        m=np.all(states==state,axis=-1)
        out[k,m]=1
    
    out+=out.T
    l=-np.linalg.eig(out)[0].real
    Ravg=l[l>1e-5].mean()
    
    out/=tc*Ravg
    
    return out
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    