#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 09:16:50 2023

@author: albertsmith
"""


import numpy as np
from scipy.linalg import expm

import multiprocessing as mp


#%% Numba attempt (I hate numba)
# from . import Defaults
# def expm(M):
#     out=np.eye(M.shape[0],dtype=Defaults['ctype'])
#     ac=np.eye(M.shape[0],dtype=Defaults['ctype'])
#     for k in range(10):
#         ac=M@ac
#         out+=ac/np.math.factorial(k+1)
#     return out



# from numba import njit
# @njit(parallel=True,nopython=True,cache=True)
# def prop(Ln,Lrf,n0,nf,tm1,tp1,dt,n_gamma):
#     print('checkpoint1')
#     U=list()
#     for Ln0 in Ln:
#         ph=np.exp(1j*2*np.pi*n0/n_gamma)
#         L=np.sum([Ln0[m+2]*(ph**(-m)) for m in range(-2,3)],axis=0)+Lrf
#         U.append(expm(L*tm1))
#         for n in range(n0+1,nf):
#             ph=np.exp(1j*2*np.pi*n/n_gamma)
#             L=np.sum([Ln0[m+2]*(ph**(-m)) for m in range(-2,3)],axis=0)+Lrf
#             U[-1]=np.dot(expm(L*dt),U[-1])
#         if tp1>1e-10:
#             ph=np.exp(1j*2*np.pi*nf/n_gamma)
#             L=np.sum([Ln0[m+2]*(ph**(-m)) for m in range(-2,3)],axis=0)+Lrf
#             U[-1]=np.dot(expm(L*dt),U[-1])
#     return U
            

#%% Parallel attempt
def prop0(X):
    Ln0,Lrf,n0,nf,tm1,tp1,dt,n_gamma=X
    
    ph=np.exp(1j*2*np.pi*n0/n_gamma)
    L=np.sum([Ln0[m+2]*(ph**(-m)) for m in range(-2,3)],axis=0)+Lrf
    U=expm(L*tm1)
    for n in range(n0+1,nf):
        ph=np.exp(1j*2*np.pi*n/n_gamma)
        L=np.sum([Ln0[m+2]*(ph**(-m)) for m in range(-2,3)],axis=0)+Lrf
        U=expm(L*dt)@U
    if tp1>1e-10:
        ph=np.exp(1j*2*np.pi*nf/n_gamma)
        L=np.sum([Ln0[m+2]*(ph**(-m)) for m in range(-2,3)],axis=0)+Lrf
        U=expm(L*dt)@U
    
    return U

def prop(Ln,Lrf,n0,nf,tm1,tp1,dt,n_gamma):    
    X=[(Ln0,Lrf,n0,nf,tm1,tp1,dt,n_gamma) for Ln0 in Ln]
    with mp.Pool(processes=mp.cpu_count()) as pool:
        U=pool.map(prop0,X)
        
    return U


