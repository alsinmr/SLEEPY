#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 09:16:50 2023

@author: albertsmith
"""


import numpy as np
from scipy.linalg import expm

import multiprocessing as mp


# from numba import njit,prange
# from functools import lru_cache
# from . import Defaults

# @njit(cache=True)
# def fact(n):
#    if n:
#       return n*fact(n-1)
#    else: 
#       return 1
     

# @njit(cache=True, fastmath=True)
# def expm(M, n = 10, facts = np.array([fact(f+1) for f in prange(30)])):
#     out=np.eye(M.shape[0],dtype=M.dtype)
#     ac=np.eye(M.shape[0], dtype=M.dtype)
#     for k in prange(n):
#         ac=M@ac
#         out+=ac/facts[k]
#     return out


# @njit(cache=True, parallel=True)
# def prop1(U, Ln,Lrf,n0,nf,tm1,tp1,dt,n_gamma, ph0, ph, phf):
#     L = np.zeros((U.shape[1],U.shape[2]),dtype=U.dtype)
#     #ph0=np.exp(1j*2*np.pi*n0/n_gamma)
#     #ph = np.array([np.exp(1j*2*np.pi*n/n_gamma) for n in prange(nf)])
#     #phf=np.exp(1j*2*np.pi*nf/n_gamma)
#     for l in prange(Ln.shape[0]):       
#         L[::] = Lrf
#         for m in prange(-2,3):
#            L += Ln[l][m+2]*(ph0**(-m))
        

#         U[l]=expm(L*tm1) 
#         for n in prange(n0+1,nf):
#             L[::] = Lrf
#             for m in prange(-2,3):
#                L += Ln[l][m+2]*(ph[n]**(-m))
#             U[l]=expm(L*dt)@U[l]

#         if tp1>1e-10:
#             L[::] = Lrf
#             for m in prange(-2,3):
#                L += Ln[l][m+2]*(phf**(-m))
#             U[l]=expm(L*tp1)@U[l]

# @lru_cache()
# def ph0_ph_phf(n0, nf, n_gamma):
#     return np.exp(1j*2*np.pi*n0/n_gamma), np.array([np.exp(1j*2*np.pi*n/n_gamma) for n in prange(nf)]), np.exp(1j*2*np.pi*nf/n_gamma)
   
# def prop(Ln,Lrf,n0,nf,tm1,tp1,dt,n_gamma):
#     ph0, ph, phf = ph0_ph_phf(n0,nf, n_gamma)
#     Ln = np.array(Ln)
#     U = np.zeros((Ln.shape[0], Ln.shape[2], Ln.shape[3]), dtype=Ln.dtype)
#     prop1(U, Ln,Lrf,n0,nf,tm1,tp1,dt,n_gamma, ph0, ph, phf)
#     return [u for u in U]


#%% Static processing
def prop_static0(X):
    L,Dt=X
    return expm(L*Dt)

def prop_static(L,Dt):
    X=[(L0,Dt) for L0 in L]
    with mp.Pool(processes=mp.cpu_count()) as pool:
        U=pool.map(prop_static0,X)
    return U


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



def prop_x_rho0(X):
    Ln,dct,Op,rho,static=X
    
    t=dct['t']
    for m,(ta,tb) in enumerate(zip(t[:-1],t[1:])):
        #Set the RF fields
        Lrf=np.zeros(Ln[0].shape)
        for k,(v1,phase,voff,Op0) in enumerate(zip(dct['v1'],dct['phase'],dct['voff'],Op)):
            Lrf+=-1j*2*np.pi*(v1*(np.cos(phase)*Op0.x+np.sin(phase)*Op0.y)-voff*Op0.z)
        
        if static:
            Ldt=(Ln[2]+Lrf)*(tb-ta)
            rho=expm_x_rho(Ldt,rho)
        else:
            pass
            #TODO
            # HERE WE NEED TO GO THROUGH THE ROTOR CYCLE, EX. SEE LIOUVILLIAN,
            # LINES 487-
        
from copy import copy
def expm_x_rho(Ldt,rho,n=20):
    """
    Calculates the product of the matrix exponential multiplied by the density
    operator. Bypasses full calculation of the matrix exponential

    Parameters
    ----------
    Ldt : array (square)
        Liouville matrix multiplied by time step
    rho : array (vector)
        Initial density matrix

    Returns
    -------
    None.

    """
    
    out=copy(rho)
    for k in range(1,n):
        rho=Ldt@rho
        out+=rho/np.math.factorial(k)
    return out
            
    
            
    
    


