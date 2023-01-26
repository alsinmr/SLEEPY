#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 10:39:58 2023

@author: albertsmith
"""

import numpy as np
from . import Defaults



def so_m(S=None):
    "Calculates Jm for a single spin"
    if S is None:S=1/2
    M=np.round(2*S+1).astype(int)    
    jm=np.zeros(M**2,dtype=Defaults['ctype'])
    jm[M::M+1]=np.sqrt(S*(S+1)-np.arange(-S+1,S+1)*np.arange(-S,S))
    return jm.reshape(M,M).astype(Defaults['ctype'])

def so_p(S=None):
    "Calculates Jp for a single spin"
    if S is None:S=1/2
    M=np.round(2*S+1).astype(int)    
    jp=np.zeros(M**2,dtype=Defaults['ctype'])
    jp[1::M+1]=np.sqrt(S*(S+1)-np.arange(-S+1,S+1)*np.arange(-S,S))
    return jp.reshape(M,M).astype(Defaults['ctype'])
        
def so_x(S=None):
    "Calculates Jx for a single spin"
    return 0.5*(so_m(S)+so_p(S)).astype(Defaults['ctype'])

def so_y(S=None):
    "Calculates Jx for a single spin"
    return 0.5*1j*(so_m(S)-so_p(S)).astype(Defaults['ctype'])

def so_z(S=None):
    "Calculates Jz for a single spin"
    if S is None:S=1/2
    M=np.round(2*S+1).astype(int)
    jz=np.zeros(M**2,dtype=Defaults['ctype'])
    jz[::M+1]=np.arange(S,-S-1,-1)
    return jz.reshape(M,M).astype(Defaults['ctype'])

def so_alpha(S=None):
    "Calculates the alpha state for a single spin"
    Sz=so_z(S)
    return np.eye(Sz.shape[0],dtype=Defaults['ctype']).astype(Defaults['ctype'])/2+Sz

def so_beta(S=None):
    "Calculates the beta state for a single spin"
    Sz=so_z(S)
    return np.eye(Sz.shape[0],dtype=Defaults['ctype']).astype(Defaults['ctype'])/2-Sz

def so_eye(S=None):
    if S is None:S=1/2
    M=np.round(2*S+1).astype(int)
    return np.eye(M,dtype=Defaults['ctype'])