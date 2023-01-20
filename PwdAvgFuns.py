#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  20 10:14:04 2021

@author: albertsmith
"""

import numpy as np

#%% Functions for powder averaging    
def pwd_JCP59(q=3):
    """
    Generates a powder average with quality quality q (1 to 12). 
    According to JCP 59 (8) 3992 (1973) (copied from Matthias Ernst's Gamma
    scripts).
    """
    
    q+=-1   #We use q as an index, switch to python indexing
    
    value1=[2,50,100,144,200,300,538,1154,3000,5000,7000,10000];
    value2=[1,7,27,11,29,37,55,107,637,1197,1083,1759];
    value3=[1,11,41,53,79,61,229,271,933,1715,1787,3763];

    count=np.arange(1,value1[q])

    alpha=2*np.pi*np.mod(value2[q]*count,value1[q])/value1[q];
    beta=np.pi*count/value1[q];
    gamma=2*np.pi*np.mod(value3[q]*count,value1[q])/value1[q];

    weight=np.sin(beta);
    weight*=1/weight.sum();

    return alpha,beta,gamma,weight