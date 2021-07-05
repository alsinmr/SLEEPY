#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:44:13 2019

@author: albertsmith
"""

import os
import numpy as np
import pandas as pd
import re

#%% Some useful tools (Gyromagnetic ratios, spins, dipole couplings)
def NucInfo(Nuc=None,info='gyro'):
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
    
    Nucs=[]
    MassNum=[]
    spin=[]
    g=[]
    Abund=[]

    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    with open(dir_path+"/GyroRatio") as f:
        data=f.readlines()
        for line in data:
            line=line.strip().split()
            MassNum.append(int(line[1]))
            Nucs.append(line[3])
            spin.append(float(line[5]))
            g.append(float(line[6]))
            Abund.append(float(line[7]))
    
    NucData=pd.DataFrame({'nuc':Nucs,'mass':MassNum,'spin':spin,'gyro':g,'abund':Abund})
    
    
    if Nuc==None:
        return NucData
    else:
        
        if Nuc=='D':
            Nuc='2H'
        
        mass=re.findall(r'\d+',Nuc)
        if not mass==[]:
            mass=int(mass[0])
            
        
        Nuc=re.findall(r'[A-Z]',Nuc.upper())
        
        if np.size(Nuc)>1:
            Nuc=Nuc[0].upper()+Nuc[1].lower()
        else:
            Nuc=Nuc[0]
            
            
            
        NucData=NucData[NucData['nuc']==Nuc]
       
        if not mass==[]:    #Use the given mass number
            NucData=NucData[NucData['mass']==mass]
        elif any(NucData['spin']==0.5): #Ambiguous input, take spin 1/2 nucleus if exists
            NucData=NucData[NucData['spin']==0.5] #Ambiguous input, take most abundant nucleus
        elif any(NucData['spin']>0):
            NucData=NucData[NucData['spin']>0]
        
        NucData=NucData[NucData['abund']==max(NucData['abund'])]
            
        
        h=6.6260693e-34
        muen=5.05078369931e-27
        
        NucData['gyro']=float(NucData['gyro'])*muen/h
#        spin=float(NucData['spin'])
#        abund=float(NucData['abund'])
#        mass=float(NucData['spin'])
        if info[:3]=='all':
            return NucData
        else:
            return float(NucData[info])

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



        
        



       