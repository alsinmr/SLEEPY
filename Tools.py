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
    
    with open(os.path.join(dir_path,"GyroRatio.txt")) as f:
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


def d2(c=0,s=None,m=None,mp=0):
    """
    Calculates components of the d2 matrix. By default only calculates the components
    starting at m=0 and returns five components, from -2,-1,0,1,2. One may also
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
        if m is None and mp is None:
            print('m or mp must be specified')
            return
        elif m is None:
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
       