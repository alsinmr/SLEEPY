#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright 2021 Albert Smith-Penzel

This file is part of pyDIFRATE

pyDIFRATE is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

pyDIFRATE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with pyDIFRATE.  If not, see <https://www.gnu.org/licenses/>.


Questions, contact me at:
albert.smith-penzel@medizin.uni-leipzig.de



Created on Wed Nov 27 13:21:51 2019

@author: albertsmith
"""

"""
Library of functions to deal with vectors and tensors, used for aligning tensors
and vectors into different frames. We assume all vectors provided are 2D numpy
arrays, with the first dimension being X,Y,Z (we do not deal with time-
dependence in these functions. This is obtained by sweeping over the trajectory.
Frames are processed at each time point separately)
"""


"""
Rotations are 
"""


import numpy as np


#%% Reverse rotation direction (passive/active)
def pass2act(cA,sA,cB,sB=None,cG=None,sG=None):
    """
    After determining a set of euler angles, we often want to apply them to go
    into the reference frame corresponding to those angles. This requires
    reversing the rotation, performed by this function
    
    -gamma,-beta,-alpha=pass2act(alpha,beta,gamma)
    
    or 
    
    cG,-sG,cB,-sB,cA,-sA=pass2act(cA,sA,cB,sB,cG,sG)
    """
    
    if sB is None:
        return -cB,-sA,-cA
    else:
        return cG,-sG,cB,-sB,cA,-sA

#%% Change sines and cosines to angles
def sc2angles(cA,sA=None,cB=None,sB=None,cG=None,sG=None):
    """
    Converts cosines and sines of angles to the angles themselves. Takes one or
    three cosine/sine pairs. Note, if an odd number of arguments is given (1 or 3),
    we assume that this function has been called using angles instead of cosines
    and sines, and simply return the input.
    """
    if sA is None:
        return cA
    elif cB is None:
        return np.mod(np.arctan2(sA,cA),2*np.pi)
    elif sB is None:
        return np.array([cA,sA,cB])
    else:
        return np.mod(np.array([np.arctan2(sA,cA),np.arctan2(sB,cB),np.arctan2(sG,cG)]),2*np.pi)
    

#%% Apply/invert rotations     
def Rz(v0,c,s=None):
    """
    Rotates a vector around the z-axis. One must provide the vector(s) and either
    the angle itself, or the cosine(s) and sine(s) of the angle(s). The number
    of vectors must match the number of angles, or only one angle is provided
    
    v=Rz(v0,c,s)
    
        or
        
    v=Rz(v0,theta)
    """
    
    if s is None:
        c,s=np.cos(c),np.sin(c)
        
    X,Y,Z=v0.copy()
    
    X,Y=c*X-s*Y,s*X+c*Y
    Z=np.ones(X.shape)*Z
    
    return np.array([X,Y,Z])

def Ry(v0,c,s=None):
    """
    Rotates a vector around the y-axis. One must provide the vector(s) and either
    the angle itself, or the cosine(s) and sine(s) of the angle(s). The number
    of vectors must match the number of angles, or only one angle is provided
    
    v=Ry(v0,c,s)
    
        or
        
    v=Ry(v0,theta)
    """
    
    if s is None:
        c,s=np.cos(c),np.sin(c)
        
    X,Y,Z=v0.copy()
    
    X,Z=c*X+s*Z,-s*X+c*Z
    Y=np.ones(c.shape)*Y
    
    return np.array([X,Y,Z])

def R(v0,cA,sA,cB,sB=None,cG=None,sG=None):
    """
    Rotates a vector using ZYZ convention. One must provide the vector(s) and 
    either the euler angles, or the cosine(s) and sine(s) of the angle(s). The 
    number of vectors must match the number of angles, or only one angle is 
    provided for alpha,beta,gamma (or the sines/cosines of alpha,beta,gamma)
    
    v=R(v0,cA,sA,cB,sB,cG,sG)
    
        or
        
    v=R(v0,alpha,beta,gamma)
    """
    if v0 is None:
        return None
    
    if sB is None:
        cA,sA,cB,sB,cG,sG=np.cos(cA),np.sin(cA),np.cos(sA),np.sin(sA),np.cos(cB),np.sin(cB)
        
    return Rz(Ry(Rz(v0,cA,sA),cB,sB),cG,sG)

def Rfull(cA,sA,cB,sB=None,cG=None,sG=None):
    """
    Returns a ZYZ rotation matrix for one set of Euler angles
    """
    
    if sB is None:
        a=cA
        b=sA
        g=cB
        cA,sA,cB,sB,cG,sG=np.cos(a),np.sin(a),np.cos(b),np.sin(b),np.cos(g),np.sin(g)
    
    return np.array([[cA*cB*cG-sA*sG,-cG*sA-cA*cB*sG,cA*sB],\
                [cA*sG+cB*cG*sA,cA*cG-cB*sA*sG,sA*sB],\
                [-cG*sB,sB*sG,cB]])

        

def Rspher(rho,cA,sA,cB,sB=None,cG=None,sG=None):
    """
    Rotates a spherical tensor, using angles alpha, beta, and
    gamma. The cosines and sines may be provided, or the angles directly.
    
    One may provide multiple rho and/or multiple angles. If a single rho vector
    is given (5,), then any shape of angles may be used, and similarly, if a single
    set of euler angles is used, then any shape of rho may be used (the first 
    dimension must always be 5). Otherwise, standard broadcasting rules apply
    (the last dimensions must match in size)
    
    rho_out = Rspher(rho,alpha,beta,gamma)
    
    or
    
    rho_out = Rspher(rho,cA,sA,cB,sB,cG,sG)- cosines and sines of the angles
    """
    

    for k,r in enumerate(rho):
        M=D2(cA,sA,cB,sB,cG,sG,mp=k-2,m=None)   #Rotate from mp=k-2 to all new components
        if k==0:
            rho_out=M*r
        else:
            rho_out+=M*r
    return rho_out    
    
    

def R2euler(R,return_angles=False):
    """
    Input a rotation matrix in cartesian coordinates, and return either the
    euler angles themselves or their cosines and sines(default)
    
    cA,sA,cB,sB,cG,sG = R2euler(R)
    
        or
    
    alpha,beta,gamma = R2euler(R,return_angles=True)
    
    R can be a list of matrices
    """
    
#    R = np.array([R]) if np.ndim(R)==2 else np.array(R)
    
    
    """
    Note that R may be the result of an eigenvector decomposition, and does
    not guarantee that R is a proper rotation matrix. We can check the sign
    on the determinant: if it is 1, it's a proper rotation, if it's -1, it's not
    Then, we just multiply each matrix by the result to have only proper
    rotations.

    """
    sgn=np.sign(np.linalg.det(R))
        
    if np.ndim(R)>2:    #Bring the dimensions of the R matrix to the first two dimensions
        for m in range(0,R.ndim-2):
            for k in range(0,R.ndim-1):R=R.swapaxes(k,k+1)
    R=R*sgn
    
    if R.ndim>2:
        cB=R[2,2]
        cB[cB>1]=1.     #Some clean-up to make sure we don't get imaginary terms later (cB cannot exceed 1- only numerical error causes this)
        cB[cB<-1]=-1.
        sB=np.sqrt(1.-cB**2)
        i,ni=sB!=0,sB==0
        cA,sA,cG,sG=np.ones(i.shape),np.zeros(i.shape),np.ones(i.shape),np.zeros(i.shape)
        cA[i]=R[2,0,i]/sB[i]    #Sign swap, 30.09.21
        sA[i]=R[2,1,i]/sB[i]
        cG[i]=-R[0,2,i]/sB[i]   #Sign swap, 30.09.21
        sG[i]=R[1,2,i]/sB[i]
        
        cG[ni]=R[0,0,ni]
        sG[ni]=-R[1,0,ni]       #Sign swap, 30.09.21
    else:
        cB=R[2,2]
        if cB>1:cB=1
        if cB<-1:cB=-1
        sB=np.sqrt(1-cB**2)
        if sB>0:
            cA=R[2,0]/sB        #Sign swap, 30.09.21
            sA=R[2,1]/sB
            cG=-R[0,2]/sB       #Sign swap, 30.09.21
            sG=R[1,2]/sB
        else:
            cA,sA=1,0
            cG=R[0,0]
            sG=-R[1,0]          #Sign swap, 30.09.21

    
    if return_angles:
        return sc2angles(cA,sA,cB,sB,cG,sG)
    else:
        return np.array((cA,sA,cB,sB,cG,sG))
    
def R2vec(R):
    """
    Given a rotation matrix, R, this function returns two vectors, v1, and v2
    that have been rotated from v10=[0,0,1] and v20=[1,0,0]
    
    v1=np.dot(R,v10)
    v2=np.dot(R,v20)
    
    If a frame is defined by a rotation matrix, instead of directly by a set of
    vectors, then v1 and v2 have the same Euler angles to rotate back to their
    PAS as the rotation matrix
    
    R may be a list of rotation matrices
    
    Note: v1, v2 are trivially given by R[:,:,2] and R[:,:,0]
    """
    R = np.array([R]) if np.ndim(R)==2 else np.array(R)
    
    v1=R[:,:,2]
    v2=R[:,:,0]
    
    return v1.T,v2.T
    
    
#%% Tensor operations
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

def D2(cA=0,sA=0,cB=0,sB=None,cG=None,sG=None,m=None,mp=0):
    """
    Calculates components of the Wigner rotation matrix from Euler angles or
    from the list of sines and cosines of those euler angles. All vectors must
    be the same size (or have only a single element)
    
    mp and m should be specified. m may be set to None, so that all components
    are returned in a 2D array
    
    D2_m_mp=D2(m,mp,cA,sA,cB,sB,cG,sG)  #Provide sines and cosines
    
        or
        
    D2_m_mp=D2(m,mp,alpha,beta,gamma) #Give the angles directly
    
    (Note that m is the final index)
    """
    if sB is None:
        cA,sA,cB,sB,cG,sG=np.cos(cA),np.sin(cA),np.cos(sA),np.sin(sA),np.cos(cB),np.sin(cB)

        
    d2c=d2(cB,sB,m,mp)
    
    "Rotation around z with alpha (mp)"
    if mp is None:
        ea1=cA-1j*sA
        eam1=cA+1j*sA
        ea2=ea1**2
        eam2=eam1**2
        ea0=np.ones(ea1.shape)
        ea=np.array([eam2,eam1,ea0,ea1,ea2])
    else:
        if mp!=0:
            ea=cA-1j*np.sign(mp)*sA
            if np.abs(mp)==2:
                ea=ea**2
        else:
            ea=1

    "Rotation around z with gamma (m)"
    if m is None:
        eg1=cG-1j*sG
        egm1=cG+1j*sG
        eg2=eg1**2
        egm2=egm1**2
        eg0=np.ones(eg1.shape)
        eg=np.array([egm2,egm1,eg0,eg1,eg2])
    else:
        if m!=0:
            eg=cG-1j*np.sign(m)*sG
            if np.abs(m)==2:
                eg=eg**2
        else:
            eg=1
            
    return ea*d2c*eg
    




def Spher2Cart(rho):
    """
    Takes a set of components of a spherical tensor and calculates its cartesian
    representation (as a vector, with components in order of Axx,Axy,Axz,Ayy,Ayz)
    
    Input may be a list (or 2D array), with each new column a new tensor
    """
    
    rho=np.array(rho,dtype=complex)

    M=np.array([[0.5,0,-np.sqrt(1/6),0,0.5],
                 [0.5*1j,0,0,0,-0.5*1j],
                 [0,0.5,0,-0.5,0],
                 [-0.5,0,-np.sqrt(1/6),0,-.5],
                 [0,.5*1j,0,.5*1j,0]])
    SZ0=rho.shape
    SZ=[5,np.prod(SZ0[1:]).astype(int)]
    out=np.dot(M,rho.reshape(SZ)).real
    return out.reshape(SZ0)
    
    
def Spher2pars(rho,return_angles=False):
    """
    Takes a set of components of a spherical tensor and calculates the parameters
    describing that tensor (delta,eta,alpha,beta,gamma)
    
    
    delta,eta,cA,sA,cB,sB,cG,sG=Spher2pars(rho)
    
        or
        
    delta,eta,alpha,beta,gamma=Spher2pars(rho,return_angles=True)
    
    
    Input may be a list (or 2D array), with each new column a new tensor (5xN)
    """

    A0=Spher2Cart(rho)  #Get the Cartesian tensor
    if A0.ndim==1:
        A0=np.atleast_2d(A0).T

    R=list()
    delta=list()
    eta=list()
    
    
    for k,x in enumerate(A0.T):
        Axx,Axy,Axz,Ayy,Ayz=x
        A=np.array([[Axx,Axy,Axz],[Axy,Ayy,Ayz],[Axz,Ayz,-Axx-Ayy]])    #Full matrix
        D,V=np.linalg.eigh(A)   #Get eigenvalues, eigenvectors 
        i=np.argsort(np.abs(D))
        D,V=D[i[[1,0,2]]],V[:,i[[1,0,2]]]     #Ordering is |azz|>=|axx|>=|ayy|
        "V should have a determinant of +1 (proper vs. improper rotation)"
        V=V*np.sign(np.linalg.det(V))
        delta.append(D[2])
        eta.append((D[1]-D[0])/D[2])
        R.append(V)
    
    delta=np.array(delta)
    eta=np.array(eta)
    euler=R2euler(np.array(R))
    
    if return_angles:
        euler=sc2angles(*euler)
       
    return np.concatenate(([delta],[eta],euler),axis=0)
        

def pars2Spher(delta,eta=None,cA=None,sA=None,cB=None,sB=None,cG=None,sG=None):
    """
    Converts parameters describing a spherical tensor (delta, eta, alpha, beta,
    gamma) into the tensor itself. All arguments except delta are optional. Angles
    may be provided, or their cosines and sines may be provided. The size of the
    elements should follow the rules required for Rspher.
    """

    if cA is None:
        cA,sA,cB,sB,cG,sG=np.array([1,0,1,0,1,0])
    
    if eta is None:
        eta=np.zeros(np.shape(delta))
    
    rho0=np.array([-0.5*eta*delta,0,np.sqrt(3/2)*delta,0,-0.5*eta*delta])
    
    return Rspher(rho0,cA,sA,cB,sB,cG,sG)
    
    
    