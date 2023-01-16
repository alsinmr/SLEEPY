#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 10:54:04 2021

@author: albertsmith
"""

import numpy as np
import copy
from types import MethodType
from .Tools import NucInfo
from .Powder import RotInter

#%% Functions to generate spin operators

def so_m(S=None):
    "Calculates Jm for a single spin"
    if S is None:S=1/2
    M=np.round(2*S+1).astype(int)    
    jm=np.zeros(M**2)
    jm[M::M+1]=np.sqrt(S*(S+1)-np.arange(-S+1,S+1)*np.arange(-S,S))
    return jm.reshape(M,M)

def so_p(S=None):
    "Calculates Jp for a single spin"
    if S is None:S=1/2
    M=np.round(2*S+1).astype(int)    
    jp=np.zeros(M**2)
    jp[1::M+1]=np.sqrt(S*(S+1)-np.arange(-S+1,S+1)*np.arange(-S,S))
    return jp.reshape(M,M)
        
def so_x(S=None):
    "Calculates Jx for a single spin"
    return 0.5*(so_m(S)+so_p(S))

def so_y(S=None):
    "Calculates Jx for a single spin"
    return 0.5*1j*(so_m(S)-so_p(S))

def so_z(S=None):
    "Calculates Jz for a single spin"
    if S is None:S=1/2
    M=np.round(2*S+1).astype(int)
    jz=np.zeros(M**2)
    jz[::M+1]=np.arange(S,-S-1,-1)
    return jz.reshape(M,M)

def so_alpha(S=None):
    "Calculates the alpha state for a single spin"
    Sz=so_z(S)
    return np.eye(Sz.shape[0])/2+Sz

def so_beta(S=None):
    "Calculates the beta state for a single spin"
    Sz=so_z(S)
    return np.eye(Sz.shape[0])/2-Sz
    
def Op(obj,Type,n):
    """
    Returns spin operator, specified by Type (see so_* at beginning of file),
    for spin with index n
    """
    assert 'so_'+Type in globals().keys(),'Unknown spin operator type'
    
    ops=getattr(obj,'__'+Type)
    if ops[n] is not None:return ops[n]
    op0=globals()['so_'+Type](obj.S[n])
    
    ops[n]=np.kron(np.kron(np.eye(obj.Mult[:n].prod()),op0),np.eye(obj.Mult[n+1:].prod()))
    
    setattr(obj,'__'+Type,ops)
    
    return ops[n]

def Op_fun(Type):
    def fun(obj,n):
        return Op(obj,Type,n)
    return fun

def getOp(obj,Type):
    @property
    def fun():
        return getattr(obj._parent,Type)(obj._n)
    return fun

#%% Class to store the spin system
class SpinOp:
    """
    Here we create a class designed for generating spin systems, similar to the 
    previous n_spin_system in matlab
    """
    def __init__(self,S=None,N=None):
        """
        Initializes the class. Provide either a list of the spins in the system,
        or set n, the number of spins in the system (all spin 1/2)
        """
        
        if N is not None:
            self.__S=np.ones(N)*1/2
        elif hasattr(S,'__len__'):
            self.__S=np.array(S)
        else:
            self.__S=np.ones(1)*S
    
        self.__Mult=(2*self.__S+1).astype(int)
        self.__N=len(self.S)
        self.__index=-1
    
        for k in globals().keys():
            if 'so_' in k:
                Type=k[3:]
                setattr(self,'__'+Type,[None for _ in range(self.__N)])
                setattr(self,Type,MethodType(Op_fun(Type),self))
                
        self.__Op=[OneSpin(self,n) for n in range(self.__N)]
        
    def __getitem__(self,n):
        "Returns the single spins of the spin system"
        return OneSpin(self,n)
    def __next__(self):
        self.__index+=1
        if self.__index==self.__N-1:
            self.__index==-1
            raise StopIteration
            return self.__Op[self.__N-1]
        else:
            return self.__Op[self.__index]
    def __iter__(self):
        return self
        
    @property
    def S(self):
        return self.__S.copy()    
    @property
    def N(self):
        return self.__N
    @property
    def Mult(self):
        return self.__Mult
        
class OneSpin:
    """Object for containing spin matrices just for one spin (but of a larger
    overall spin system)
    """
    def __init__(self,spinops,n):
        self._parent=spinops
        self._n=n
        
"Here, we add all the required properties to OneSpin"        
for Type in dir(SpinOp(N=1)):
    if 'so_'+Type in globals().keys():
        setattr(OneSpin,Type,property(lambda self,t=Type:getattr(self._parent,t)(self._n)))


#%% Class to store the information about the spin system

        
class ExperSys():
    """
    Stores various information about the spin system. Initialize with a list of
    all nuclei in the spin system.
    """
    def __init__(self,v0H,Nucs):
        self.v0H=v0H
        self.B0=self.v0H*1e6/NucInfo('1H')
        self.Nucs=np.atleast_1d(Nucs).squeeze()
        self.N=len(self.Nucs)
        self.S=np.array([NucInfo(nuc,'spin') for nuc in self.Nucs])
        self.gamma=np.array([NucInfo(nuc,'gyro') for nuc in self.Nucs])
        self.Op=SpinOp(self.S)
        self.__index=-1
        self.__Ninter=0
        
        
        self.inter_types=dict()
        for k in globals().keys():
            if hasattr(globals()[k],'__code__'):
                setattr(self,k[6:],list())
                if 'int1i_' in k:
                    self.inter_types[k[6:]]=['int1i',['i1','value']]
                elif 'int2i_' in k:
                    self.inter_types[k[6:]]=['int2i',['i1','i1','value']]
                elif 'int1a_' in k:
                    self.inter_types[k[6:]]=['int1a',['i1','delta','eta','euler']]
                elif 'int2a_' in k:
                    self.inter_types[k[6:]]=['int2a',['i1','i2','delta','eta','euler']]
    
    @property
    def Ninter(self):
        return self.__Ninter
                
    def set_inter(self,Type,i1,i2=None,value=None,delta=None,eta=None,euler=None):
        """
        Adds an interaction to the total Hamiltonian. We list the required arguments
        for each type of interaction
        
        Spin-Field:
            Isotropic: i1 (spin index) and value (chemical shift in ppm)
            Anisotropic: i1 (spin index) and anisotropy (delta, in ppm). Asymmetry
                and Euler angles also optional
        Spin-Spin:
            Isotropic: i1,i2 (spin indices) and value (J coupling gin Hz)
            Anisotropic: i1,i2 (spin indices) and anisotropy (delta, in Hz. Asymmetry
                and Euler angles also optional
        """
        
        assert Type in self.inter_types.keys(),"Unknown interaction type"
        
        if 'a' in self.inter_types[Type][0]:
            assert delta is not None,'delta required for anisotropic interactions'
            assert value is None,'value only used for isotropic interactions'
        if 'i' in self.inter_types[Type][0][-1]:
            assert delta is None and eta is None and euler is None,'delta, eta, and euler only used for anisotropic interactions'
            assert value is not None,'value required for isotropic interactions'
        if '2' in self.inter_types[Type][0]:
            assert i1 is not None and i2 is not None,'i1 and i2 required for 2-spin interactions'
        if '1' in self.inter_types[Type][0]:
            assert i1 is not None and i2 is None,'Only use i1 (and not i2) for spin-field interactions'
        assert i1<self.N,'i1 cannot exceed {0} ({1} spins in system)'.format(self.N+1,self.N)
        if i2 is not None:
            assert i2<self.N,'i2 cannot exceed {0} ({1} spins in system)'.format(self.N+1,self.N)
        
        if i2 is not None:
            i1,i2=np.sort([i1,i2])
        
        args={'Type':Type}
        for k in ['i1','i2','value','delta','eta','euler']:
            if locals()[k] is not None:
                args[k]=locals()[k]
        
        inter=getattr(self,Type)
        
        for k,i in enumerate(inter):
            if i2 is None:
                if i1==i['i1']:
                    print('Warning: Overwriting existing interaction')
                    inter[k]=args
                    break
            else:
                if i1==i['i1'] and i2==i['i2']:
                    print('Warning: Overwriting existing interaction')
                    inter[k]=args
                    break
        else:
            inter.append(args)
            self.__Ninter+=1
                             
    def get_inter(self,n=None,Type=None,i1=None,i2=None):
        """
        Get an interaction, for a given index of the powder average (if isotropic,
        then this argument is ignored).
        
        There are several indexing options:
            1) Absolute index. Get the nth interaction, sweeping through all 
            interaction types (provide n)
            2) Type and index. Get the nth interaction of give Type (provide Type and n)
            3) Type and spin index. Get the index for Type between spin(s) i1 (and i2)
            (provide Type and i1 and optionally i2)
            
        """
        if Type is None:
            assert n<self.__Ninter,'Index exceeds number of defined interactions'
            i=0
            Types=[k for k in self.inter_types.keys()]
            while n>=len(getattr(self,Types[i])):
                n+=-len(getattr(self,Types[i]))
                i+=1
            return getattr(self,Types[i])[n]
        else:        
            assert Type in self.inter_types.keys(),"Unknown interaction type"
            if n is not None:
                assert n<len(getattr(self,Type)),"Only {0} interactions have been defined for type {1}"\
                    .format(len(getattr(self,Type)),Type)
                return getattr(self,Type)[n]        
            else:
                if i1 is not None and i2 is not None:i1,i2=np.sort([i1,i2])
                if self.inter_types[Type][0][3]=='2':
                    assert i1 is not None and i2 is not None,"i1 and i2 required for interactions of type {0}".format(Type)
                    for i in getattr(self,Type):
                        if i1==i['i1'] and i2==i['i2']:
                            return i
                    else:
                        assert False,'Could not find interaction {0} with indices {1} and {2}'.format(Type,i1,i2)
                else:
                    assert i1 is not None and i2 is None,"Only one index (i1) for interactions of type {0}".format(Type)
                    for i in getattr(self,Type):
                        if i1==i['i1']:
                            return i
                    else:
                        assert False,'Could not find interaction {0} with indices {1} and {2}'.format(Type,i1,i2)

    def get_abs_index(self,n=None,Type=None,i1=None,i2=None):
        """
        Get the absolute index of a given interaction
        """
        
        if Type is None:
            assert n<self.__Ninter,"n must be less than the number of defined interactions ({0})".format(self.__Ninter)
            return n
        else:
            assert Type in self.inter_types.keys(),'Interaction {0} is not defined'.format(Type)
            n0=0
            for t in self.inter_types.keys():
                if Type==t:
                    break
                else:
                    n0+=len(getattr(self,t))
            if n is None:
                if i2 is not None:i1,i2=np.sort([i1,i2])
                for k,m in enumerate(getattr(self,Type)):
                    if ('i2' in m.keys() and i1==m['i1'] and i2==m['i2']) or i1==m['i1']:
                        return n0+k
                else:
                    assert False,"Interaction {0} with given indices not defined"
            else:
                assert n0+n<self.__Ninter,""
                return n0+n
                    
    
    def __getitem__(self,n):
        """
        Returns parameters for the nth interaction. Indexing sweeps through 
        each interaction type sequentially
        """
        
        return self.get_inter(n=n)
            

    
    def __next__(self):
        self.__index+=1
        if self.__index==self.__Ninter-1:
            self.__index==-1
            raise StopIteration
            return self.__getitem__(self.__Ninter-1)
        else:
            return self.__getitem__(self.__index)
    def __iter__(self):
        return self
   

#%% Class to store Hamiltonians
# Probably we will replace this Hamiltonian class with something simpler. 
# I think a good idea is to make the Hamiltonian iterable. The outer loop is
# over the powder average. The inner loop should return a time step in the
# rotor cycle. Then, for a given element of the powder average, we should
# have the attribute inter, which will contain each of the interactions, and
# secondly, each interaction should contain its components for rotation.
class Hamiltonian():
    """
    Stores a Hamiltonian, and returns its value for a particular orientation
    """
    def __init__(self,exp_sys,channels=None,pwdavg=None,rotor_angle=np.arccos(np.sqrt(1/3))):
        """
        Initializes the Hamiltonian. requirements are A, a matrix containing
        the rotating components of the Hamiltonizn (Nx3), and Op, the spin
        operator corresponding to the interaction.
        """
        self.__exp_sys=copy.copy(exp_sys)   
        self.channels=np.array(channels)
        self.__pwdavg=copy.copy(pwdavg)
        self.__pwdavg.clear_inter()
        self.__rotor_angle=np.array(rotor_angle)
        
        "Save orientation dependence"        
        for i in self.__exp_sys:
            i=copy.deepcopy(i)
            Type=i.pop('Type')
            i0=self.__exp_sys.inter_types[Type][0]
            if i0[-1]=='a':
                assert pwdavg is not None,"pwdavg object required for anisotropic interations"
                index=[i.pop('i1'),i.pop('i2')] if i0[-2]=='2' else i.pop('i1')
                self.__pwdavg.new_inter(Type=Type,index=index,**i)
    
    @property                
    def exp_sys(self):
        return copy.copy(self.__exp_sys)
    
    @property
    def pwdavg(self):
        return copy.copy(self.__pwdavg)
    
    @property
    def rotor_angle(self):
        return self.__rotor_angle.copy()
    
    def get_inter(self,n=None,Type=None,i1=None,i2=None,pwd_ind=0):
        """
        Get an interaction, for a given index of the powder average (if isotropic,
        then this argument is ignored).
        
        There are several indexing options:
            1) Absolute index. Get the nth interaction, sweeping through all 
            interaction types (provide n)
            2) Type and index. Get the nth interaction of give Type (provide Type and n)
            3) Type and spin index. Get the index for Type between spin(s) i1 (and i2)
            (provide Type and i1 and optionally i2)
            
        """
        
        args=self.__exp_sys.get_inter(Type=Type,n=n,i1=i1,i2=i2)
        Type,i1,i2=[args[k] for k in ['Type','i1','i2']]
        
        fun=globals()[self.__exp_sys.inter_types[Type][0]+'_'+Type]
        t0=self.__exp_sys.inter_types[Type][0]
        if t0=='int1i':return fun(self,i1)
        if t0=='int2i':return fun(self,i1,i2)
        if t0=='int1a':return fun(self,i1,pwd_ind)
        if t0=='int2a':return fun(self,i1,i2,pwd_ind)        
    
    def __call__(self,ind_pwd,in_gamma,v1=None,phase=None,offset=None):
        """
        Returns the Hamiltonian for the qth element of the powder average, 
        including an additional gamma rotation
        """
        
        
        return 
    
class Ham_pwd_n():
    """
    Container for a Hamiltonian for a particular orientation of the powder average
    """
    def __init__(self,H):
        """
        Provide the 5 rotating components of the Hamiltonian
        """
        self.__H=H
        self.__index=-3
        
    def __getitem__(self,n):
        """
        Returns the requested rotating component (note: list from -2 to 2, not 
        0 to 4) 
        """
        return self.__H[n+2]
    
    def __next__(self):
        self.__index+=1
        if self.__index==2:
            self.__index=-3           
            raise StopIteration
            return self.__getitem__(2)
        else:
            return self.__getitem__(self.__index)
    def __iter__(self):
        return self
    
    def __call__(self,gamma):
        """
        Return the Hamiltonian for a particular gamma angle (in radians)
        """
        H=np.sum([np.exp(-1j*gamma*m)*self[m] for m in range(-2,3)],axis=0)
        return H
#%% Calculate rotating components of interactions 
"""Each function should be preceded by int1a, int2a, int1i, or int2i. The number
is if it is a 1 spin (spin-field) or 2 spin (spin-spin) interaction. The letter
refers to if the interaction is isotropic (i, for no orientation dependence) or
anisotropic (a, for orientational dependence). The first argument should always
be an instance of the ExperSys class. Subsequent arguments depend on the 
interaction type

int1i_*(exp_sys,i1,...)
int12_*(exp_sys,i1,i2,...)
int1a_*(exp_sys,i1,pwd_ind,gamma,...)
int2a_*(exp_sys,i1,i2,pwd_ind,gamma,...)



"""
def int2a_dipole(ham,i1,i2,pwd_ind):
    """
    Calculates the dipolar interaction
    """
    
    es=ham.exp_sys
    i=es.get_abs_index(Type='dipole',i1=i1,i2=i2)
    
    A=ham.pwdavg[i].Azz[pwd_ind]
    
    S,I=es.Op[i1],es.Op[i2]
    
    if es.Nucs[i1]==es.Nucs[i2]:
        H0=np.sqrt(2/3)*(S.z*I.z-0.5*(S.x*I.x+S.y*I.y))
    else:
        H0=np.sqrt(2/3)*S.z*I.z
    
    H=[A0*H0 for A0 in A]
    
    return Ham_pwd_n(H)
    

def int2i_J(ham,i1,i2):
    """
    Calucates the J interaction
    """
    pass

def int1i_CS(ham,i1):
    """
    Calculates the chemical shift interaction
    """
    pass

def int1a_CSA(ham,i1,pwd_ind):
    """
    Calculates the chemical shift anisotropy
    """
    pass