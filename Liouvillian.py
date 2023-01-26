#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 15:46:57 2023

@author: albertsmith
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 15:06:02 2023

@author: albertsmith
"""

import numpy as np
from copy import copy
import warnings
from scipy.linalg import expm
from .Propagator import Propagator
from . import Defaults
from .Tools import Ham2Super
from .Hamiltonian import Hamiltonian
from . import RelaxMat


# import importlib.util
# numba=importlib.util.find_spec('numba') is not None
# if numba:
#     from .Parallel import prop

from .Parallel import prop

class Liouvillian():
    def __init__(self,H:list,kex=None):
        """
        Creates the full Liouvillian, and provides some functions for 
        propagation of the Liouvillian

        Parameters
        ----------
        H : list
            List of Hamiltonians or alternatively ExpSys
        kex : np.array, optional
            Exchange Matrix. The default is None.

        Returns
        -------
        None.

        """
        if hasattr(H,'shape') or hasattr(H,'B0'):H=[H]
        self.H=[*H]
        
        for k,H in enumerate(self.H):
            if not(hasattr(H,'Hinter')) and hasattr(H,'B0'):
                H=Hamiltonian(H)
                self.H[k]=H
            assert hasattr(H,'Hinter'),'Liouvillian must be provided with Hamiltonian or ExpSys objects'
            assert H.pwdavg==self.pwdavg,"All Hamiltonians must have the same powder average"
            if H.rf is not self.rf:
                H.expsys._rf=self.rf

        
        self.kex=kex
        self.sub=False
        
        self._Lex=None
        self._index=-1
        self._Lrelax=None
        self._Lrf=None
        self._Ln=None
        self._fields=self.fields
        self.relax_info=[]  #Keeps a short record of what kind of relaxation is used
    

    
    @property
    def _ctype(self):
        return Defaults['ctype']
    
    @property
    def _rtype(self):
        return Defaults['rtype']
    
    @property
    def _parallel(self):
        return Defaults['parallel']
    
    def reset_prop_time(self,t:float=0):
        """
        Resets the current time for propagators to t
        
        (L.expsys._tprop=t)

        Parameters
        ----------
        t : float, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        None.

        """
        self.expsys._tprop=t
    
    @property
    def isotropic(self):
        return np.all([H0.isotropic for H0 in self.H])
    
    @property
    def pwdavg(self):
        return self.H[0].pwdavg
    
    @property
    def Peq(self):
        for ri in self.relax_info:
            if 'Peq' in ri[1] and ri[1]['Peq']:
                return True
        return False
    
    @property
    def expsys(self):
        """
        Returns the expsys of the first stored Hamiltonian

        Returns
        -------
        ExpSys
            Description of the experimental conditions and system

        """
        return self.H[0].expsys
    
    @property
    def taur(self):
        """
        Length of a rotor period

        Returns
        -------
        None.

        """
        if self.H[0].isotropic:return None
        return 1/self.expsys.vr
    
    @property
    def dt(self):
        """
        Time step for changing the rotor angle

        Returns
        -------
        float

        """
        return self.taur/self.expsys.n_gamma
    
    @property
    def shape(self):
        """
        Shape of the resulting Liouvillian

        Returns
        -------
        tuple

        """
        return np.prod(self.H[0].shape)*len(self.H),np.prod(self.H[0].shape)*len(self.H)
    
    @property
    def rf(self):
        return self.H[0].rf
    
    @property
    def fields(self):
        return self.rf.fields
    
    def __setattr__(self,name,value):
        """
        Resets certain parameters if edits occur

        Parameters
        ----------
        name : str
            Parameter name.
        value : TYPE
            Parameter value.

        Returns
        -------
        None.

        """
        
        if name=='kex':
            self._Lex=None
            value=np.array(value)
        
        super().__setattr__(name,value)
    
    def __getitem__(self,i:int):
        """
        Goes to a particular item of the powder average

        Parameters
        ----------
        i : int
            Element of the powder average to go to.

        Returns
        -------
        Liouvillian

        """
        out=copy(self)
        
        out.H=[H0[i] for H0 in self.H]
        out.sub=True
        return out
    
    def add_relax(self,M=None,Type:str=None,**kwargs):
        """
        Add explicit relaxation to the Liouvillian. This is provided by a matrix,
        M, directly. The matrix itself can be produced with the RelaxationMatrix
        class. Note that the matrix can either have the same shape as the full
        Liouvillian, or the shape for just one Hamiltonian. For example, for
        a two-spin 1/2 system in two-site exchange, M may have size 16x16 or
        32x32. The 32x32 matrix allows different relaxation properties for the
        two sites.

        Parameters
        ----------
        M : np.array, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        if M is None:
            if hasattr(RelaxMat,Type):
                M=getattr(RelaxMat,Type)(expsys=self.expsys,**kwargs)
                self.relax_info.append((Type,kwargs))
            else:
                warnings.warn(f'Unknown relaxation type: {Type}')
                return
        
        q=np.prod(self.H[0].shape)
        self.Lrelax  #Call just to pre-allocate
        if M.shape[0]==q:
            for k,H0 in enumerate(self.H):
                self._Lrelax[k*q:(k+1)*q][:,k*q:(k+1)*q]+=M
        elif M.shape[0]==self.shape[0]:
            self._Lrelax+=M
        else:
            assert False,f"M needs to have size ({q},{q}) or {self.shape}"   
    
    def clear_relax(self):
        """
        Removes all explicitely defined relaxation

        Returns
        -------
        None.

        """
        self.relax_info=[]
        self._Lrelax=None
        
    def validate_relax(self):
        """
        Checks if systems with T1 relaxation have T2 relaxation. Also returns
        True if the system relaxes to an equilibrium value

        Returns
        -------
        None.

        """
        Long=False
        Peq=False
        Trans=False
        for ri in self.relax_info:
            if ri[0] in ['T1']:  #Check for Longitudinal relaxation
                Long=True
                if 'Peq' in ri[1] and ri[1]['Peq']:
                    Peq=True
            elif ri[0] in ['T2']: #Check for Transverse relaxation
                Trans=True
        if Long and Peq and not Trans:
            warnings.warn('T1 relaxation and Peq included without T2 relaxation. System can diverge')
        elif Long and not Trans:
            warnings.warn('T1 relaxation included without T2 relaxation. Unphysical system')
    
    
    @property
    def Lrelax(self):
        if self._Lrelax is None:
            self._Lrelax=np.zeros(self.shape,dtype=self._rtype)
        return self._Lrelax 
    
    @property
    def Lex(self):
        """
        Returns the exchange component of the Liouvillian

        Returns
        -------
        np.array

        """
        
        if self._Lex is None:
            if self.kex is None or self.kex.size!=len(self.H)**2 or self.kex.ndim!=2:
                self.kex=np.zeros([len(self.H),len(self.H)],dtype=self._rtype)
                if len(self.H)>1:print('Warning: Exchange matrix was not defined')
            self._Lex=np.kron(self.kex.astype(self._rtype),np.eye(np.prod(self.H[0].shape),dtype=self._rtype))
            
        return self._Lex
    
    def Ln_H(self,n:int):
        """
        Returns the nth rotating component of the Liouvillian resulting from
        the Hamiltonians. That is, contributions from exchange and relaxation
        matrices are not included.
        
        Only works if we are at a particular index of the Liouvillian
        L[0].Ln_H(0)
        Other

        Parameters
        ----------
        n : int
            Index of the rotating component (-2,-1,0,1,2).

        Returns
        -------
        np.array

        """
        assert self.sub,"Calling Ln_H requires indexing to a specific element of the powder average"
        out=np.zeros(self.shape,dtype=self._ctype)
        q=np.prod(self.H[0].shape)
        for k,H0 in enumerate(self.H):
            out[k*q:(k+1)*q][:,k*q:(k+1)*q]=H0.Ln(n)
        out*=-1j*2*np.pi
        return out
    
    def Ln(self,n:int):
        """
        Returns the nth rotation component of the total Liouvillian. 

        Parameters
        ----------
        n : int
            DESCRIPTION.

        Returns
        -------
        None.

        """
            
        assert self.sub,"Calling Ln requires indexing to a specific element of the powder average"
        
        if self._Ln is None:
            self._Ln=[self.Ln_H(n) for n in range(-2,3)]
            self._Ln[2]+=self.Lex+self.Lrelax
            
        return self._Ln[n+2]
        
        # out=self.Ln_H(n)
        # if n==0:  #Only add these terms to the n=0 term
        #     out+=self.Lex
        #     if self.Lrelax is not None:
        #         out+=self.Lrelax
        # return out
    
    @property
    def Lrf(self):
        """
        Liouville matrix due to RF field

        Returns
        -------
        None.

        """
        
        if self._fields!=self.fields:
            self._Lrf=None
                
        if self._Lrf is None:
            self._Lrf=np.zeros(self.shape,dtype=self._ctype)
            n=self.H[0].shape[0]**2
            Lrf0=Ham2Super(self.rf())
            for k in range(len(self.H)):
                self._Lrf[k*n:(k+1)*n][:,k*n:(k+1)*n]=Lrf0
            self._Lrf*=-1j*2*np.pi
            self._fields=copy(self.fields)
                
        return self._Lrf
    
    def L(self,step):
        """
        Returns the Liouvillian for a given step in the rotor cycle (t=step*L.dt)

        Parameters
        ----------
        step : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # Ln=[self.Ln(n) for n in range(-2,3)]
        # n_gamma=self.expsys.n_gamma
        # ph=np.exp(1j*2*np.pi*step/n_gamma)
        # return np.sum([Ln0*ph**(-m) for Ln0,m in zip(Ln,range(-2,3))],axis=0)+self.Lrf
    
        ph=np.exp(1j*2*np.pi*step/self.expsys.n_gamma)
        return np.sum([self.Ln(m)*(ph**(-m)) for m in range(-2,3)],axis=0)+self.Lrf
    
    
    def U(self,t0:float=None,Dt:float=None):
        """
        Calculates the propagator between times t0 and t0+Dt. By default, t0 will
        be set to align with the end of the last propagator calculated for 
        this system. By default, Dt will be one rotor period. 

        Parameters
        ----------
        t0 : float, optional
            Initial time for the propagator. The default is 0.
        Dt : float, optional
            Length of the propagator. 

        Returns
        -------
        U  :  Propagator

        """
        
        # assert self.sub,"Calling L.U requires indexing to a specific element of the powder average"
    
        self.validate_relax()
    
        if self.isotropic:
            assert Dt is not None,"For isotropic systems, one must specify Dt"
            t0=0
        else:
            if t0 is None:t0=self.expsys._tprop%self.taur
            if Dt is None:Dt=self.taur

        tf=t0+Dt
        
        self.expsys._tprop=0 if self.taur is None else tf%self.taur  #Update current time
        
        if self.sub:
            if self.isotropic:
                L=self.L(0)
                # U=expm(L*Dt)

                d,v=np.linalg.eig(L)
                U=v@np.diag(np.exp(d*Dt))@np.linalg.pinv(v)

                return Propagator(U,t0=t0,tf=tf,taur=self.taur,L=self,isotropic=self.isotropic)
            else:
                dt=self.dt
                n0=int(t0//dt)
                nf=int(tf//dt)
                
                tm1=t0-n0*dt
                tp1=tf-nf*dt
                
                if tm1<=0:tm1=dt
                
                L=self.L(n0)
                U=expm(L*tm1)
                    
                for n in range(n0+1,nf):
                    L=self.L(n)
                    U=expm(L*dt)@U
                if tp1>1e-10:
                    L=self.L(nf)
                    U=expm(L*tp1)@U
                return Propagator(U,t0=t0,tf=tf,taur=self.taur,L=self,isotropic=self.isotropic)
        else:
            if self._parallel and not(self.isotropic):
                dt=self.dt
                n0=int(t0//dt)
                nf=int(tf//dt)
                
                tm1=t0-n0*dt
                tp1=tf-nf*dt
                
                if tm1<=0:tm1=dt
                Ln=[[L0.Ln(k) for k in range(-2,3)] for L0 in self]
                U=prop(Ln,Lrf=np.array(self.Lrf),n0=n0,nf=nf,tm1=tm1,tp1=tp1,dt=dt,n_gamma=int(self.expsys.n_gamma))
                return Propagator(U=U,t0=t0,tf=tf,taur=self.taur,L=self,isotropic=self.isotropic)
                
            else:
                U=[L0.U(t0=t0,Dt=Dt).U for L0 in self]
            return Propagator(U=U,t0=t0,tf=tf,taur=self.taur,L=self,isotropic=self.isotropic)

    def Ueye(self,t0:float=None):
        """
        Returns a propagator with length zero (identity propagator)

        Returns
        -------
        t0 : float, optional
            Initial time for the propagator. The default is 0.
            
        U  :  Propagator

        """
        
        if self.isotropic:
            t0=0
        else:
            if t0 is None:t0=self.expsys._tprop%self.taur
            
        return self.U(t0,0)

    def Ueig(self):
        """
        Returns eigenvalues and eigenvectors of U for one rotor period. Can
        be used for fast propagation in the eigenbasis

        Returns
        -------
        tuple

        """
        
        # d,v=eig(self.U(),k=self.shape[0]-1)
        d,v=np.linalg.eig(self.U())
        i=np.abs(d)>1
        d[i]/=np.abs(d[i])
        return d,v
    
    def __len__(self):
        if self.pwdavg is None:return 1
        return self.pwdavg.N
    
    def __next__(self):
        self._index+=1
        if self._index==len(self):
            self._index=-1           
            raise StopIteration
        else:
            return self[self._index]
    
    def __iter__(self):
        self._index=-1
        return self
    
    def __repr__(self):
        out='Liouvillian with Hamiltonians:\n'
        for k,H in enumerate(self.H):
            out+=f'Hamiltonian #{k}\n'
            out+=H.__repr__()
            out+='\n'
            
        if len(self.H)>1:
            out+='Coupled by exchange matrix:\n'
            out+=self.kex.__repr__()
            
        if np.any(self.Lrelax)>0:
            out+='Explicit relaxation\n'
            for ri in self.relax_info:
                out+=f'{ri[0]} with arguments: '+','.join([f'{k} = {v}' for k,v in ri[1].items()])+'\n'
        return out
    
