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
from .Propagator import Propagator,PropCache
from . import Defaults
from .Tools import Ham2Super,BlockDiagonal
from .Hamiltonian import Hamiltonian
from . import RelaxMat
from .RelaxClass import RelaxClass
from .Sequence import Sequence
from .Para import ParallelManager, StepCalculator
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# import importlib.util
# numba=importlib.util.find_spec('numba') is not None
# if numba:
#     from .Parallel import prop

from .Parallel import prop,prop_static

class Liouvillian():
    def __init__(self,*ex,kex=None):
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
        if len(ex)==1:ex=ex[0]
        if hasattr(ex,'shape') or hasattr(ex,'B0'):ex=[ex]
        self.H=[*ex]
        
        for k,H in enumerate(self.H):
            if not(hasattr(H,'Hinter')) and hasattr(H,'B0'):
                H=Hamiltonian(H)
                self.H[k]=H
            assert hasattr(H,'Hinter'),'Liouvillian must be provided with Hamiltonian or ExpSys objects'
            assert H.pwdavg==self.pwdavg,"All Hamiltonians must have the same powder average"
            if H.rf is not self.rf:
                H.expsys._rf=self.rf
                
        self._children=[]  #Store reduced children of L for clearling the cache

        self._PropCache=PropCache(self)
        
        self._shape=np.prod(self.H[0].shape)*len(self.H),np.prod(self.H[0].shape)*len(self.H)
        
        
        # self.sub=False
        
        self._Lex=None
        self._index=-1
        self._Lrelax=None
        self._Lrf=None
        self._Ln=None
        # self._Ln_H=None
        self._LrelaxOS=RelaxClass(self)
        if Defaults['cache']:self._Ln_H=[[None for _ in range(5)] for _ in range(len(self))]
        
        self._fields=self.fields
        
        if kex is None:kex=np.zeros([len(self.H),len(self.H)])
        self.kex=kex
        
        self.relax_info=[]  #Keeps a short record of what kind of relaxation is used
    
        
        
    def getBlock(self,block):
        """
        Returns a reduced version of this Liouvillian defined by a given
        block.

        Parameters
        ----------
        block : TYPE
            DESCRIPTION.

        Returns
        -------
        LiouvilleBlock 

        """
        return LiouvilleBlock(self, block)
    
    @property
    def block(self):
        return np.ones(self.shape[0],dtype=bool)
    
    def clear_cache(self):
        self._Ln_H=None
        self._PropCache.reset()
        if self._LrelaxOS is not None:self.LrelaxOS.clear_cache()
        for child in self._children: # A bit tricky to reset the reduced Liouvillian
            child._L=self
            child._PropCache=PropCache(child)
            child._LrelaxOS=RelaxClass(child)
            
        # for p in range(len(self.H)):  #Useful if the powder average is changed
        #     for q in range(len(self.H[0].Hinter)):
        #         if self.H[p].Hinter[q].rotInter is not None:
        #             self.H[p].Hinter[q].rotInter._Azz=None
        #             self.H[p].Hinter[q].rotInter._A=None
        #             self.H[p].Hinter[q].rotInter._Afull=None
            
        return self
    
    @property
    def reduced(self):
        return False
    
    @property
    def sub(self):
        if self._index==-1:return False
        return True

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
    def static(self):
        return self.expsys.vr==0 or self.isotropic
    
    @property
    def pwdavg(self):
        return self.H[0].pwdavg
    
    @property
    def Peq(self):
        # for ri in self.relax_info:
        #     if 'Peq' in ri[1] and ri[1]['Peq']:
        #         return True
        if self._LrelaxOS.Peq:return True
        for ri in self.relax_info:
            if ri[0]=='recovery':return True
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
        if self.isotropic or self.static:return None
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
        return self._shape
    
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
            
            if value is not None:
                value=np.array(value)
                assert value.shape[0]==value.shape[1],"Exchange matrix must be square"
                assert value.shape[0]==len(self.H),f"For {len(self.H)} Hamiltonians, exchange matrix must be {len(self.H)}x{len(self.H)}"
                if np.any(np.diag(value)>0):
                    warnings.warn("Diagonals of exchange matrix should not be positive")
                elif np.any(np.abs(value.sum(0))>1e-10*np.mean(-np.diag(value))):
                    warnings.warn("Invalid exchange matrix. Columns should sum to 0. Expect unphysical behavior.")
        
                self._Lex=None
                super().__setattr__(name,value)
                self.clear_cache()
                return
        
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
        Liouvillian''

        """
        out=copy(self)
        out._LrelaxOS=copy(out.LrelaxOS)
        
        out.H=[H0[i] for H0 in self.H]
        out._Ln=None
        out._Ln_H=None
        out._index=i
        out._PropCache=self._PropCache
        out._PropCache.L=out 
        out._LrelaxOS.L=out
        # The above line bothers me. If we extract a particular element and
        # then later another element, the first element's propagator cache
        # references the wrong element of the Liouvillian.
        return out
    
    def add_SpinEx(self,i:list,tc:float):
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
        always be cyclic, with equal populations.
        
        The correlation time is the inverse of the mean hopping rate constant. For
        two- and three-site exchange, there is only one unique hopping rate, but for
        higher numbers of states, the mean will be calculated.
        
        Note that this is implemented via the relaxation module (RelaxMat), and
        can equivalently be introduced by running add_relax with Type='SpinExchange'
    
        Parameters
        ----------
        i : list
            List of the spins in exchange. E.g. i=[1,2] would cause swaps between
            spins 1 and 2. i=[1,2,3] might represent a methyl rotation, yielding
            exchange such that either 1->2, 2->3, 3->1, and 1->3, 2->1, 3->2.
        tc : float
            Correlation time of the exchange (inverse of rate constant)
    
        Returns
        -------
        self
    
        """
        
        self.add_relax(Type='SpinExchange',i=i,tc=tc)
        
        return self
    
    def add_relax(self,M=None,Type:str=None,OS:bool=False,state:int=None,**kwargs):
        """
        Add explicit relaxation to the Liouvillian. This is either provided
        directly by the user via a matrix, or by type, where currently T1, T2, 
        and recovery are provided, where recovery forces the simulation to go 
        towards thermal equilibrium.
        
        T1: provide T1 and i, specifying the spin's index
        T2: provide T2 and i, specifying the spin's index
        
        In Orientation-specific mode (OS=True), the recovery towards thermal
        equilibrium needs to be specified when setting T1, by including
        Thermal=True as an argument to add_relax
        
        
        The exchange matrix can be explicitely provided by M directly. Note 
        that the matrix can either have the same shape as the full Liouvillian,
        or the shape for just one Hamiltonian. For example, for a two-spin 1/2 
        system in two-site exchange, M may have size 16x16 or 32x32. The 32x32
        matrix allows different relaxation properties for the two sites.

        Parameters
        ----------
        M : np.array, optional
            Provide the relaxation matrix explicitly. The default is None.
        Type : str, optional
            Type of relaxation to include. The default is None.
        OS : bool, optional
            Orientation-specific relaxation. Can be necessary where the 
            quantization axis of a spin-system is not along the z-axis.
            The default is False.
        state : int, optional
            For states in exchange, this index allows us to specify relaxation 
            for each state separately. The default is None.
        **kwargs : TYPE
            Arguments sent to the relaxation .

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
    

        self._PropCache.reset()
        
        if isinstance(M,str): #In case Type is input as the first argument, just fix for the user
            Type=M
            M=None
        
        if hasattr(self,'recovery'):
            warnings.warn('recovery should always be the last term added to Lrelax')
        
        if Type in ['DynamicThermal']:OS=True   #List methods only in RelaxClass here
        
        if OS:
            if state is not None:
                getattr(self.LrelaxOS,Type)(state=state,**kwargs)
            else:
                getattr(self.LrelaxOS,Type)(**kwargs)
            kwargs.update({'OS':OS})
            self.relax_info.append((Type,kwargs))
            return self
        
        
                
        
        if M is None:
            if Type=='recovery':
                M=RelaxMat.recovery(expsys=self.expsys,L=self)
                self.relax_info.append(('recovery',{'OS':OS}))
            # elif Type=='Thermal':
            #     self.LrelaxOS.Thermal()
            #     self.relax_info.append(('Thermal',{}))
            #     return self
            elif hasattr(RelaxMat,Type):
                M=getattr(RelaxMat,Type)(expsys=self.expsys,**kwargs)
                self.relax_info.append((Type,kwargs))
            else:
                warnings.warn(f'Unknown relaxation type: {Type}')
                return self
        
        q=np.prod(self.H[0].shape)
        self.Lrelax  #Call to make sure it's pre-allocated
        if M.shape[0]==self.shape[0]:
            self._Lrelax+=M
        elif M.shape[0]==q:
            if state is not None:
                self._Lrelax[state*q:(state+1)*q][:,state*q:(state+1)*q]+=M
            else:
                for k in range(len(self.H)):
                    self._Lrelax[k*q:(k+1)*q][:,k*q:(k+1)*q]+=M
        else:
            assert False,f"M needs to have size ({q},{q}) or {self.shape}"   
            
        if state is not None:
            kwargs['state']=state
            
        return self
    
    def clear_relax(self):
        """
        Removes all explicitely defined relaxation

        Returns
        -------
        None.

        """
        self._PropCache.reset()
        
        self.relax_info=[]
        self._Lrelax=None
        self._LrelaxOS.clear()
        if hasattr(self,'recovery'):
            delattr(self,'recovery')
        self.clear_cache()
        return self
        
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
            self._Lrelax=np.zeros(self.shape,dtype=self._ctype)
        return self._Lrelax
    
    
    @property
    def LrelaxOS(self):
        """
        Returns the orientation-specific relaxation matrix for the current 
        rotor step and orientation

        Parameters
        ----------
        step : int, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        return self._LrelaxOS
    
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
        # self._Ln_H=None
        # if self._Ln_H is not None and self._Ln_H[self._index][n+2] is not None:
        #     return copy(self._Ln_H[self._index][n+2])
        
        out=np.zeros(self.shape,dtype=self._ctype)
        q=np.prod(self.H[0].shape)
        for k,H0 in enumerate(self.H):
            # TODO The next line should not be necessary. H0 should automatically be updated to its index
            # Maybe somewhere we just changed the index instead of calling for the specific item
            
            H0=H0[H0._index] 
            out[k*q:(k+1)*q][:,k*q:(k+1)*q]=H0.Ln(n)
        out*=-1j*2*np.pi
        
        
        # if self._Ln_H is not None:self._Ln_H[self._index][n+2]=out
        
        return copy(out)
    
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
    
    def L(self,step:int):
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
        return np.sum([self.Ln(m)*(ph**(-m)) for m in range(-2,3)],axis=0)+self.Lrf+self.LrelaxOS(step)    
    
    def Lcoh(self,step:int):
        """
        Returns the coherent Liouvillian for a given step in the rotor cycle (t=step*L.dt)

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
        out=np.sum([self.Ln_H(m)*(ph**(-m)) for m in range(-2,3)],axis=0)   
        return out
    
    def U(self,Dt:float=None,t0:float=None,calc_now:bool=False):
        """
        Calculates the propagator between times t0 and t0+Dt. By default, t0 will
        be set to align with the end of the last propagator calculated for 
        this system. By default, Dt will be one rotor period. 
        
        Note that the propagator in general will not be calculated until required.
        To force calculation on creation, set calc_now to True.

        Parameters
        ----------
        Dt : float, optional
            Length of the propagator. 
        t0 : float, optional
            Initial time for the propagator. The default is None, which sets t0
            to the end of the last calculated propagator
        calc_now : bool, optional.
            Calculates the propagator immediately, as opposed to only when required

        Returns
        -------
        U  :  Propagator

        """
        
        # assert self.sub,"Calling L.U requires indexing to a specific element of the powder average"
    
        self.validate_relax()
    
        if self.static:
            assert Dt is not None,"For static/isotropic systems, one must specify Dt"
            t0=0
        else:
            if t0 is None:t0=self.expsys._tprop%self.taur
            if Dt is None:Dt=self.taur

        tf=t0+Dt
        
        self.expsys._tprop=0 if self.taur is None else tf%self.taur  #Update current time
        
        voff=np.array([x[-1] for x in self.rf.fields.values()])
        ph_acc=(voff*Dt)*2*np.pi
        
        if calc_now:
            if self.sub:
                if self.static:
                    L=self.L(0)
                    # U=expm(L*Dt)
    
                    d,v=np.linalg.eig(L)
                    

                    
                    U=v@np.diag(np.exp(d*Dt))@np.linalg.pinv(v)
    
                    return Propagator(U,t0=t0,tf=tf,taur=self.taur,L=self,isotropic=self.isotropic,phase_accum=ph_acc)
                else:
                    # dt=self.dt
                    # n0=int(t0//dt)
                    # nf=int(tf//dt)
                    
                    # tm1=t0-n0*dt
                    # tp1=tf-nf*dt
                    
                    # if tm1<=0:tm1=dt
                    
                    dt=self.dt
                    n0,nf,tm1,tp1=StepCalculator(t0=t0,Dt=Dt,dt=dt)
                    
                    if tm1==dt:
                        U=self._PropCache[n0]
                    else:
                        L=self.L(n0)
                        U=expm(L*tm1)
                        
                        
                    for n in range(n0+1,nf):
                        U=self._PropCache[n]@U
                        # L=self.L(n)
                        # U=expm(L*dt)@U

                    if tp1>1e-10:
                        if tp1==dt:
                            U=self._PropCache[nf]@U
                        else:
                            L=self.L(nf)
                            U=expm(L*tp1)@U
                    return Propagator(U,t0=t0,tf=tf,taur=self.taur,L=self,isotropic=self.isotropic,phase_accum=ph_acc)
            else:
                if self.isotropic:
                    U=[L0.U(t0=t0,Dt=Dt,calc_now=calc_now).U for L0 in self]
                    return Propagator(U=U,t0=t0,tf=tf,taur=self.taur,L=self,isotropic=self.isotropic,phase_accum=ph_acc)
                else:
                    pm=ParallelManager(L=self,t0=t0,Dt=Dt)
                    U=pm()
                    return Propagator(U=U,t0=t0,tf=tf,taur=self.taur,L=self,isotropic=self.isotropic,phase_accum=ph_acc)
                # if self._parallel and not(self.static):
                #     dt=self.dt
                #     n0=int(t0//dt)
                #     nf=int(tf//dt)
                    
                #     tm1=t0-n0*dt
                #     tp1=tf-nf*dt
                    
                #     if tm1<=0:tm1=dt
                #     # Ln=[[L0.Ln(k) for k in range(-2,3)] for L0 in self]
                #     # U=prop(Ln,Lrf=np.array(self.Lrf),n0=n0,nf=nf,tm1=tm1,tp1=tp1,dt=dt,n_gamma=int(self.expsys.n_gamma))
                    
                #     pm=ParallelManager(L=self,n0=n0,nf=nf,tm1=tm1,tp1=tp1,dt=dt,n_gamma=self.pwdavg.n_gamma)
                #     U=pm()
                    
                    
                #     return Propagator(U=U,t0=t0,tf=tf,taur=self.taur,L=self,isotropic=self.isotropic)
                # elif self._parallel and not(self.isotropic) and False:  #Why doesn't this work?
                #     L=[L0.L(0) for L0 in self]
                #     U=prop_static(L,Dt=tf-t0)
                # else:
                #     U=[L0.U(t0=t0,Dt=Dt,calc_now=calc_now).U for L0 in self]
                # return Propagator(U=U,t0=t0,tf=tf,taur=self.taur,L=self,isotropic=self.isotropic)
        else:
            dct=dict()
            dct['t']=[t0,tf]
            dct['v1']=np.zeros([len(self.fields),2])
            dct['phase']=np.zeros([len(self.fields),2])
            dct['voff']=np.zeros([len(self.fields),2])
            for k,v in self.fields.items():
                dct['v1'][k],dct['phase'][k],dct['voff'][k]=v
            return Propagator(U=dct,t0=t0,tf=tf,taur=self.taur,L=self,isotropic=self.isotropic,phase_accum=ph_acc)
            
        
    def Ueye(self,t0:float=None):
        """
        Returns a propagator with length zero (identity propagator)

        Returns
        -------
        t0 : float, optional
            Initial time for the propagator. The default is 0.
            
        U  :  Propagator

        """
        
        if self.static:
            t0=0
        else:
            if t0 is None:t0=self.expsys._tprop%self.taur
        
        return Propagator(U=[np.eye(self.shape[0]) for _ in range(len(self))],
                          t0=t0,tf=t0,taur=self.taur,L=self,isotropic=self.isotropic,phase_accum=0)    
    
    def Udelta(self,channel,phi:float=np.pi,phase:float=0,t0:float=None):
        """
        Provides a delta pulse on the chosen channel or specific spin. Channel
        is provided with the nucleus name ('1H','13C','etc'), a specific spin
        or spins is provided by setting channel to an integer or list or 
        integers

        Parameters
        ----------
        channel : TYPE
            DESCRIPTION.
        phi : float, optional
            DESCRIPTION. The default is np.pi.
        phase : float, optional
            DESCRIPTION. The default is 0.
        t0 : float, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        
        if self.static:
            t0=0
        else:
            if t0 is None:t0=self.expsys._tprop%self.taur
        
        if isinstance(channel,str):
            if channel.lower()=='e':channel='e-'
            i=np.argwhere([channel==Nuc for Nuc in self.expsys.Nucs])[:,0]
        else:
            i=np.atleast_1d(channel)
                
        H=np.zeros(self.H[0].shape,dtype=self._ctype)
        for i0 in i:
            Op=self.expsys.Op[i0]
            H+=np.cos(phase)*Op.x+np.sin(phase)*Op.y
            
        L0=Ham2Super(H)
        L=np.zeros(self.shape,dtype=self._ctype)
        
        n=self.H[0].shape[0]**2
        for k in range(len(self.H)):
            L[k*n:(k+1)*n][:,k*n:(k+1)*n]=L0
        U=expm(-1j*phi*L)
        
        return Propagator(U=[U for _ in range(len(self))],
                          t0=t0,tf=t0,taur=self.taur,L=self,isotropic=self.isotropic,phase_accum=0)

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
    
    def Sequence(self,Dt:float=None,cyclic:bool=True,rho=None) -> Sequence:
        """
        Returns a Sequence object initialized from this Liouvillian

        Parameters
        ----------
        Dt : float, optional
            Timestep for the sequence. Typically only used if one intends to 
            make an empty sequence (no pulses). The default is None.
        cyclic : bool, optional
            If the sequence is execute for a Dt longer than the default sequence
            length, then setting cyclic to True will cause the sequence to
            repeat. If False, then the final state of the sequence will be
            retained. The default is False.
        rho : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        Sequence
            Pulse sequence object with this Liouvillian.

        """
        if Dt is not None:
            seq=Sequence(self,cyclic=cyclic,rho=rho)
            seq.add_channel(self.expsys.Nucs[0],t=Dt)
            return seq
        return Sequence(self,cyclic=cyclic,rho=rho)
    
    @property
    def ex_pop(self):
        """
        Returns the populations resulting from chemical exchange.

        Returns
        -------
        None.

        """
        self.Lex  #Forces a default kex if not defined
        if np.abs(self.kex).max()==0:  #All zeros– assume uniform population
            return np.ones(self.kex.shape[0])/self.kex.shape[0]
        d,v=np.linalg.eig(self.kex)
        pop=v[:,np.argmax(d)]
        pop/=pop.sum()
        return pop
        
        
    
    def rho_eq(self,Hindex:int=None,pwdindex:int=None,step:int=None,sub1:bool=False):
        """
        Returns the equilibrium density operator for a given Hamiltonian and
        element of the powder average.
        

        Parameters
        ----------
        Hindex : int, optional
            Index of the Hamiltonian, i.e. in case there are multiple 
            Hamiltonians undergoing exchange. The default is None, which
            will return the equilibrium density operator weighted by the 
            populations resulting from the exchange matrix.
        pwdindex : int, optional
            Index of the element of the powder average. Should not have an 
            influence unless the rotor is not at the magic angle or no 
            spinning is included (static, anisotropic). The default is 0.
        step : int, optional
            If provided, uses rho_eq for the particular step in the rotor cycle
            required. Otherwise, the average Hamiltonian will be used, that is,
            the rotating components will be omitted.
        sub1 : bool, optional
            Subtracts the identity from the density matrix. Primarily for
            internal use.
            The default is False

        Returns
        -------
        None.

        """
        if pwdindex is None:
            pwdindex=0 if self._index==-1 else self._index
            
        if Hindex is None:
            pop=self.ex_pop
            N=self.block.shape[0]//len(pop)
            rho_eq=np.zeros(self.block.shape[0],dtype=self._ctype)
            for k,p in enumerate(pop):
                rho_eq[N*k:N*(k+1)]=self.rho_eq(k,pwdindex=pwdindex,step=step,sub1=sub1)*p
            return rho_eq[self.block]
            # pop=self.ex_pop
            
            
            # # 6 July 2024. Why were we calculating this here and not just getting it from the Hamiltonian?
            # H0=list()
            # for H in self.H:
            #     if self.static and not(self.isotropic): #Include all terms Hn
            #         H0.append(np.sum([H[pwdindex].Hn(m) for m in range(-2,3)],axis=0))
            #     else:
            #         H0.append(H[pwdindex].Hn(0))
            #     for k,LF in enumerate(self.expsys.LF):
            #         if not(LF):
            #             H0[-1]+=H.expsys.v0[k]*self.expsys.Op[k].z
            
            
            # ##Approach 1: same rho_eq0 for all states in exchange
            # # H=(np.array(H0).T*pop).sum(-1).T
            # # rho_eq0=expm(6.62607015e-34*H/(1.380649e-23*self.expsys.T_K))
            # # rho_eq0/=np.trace(rho_eq0)
            # # if sub1:
            # #     eye=np.eye(rho_eq0.shape[0])
            # #     rho_eq0-=np.trace(rho_eq0@eye)/rho_eq0.shape[0]*eye
            # # rho_eq=np.zeros(self.shape[0],dtype=self._ctype)
            # # n=self.H[0].shape[0]**2
            # # for k,p in enumerate(pop):
            # #     rho_eq[k*n:(k+1)*n]=rho_eq0.flatten()*p
            
            # ##Approach 2: different rho_eq0 for each state in exchange
            
            # rho_eq=np.zeros(self.shape[0],dtype=self._ctype)
            # n=self.H[0].shape[0]**2
            # for k,(H,p) in enumerate(zip(H0,pop)):
            #     rho_eq0=expm(6.62607015e-34*H/(1.380649e-23*self.expsys.T_K))
            #     rho_eq0/=np.trace(rho_eq0)
            #     if sub1:
            #         eye=np.eye(rho_eq0.shape[0])
            #         rho_eq0-=np.trace(rho_eq0@eye)/rho_eq0.shape[0]*eye
            #     rho_eq[k*n:(k+1)*n]=rho_eq0.flatten()*p
                
        
            # return rho_eq
        else:
            return self.H[Hindex].rho_eq(pwdindex=pwdindex,step=step,sub1=sub1).reshape(self.block.shape[0]//len(self.H))
        
        
    @property    
    def Energy(self):
        """
        Energy for each of the NxNxnHam states in the Liouvillian, including 
        energy from the Larmor frequency (regardless of whether in lab frame).
        Neglects rotating terms, Hn, for n!=0

        Returns
        -------
        None.

        """
        Energy=np.zeros(self.shape[0])
        N=self.H[0].shape[0]**2
        for k,H in enumerate(self.H):
            Energy[k*N:(k+1)*N]=H.Energy
        return Energy
    
    def Energy2(self,step:int):
        """
        Energy for each of the NxNxnHam states in the Liouvillian, including 
        energy from the Larmor frequency (regardless of whether in lab frame).
        Neglects rotating terms, Hn, for n!=0

        Returns
        -------
        None.

        """
        Energy=np.zeros(self.shape[0])
        N=self.H[0].shape[0]**2
        for k,H in enumerate(self.H):
            Energy[k*N:(k+1)*N]=H.Energy2(step)
        return Energy

    
    def plot(self,what:str='L',seq=None,rho=None,cmap:str=None,mode:str='log',colorbar:bool=True,
             step:int=0,block:int=None,ax=None) -> plt.axes:
        """
        Visualizes the Liouvillian matrix. Options are what to view (what) and 
        how to display it (mode), as well as colormaps and one may optionally
        provide the axis.
        
        Note, one should index the Liouvillian before running. If this is not
        done, then we jump to the halfway point of the powder average
        
        what:
        'L' : Full Liouvillian. Optionally specify time step
        'Lcoh' : Coherent Hamiltonian. Optionally specify time step
        'Lrelax' : Full relaxation matrix
        'Lrf' : Applied field matrix
        'recovery' : Component of relaxation matrix responsible for magnetizaton recovery
        'L0', 'L1', 'L2', 'L-1', 'L-2' : Liouvillians from different components of the
        Hamiltonian (does not include relaxaton / RF)
        
        mode:
        'abs' : Colormap of the absolute value of the plot
        'log' : Similar to abs, but on a logarithmic scale
        'signed' : Usually applied for real matrices (i.e. relaxation), which
                    shifts the data to show both negative and positive values
                    (imaginary part will be omitted)
        'spy' : Black/white for nonzero/zero (threshold applied at 1/1e6 of the max)



        Parameters
        ----------
        what : str, optional
            Specifies which Liouville matrix to plot. The default is 'L'.
        seq : Sequence, optional
            Include a sequence, which is used to determine what channels will
            have rf turn on at some point. Uses the max v1 setting for each
            channel in the sequence for plotting.
        rho : Rho, optional
            Providing rho will cause the plot to only include terms in the 
            Liouvillian required for propagating and detecting rho.
        cmap : str, optional
            Colormap used for plotting. The default is 'YOrRd'.
        mode : str, optional
            Plotting mode. The default is 'abs'.
        colorbar : bool, optional
            Turn color bar on/off. The default is True.
        step : int, optional
            Specify which step in the rotor period to plot. The default is 0.
        ax : plt.axis, optional
            Provide an axis to plot into. The default is None.

        Returns
        -------
        plt.axes
            Returns the plot axis object

        """
    
        if seq is not None:
            assert seq.L is self,"Sequence was not generated from this Liouvillian"
    
        if rho is not None:
            if seq is None:seq=self.Sequence()
            _,seq=rho.ReducedSetup(seq)
            return seq.L.plot(what=what,seq=seq,rho=None,cmap=cmap,mode=mode,
                       colorbar=colorbar,step=step,ax=ax)
    
        mode=mode.lower()
        
        if what=='Lrelax' and np.max(np.abs(self.Lrelax))==0 and self.LrelaxOS.active:
            what='LrelaxOS'
    
        if ax is None:
            fig,ax=plt.subplots()
        else:
            fig=None
            
        if seq is not None:
            fields=copy(self.rf.fields)
            for k,(v1,phase,voff) in enumerate(zip(seq.v1,seq.phase,seq.voff)):
                if np.any(v1):
                    i=np.argmax(v1)
                    self.rf.add_field(k,v1=v1[i],phase=phase[i],voff=voff[i])
                    
        
        if cmap is None:
            if mode == 'abs' or mode=='log':
                cmap='YlOrRd'
            elif mode == 'signed':
                cmap='BrBG'
            elif mode == 'spy':
                cmap= 'binary'
                
        if what in ['L0','L1','L2','L-1','L-2']:
            x=self[len(self)//2].Ln_H(int(what[1:])) if self._index==-1 else self.Ln_H(int(what[1:]))
        # elif what=='Lcoh':
        #     ph=np.exp(1j*2*np.pi*step/self.expsys.n_gamma)
        #     index=len(self)//2 if self._index==-1 else self._index
        #     x=np.sum([self[index].Ln_H(k)*ph**(-k) for k in range(-2,3)],axis=0)
        else:
            x=getattr(self[len(self)//2] if self._index==-1 else self,what)
            if hasattr(x,'__call__'):
                x=x(step)
                
        if mode=='log' and np.max(np.abs(x[x!=0]))==np.min(np.abs(x[x!=0])):
            mode='abs'
        
        sc0,sc1,sc=1,1,1
        if mode=='abs':
            x=np.abs(x)
            sc=x.max()
            x/=sc
        elif mode in ['re','im']:
            x=copy(x.real if mode=='re' else x.imag)
            sc=np.abs(x).max()
            x/=sc*2
            x+=.5
        elif mode=='spy':
            cutoff=np.abs(x).max()*1e-6
            x=np.abs(x)>cutoff
        elif mode=='log':
            x=np.abs(x)
            i=np.logical_not(x==0)
            if i.sum()!=0:
                if x[i].min()==x[i].max():
                    sc0=sc1=np.log10(x[i].max())
                    x[i]=1
                else:
                    x[i]=np.log10(x[i])
                    sc0=x[i].min()
                    x[i]-=sc0
                    x[i]+=x[i].max()*.2
                    sc1=x[i].max()
                    x[i]/=sc1
                    
                    sc1=sc1/1.2+sc0
        else:
            assert 0,'Unknown plotting mode (Try "abs", "re", "im", "spy", or "log")'
        
        if block is not None:
            assert isinstance(block,int),'block must be an integer'
            bi=BlockDiagonal(self[len(self)//2].L(0))
            assert block<len(bi),f"block must be less than the number of independent blocks in the Liouville matrix ({len(bi)})"
            bi=bi[block]
            x=x[bi][:,bi]
        elif hasattr(self,'block'):
            bi=self.block
        else:
            bi=np.ones(len(x),dtype=bool)
        
        hdl=ax.imshow(x,cmap=cmap,vmin=0,vmax=1)
        
        if colorbar and mode!='spy':
            hdl=plt.colorbar(hdl)
            if mode=='abs':
                hdl.set_ticks(np.linspace(0,1,6))
                hdl.set_ticklabels([f'{q:.2e}' for q in np.linspace(0,sc,6)])
                hdl.set_label(r'$|L_{n,n}|$')
            elif mode=='log':
                hdl.set_ticks(np.linspace(0,1,6))
                labels=['0',*[f'{10**q:.2e}' for q in np.linspace(sc0,sc1,5)]]
                hdl.set_ticklabels(labels)
                hdl.set_label(r'$|L_{n,n}|$')
            elif mode in ['re','im']:
                hdl.set_ticks(np.linspace(0,1,5))
                labels=[f'{q:.2e}' for q in np.linspace(-sc,sc,5)]
                hdl.set_ticklabels(labels)
                hdl.set_label(r'$L_{n,n}$')
            
        labels=self.expsys.Op.Llabels
        if labels is not None:
            if len(self.H)>1:
                label0=[]
                for k in range(len(self.H)):
                    for l in labels:
                        label0.append('|'+l+fr'$\rangle_{{{k+1}}}$')
            else:
                label0=['|'+l+r'$\rangle$' for l in labels]
            label0=np.array(label0)[bi]
            
            
            def format_func(value,tick_number):
                value=int(value)
                if value>=len(label0):return ''
                elif value<0:return ''
                return label0[value]

            ax.set_xticklabels('',rotation=-90)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
            
            if len(self.H)>1:
                label1=[]
                for k in range(len(self.H)):
                    for l in labels:
                        label1.append(r'$\langle$'+l+fr'$|_{{{k+1}}}$')
            else:
                label1=[r'$\langle$'+l+'|' for l in labels]
            label1=np.array(label1)[bi]    
            
                
            def format_func(value,tick_number):
                value=int(value)
                if value>=len(label0):return ''
                elif value<0:return ''
                return label1[value]
            ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
            
        ax.xaxis.set_major_locator(MaxNLocator(min([bi.sum(),20]),integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(min([bi.sum(),20]),integer=True))
        if fig is not None:fig.tight_layout()
        
        if seq is not None:
            self.rf.fields=fields
            
        return ax
            
            
                
            
        
        
    
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
        out='Liouvillian under the following conditions:\n\t'
        for k,H in enumerate(self.H):
            
            if k:
                out+=f'Hamiltonian #{k}\n\t'
                out+=H.__repr__().split('\n',1)[1].split('\n',7)[-1].rsplit('\n',1)[0].replace('\n','\n\t')
                out+='\n\t'
            else:
                out+='\n\t'.join(H.__repr__().split('\n')[1:7])
                out+='\n\nThe individual Hamiltonians have the following interactions\n\t'
                out+=f'Hamiltonian #{k}\n\t'
                out+=H.__repr__().split('\n',1)[1].split('\n',7)[-1].rsplit('\n',1)[0].replace('\n','\n\t')
                out+='\n\t'
            
        if len(self.H)>1:
            out+='\nHamiltonians are coupled by exchange matrix:\n\t'
            out+=self.kex.__repr__().replace('\n','\n\t')
            
        if self.relax_info:
            out+='\n\nExplicit relaxation\n'
            for ri in self.relax_info:
                if len(ri[1]):
                    out+=f'\t{ri[0]} with arguments: '+', '.join([f'{k} = {v}' for k,v in ri[1].items()])+'\n'
                else:
                    out+=f'\t{ri[0]}'
        out+='\n\n'+super().__repr__()
        return out
    
class LiouvilleBlock(Liouvillian):
    def __init__(self,L,block):
        self.__dict__=copy(L.__dict__)
        self._L=L
        self._block=block
        self._PropCache=PropCache(self)
        self._LrelaxOS=RelaxClass(self)
        self._shape=block.sum(),block.sum()
        self.LrelaxOS.methods=L.LrelaxOS.methods
        
    def __getitem__(self,i:int):
        # TODO edit accordingly
        """
        Goes to a particular item of the powder average

        Parameters
        ----------
        i : int
            Element of the powder average to go to.

        Returns
        -------
        Liouvillian''

        """
        out=copy(self)
        out._LrelaxOS=copy(out.LrelaxOS)
        
        out._L=self._L[i]
        
        out._index=i
        out._PropCache=self._PropCache
        out._PropCache.L=out 
        out._LrelaxOS.L=out
        # The above line bothers me. If we extract a particular element and
        # then later another element, the first element's propagator cache
        # references the wrong element of the Liouvillian.
        return out
        
    # def L(self,step):
    #     return self._L[self._index].L(step)[self.block][:,self.block]
    
    # def Ln(self,n:int):
    #     return self._L[self._index].Ln(n)[self.block][:,self.block]
    
    def L(self,step):
        return self._L.L(step)[self.block][:,self.block]
    
    def Ln(self,n:int):
        return self._L.Ln(n)[self.block][:,self.block]
    
    @property
    def Lrf(self):
        return self._L.Lrf[self.block][:,self.block]
    
    @property
    def Lex(self):
        return self._L.Lex[self.block][:,self.block]
        
    
    @property
    def shape(self):
        return self._shape
    
    @property
    def reduced(self):
        return True
    
    @property
    def block(self):
        return self._block
    
    
        
    
    
        
    