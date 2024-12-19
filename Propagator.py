#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 21:29:29 2023

@author: albertsmith
"""

import numpy as np
from fractions import Fraction
import warnings
from copy import copy
from scipy.linalg import expm
from . import Defaults
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

tol=1e-10

class Propagator():
    def __init__(self,U,t0,tf,taur,L,isotropic,phase_accum):
        self.U=U
        self.pwdavg=False if hasattr(U,'shape') else True
        self.t0=t0
        self.tf=tf
        self.taur=taur
        self.L=L
        self._index=-1
        self._isotropic=isotropic
        self._eig=None
        self.phase_accum=phase_accum%(2*np.pi)
        
    
    @property
    def calculated(self):
        if isinstance(self.U,dict):return False
        return True
    
    @property
    def isotropic(self):            #Not sure this is necessary. Why not just get the value out of L?
        return self._isotropic
        
    @property
    def Dt(self):
        return self.tf-self.t0
    
    @property
    def expsys(self):
        return self.L.expsys
    
    @property
    def shape(self):
        return self.L.shape
    
    @property
    def static(self):
        return self.L.static
    
    @property
    def reduced(self):
        return self.L.reduced
    
    @property
    def block(self):
        return self.L.block
    
    def getBlock(self,block):
        """
        Returns a object that has been reduced for a particular block. Provide
        the logical index for the given block.

        Parameters
        ----------
        block : np.array (bool type)
            Logical array specifying the block to propagate.

        Returns
        -------
        Propagator

        """
        out=copy(self)
        out.L=self.L.getBlock(block)
        if self.calculated:
            for k,U in enumerate(self.U):
                out.U[k]=U[block][:,block]
                
        return out
            
    def eig(self,back_calc:bool=True):
        """
        Calculates eigenvalues/eigenvectors of all stored propagators. Stored for
        later usage (self._eig). Subsequent operations will be performed in
        the eigenbasis where possible. Note that we will also ensure that no
        eigenvalues have an absolute value greater than 1. For systems that 
        deviate due to numerical error, this approach may stabilize the system.

        Parameters
        ----------
        back_calc : bool, optional
            Back-calculates the stored propagators from the eigenvalues which
            have been corrected such that they do not have magnitude greater than
            one. This may stabilize some calculations.

        Returns
        -------
        self

        """
        if self._eig is None:
            self._eig=list()
            Unew=list()
            for k,U in enumerate(self):
                d,v=np.linalg.eig(U)
                vi=np.linalg.pinv(v)
                # d,v=eigs(U,k=U.shape[0]//2)
                # We do this because anything above 1 would be producing magnetization
                # The most we can have is 1, which represents equilibrium
                # under the current conditions
                
                # Is this correct, though? Does DNP produce magnetization? At
                # the moment, Solid Effect works. So probably ok.
                
                # No eigenvalue can grow in time
                dabs=np.abs(d) 
                i=dabs>1
                d[i]/=dabs[i]
                if self.L.Peq and not self.reduced:
                    # This approach is not valid for some reduced matrices....
                    
                    # We do this because there does indeed need to be an equilibrium
                    # state. It is possible, however, that without relaxation,
                    # the equilibrium state is never accessed. Still, it
                    # doesn't hurt to enforce it's existence. Note that we're
                    # just cleaning up numerical error that might have the
                    # equilibrium deviate slightly from 1, leading to slow
                    # decay of all magnetization. It won't increase only because
                    # we already cleaned that up several lines above.
                    #
                    # Actually....it's totally wrong to do this. Ignore the above
                    # I'm leaving my comments in case I remember why
                    # I thought this was a good idea...oops
                    #
                    # And now I don't know why I thought it was wrong...I think
                    # I'm mixing up what the Liouvillian and the propagator
                    # should do.
                    
                    # And a final comment: I think it's always ok for the full
                    # matrix to set one state to have an eigenvalue of 1, 
                    # representing the equilibrium position of the density
                    # matrix. This is analogous to the null-space of the 
                    # Liouvillian (which, btw, if only coherent, then has
                    # a larger null-space, but once relaxation/dynamics come
                    # in, then the null-space seems to reduce to 1 element)
                    
                    # However, it is not correct under irradiation to force
                    # the corresponding eigenvector to be equal to the equilibrium
                    # density operator. In particular, it destroys DNP
                    # transfers.
                    
                    # and finally, if the matrix is reduced, it is possible
                    # that we have excluded the eigenvector with value of 1,
                    # so we should also not enforce this.
                    
                    # n=self.L.H[0].shape[0]
                    # Force the equilibrium value to exist.
                    i=np.argmax(np.abs(d))
                    d[i]=1.
                    # i=np.argsort(d.real)[-n:]
                    
                    # The
                    # v[:,i]=self.L.rho_eq(pwdindex=k)
                    # v[:,i]/=np.sqrt((v[:,i].conj()*v[:,i]).sum())
                    
                    # q=np.eye(self.L.H[0].shape[0],dtype=bool).flatten()
                    # q=np.logical_not(np.tile(q,len(self.L.H)))
                    # v[:,i][q]=0
                    
                    # i=np.abs(d)<1e-12
                    # d[i]=0
                self._eig.append((d,v,vi))
                if back_calc:
                    Unew.append(v@np.diag(d)@vi)
            if back_calc:self.U=Unew
        return self
        
    
    def calcU(self):
        """
        We have the option when generating U, to not actually calculate its value
        until an operation requiring U is performed. This is potentially useful
        since operations rho*U do not actually require ever calculating U
        explicitely. This function calculates and stores U.
        
        Note that this function runs when self.U is a dictionary instead of
        a list, with keys t, v1, phase, and voff. Once run, U is replaced by
        the propagator matrices for each element of the powder average.

        Returns
        -------
        None.

        """
        
        
        if not(self.calculated):
            ct=self.expsys.current_time  #This operation should not change the current time
            
            dct=self.U
            t=dct['t']
            
            L=self.L
            
            ini_fields=copy(L.fields)
    
            U=L.Ueye(t[0])
            for m,(ta,tb) in enumerate(zip(t[:-1],t[1:])):
                for k,(v1,phase,voff) in enumerate(zip(dct['v1'],dct['phase'],dct['voff'])):
                    L.fields[k]=(v1[m],phase[m],voff[m])
                U0=self.L.U(t0=ta,Dt=tb-ta,calc_now=True)
                U=U0*U
                
            L.fields.update(ini_fields)
            
            self.U=U.U
            
            self.expsys._tprop=ct  #This operation should not change the current time
        return self
         
    @property
    def rotor_fraction(self):
        """
        Determines how we may divide the propagator into the rotor cycle. For
        example, if the propagator has a length of 0.4 rotor cycles, then 
        every 2 rotor cycles, corresponding to 5 propagator steps, we may then
        recycle the propagator. Then, this function would return a 2 and a 5

        Returns
        -------
        tuple

        """
        out=Fraction(self.Dt/self.taur).limit_denominator(100000)
        return out.numerator,out.denominator
        
    def __mul__(self,U):
        if str(U.__class__)!=str(self.__class__):
            return NotImplemented
        
        if self.tf==self.t0:
            self.t0=U.tf
            self.tf=U.tf
        if U.tf==U.t0:
            U.t0=self.t0
            U.tf=self.t0
        
        if not(self.static) and np.abs((self.t0-U.tf)%self.taur)>tol and np.abs((U.tf-self.t0)%self.taur)>tol:
            warnings.warn(f'\nFirst propagator ends at {U.tf%self.taur} but second propagator starts at {self.t0%U.taur}')
        
        assert U.shape[0]==self.shape[0],"Propagator shapes do not match"
        assert U.block.sum(0)==self.block.sum(0),"Different matrix reduction applied to propagators (cannot be multiplied)"
        if not(np.all(U.block==self.block)):
            warnings.warn(f'\nMatrix blocks do not match. This is almost always wrong')
        
        if self.pwdavg:
            assert U.pwdavg,"Both propagators should have a powder average or both not"
        else:
            assert not(U.pwdavg),"Both propagators should have a powder average or both not"

        if U is self:
            Uout=[U1@U1 for U1 in self]
        else:
            Uout=[U1@U2 for U1,U2 in zip(self,U)]
        
        if not(self.pwdavg):Uout=Uout[0]
        return Propagator(Uout,t0=U.t0,tf=U.tf+self.Dt,taur=self.taur,L=self.L,isotropic=self.isotropic,phase_accum=self.phase_accum+U.phase_accum)
    
    # def __rmul__(self,U):
    #     if U==1:
    #         return self
        
    
    def __pow__(self,n):
        """
        Raise the operator to a given power
        
        Probably, we should consider when to use eigenvalues and when to do
        a direct calculation. Currently always uses eigenvalues, which always
        takes about the same amount of time regardless of n. For a 32x32 matrix,
        n=100 is roughly equally as fast for both operations.  
        
        But, this is the rule for a 32x32 L matrix. Scaling is supposedly
        O(n^3) for matrix multiplication
        O(n^2) for eigenvalue decomposition.

        Parameters
        ----------
        n : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        if n==np.inf or (isinstance(n,str) and n.lower()=='inf'):
            self.eig()
            Uout=list()
            for d,v,vi in self._eig:
                i=d==1.
                if np.any(i):
                    Uout.append(v[:,i]@vi[i])
                else:
                    Uout.append(np.zeros(v.shape,dtype=v.dtype))
                
            out=Propagator(Uout,t0=self.t0,tf=self.t0+self.Dt,taur=self.taur,L=self.L,isotropic=self.isotropic,phase_accum=0)
            return out        

        if not(self.static) and (self.Dt%self.taur>tol or (self.taur-self.Dt)%self.taur>tol):
            warnings.warn('Power of a propagator should only be used if the propagator length is an integer multiple of rotor periods')

        if not(self.static) and not(np.int64(n)==n):
            warnings.warn('Warning: non-integer powers may not accurately reflect state of propagator in the middle of a rotor period')

    
        self.eig()

        Uout=list()
        _eig=list()
        for d,v,vi in self._eig:
            D=d**n
            Uout.append(v@np.diag(D)@vi)
            _eig.append((D,v,vi))
        
        # for U in self:
        #     d,v=np.linalg.eig(U)
        #     i=np.abs(d)>1
        #     d[i]/=np.abs(d[i]) #Avoid growth from numerical error
        #     D=d**n
        #     Uout.append(v@np.diag(D)@np.linalg.pinv(v))
            
        if not(self.pwdavg):Uout=Uout[0]
        
        out=Propagator(Uout,t0=self.t0,tf=self.t0+self.Dt*n,taur=self.taur,L=self.L,isotropic=self.isotropic,phase_accum=self.phase_accum*n)
        out._eig=_eig
        return out   

    def plot(self,index:int=None,mode:str='log',cmap:str=None,colorbar:bool=True,
             ax=None) -> plt.axes:
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
    
        mode=mode.lower()
        
    
        if ax is None:
            fig,ax=plt.subplots()
        else:
            fig=None
                    
        
        if cmap is None:
            if mode == 'abs' or mode=='log':
                cmap='YlOrRd'
            elif mode == 'signed':
                cmap='BrBG'
            elif mode == 'spy':
                cmap= 'binary'
               
        x=self[len(self)//2 if index is None else index]
                
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
        
        if hasattr(self,'block'):
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
            if len(self.L.H)>1:
                label0=[]
                for k in range(len(self.L.H)):
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
            
            if len(self.L.H)>1:
                label1=[]
                for k in range(len(self.L.H)):
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
            
        return ax         
    
    def __getitem__(self,i:int):
        """
        Return the ith propagator

        Parameters
        ----------
        i : int
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        if not(self.pwdavg):return self.U
        self.calcU()
        return self.U[i%len(self)]
    
    def __len__(self):
        if not(self.pwdavg):return 1
        return self.L.pwdavg.N
    
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
        out=f'Propagator with length of {self.Dt*1e6:.3f} microseconds (t0={self.t0*1e6:.3f},tf={self.tf*1e6:.3f})\n'
        out+='Constructed from the following Liouvillian:\n\t'
        out+=self.L.__repr__().replace('\n','\n\t').rsplit('\n',1)[0]
        out+='\n'+super().__repr__()
        
        return out
                
    

try:
    from multiprocessing.shared_memory import SharedMemory
    SM=True
except:
    SM=False
    
class PropCache():
    def __init__(self,L):
        """
        Stores propagators that may be later recycled.

        Parameters
        ----------
        L : TYPE
            DESCRIPTION.
        active : bool, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        None.

        """
        self.shared_memory=SM

        
        self.L=L
        self._sm0=[]
        self._sm1=[]
        self.reset()
        
        
    def reset(self):
        self.fields=[]
        self._U=[]
        self._calc_index=[]
        for sm in [*self._sm0,*self._sm1]:
            if sm is None:continue
            sm.unlink()
        self._sm0=[]
        self._sm1=[]
        self.cache=Defaults['cache']
        return self
        
        
    @property
    def pwdavg(self):
        return self.L.pwdavg
    
    
    #%% Return the applied-field specific information
    @property
    def field(self):
        return tuple(x for x in self.L.rf.fields.values())
    
    @property
    def field_index(self):
        if self.field not in self.fields:
            self.add_field()
        return self.fields.index(self.field)
    
    @property
    def U(self):
        return self._U[self.field_index]
    @U.setter
    def U(self,U):
        self._U.append(U)
    
    @property
    def calc_index(self):
        if not(self.cache):return None
        return self._calc_index[self.field_index]
    @calc_index.setter
    def calc_index(self,x):
        self._calc_index.append(x)
    
    @property
    def sm0(self):
        if not(self.cache):return None
        return self._sm0[self.field_index]
    @sm0.setter
    def sm0(self,sm0):
        self._sm0.append(sm0)
    
    @property
    def sm1(self):
        if not(self.cache):return None
        return self._sm1[self.field_index]
    @sm1.setter
    def sm1(self,sm1):
        self._sm1.append(sm1)
    
    
    @property
    def nbytes(self):
        nb=np.zeros(1,dtype=Defaults['ctype']).nbytes
        return np.prod(self.SZ)*nb
        
    #%% Sizes/indices
    @property
    def SZ(self):
        return (self.pwdavg.N,(self.pwdavg.n_gamma if self.pwdavg._gamma_incl else 1),*self.L.shape)
    
    def index(self,n):
        if self.pwdavg._gamma_incl:
            return self.L._index
        return (self.L._index+n*self.pwdavg.n_alpha)%self.pwdavg.N
    
    def step_index(self,n):
        if self.pwdavg._gamma_incl:
            return n
        return 0
 
    #%% Add field + data management
    def add_field(self):
        if not(self.cache):return
        if self.field not in self.fields:
            self.fields.append(self.field)
            if Defaults['parallel'] and self.shared_memory:
                self.sm0=SharedMemory(create=True,size=np.prod(self.SZ[:2]))
                self.sm1=SharedMemory(create=True,size=self.nbytes)
                self.calc_index=np.ndarray(shape=self.SZ[:2],dtype=bool,buffer=self.sm0.buf)
                self.U=np.ndarray(shape=self.SZ,dtype=Defaults['ctype'],buffer=self.sm1.buf)
            else:
                self.sm0=None
                self.sm1=None
                self.calc_index=np.zeros(self.SZ[:2],dtype=bool)
                self.U=np.zeros(self.SZ,dtype=Defaults['ctype'])          
        return self
    
    def __del__(self,*args):
        for sm in [*self._sm0,*self._sm1]:
            if sm is None:continue
            sm.unlink()
        
    
    #%% Return propagators
    def __getitem__(self,n:int):
        return self.get_prop(n%len(self))
    
    def __len__(self):
        return self.pwdavg.n_gamma
    
    def get_prop(self,n:int):
        if not(self.cache):return expm(self.L.L(n)*self.L.dt)
        U=self.U
        i0,i1=self.index(n),self.step_index(n)
        if not(self.calc_index[i0,i1]):
            U[i0,i1]=expm(self.L.L(n)*self.L.dt)
            self.calc_index[i0,i1]=True
            
        return U[i0,i1]

            
    
        
    
            
        