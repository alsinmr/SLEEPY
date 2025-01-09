#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 22:32:33 2023

@author: albertsmith
"""

from copy import copy
import numpy as np
import warnings
import matplotlib.pyplot as plt
from . import Defaults
from .Tools import NucInfo,BlockDiagonal,ApodizationFun
import re


ctype=Defaults['ctype']
rtype=Defaults['rtype']
tol=1e-10

class Rho():
    def __init__(self,rho0,detect,Reduce:bool=True,L=None):
        """
        Creates an object that contains both the initial density matrix and
        the detector matrix. One may then apply propagators to the density
        matrix and detect the magnetization.

        Strings for specifying operators:
                
        S0x, S1y, S2alpha:
            Specifies a spin by index (S0, S1, etc.), followed by the operator
            type (x,y,z,p,m,alpha,beta)
            
        13Cx, 1Hy, 15Np:
            Specifies the channel followed by the operator type (sum of all nuclei of that type)
            
        Custom operators may be produced by adding together the matrices found
        in expsys.Op
        

        Parameters
        ----------
        rho0 : Spinop or, str, optional
            Initial density matrix, specify by string or the operator itself. 
            Operators may be found in expsys.Op.
        detect : Detection matrix or list of matrices, specify by string or the
            operator itself. Operators may be found in expsys.Op. Multiple 
            detection matrices may be specified by providing a list of operators.
        Reduce : Flag to determine if we may reduce the size of the Liouvillian
            and only compute components required for propagation and detection.
            Default is True

        Returns
        -------
        None.

        """
        
        
        self.rho0=rho0
        self.rho=copy(rho0)
        if not(isinstance(detect,list)) and not(isinstance(detect,tuple)):detect=[detect]  #Make sure a list
        self.detect=detect
        self._L=None
        
        
        self._awaiting_detection=False  #Detection hanging because L not defined
        self._taxis=[]
        self._t=None
        
        
        self.Reduce=Reduce
        self._BDP=False #Flag to indicate that Block-diagonal propagation was used.
        # self._Setup()
        self.apodize=False
        self._block=None
        self._phase_accum=None
        self._phase_accum0=None
        self.apod_pars={'WDW':'em','LB':None,'SSB':2,'GB':15,'SI':None}
        
        if L is not None:self.L=L
    
    @property
    def _rtype(self):
        return Defaults['rtype']

    @property
    def _ctype(self):
        return Defaults['ctype']
    
    @property
    def isotropic(self):
        return self.L.isotropic
    
    @property
    def static(self):
        return self.L.static
        
    @property
    def shape(self):
        if self.L is None:return None
        return self.L.shape[:1]
    @property
    def L(self):
        return self._L
    
    @L.setter
    def L(self,L):
        if L is not self.L and len(self.t_axis):
            warnings.warn("Internal Liouvillian does not match propagator's Liouvillian, although system has already been propagated")
         
        self._L=L
        self._Setup()
        
    
    @property
    def n_det(self):
        return len(self.detect)
    
    @property
    def expsys(self):
        return self.L.expsys
    
    @property
    def pwdavg(self):
        return self.L.pwdavg
        
    @property
    def Op(self):
        return self.expsys.Op
    
    @property
    def taur(self):
        return self.L.taur
        
    @property
    def t(self):
        """
        Current time (sum of length of all applied propagators)

        Returns
        -------
        float
            time.

        """
        return self._t if self._t is not None else 0
    
    @t.setter
    def t(self,t):
        self._t=t
    
    @property
    def t_axis(self):
        """
        Time axis corresponding to when detection was performed.

        Returns
        -------
        array
            Array of all times at which the detection was performed.

        """
        if self._tstatus:
            return np.sort(self._taxis)
        else:
            return np.array(self._taxis)
    
    @property
    def _tstatus(self):
        """
        Returns an integer indicating what the time axis can be used for.
        
        0   :   Unusable (possibly constant time experiment)
        1   :   Ideal usage (uniformly spaced)
        2   :   Acceptable usage (unique values)
        3   :   Non-ideal usage (1-2 duplicate values- likely due to poor coding)

        Returns
        -------
        None.

        """
        
        if len(self._taxis)==1 or len(self._taxis)==0:return 1
        
        unique=np.unique(self._taxis)
        if unique.size+2<len(self._taxis):  #3 or more non-unique values
            return 0
        if unique.size<len(self._taxis): #1-2 duplicate values
            return 3
        diff=np.diff(unique)
        if diff.min()*(1+1e-10)>diff.max():   #Uniform spacing within error
            return 1
        return 2
    
    @property
    def phase_accum(self):
        return np.array(self._phase_accum).T
    
    @property
    def reduced(self):
        if self.L is None:return False
        return self.L.reduced
    
    @property
    def block(self):
        if self.L is None:return None
        return self.L.block
    
    def Blocks(self,*seq_U):
        """
        Returns a list of logical indices, where each list element consists of
        a block of the Liouvillian that needs to be calculated. Note that not
        all blocks need to be calculated, but this depends on the value of the
        current density matrix and the detection operator.

        Parameters
        ----------
        seq_U : Propagators or sequences to be used in the block-diagonalization
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if self.L is None:self.L=seq_U[0].L  #Initialize self if necessary
        
        
        ini_fields=copy(self.L.rf.fields) #Store initial field settings
        rf=self.L.rf
        for k in rf.fields:rf.add_field(k) #Set all fields first to zero
        
        x=np.zeros(self.L.shape,dtype=bool)
        for seq0 in seq_U:
            if hasattr(seq0,'rf') or not(seq0.calculated):
                v10=seq0.v1 if hasattr(seq0,'rf') else seq0.U['v1']
                    
                for k,v1 in enumerate(v10):
                    if np.any(v1):rf.add_field(k,v1=1) #Turn field on
                    x+=self.L.Lrf.astype(bool)  #Add it to the logical matrix
                    rf.add_field(k) #Turn field off
            else:
                x+=np.abs(seq0[0])>1e-5  #Tolerance ok?
                # We try to avoid any weird orientations that are missing cross terms so check a few orientations
                x+=np.abs(seq0[len(self.L)//2])>1e-5
                x+=np.abs(seq0[len(self.L)//3])>1e-5  #Tolerance ok?
                x+=np.abs(seq0[len(self.L)//4])>1e-5  #Tolerance ok?

        
        rf.fields=ini_fields
        
        x+=self.L[0].L(0).astype(bool)
        # We try to avoid any weird orientations that are missing cross terms so check a few orientations
        x+=self.L[len(self.L)//2].L(0).astype(bool)
        x+=self.L[len(self.L)//3].L(0).astype(bool)
        x+=self.L[1*len(self.L)//4].L(0).astype(bool)
        
        
        B=BlockDiagonal(x)
        blocks=[]
        rho=np.array(self._rho,dtype=bool).sum(0).astype(bool)
        detect=np.array(self._detect,dtype=bool).sum(0).astype(bool)
        for b in B:
            if np.any(rho[b]) and np.any(detect[b]):
                blocks.append(b)
        return blocks
    
    def getBlock(self,block):
        """
        Returns a Rho object that has been reduced for a particular block. Provide
        the logical index for the given block

        Parameters
        ----------
        block : np.array (bool type)
            Logical array specifying the block to propagate.

        Returns
        -------
        Rho

        """
        
        rho=copy(self)
        if not(self.L.reduced):
            rho._L=self.L.getBlock(block)
        rho._rho0=[rho0[block] for rho0 in self._rho0] if isinstance(self._rho0,list) else self._rho0[block]
        # rho._rho0=self._rho0[block]
        rho._detect=[d[block] for d in self._detect]
        rho._rho=[r[block] for r in self._rho]
        rho._Ipwd=[[[] for _ in range(len(self._detect))] for _ in range(len(self.L))]
        rho._taxis=[]
        self._phase_accum=list()
        rho.Reduce=False
        
        return rho
    
    def ReducedSetup(self,*seq_U):
        """
        Sets up reduced matrices for the density matrix and all provided sequences.
        Note that one should prepare all sequences to be used in the simulation
        and enter them here. All operations are 

        Parameters
        ----------
        *seq : TYPE
            DESCRIPTION.

        Returns
        -------
        tuple
            rho,*seq_U_red
            Reduced density matrix and reduced sequences or propagators

        """
        block=np.sum(self.Blocks(*seq_U),axis=0).astype(bool)
        rho=self.getBlock(block)
        seq_red=[s.getBlock(block) for s in seq_U]
        
        if Defaults['verbose']:
            if block.sum()==0:
                warnings.warn('Combination of sequence, initial density operator, and detection operator will not yield any signal (errors likely to follow)')
            else:
                if block.sum()==len(block):
                    rho.Reduced=False
                    rho.Reduce=False
                    return (rho,*seq_U)
                else:
                    print(f'State-space reduction: {block.__len__()}->{block.sum()}')
        
        for x in seq_red:x.L=rho.L
        
        self.L._children.append(rho.L)
        
        return (rho,*seq_red)
    
    def copy_reduced(self):
        """
        Returns a new rho object which has been reduced using the same elements
        as the current rho object

        Returns
        -------
        Rho
            Rho object with reduced dimensionality

        """
        return Rho(rho0=self.rho0,detect=self.detect,L=self.L._L).getBlock(self.block)
    
    # Is this function used anywhere?
    def _reduce(self,*seq):
        """
        Reduces rho (self) and all provided sequences for faster propagation.
        One should do the reduction using ALL sequences that will be applied
        to rho.

        Parameters
        ----------
        *seq : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        blocks=self.Blocks(*seq)
        block=np.sum(blocks,axis=0).astype(bool)
        self._rho0=[rho0[block] for rho0 in self._rho0] if isinstance(self._rho0,list) else self._rho0[block]
        # self._rho0=self._rho0[block]
        self._detect=[d[block] for d in self._detect]
        self._rho=[r[block] for r in self._rho]
        self.Reduce=False
        self.block=block
        if seq.reduced:
            self._L=seq.L
        
        return self
        
    
    def downmix(self,DM:bool=True,t0:float=None,baseline:bool=False):
        """
        Takes the stored signals and downmixes them if they were recorded in the
        lab frame (result replaces Ipwd). Only applied to signals detected with
        + or - (i.e., the complex signal is required!)
        
        Not heavily tested. 
        
        Parameters
        ----------
        DM : bool, optional
            Determines whether we are downmixing or removing downmixing. Set to
            False to remove downmixing. Default is True
        t0  : float, optional
            Effectively a phase-correction parameter. By default, t0 is None,
            which means we will subtract away self.t_axis[0] from the time
            axis. The parameter to be subtracted may be adjusted by setting t0
            Default is None.
            
        baseline : bool, optional
            Subtracts away a baseline that caused by a strongly tilted 
            eigenbasis. Default is False

        Returns
        -------
        None.

        """
        
        # from scipy.signal import butter,filtfilt
        
        if not(DM):
            self._Ipwd_DM=None
            return self
        
        if self._Ipwd_DM is not None:
            print('Already downmixed')
            return self
        
        if t0 is None:t0=self.t_axis[0]
        
        #A new, more general attempt
        co=[np.tile(o.coherence_order,len(self.L.H)).T[self.block] for o in self.expsys.Op]
        Idm=np.zeros((self.Ipwd.shape[1],self.Ipwd.shape[0],self.Ipwd.shape[2]),dtype=self.Ipwd.dtype)
        for k,detect in enumerate(self._detect):
            detect=detect.astype(bool)
            q=np.ones(len(self.t_axis),dtype=ctype)
            for i,(ph_acc,co0) in enumerate(zip(self.phase_accum,co)):
                if self.expsys.LF[i]:  #Add phase from LF rotation if necessary
                    ph_acc+=self.expsys.v0[i]*2*np.pi*(self.t_axis-t0)
                if np.unique(co0[detect]).__len__()==1:
                    q*=np.exp(1j*ph_acc*co0[detect][0])
                elif np.unique(np.abs(co0[detect])).__len__()==1:
                    q*=np.exp(1j*ph_acc*np.abs(co0[detect][0]))
                else:
                    warnings.warn(f'Inconsistent coherence orders in detection matrix {k}, downmixing aborted for this matrix')
                    break
            
            Ipwd=self.Ipwd[:,k]
            if baseline:
                Ipwd=(Ipwd.T-Ipwd.mean(-1)).T
            Idm[k]=Ipwd*q
            # for m in range(self.pwdavg.N):
            #     self._Ipwd[m][k]=Idm[m].tolist()
        self._Ipwd_DM=np.swapaxes(Idm,0,1)
        
                
        return self
            
            
            
        
        # # Old approach
        # for k,detect in enumerate(self.detect):
        #     OpName,_=self.OpScaling(detect)
        #     v0=None
        #     q=None
        #     if OpName[0]=='S':
        #         i=int(OpName[1])   #At the moment, I'm assuming this program won't work with 11 spins...
        #         if self.expsys.LF[i] and OpName[2:] in ['p','m']: #Downmix required
        #             v0=self.expsys.v0[i]*(-1 if OpName[2:]=='p' else 1)
        #     else:
        #         Nuc,a=self.parseOp(OpName)
        #         i=self.expsys.Nucs.tolist().index(Nuc)
        #         if self.expsys.LF[i] and a in ['p','m']:
        #             v0=self.expsys.v0[i]*(-1 if a=='p' else 1)
                    
        #     if v0 is not None:
        #         Ipwd=self.Ipwd[:,k]
        #         Idm=Ipwd*np.exp(1j*v0*2*np.pi*(self.t_axis-t0))
        #         for m in range(self.pwdavg.N):
        #             self._Ipwd[m][k]=Idm[m]
                
    
    @property
    def Ipwd(self):
        """
        Npwd x Nd x Nt matrix of detected amplitudes, where Npwd is the number
        of angles in the powder average, Nd is the number of detection matrices
        that have been defined, and Nt is the number of time points stored.

        Returns
        -------
        None.

        """
        if len(self.t_axis):
            if self._Ipwd_DM is not None:
                return self._Ipwd_DM
            if self._tstatus:
                i=np.argsort(self._taxis)
                return np.array(self._Ipwd).T[i].T
            else:
                return np.array(self._Ipwd)
    
    @property
    def I(self):
        """
        Nd x Nt matrix of detected amplitude (powder average applied), where Nd
        is the number of detection matrices that have been defined and Nt is
        the number of time points stored

        Returns
        -------
        None.

        """
        return (self.Ipwd.T*self.pwdavg.weight).sum(-1).T
    
    @property
    def v_axis(self):
        """
        Frequency axis for the Fourier transform of the signal

        Returns
        -------
        None.

        """
        ZF=len(self.t_axis)*2 if self.apod_pars['SI'] is None else int(self.apod_pars['SI'])
        if self._tstatus!=1:
            return np.arange(ZF)
            
            
        
        v=1/(self.t_axis[1]-self.t_axis[0])/2*np.linspace(-1,1,ZF)
        v-=np.diff(v[:2])/2
        return v
        
    
    @property
    def FT(self):
        """
        Fourier transform of the time-dependent signal

        Returns
        -------
        np.array
            FT, including division of the first time point by zero.

        """
        if self._tstatus!=1:
            warnings.warn('Time points are not equally spaced. FT will be incorrect')

        I=np.concatenate((self.I[:,:1]/2,self.I[:,1:]),axis=1)
        if self.apodize:
            apod=ApodizationFun(self.t_axis,**self.apod_pars)
            I*=apod

        ZF=I.shape[1]*2 if self.apod_pars['SI'] is None else int(self.apod_pars['SI'])
            
        return np.fft.fftshift(np.fft.fft(I,n=ZF,axis=1),axes=[1])
    
    @property
    def FTpwd(self):
        """
        Fourier transform of the time-dependent signal

        Returns
        -------
        np.array
            FT, including division of the first time point by zero.

        """
        
        if self._FTpwd is None:
            if self._tstatus!=1:
                warnings.warn('Time points are not equally spaced. FT will be incorrect')
    
            I=np.concatenate((self.Ipwd[:,:,:1]/2,self.Ipwd[:,:,1:]),axis=-1)
            if self.apodize:
                ap=self.apod_pars
                wdw=ap['WDW'].lower()
                t=self.t_axis
                LB=ap['LB'] if ap['LB'] is not None else 5/t[-1]/np.pi
                
                if wdw=='em':
                    apod=np.exp(-t*LB*np.pi)
                elif wdw=='gm':
                    apod=np.exp(-np.pi*LB*t+(np.pi*LB*t**2)/(2*ap['GB']*t[-1]))
                elif wdw=='sine':
                    if ap['SSB']>=2:
                        apod=np.sin(np.pi*(1-1/ap['SSB'])*t/t[-1]+np.pi/ap['SSB'])
                    else:
                        apod=np.sin(np.pi*t/t[-1])
                elif wdw=='qsine':
                    if ap['SSB']>=2:
                        apod=np.sin(np.pi*(1-1/ap['SSB'])*t/t[-1]+np.pi/ap['SSB'])**2
                    else:
                        apod=np.sin(np.pi*t/t[-1])**2
                elif wdw=='sinc':
                    apod=np.sin(2*np.pi*ap['SSB']*(t/t[-1]-ap['GB']))
                elif wdw=='qsinc':
                    apod=np.sin(2*np.pi*ap['SSB']*(t/t[-1]-ap['GB']))**2
                else:
                    warnings.warn(f'Unrecognized apodization function: "{wdw}"')
                    apod=np.ones(t.shape)
                I*=apod
    
                
            self._FTpwd=np.fft.fftshift(np.fft.fft(I,n=I.shape[-1]*2,axis=-1),axes=[-1])
        
        return self._FTpwd
    
    def _Setup(self):
        """
        At initialization, we do not require Rho to know the spin-system yet. 
        However, for most functions, this is in fact required. Therefore, at
        the first operation with a propagator, we will run _Setup to finalize
        the Rho setup.

        Returns
        -------
        None.

        """
        
        self._Ipwd=[[list() for _ in range(self.n_det)] for _ in range(self.pwdavg.N)]
        self._Ipwd_DM=None #Downmixed signal
        self._FTpwd=None
        self._taxis=list()
        self._phase_accum=list()
        
        
        if isinstance(self.rho0,str) and self.rho0=='Thermal':
            # rhoeq=self.L.rho_eq(sub1=True)
            step=0 if self.taur is None else self.t//self.L.dt
            rhoeq=[]
            for L in self.L:
                rhoeq.append(L.rho_eq(step=step,sub1=not(self.L.Peq)))

                # if self.L.Peq:
                #     eye=np.tile(np.ravel(self.expsys.Op[0].eye),len(self.L.H))[self.block]
                #     rhoeq[-1]+=eye/self.expsys.Op.Mult.prod()
            self._rho0=rhoeq
        else:
            self._rho0=self.Op2vec(self.strOp2vec(self.rho0))
        self._detect=[self.Op2vec(self.strOp2vec(det,detect=True),detect=True) for det in self.detect]
        
        for k,det in enumerate(self._detect):
            if np.any(np.isnan(det)):
                warnings.warn(f'Detector {k} is not valid')
                
        self._phase_accum0=np.zeros(self.expsys.nspins)
        self.reset()
        if self.L.reduced:
            warnings.warn('Reduced Liouvillian applied to uninitialized propagator. Make sure reduction was perfomed with same Rho')
            rho=self.getBlock(self.L.block)
            self.__dict__=rho.__dict__
            
        if self.L.LrelaxOS is not None and self.L.LrelaxOS.sc!=1:
            one=np.tile(np.eye(self.L.H[0].shape[0]).flatten(),len(self.L.H))
            # one/=one.sum()
            if isinstance(self._rho0,list):
                for k in range(self.pwdavg.N):
                    self._rho0[k]-=(self._rho0[k]@one)*one/one.sum()
                    self._rho0[k]+=one/self.L.LrelaxOS.sc/one.sum()
            else:
                self._rho0-=(self._rho0@one)*one/one.sum()
                self._rho0+=(one/self.L.LrelaxOS.sc)/one.sum()
                
                # if self.rho0=='Thermal':
                #     Dt=0.1 if self.L.static else self.L.taur
                #     self.L.U(Dt)**np.inf*self
                #     self._rho0=copy(self._rho)
                
            
            
            # if self.rho0=='Thermal':
            #     Dt=0.1 if self.L.static else self.L.taur
            #     self.L.U(Dt)**np.inf*self
            #     self._rho0=copy(self._rho)
            #     self.reset()
            #     print('checkpoint')
            # else:
            #     warnings.warn('DynamicThermal should be initialized with Thermal')
        
        
    def reset(self):
        """
        Resets the density matrices back to rho0

        Returns
        -------
        None.

        """
        if self.L is not None:
            self._rho=copy(self._rho0) if isinstance(self._rho0,list) else [self._rho0 for _ in range(self.pwdavg.N)]
            self._phase_accum0=np.zeros(self.expsys.nspins)
        self._t=None
        
        return self
    
    def clear(self,data_only:bool=False):
        """
        Clears variables in order to start over propagation. 
        
        Note that if you want to set the system back to the initial rho0 value,
        but want to retain the amplitudes and times recorded, run rho.reset()
        instead of rho.clear() 

        Parameters
        ----------
        data_only : bool, optional
            Only clear the data, but retain the Liouvillian. Useful if the
            density matrix has been reduced. The default is False.

        Returns
        -------
        self

        """
        
        L=self.L
        
        self._t=None
        
        self._Ipwd=[[]]
        self._FTpwd=None
        self._taxis=list()
        self._rho=list() #Storage for numerical rho
        self._Ipwd_DM=None
        
        self._L=None
        self._BDP=False

        if data_only:
            warnings.filterwarnings("ignore", 
                message="Reduced Liouvillian applied to uninitialized propagator. Make sure reduction was perfomed with same Rho")
            self.L=L
            warnings.filterwarnings("default", 
                message="Reduced Liouvillian applied to uninitialized propagator. Make sure reduction was perfomed with same Rho")
        
        return self
        # if self._L is not None:
        #     self._Setup()
        
    def prop(self,U):
        """
        Propagates the density matrix by the provided propagator or sequence

        Parameters
        ----------
        U : Propagator
            Propagator object.

        Returns
        -------
        None.

        """
        
        if hasattr(U,'add_channel'):U=U.U() #Sequence provided
        
        if self._BDP:
            warnings.warn('Block-diagonal propagation was previously used. Propagator is set to time point BEFORE block-diagonal propagation.')
                   
        if self.L is None:
            self.L=U.L

        if self._awaiting_detection:  #Use this if detect was called before L was assigned
            self._awaiting_detection=False
            if self._t is None:self._t=U.t0
            self()
            
         
        if self._t is None:
            self._t=U.t0
            
        if not(self.static) and np.abs((self.t-U.t0)%self.taur)>tol and np.abs((U.t0-self.t)%self.taur)>tol:
            if not(U.t0==U.tf):
                warnings.warn('The initial time of the propagator is not equal to the current time of the density matrix')
        
        assert U.shape[0]==self.shape[0],"Detector and propagator shapes do not match"
        assert U.block.sum(0)==self.block.sum(0),"Different matrix reduction applied to propagators (cannot be multiplied)"
        if not(np.all(U.block==self.block)):
            warnings.warn('\nMatrix blocks do not match. This is almost always wrong')
        
        
            
        if U.calculated:
            self._rho=[U0@rho for U0,rho in zip(U,self._rho)]
            self._t+=U.Dt
            
        else:
            # This approach is incomplete and should not be used!
            
            dct=U.U
            t=dct['t']
            L=U.L
            ini_fields=copy(L.fields)
            
            for m,(ta,tb) in enumerate(zip(t[:-1],t[1:])):
                for k,(v1,phase,voff) in enumerate(zip(dct['v1'],dct['phase'],dct['voff'])):
                    L.fields[k]=(v1[m],phase[m],voff[m])
            
                    #TODO
                    # AT THIS POINT, WE NEED TO APPLY THE PROPAGATOR TO RHO
                    # OVER DIFFERENT ROTOR POSITIONS. FOR EXAMPLE, CHECK THE
                    # L.U MACHINERY. ALSO SHOULD BE SET UP FOR PARALLEL PROCESSING
                    
            
            L.fields.update(ini_fields)  #Return fields to their initial state
            self._t+=U.Dt
        
        self._phase_accum0+=U.phase_accum
        self._phase_accum0%=2*np.pi
        
        return self
                    
                
        
    def __rmul__(self,U):
        """
        Runs rho.prop(U) and returns self.
        
        This is the usual mechanism for accessing rho.prop

        Parameters
        ----------
        U : Propagator
            Propagator object.

        Returns
        -------
        self

        """
        if hasattr(U,'add_channel'):U=U.U() #Sequence provided
        U.calcU()
        return self.prop(U)
    
    def __mul__(self,U):
        """
        Runs rho.prop(U) and returns self
        
        This isn't really fully implemented. Would run if we execute rho*U

        Parameters
        ----------
        U : Propagator
            Propagator object.

        Returns
        -------
        self

        """
        if hasattr(U,'add_channel'):U=U.U() #Sequence provided
        U.calcU()   #This line should be removed if we were to perform faster
                    #approach for multiplying rho*U
        return self.prop(U)
        
    
    def Detect(self):
        """
        Evaluates the density matrices at the current time point and stores
        the result

        Returns
        -------
        None.

        """
        self._Ipwd_DM=None
        if self.L is None:
            if self._awaiting_detection:
                warnings.warn('Detection called twice before applying propagator. Second call ignored')
            self._awaiting_detection=True
            return self
        
        self._taxis.append(self.t)
        self._phase_accum.append(self._phase_accum0)
        for k,rho in enumerate(self._rho):
            for m,det in enumerate(self._detect):
                self._Ipwd[k][m].append((rho*det).sum())
        return self
    
        
    def __call__(self):
        return self.Detect()
    
    def __getitem__(self,i:int):
        """
        Return the density operator for the ith element of the powder average

        Parameters
        ----------
        i : int
            Index for the density operator.

        Returns
        -------
        None.

        """
        i%=len(self)
        return self._rho[i]
    
    def __len__(self):
        if self.L is not None:
            return self.L.__len__()
    
    def DetProp(self,U=None,seq=None,n:int=5000,n_per_seq:int=1):
        """
        Executes a series of propagation/detection steps. Detection occurs first,
        followed by propagation, with the sequence repeated for n steps. 
        If n>100, then we will use eigenvalue decomposition for the propagation

        Parameters
        ----------
        U : Propagator
            Propagator applied. Should be an integer number of rotor periods
        seq : Sequence 
             Alternative to providing a propagator, which does not need to
             be an integer multiple of rotor periods
        n : int, optional
            Number of time steps. The default is 1.
        n_per_seq : int, optional 
            Allows one to break a sequence into steps, e.g. to obtain a larger
            spectral width.

        Returns
        -------
        self

        """
        assert not(U is None and seq is None),"Either U or seq must be defined"
        
        
        if self._BDP:
            warnings.warn('Block-diagonal propagation was previously used. Propagator is set to time point BEFORE block-diagonal propagation.')
        
        if seq is None and not(hasattr(U,'calcU')):
            seq=U
            U=None
            
        
        if U is not None and seq is not None:
            warnings.warn('Both U and seq are defined. seq will not be used')
            seq=None
        
        if self.L is None:
            if U is not None:
                self.L=U.L
            else:
                self.L=seq.L
        
        # Block-diagonal propagation
        if seq is not None and self.Reduce:
            rb,sb=self.ReducedSetup(seq)
            
            if not(np.all(rb.block)):
                rb.DetProp(seq=sb,n=n,n_per_seq=n_per_seq)
                Ipwd=rb.Ipwd
                for k in range(Ipwd.shape[0]):
                    for j in range(Ipwd.shape[1]):
                        self._Ipwd[k][j].extend(Ipwd[k,j])
                self._taxis.extend(rb._taxis)
                self._phase_accum.extend(rb._phase_accum)
                self._BDP=True
                self._t=rb._t
                return self
                
            
        

        
        
        if U is not None:
            if self._t is None:self._t=U.t0
            if not(self.static) and np.abs((self.t-U.t0)%self.taur)>tol and np.abs((U.t0-self.t)%self.taur)>tol:
                warnings.warn('The initial time of the propagator is not equal to the current time of the density matrix')
            if not(self.static) and np.abs(U.Dt%self.taur)>tol and np.abs((U.Dt%self.taur)-self.taur)>tol:
                warnings.warn('The propagator length is not an integer multiple of the rotor period')
         
        elif self._t is None:
            self._t=0
        
        if U is not None:
            if n>=100:
                self()  #This is the initial detection
                U.eig()
                for k,((d,v,vi),rho) in enumerate(zip(U._eig,self)):
                    rho0=vi@rho
                    dp=np.cumprod(np.repeat([d],n,axis=0),axis=0)
                    rho_d=dp*rho0
                    self._rho[k]=v@rho_d[-1]
                    for m,det in enumerate(self._detect):
                        det_d=det@v
                        self._Ipwd[k][m].extend((det_d*rho_d[:-1]).sum(-1))
                
                self._taxis.extend([self.t+k*U.Dt for k in range(1,n)])
                self._t+=n*U.Dt
                self._phase_accum.extend([(self._phase_accum0+k*U.phase_accum)%(2*np.pi) for k in range(1,n)])
                self._phase_accum0=(self._phase_accum0+n*U.phase_accum)%(2*np.pi)
                    
            else:
                for _ in range(n):
                    U*self()
        else:
            if self.static:
                nsteps=n_per_seq
                
            else:
            
                if (seq.Dt%seq.taur<tol or -seq.Dt%seq.taur<tol) and n_per_seq==1:
                    U=seq.U(t0=self.t,Dt=seq.Dt)  #Just generate the propagator and call with U
                    self.DetProp(U=U,n=n)
                    return self
                
                if seq.Dt<seq.taur:
                    for k in range(1,n):
                        nsteps=np.round(k*seq.taur/seq.Dt,0).astype(int)
                        if nsteps*seq.Dt%seq.taur < tol:break
                        if seq.taur-(nsteps*seq.Dt%seq.taur) < tol:break
                    else:
                        nsteps=n
                                        
                else:
                    for k in range(1,n):
                        nsteps=np.round(k*seq.Dt/seq.taur,0).astype(int)
                        if nsteps*seq.Dt%seq.taur < tol:break
                        if seq.taur-(nsteps*seq.Dt%seq.taur) < tol:break
                    else:
                        nsteps=n
                    k,nsteps=nsteps,k
                nsteps*=n_per_seq if nsteps<n else n
                
                if Defaults['verbose']:
                    print(f'Prop: {nsteps} step{"" if nsteps==1 else "s"} per every {k} rotor period{"" if k==1 else "s"}')
            
            
            seq.reset_prop_time(self.t)
            
            
            Dt=seq.Dt/n_per_seq
            # U=[seq.U(Dt=Dt,t0_seq=k*Dt) for k in range(nsteps)]
            # Eliminating the second argument should let us set t0_seq to change the starting time on the sequence
            # This would be special usage, but maybe useful occasionally
            # seq should keep track itself of the current time
            U=[seq.U(Dt=Dt) for k in range(nsteps)]
            
            if n//nsteps>100:
                U0=[]
                Ipwd=np.zeros([len(self),len(self._detect),n],dtype=ctype)
                phase_accum=np.ones([n,self.expsys.nspins],dtype=rtype)*self._phase_accum0
                
                rho00=copy(self._rho)
                for q in range(nsteps):  #Loop over the starting time
                    n0=n//nsteps+(q<n%nsteps)
                    U0=U[q]
                    for m in range(q+1,q+nsteps):U0=U[m%nsteps]*U0 #Calculate propagator for 1 rotor period starting U[q]
                    U0.eig()
                    phase_accum[q::nsteps]+=U0.phase_accum*np.repeat([np.arange(n0)],self.expsys.nspins,axis=0).T
                    for k,((d,v,vi),rho0) in enumerate(zip(U0._eig,rho00)):  #Sweep over the powder average
                        rho0=vi@rho0
                        dp=np.concatenate([np.ones([1,d.size],dtype=ctype),
                                           np.cumprod(np.repeat([d],n0-1,axis=0),axis=0)],axis=0)
                        rho_d=dp*rho0
                        for m,det in enumerate(self._detect):
                            det_d=det@v
                            Ipwd[k][m][q::nsteps]=(det_d*rho_d).sum(-1)
                        if q==nsteps-1:
                            self._rho[k]=v@rho_d[-1]
                            
                        rho00[k]=U[q][k]@rho00[k] #Step forward by 1/nsteps rotor period for the next step
                        
                for k in range(len(self)):
                    for m in range(len(self._detect)):
                        self._Ipwd[k][m].extend(Ipwd[k][m].tolist())
                
                self._taxis.extend([self.t+k*Dt for k in range(0,n)])
                
                self._phase_accum.extend([pa%(2*np.pi) for pa in phase_accum])
                self._phase_accum0=self._phase_accum[-1]
                self._t+=n*Dt
                        
            else:
                for k in range(n):
                    U[k%nsteps]*self()
            # t0,rho0=self.t,copy(self._rho)  #We need to keep the starting state in case this has already been propagated
            
            # Ua=seq.L.Ueye(t0=t0)
            # for k in range(nsteps):
            #     self.reset(t0=t0)  #This line and next set rho back to its starting state
            #     self._rho=copy(rho0)
                
            #     Ua*self  #Accumlate (k-1) steps to the density matrix
            #     Ua=U[k]*Ua  #Accumulate the kth step into the propagator
            #     Ur=seq.L.Ueye(t0=self.t) #Propagator for one rotor period                
            #     for m in range(nsteps):
            #         Ur=U[(k+m)%nsteps]*Ur
            #     print(Ua.Dt,Ur.t0,Ur.Dt,self.t)
                
            #     self.DetProp(U=Ur,n=n//nsteps+(k<n%nsteps))
            
            
        return self
    
    def parseOp(self,OpName):
        """
        Determines nucleus and operator type from the operator string

        Parameters
        ----------
        OpName : TYPE
            DESCRIPTION.

        Returns
        -------
        tuple 
            (Nuc,optype)

        """
            
        
        if OpName.lower()=='zero':
            Nuc='None'
            a=''
        elif OpName[-1] in ['x','y','z','p','m']:
            a=OpName[-1]
            Nuc=OpName[:-1]
        elif len(OpName)>3 and OpName[-3:]=='eye':
            a='eye'
            Nuc=OpName[:-3]
        elif 'alpha' in OpName:
            a='alpha'
            Nuc=OpName[:-5]
        elif 'beta' in OpName:
            a='beta'
            Nuc=OpName[:-4]
        else:
            return None
        if Nuc=='e':Nuc='e-'
        return Nuc,a
    
    def OpScaling(self,OpName):
        """
        Determines if the operator (given as string) contains a scaling factor.
        Can be indicated by presence of * operator in string, or simply a
        minus sign may be included.

        Parameters
        ----------
        OpName : TYPE
            DESCRIPTION.

        Returns
        -------
        OpName : TYPE
            DESCRIPTION.
        scale : TYPE
            DESCRIPTION.

        """
        
        scale=1.
        if '*' in OpName:
            p,q=OpName.split('*')
            if len(re.findall('[A-Z]',p)):
                scale=complex(q) if 'j' in q else float(q)
                OpName=p
            else:
                scale=complex(p) if 'j' in p else float(p)
                OpName=q
        elif OpName[0]=='-':
            scale=-1
            OpName=OpName[1:]
        return OpName,scale
    
    def strOp2vec(self,OpName:str,detect:bool=False):
        """
        Converts the string for an operator into a matrix

        Strings for specifying operators:
                
        S0x, S1y, S2alpha:
            Specifies a spin by index (S0, S1, etc.), followed by the operator
            type (x,y,z,p,m,alpha,beta)
            
        13Cx, 1Hy, 15Np:
            Specifies the channel followed by the operator type (sum of all nuclei of that type)

        Parameters
        ----------
        OpName : str
            Name of the desired operator.
        detect : bool
            Indicates if this is the detection operator

        Returns
        -------
        Operator matrix

        """
        
        if not(isinstance(OpName,str)):return OpName #Just return if already a matrix
        
        if '+' in OpName:
            OpNames=OpName.split('+')
            return np.sum([self.strOp2vec(op) for op in OpNames],axis=0)
        
        OpName,scale=self.OpScaling(OpName)
        
        if OpName[0]=='S':
            i=int(OpName[1])   #At the moment, I'm assuming this program won't work with 11 spins...
            Op=getattr(self.Op[i],OpName[2:])*scale
            
            if self.L.Peq and not(detect):
                Peq=self.expsys.Peq[i]
                Op*=Peq/self.expsys.Op.Mult.prod()*2 #Start out at thermal polarization
                Op+=self.expsys.Op[0].eye/self.expsys.Op.Mult.prod()
            return Op
        
        # if OpName=='Thermal':
        #     Op=np.zeros(self.Op.Mult.prod()*np.ones(2,dtype=int),dtype=self._ctype)
        #     for op,peq,mult in zip(self.expsys.Op,self.expsys.Peq,self.expsys.Op.Mult):
        #         Op+=op.z*peq
        #     if self.L.Peq:
        #         Op+=op.eye/self.expsys.Op.Mult.prod()
        #     return Op
        
        
        Op=np.zeros(self.Op.Mult.prod()*np.ones(2,dtype=int),dtype=self._ctype)
        
        Nuc,a=self.parseOp(OpName)
        
        i=self.expsys.Nucs==Nuc
        if OpName.lower()=='zero':
            i0=0
        elif not(np.any(i)):
            warnings.warn('Nucleus is not in the spin system or was not recognized')
        for i0 in np.argwhere(i)[:,0]:
            Op+=getattr(self.Op[i0],a)*scale
        
        if self.L.Peq and not(detect):
            Peq=self.expsys.Peq[i0]
            Op*=Peq/self.expsys.Op.Mult.prod()*2  #Start out at thermal polarization
            Op+=self.expsys.Op[0].eye/self.expsys.Op.Mult.prod()
            # for op0,mult in zip(self.expsys.Op,self.expsys.Op.Mult):
            #     Op+=op0.eye/mult  #Add in the identity for relaxation to thermal equilibrium

        return Op
    
    def Op2vec(self,Op,detect:bool=False):
        """
        Converts a matrix operator for one Hamiltonian into a vector for the
        full Liouville space. Required for initial density matrix and
        detection

        Parameters
        ----------
        Op : np.array
            Square matrix for rho or detection
        detect : bool, optional
            Set to true for the detection vectors, where we need to take the
            conjugate of the matrix. The default is False.

        Returns
        -------
        Operator vector
        
        """
        nHam=len(self.L.H)
        
        # I'm not 100% sure that I shouldn't be taking the transpose of both the 
        # initial and detection operators. Or maybe I should not be taking the 
        # transpose of either operator.
        
        
        if detect:
            Op=Op.T.conj()
            # Op/=np.abs(np.trace(Op.T.conj()@Op))*self.expsys.Op.Mult.prod()/2
            Op/=np.abs(np.trace(Op.T.conj()@Op))
            if (self.L.Peq or self.rho0=='Thermal'):Op*=self.expsys.Op.Mult.prod()/2
            return np.tile(Op.reshape(Op.size),nHam)
        else:
            # Op/=np.abs(np.trace(Op.T.conj()@Op))
            Op=Op.reshape(Op.size)
            pop=self.L.ex_pop
            # d,v=np.linalg.eig(self.L.kex)
            # pop=v[:,np.argmax(d)]    #We need to make sure we start at equilibrium
            # pop/=pop.sum()
            out=np.zeros([Op.size*nHam],dtype=self._ctype)
            for k,pop0 in enumerate(pop):
                out[k*Op.size:(k+1)*Op.size]=Op*pop0
            return out
        
        
    def plot(self,det_num:int=None,pwd_index:int=None,ax=None,FT:bool=False,mode:str='Real',apodize=False,axis='kHz/ms',**kwargs):
        """
        Plots the amplitudes as a function of time or frequency

        Parameters
        ----------
        det_num : int, optional
            Which detection operator to plot. The default is None (all detectors).
        ax : plt.axis, optional
            Specify the axis to plot into. The default is None.
        FT : bool, optional
            Plot the Fourier transform if true. The default is False.
        mode : str, optional
            Determines what to plot. Options are 'Real', 'Imag', 'Abs', and 'ReIm'
            The default is 'Real'
        apodize : bool, optional
            Apodize the signal with decaying exponential, with time constant 1/5
            of the time axis (FT signal only)
        axis : str, optional
            Specify the type of axis. Currently, 'Hz', 'kHz', 'MHz', and 'ppm'
            are implemented. 'ppm' is only valid if the detector 

        Returns
        -------
        None.

        """
        if ax is None:ax=plt.figure().add_subplot(111)
        
        for x in self.L.relax_info:
            if x[0]=='DynamicThermal':
                if np.abs(self.I).max()>np.abs(self.expsys.Peq).max()*1.01:  #1% tolerance?
                    warnings.warn('Diverging system due to unfavorable scaling')

        def det2label(detect):
            if isinstance(detect,str):
                if detect[0]=='S':
                    x='S'+r'_'+detect[1]
                    a=detect[2:]
                    Nuc=''
                else:
                    Nuc,a=self.parseOp(self.OpScaling(detect)[0])
                    mass=re.findall(r'\d+',Nuc)
                    if Nuc!='e-':
                        Nuc=re.findall(r'[A-Z]',Nuc.upper())[0]
                    else:
                        Nuc='e'
                    x=(r'^{'+mass[0]+'}' if len(mass) else r'')+(Nuc if Nuc=='e' else Nuc.capitalize())
                
                if a in ['x','y','z']:
                    a=r'_'+a
                elif a in ['alpha','beta']:
                    a=r'^\alpha' if a=='alpha' else r'^\beta'
                elif a in ['p','m']:
                    a=r'^+' if a=='p' else r'^-'
                else:
                    a=a+r''
                if '_' in x and '_' in a:
                    x=x.replace('_','_{')
                    a=a[1:]+'}'
                

                return r'<'+x+'$'+a+'$>' if Nuc=='e' else r'<$'+x+a+'$>'

            else:
                return r'<Op>'
                
        
        if det_num is None or hasattr(det_num,'__len__'):
            det_num=np.arange(len(self._detect)) if det_num is None else np.array(det_num,dtype=int)
            h=[]
            for det_num in det_num:
                kids=self.plot(det_num=det_num,pwd_index=pwd_index,ax=ax,FT=FT,
                               mode=mode,apodize=apodize,axis=axis,**kwargs).get_children()
                i=np.array([isinstance(k,plt.Line2D) for k in kids],dtype=bool)
                h.append(np.array(kids)[i][-1])
            if det_num:
                ax.set_ylabel('<Op>')
                # ax.legend(h,[det2label(detect) for detect in self.detect])
                ax.legend()
            return ax
        
        ap=self.apodize
        self.apodize=apodize
        
        label=det2label(self.detect[det_num])
        
        if FT:
            if axis.lower()=='ppm' and self.parseOp(self.detect[det_num]) is not None:
                Nuc,_=self.parseOp(self.detect[det_num]) 
                v0=NucInfo(Nuc)*self.expsys.B0
                v_axis=self.v_axis/v0*1e6
                mass,name=''.join(re.findall(r'\d',Nuc)),''.join(re.findall('[A-Z]',Nuc.upper()))
                label=r"$\delta$($^{"+mass+r"}$"+name+") / ppm"
            elif axis.lower()=='mhz':
                v_axis=self.v_axis/1e6
                label=r'$\nu$ / MHz'
            elif axis.lower()=='khz':
                v_axis=self.v_axis/1e3
                label=r'$\nu$ / kHz'
            elif axis.lower()=='points':
                v_axis=np.arange(len(self.v_axis))
                label='points'
            else:
                v_axis=self.v_axis
                label=r'$\nu$ / Hz'
            
            
            if pwd_index is None:
                Re=self.FT[det_num].real
                Im=self.FT[det_num].imag
            else:
                Re=self.FTpwd[pwd_index][det_num].real
                Im=self.FTpwd[pwd_index][det_num].imag
            
            if mode.lower()=='reim':
                ax.plot(v_axis,Re,label=f'Re[{label}]',**kwargs)
                ax.plot(v_axis,Im,label=f'Im[{label}]',**kwargs)
                ax.legend(('Re','Im'))
            elif mode[0].lower()=='r':
                ax.plot(v_axis,Re,label=label,**kwargs)
            elif mode[0].lower()=='a':
                ax.plot(v_axis,np.abs(Re+1j*Im),label=f'Abs[{label}]',**kwargs)
            elif mode[0].lower()=='i':
                ax.plot(v_axis,Im,label=f'Im[{label}]',**kwargs)
            else:
                assert 0,'Unrecognized plotting mode'
                
            ax.set_xlabel(label)
            ax.set_ylabel('I / a.u.')
            ax.xaxis.set_inverted(True)
        else:
            if self._tstatus:
                x=self.t_axis
                if axis.lower()=='s':
                    xlabel='t / s'
                elif axis.lower() in ['microseconds','us']:
                    x*=1e6
                    xlabel=r't / $\mu$s'
                elif axis.lower()=='ns':
                    x*=1e9
                    xlabel='t / ns'
                elif axis.lower()=='points':
                    x=np.arange(len(self.t_axis))
                    xlabel='points'
                else:
                    x*=1e3
                    xlabel='t / ms'
            else:
                x=np.arange(len(self.t_axis))
                xlabel='Acquisition Number'


            if pwd_index is None:
                Re,Im=self.I[det_num].real,self.I[det_num].imag
            else:
                Re,Im=self.Ipwd[pwd_index][det_num].real,self.I[det_num].imag

                
            if mode.lower()=='reim':
                ax.plot(x,Re,label=f'Re[{label}]',**kwargs)
                ax.plot(x,Im,label=f'Im[{label}]',**kwargs)
                ax.legend(('Re','Im'))
            elif mode[0].lower()=='r':
                    ax.plot(x,Re,label=label,**kwargs)
            elif mode[0].lower()=='a':
                ax.plot(x,np.abs(Re+1j*Im),label=f'Abs[{label}]',**kwargs)
            elif mode[0].lower()=='i':
                ax.plot(x,Im,label=f'Im[{label}]',**kwargs)
            else:
                assert 0,'Unrecognized plotting mode'
                
            ax.set_ylabel(det2label(self.detect[det_num]))
            ax.set_xlabel(xlabel)
            
            # if self._tstatus==0:
            #     if mode.lower()=='reim':
            #         ax.plot(np.arange(len(self.t_axis)),self.I[det_num].real,label=f'Re[{label}]',**kwargs)
            #         ax.plot(np.arange(len(self.t_axis)),self.I[det_num].imag,label=f'Im[{label}]',**kwargs)
            #         ax.legend(('Re','Im'))
            #     elif mode[0].lower()=='r':
            #         ax.plot(np.arange(len(self.t_axis)),self.I[det_num].real,label=label,**kwargs)
            #     elif mode[0].lower()=='a':
            #         ax.plot(np.arange(len(self.t_axis)),np.abs(self.I[det_num]),label=f'Abs[{label}]',**kwargs)
            #     elif mode[0].lower()=='i':
            #         ax.plot(np.arange(len(self.t_axis)),self.I[det_num].imag,label=f'Im[{label}]',**kwargs)
            #     else:
            #         assert 0,'Unrecognized plotting mode'
                
            #     ax.set_ylabel('<'+self.detect[det_num]+'>')
            #     ax.set_xlabel('Acquisition Number')
                
            # else:
            #     if axis.lower() in ['microseconds','us']:
            #         t_axis=self.t_axis*1e6
            #         label=r'$t$ / $\mu$s'
            #     elif axis.lower()=='s':
            #         t_axis=self.t_axis
            #         label=r'$t$ / s'
            #     elif axis.lower()=='ns':
            #         t_axis=self.t_axis*1e9
            #         label=r'$t$ / ns'
            #     else:
            #         t_axis=self.t_axis*1e3
            #         label=r'$t$ / ms'
                    
            #     if mode.lower()=='reim':
            #         ax.plot(t_axis,self.I[det_num].real,**kwargs)
            #         ax.plot(t_axis,self.I[det_num].imag,**kwargs)
            #         ax.legend(('Re','Im'))
            #     elif mode[0].lower()=='r':
            #         ax.plot(t_axis,self.I[det_num].real,**kwargs)
            #     elif mode[0].lower()=='a':
            #         ax.plot(t_axis,np.abs(self.I[det_num]),**kwargs)
            #     elif mode[0].lower()=='i':
            #         ax.plot(t_axis,self.I[det_num].imag,**kwargs)
            #     else:
            #         assert 0,'Unrecognized plotting mode'
                
            #     ax.set_ylabel(det2label(self.detect[det_num]))
            #     ax.set_xlabel(label)
        self.apodize=ap
        return ax
            
    def extract_decay_rates(self,U,det_num:int=0,mode='pwdavg'):
        """
        Uses eigenvalue decomposition to determine all relaxation rates present
        for a density matrix, rho, and their corresponding amplitudes, based
        on detection with the stored detection operators. Note that the
        returned rate constants will be based on the current rho values. If
        you want to start from rho0, make sure to first run reset.
        
        This will calculate frequencies (f) and decay rates (R) from the 
        imaginary and real part of the eigenvalues of the propagator. 
        Weightings (A) will be determined from the eigenvectors and from the 
        density matrix. Note that the weights do no include weighting from
        the powder average.
        
        Depending on the program mode, some averaging will be performed.
        
        mode:
        
        'pwdavg' :  Oscillating terms eliminated. Averaging performed over all
                    non-oscillating terms and over the powder average to yield 
                    a single rate constant
                    
                returns: rate (float)
                    
        'avg'    :  Oscillating terms eliminated. Averaging performed over all
                    non-oscillating terms separated for each element of the
                    powder average. Returns a list of rates and weights the
                    same length as the powder average. The weight is the 
                    downscaling resulting from elimination of the oscillation
                    terms. To obtain the powder-averaged rate constant, one
                    would still need to take the product of this weight with
                    the powder average weight
                    
                returns rates (np.array of floats), weights (np.array of floats)
                    
        'rates'  :  Oscillating terms eliminated. Returns lists of rates and
                    weights. The lists are the length of the powder average
                    and arrays inside the lists correspond to all non-oscillating
                    rates
                    
                returns rates (list of arrays of floats), weights (list of arrays of floats)
                
        'wt_rates': Takes the weights and rates returned by rates and the 'rates'
                    options, and multiplies the weights by the powder average
                    weighting. Two 1D arrays are returns corresponding to the
                    rates and weights. Rates are sorted
                returns rates (1d array), weights (1d array)
                    
        'all'    :  No terms eliminated. Returns an array of rates, frequencies,
                    and amplitudes. The first dimension of the array runs down
                    the powder average, and the second across the propagator
                    dimension. In case reduced bases are used, this is usually
                    less than the full propagator dimension. The missing terms,
                    however, would all have amplitude of zero.
                    
                returns rates (2d array), frequencies (2d array)
                        
        
        

        Parameters
        ----------
        U : TYPE
            DESCRIPTION.
        det_num : int, optional
            Which detector to use. The default is 0.
        mode : str, optional.
            Which averaging mode to use. The default is 'pwdavg'.

        Returns
        -------
        None.

        """
        
        if mode.lower()=='pwdavg':
            pwdavg,avg,decay_only=True,True,True
        elif mode.lower()=='avg':
            pwdavg,avg,decay_only=False,True,True
        elif mode.lower()=='rates' or mode.lower()=='wt_rates':
            pwdavg,avg,decay_only=False,False,True
        else:
            pwdavg,avg,decay_only=False,False,False
            
        
        if self.L is None:self.L=U.L

        if not(self.reduced):
            r,U=self.ReducedSetup(U)
            return r.extract_decay_rates(U,det_num=det_num,mode=mode)
        
        if not(hasattr(U,'calcU')):  #Sequence instead of U was provided
            U=U.U()   #Calculate U from sequence
            



        R=np.zeros([U.L.pwdavg.N,U.shape[0]],dtype=float)
        f=np.zeros([U.L.pwdavg.N,U.shape[0]],dtype=float)
        A=np.zeros([U.L.pwdavg.N,U.shape[0]],dtype=float)
           
        U.eig()
        for k,(rho0,(d,v,vi)) in enumerate(zip(self._rho,U._eig)):
            # d,v=np.linalg.eig(U0)
            rhod=vi@rho0
            det_d=self._detect[det_num]@v
            
            A[k]=(rhod*det_d).real  #Amplitude
            R[k]=-np.log(d).real/U.Dt #Decay rate
            f[k]=np.log(d).imag/U.Dt #Frequency
            

        if not(decay_only):
            return R,f,A

        Rout=list()
        Aout=list()
        
        for R0,A0,f0 in zip(R,A,f):
            i=np.logical_and(np.abs(f0)<1e-5,np.abs(A0)>1e-8)  #non-oscillating terms
            Aout.append(A0[i])
            Rout.append(R0[i])

        if not(avg):
            if not(mode.lower()=='wt_rates'):
                return Rout,Aout
            
            R=[]
            A=[]
            for R0,A0,wt in zip(Rout,Aout,self.L.pwdavg.weight):
                i=np.argmin(np.abs(R0)) #Find the equilibrium rate
                i=np.arange(len(R0))!=i
                R.extend(R0[i])
                A.extend(A0[i]*wt)
                
            return np.array(R),np.array(A)
            
            
            
        #avg = True   
        Aavg=list()
        Ravg=list()
        for R0,A0 in zip(Rout,Aout):
            # i=np.logical_and(np.abs(f0)<1e-5,np.abs(A0)>1e-2)  #non-oscillating terms (??)
            # Aout.append(A0[i].sum())
            # Rout.append((R0[i]*A0[i]).sum()/Aout[-1])
            Aavg.append(A0.sum())
            Ravg.append((R0*A0).sum()/Aavg[-1] if Aavg[-1] else 0)
            
        # R=np.array(Rout)
        # A=np.array(Aout)
            
        R=np.array(Ravg)
        A=np.array(Aavg)
        
        if not(pwdavg):
            return R,A

        # pwdavg=True
        wt=U.L.pwdavg.weight*A
        if wt.sum()==0:
            warnings.warn("Only oscillating terms found in powder average")
            Ravg=np.nan
        else:
            wt/=wt.sum()
            Ravg=(R*wt).sum()
        return Ravg
            
        
        

        
        
    
    def R_dist(self,U,det_num:int=0,nbins=None):
        """
        

        Parameters
        ----------
        U : TYPE
            DESCRIPTION.
        det_num : int, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        None.

        """
        
        R,A=self.extract_decay_rates(U=U,det_num=det_num)
        nbins=U.L.pwdavg.N//2
        bins=np.linspace(R.min(),R.max(),nbins)
        I=np.zeros(bins.shape)
        
        # bl=np.concatenate(([0],bins[:-1]))
        bl=bins
        br=np.concatenate((bins[1:],[np.inf]))
        
        for k,(R0,A0) in enumerate(zip(R,A)):
            i=np.logical_and(R0>=bl,R0<br)
            I[i]+=A0*U.L.pwdavg.weight[k]/A.sum()*len(A)
            
        return bins,I
    
    def plot_R_dist(self,U,det_num:int=0,ax=None):
        """
        Plots a histogram showing the distribution of relaxation rate
        constants resulting from the powder average

        Parameters
        ----------
        U : Propagator
            Propagator to investigate
        det_num : int, optional
            DESCRIPTION. The default is 0.
        ax : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        if ax is None:ax=plt.figure().add_subplot(111)
        
        
        bins,I=self.R_dist(U,det_num)
            
        ax.bar(bins-(bins[1]-bins[0])/1,I,width=(bins[1]-bins[0])*.5)
        ax.set_xlabel(r'R / s$^{-1}$')
        ax.set_ylabel('Weight')
        
        
    def __repr__(self):
        out='Density Matrix/Detection Operator\n'
        out+='rho0: '+(f'{self.rho}' if isinstance(self.rho,str) else 'user-defined matrix')+'\n'
        for k,d in enumerate(self.detect):
            out+=f'detect[{k}]: '+(f'{d}' if isinstance(d,str) else 'user-defined matrix')+'\n'
        if self.t is not None:out+=f'Current time is {self.t*1e6:.3f} microseconds\n'
        out+=f'{len(self.t_axis)} time points have been recorded'
        if self.L is None:
            out+='\n[Currently uninitialized (L is None)]'
        out+='\n\n'+super().__repr__()
        return out
        
            
            
            

        
            
        