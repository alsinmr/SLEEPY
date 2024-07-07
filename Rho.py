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
from .Tools import NucInfo,BlockDiagonal
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
        
        if L is not None:self.L=L
        
        self.Reduce=Reduce
        self._BDP=False #Flag to indicate that Block-diagonal propagation was used.
        # self._Setup()
        self.apodize=False
        self._block=None
        self._phase_accum=None
        self._phase_accum0=None
    
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
    def L(self):
        return self._L
    
    @property
    def shape(self):
        if self.L is None:return None
        return self.L.shape[:1]
    
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
        return self.expsys.taur
        
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
    
    def Blocks(self,*seq):
        """
        Returns a list of logical indices, where each list element consists of
        a block of the Liouvillian that needs to be calculated. Note that not
        all blocks need to be calculated, but this depends on the value of the
        current density matrix and the detection operator.
        
        Block propagation is only implemented for the DetProp function, and 
        disables operation on the density matrix, since we no longer have the
        full state of the system when using block propagation.

        Parameters
        ----------
        seq : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if self.L is None:self.L=seq[0].L  #Initialize self if necessary
        
        ini_fields=copy(seq[0].rf.fields) #Store initial field settings
        
        x=np.zeros(self.L.shape,dtype=bool)
        for seq0 in seq:
            
            for k in seq0.rf.fields:seq0.rf.add_field(k) #Set all fields first to zero
            
            for k,v1 in enumerate(seq0.v1):
                if np.any(v1):seq0.rf.add_field(k,v1=1) #Turn field on
                x+=self.L.Lrf.astype(bool)  #Add it to the logical matrix
                seq0.rf.add_field(k) #Turn field off
            
        x+=self.L[0].L(0).astype(bool)
        # We try to avoid any weird orientations that are missing cross terms so check a few orientations
        x+=self.L[len(self.L)//2].L(0).astype(bool)
        x+=self.L[len(self.L)//3].L(0).astype(bool)
        x+=self.L[1*len(self.L)//4].L(0).astype(bool)
        
        seq[0].rf.fields=ini_fields
        
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
        rho.L=self.L.getBlock(block)
        rho._rho0=self._rho0[block]
        rho._detect=[d[block] for d in self._detect]
        rho._rho=[r[block] for r in self._rho]
        rho._Ipwd=[[[] for _ in range(len(self._detect))] for _ in range(len(self.L))]
        rho._taxis=[]
        rho.Reduce=False
        
        
        
        return rho
    
    def ReducedSetup(self,*seq):
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
        None.

        """
        block=np.sum(self.Blocks(*seq),axis=0).astype(bool)
        rho=self.getBlock(block)
        seq_red=[s.getBlock(block) for s in seq]
        
        print(f'State-space reduction: {block.__len__()}->{block.sum()}')
        
        return rho,*seq_red
    
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
        self._rho0=self._rho0[block]
        self._detect=[d[block] for d in self._detect]
        self._rho=[r[block] for r in self._rho]
        self.Reduce=False
        self.block=block
        if seq.reduced:
            self._L=seq.L
            
        print('')
        
        return self
        
    
    def downmix(self,t0:float=None):
        """
        Takes the stored signals and downmixes them if they were recorded in the
        lab frame (result replaces Ipwd). Only applied to signals detected with
        + or - (i.e., the complex signal is required!)
        
        Not heavily tested. 
        
        Parameters
        ----------
        t0  : float, optional
            Effectively a phase-correction parameter. By default, t0 is None,
            which means we will subtract away self.t_axis[0] from the time
            axis. The parameter to be subtracted may be adjusted by setting t0
            Default is None

        Returns
        -------
        None.

        """
        
        # from scipy.signal import butter,filtfilt
        
        if t0 is None:t0=self.t_axis[0]
        
        #A new, more general attempt
        co=[np.tile(o.coherence_order,len(self.L.H)).T[self.block] for o in self.expsys.Op]
        for k,detect in enumerate(self._detect):
            stop=False
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
                    warnings.warn(f'Inconsistent coherence orders in detection matrix {k}, downmixing aborted')
                    stop=True
                    break
            if stop:break #Break out of the second for-loop
            
            Ipwd=self.Ipwd[:,k]
            Idm=Ipwd*q
            for m in range(self.pwdavg.N):
                self._Ipwd[m][k]=Idm[m]
                
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
        v=1/(self.t_axis[1]-self.t_axis[0])/2*np.linspace(-1,1,len(self.t_axis)*2)
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
        I=np.concatenate((self.I[:,:1]/2,self.I[:,1:]),axis=1)
        if self.apodize:
            I*=np.exp(-np.arange(I.shape[-1])/I.shape[-1]*5)
        if self._tstatus!=1:
            warnings.warn('Time points are not equally spaced. FT will be incorrect')
            
        return np.fft.fftshift(np.fft.fft(I,n=I.shape[1]*2,axis=1),axes=[1])
        
    
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
        self._taxis=list()
        self._phase_accum=list()
        
        
        if isinstance(self.rho0,str) and self.rho0=='Thermal':
            rhoeq=self.L.rho_eq(sub1=True)
            if self.L.Peq:
                eye=np.tile(np.ravel(self.expsys.Op[0].eye),len(self.L.H))
                rhoeq+=eye/self.expsys.Op.Mult.prod()
            self._rho0=rhoeq
        else:
            self._rho0=self.Op2vec(self.strOp2vec(self.rho0))
        self._detect=[self.Op2vec(self.strOp2vec(det,detect=True),detect=True) for det in self.detect]
        self._phase_accum0=np.zeros(self.expsys.nspins)
        self.reset()
        
    def reset(self):
        """
        Resets the density matrices back to rho0

        Returns
        -------
        None.

        """
        if self.L is not None:
            self._rho=[self._rho0 for _ in range(self.pwdavg.N)]
            self._phase_accum0=np.zeros(self.expsys.nspins)
        self._t=None
        
        return self
    
    def clear(self):
        """
        Clears variables in order to start over propagation. 
        
        Note that if you want to set the system back to the initial rho0 value,
        but want to retain the amplitudes and times recorded, run rho.reset()
        instead of rho.clear()

        Parameters
        ----------
        clear_all : bool, optional
            Completely reset rho. The default is False.

        Returns
        -------
        None.

        """
        
        self._t=None
        
        self._Ipwd=[[]]
        self._taxis=list()
        self._rho=list() #Storage for numerical rho
        self._L=None
        self._BDP=False
        
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
            
            # blocks=self.Blocks(seq)
            # block=np.sum(blocks,0).astype(bool)
            # if not(np.all(np.sum(blocks,axis=0))):
            #     print(f'State-space reduction: {blocks[0].__len__()}->{block.sum()}')
            #     #Block diagonalization doesn't really help if we still have to calculate all blocks
                
            #     rb=self.getBlock(block)
            #     sb=seq.getBlock(block)
            
            rb.DetProp(seq=sb,n=n,n_per_seq=n_per_seq)
            Ipwd=rb.Ipwd
            for k in range(Ipwd.shape[0]):
                for j in range(Ipwd.shape[1]):
                    self._Ipwd[k][j].extend(Ipwd[k,j])
            self._taxis.extend(rb._taxis)
            self._BDP=True
            self._t=rb._t
            return self
                
            
        

        
        
        if U is not None:
            if self._t is None:self._t=U.t0
            if not(self.static) and np.abs((self.t-U.t0)%self.taur)>tol and np.abs((U.t0-self.t)%self.taur)>tol:
                warnings.warn('The initial time of the propagator is not equal to the current time of the density matrix')
            if not(self.static) and np.abs(U.Dt%self.taur)>tol and np.abs((self.taur-U.Dt)%self.taur)>tol:
                warnings.warn('The propagator length is not an integer multiple of the rotor period')
         
        elif self._t is None:
            self._t=0
        
        if U is not None:
            if n>=100:
                self()  #This is the initial detection
                U.eig()
                for k,((d,v),rho) in enumerate(zip(U._eig,self)):
                    rho0=np.linalg.pinv(v)@rho
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
            # TODO set n_per_seq functionality here
            if self.static:
                return self.DetProp(U=seq.U(),n=n)
            
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
                
            print(f'Prop: {nsteps} step{"" if nsteps==1 else "s"} per every {k} rotor period{"" if k==1 else "s"}')
            
            
            seq.reset_prop_time(self.t)
            
            
            Dt=seq.Dt/n_per_seq
            
            U=[seq.U(Dt=Dt,t0_seq=k*Dt) for k in range(nsteps)]
            
            
            if n//nsteps>100:
                self()  #Detect once before propagation
                U0=[]
                Ipwd=np.zeros([len(self),len(self._detect),n],dtype=Defaults['ctype'])
                
                rho00=copy(self._rho)
                for q in range(nsteps):  #Loop over the starting time
                    n0=n//nsteps+(q<n%nsteps)
                    U0=U[q]
                    for m in range(q+1,q+nsteps):U0=U[m%nsteps]*U0 #Calculate propagator for 1 rotor period starting U[q]
                    U0.eig()
                    for k,((d,v),rho0) in enumerate(zip(U0._eig,rho00)):  #Sweep over the powder average
                        rho0=np.linalg.pinv(v)@rho0
                        dp=np.cumprod(np.repeat([d],n0,axis=0),axis=0)
                        rho_d=dp*rho0
                        for m,det in enumerate(self._detect):
                            det_d=det@v
                            Ipwd[k][m][q::nsteps]=(det_d*rho_d).sum(-1)
                        if q==nsteps-1:
                            self._rho[k]=v@rho_d[-1]
                            
                        rho00[k]=U[q][k]@rho00[k] #Step forward by 1/nsteps rotor period for the next step
                        
                for k in range(len(self)):
                    for m in range(len(self._detect)):
                        self._Ipwd[k][m].extend(Ipwd[k][m][:-1].tolist())
                
                self._taxis.extend([self.t+k*Dt for k in range(1,n)])
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
        
        
    def plot(self,det_num:int=None,ax=None,FT:bool=False,mode:str='Real',apodize=False,axis='kHz/ms',**kwargs):
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
                
        
        if det_num is None:
            h=[]
            for det_num in range(len(self._detect)):
                kids=self.plot(det_num=det_num,ax=ax,FT=FT,mode=mode,apodize=apodize,axis=axis,**kwargs).get_children()
                i=np.array([isinstance(k,plt.Line2D) for k in kids],dtype=bool)
                h.append(np.array(kids)[i][-1])
            if det_num:
                ax.set_ylabel(r'<Op>')
                ax.legend(h,[det2label(detect) for detect in self.detect])
            return ax
        
        ap=self.apodize
        self.apodize=apodize
        
        
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
            else:
                v_axis=self.v_axis
                label=r'$\nu$ / Hz'
            
            if mode.lower()=='reim':
                ax.plot(v_axis,self.FT[det_num].real,**kwargs)
                ax.plot(v_axis,self.FT[det_num].imag,**kwargs)
                ax.legend(('Re','Im'))
            elif mode[0].lower()=='r':
                ax.plot(v_axis,self.FT[det_num].real,**kwargs)
            elif mode[0].lower()=='a':
                ax.plot(v_axis,np.abs(self.FT[det_num]),**kwargs)
            elif mode[0].lower()=='i':
                ax.plot(v_axis,self.FT[det_num].imag,**kwargs)
            else:
                assert 0,'Unrecognized plotting mode'
                
            ax.set_xlabel(label)
            ax.set_ylabel('I / a.u.')
            ax.invert_xaxis()
        else:
            if self._tstatus==0:
                if mode.lower()=='reim':
                    ax.plot(np.arange(len(self.t_axis)),self.I[det_num].real,**kwargs)
                    ax.plot(np.arange(len(self.t_axis)),self.I[det_num].imag,**kwargs)
                    ax.legend(('Re','Im'))
                elif mode[0].lower()=='r':
                    ax.plot(np.arange(len(self.t_axis)),self.I[det_num].real,**kwargs)
                elif mode[0].lower()=='a':
                    ax.plot(np.arange(len(self.t_axis)),np.abs(self.I[det_num]),**kwargs)
                elif mode[0].lower()=='i':
                    ax.plot(np.arange(len(self.t_axis)),self.I[det_num].imag,**kwargs)
                else:
                    assert 0,'Unrecognized plotting mode'
                
                ax.set_ylabel('<'+self.detect[det_num]+'>')
                ax.set_xlabel('Acquisition Number')
                
            else:
                if axis.lower() in ['microseconds','us']:
                    t_axis=self.t_axis*1e6
                    label=r'$t$ / $\mu$s'
                elif axis.lower()=='s':
                    t_axis=self.t_axis
                    label=r'$t$ / s'
                elif axis.lower()=='ns':
                    t_axis=self.t_axis*1e9
                    label=r'$t$ / ns'
                else:
                    t_axis=self.t_axis*1e3
                    label=r'$t$ / ms'
                    
                if mode.lower()=='reim':
                    ax.plot(t_axis,self.I[det_num].real,**kwargs)
                    ax.plot(t_axis,self.I[det_num].imag,**kwargs)
                    ax.legend(('Re','Im'))
                elif mode[0].lower()=='r':
                    ax.plot(t_axis,self.I[det_num].real,**kwargs)
                elif mode[0].lower()=='a':
                    ax.plot(t_axis,np.abs(self.I[det_num]),**kwargs)
                elif mode[0].lower()=='i':
                    ax.plot(t_axis,self.I[det_num].imag,**kwargs)
                else:
                    assert 0,'Unrecognized plotting mode'
                
                ax.set_ylabel(det2label(self.detect[det_num]))
                ax.set_xlabel(label)
        self.apodize=ap
        return ax
            
    def extract_decay_rates(self,U,det_num:int=0,avg:bool=True,pwdavg:bool=False,reweight=False):
        """
        Uses eigenvalue decomposition to determine all relaxation rates present
        for a density matrix, rho, and their corresponding amplitudes, based
        on detection with the stored detection operators. Note that the
        returned rate constants will be based on the current rho values. If
        you want to start from rho0, make sure to first run reset.
        
        Note, I am not sure what this will return for 'p' and 'm' operators
        
        For an nxn Liouville matrix, and N powder elements, we obtain with
        the following settings

        avg = False, pwdavg = False:
            Returns     R : (N,n) real matrix containing the eigenvector 
                            specific relaxation rate constants (1/s)
                        f : (N,n) real matrix containing the eigenvector
                            specific frequencies (rad/s). These frequencies
                            can be back-folded, so the maximum absolute frequency
                            is 1/(2*U.dt).
                        A : (N,n) real matrix containing the eigenvector
                            specific amplitude. 
                            
        avg = True, pwdavg = False
            Returns     R : (N,) real matrix containing the rate constants
                            for each element of the powder average
                        A : (N,) real matrix containing the amplitude for that
                            element of the powder average (if Rho.reset())
                            is run at the beginning, then this is usually all
                            ones (or constant)
                            
        avg =True, pwdavg = True
            Returns      R : Returns the powder-averaged relaxation rate constant
                        
        
        

        Parameters
        ----------
        U : TYPE
            DESCRIPTION.
        det_num : int, optional
            DESCRIPTION. The default is 0.
        avg : bool, optional
            DESCRIPTION. The default is True.
        pwdavg : bool, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        
        if not(avg):assert not(pwdavg),"If avg is False, then pwdavg must also be false"
        
        if self.L is None:self.L=U.L
        

        R=np.zeros([U.L.pwdavg.N,U.shape[0]],dtype=float)
        f=np.zeros([U.L.pwdavg.N,U.shape[0]],dtype=float)
        A=np.zeros([U.L.pwdavg.N,U.shape[0]],dtype=float)
           
        U.eig()
        for k,(rho0,(d,v)) in enumerate(zip(self._rho,U._eig)):
            # d,v=np.linalg.eig(U0)
            rhod=np.linalg.pinv(v)@rho0
            det_d=self._detect[det_num]@v
            
            A[k]=(rhod*det_d).real  #Amplitude
            R[k]=-np.log(d).real/U.Dt #Decay rate
            f[k]=np.log(d).imag/U.Dt #Frequency
            

        if avg:
            Rout=list()
            Aout=list()
            for R0,A0,f0 in zip(R,A,f):
                i=np.logical_and(np.abs(f0)<1e-5,np.abs(A0)>1e-2)  #non-oscillating terms (??)
                Aout.append(A0[i].sum())
                Rout.append((R0[i]*A0[i]).sum()/Aout[-1])
                # print([f'{R00:.2f}' for R00 in R0[i]])
                # print([f'{R00:.2f}' for R00 in A0[i]])
            R=np.array(Rout)
            A=np.array(Aout)
                
            # R=(R*A).sum(-1)
            # R/=A.sum(-1)
            # A=A.sum(-1)
            if pwdavg:
                wt=U.L.pwdavg.weight*(1 if reweight else A)
                wt/=wt.sum()
                Ravg=(R*wt).sum()
                return Ravg
                
            return R,A
        
        return R,f,A
    
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
        
            
            
            

        
            
        