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
from .Tools import NucInfo
import re


ctype=Defaults['ctype']
rtype=Defaults['rtype']
tol=1e-10

class Rho():
    def __init__(self,rho0,detect,L=None):
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
        
        # self._Setup()
        self._apodize=False
    
    def __setattr__(self,name,value):
        if name=="L":
            if value is not self.L and len(self.t_axis):
                warnings.warn("Internal Liouvillian does not match propagator's Liouvillian, although system has already been propagated")
             
            super().__setattr__('_L',value)
            self._Setup()
            return
        
        super().__setattr__(name,value)
    
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
        return self._t
    
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
        assert self._tstatus==1,"Uniform time-axis spacing required for downmixing"
        
        # from scipy.signal import butter,filtfilt
        
        if t0 is None:t0=self.t_axis[0]
        
        for k,detect in enumerate(self.detect):
            OpName,_=self.OpScaling(detect)
            v0=None
            if OpName[0]=='S':
                i=int(OpName[1])   #At the moment, I'm assuming this program won't work with 11 spins...
                if self.expsys.LF[i] and OpName[2:] in ['p','m']: #Downmix required
                    v0=self.expsys.v0[i]*(-1 if OpName[2:]=='p' else 1)
            else:
                Nuc,a=self.parseOp(OpName)
                i=self.expsys.Nucs.tolist().index(Nuc)
                if self.expsys.LF[i] and a in ['p','m']:
                    v0=self.expsys.v0[i]*(-1 if a=='p' else 1)
                    
            if v0 is not None:
                Ipwd=self.Ipwd[:,k]
                Idm=Ipwd*np.exp(1j*v0*2*np.pi*(self.t_axis-t0))
                for m in range(self.pwdavg.N):
                    self._Ipwd[m][k]=Idm[m]
                
                
    
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
        if self._apodize:
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
        
        if self.rho0=='Thermal':
            rhoeq=self.L.rho_eq(sub1=True)
            if self.L.Peq:
                eye=np.tile(np.ravel(self.expsys.Op[0].eye),len(self.L.H))
                rhoeq+=eye/self.expsys.Op.Mult.prod()
            self._rho0=rhoeq
        else:
            self._rho0=self.Op2vec(self.strOp2vec(self.rho0))
        self._detect=[self.Op2vec(self.strOp2vec(det,detect=True),detect=True) for det in self.detect]
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
        self._t=None
    
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
        
        # if not(hasattr(U,'__getitem__')):U=U.U()
        
                   
        if self.L is None:
            self.L=U.L

        if self._awaiting_detection:  #Use this if detect was called before L was assigned
            self._awaiting_detection=False
            if self._t is None:self._t=U.t0
            self()
            
         
        if self._t is None:
            self._t=U.t0
            
        if not(self.static) and np.abs((self.t-U.t0)%self.taur)>tol and np.abs((U.t0-self.t)%self.taur)>tol:
            warnings.warn('The initial time of the propagator is not equal to the current time of the density matrix')
        
            
        if U.calculated:
            self._rho=[U0@rho for U0,rho in zip(U,self._rho)]
            self._t+=U.Dt
            return self
        else:
            
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
                    
                    
                
        
    def __rmul__(self,U):
        """
        Runs rho.prop(U) and returns self

        Parameters
        ----------
        U : Propagator
            Propagator object.

        Returns
        -------
        self

        """
        U.calcU()
        return self.prop(U)
    
    def __mul__(self,U):
        """
        Runs rho.prop(U) and returns self

        Parameters
        ----------
        U : Propagator
            Propagator object.

        Returns
        -------
        self

        """
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
    
    def DetProp(self,U=None,seq=None,n:int=1):
        """
        Executes a series of propagation/detection steps. Detection occurs first,
        followed by propagation for n steps. If n>100, then we will use
        eigenvalue decomposition for the propagation

        Parameters
        ----------
        U : Propagator
            Propagator applied. Should be an integer number of rotor periods
        n : int, optional
            Number of steps. The default is 1.

        Returns
        -------
        self

        """
        
        assert not(U is None and seq is None),"Either U or seq must be defined"
        if U is not None and seq is not None:
            warnings.warn('Both U and seq are defined. seq will not be used')
        
        if self.L is None:
            if U is not None:
                self.L=U.L
            else:
                self.L=seq.L
        
        
        if U is not None:
            if self._t is None:self._t=U.t0
            if not(self.static) and np.abs((self.t-U.t0)%self.taur)>tol and np.abs((U.t0-self.t)%self.taur)>tol:
                warnings.warn('The initial time of the propagator is not equal to the current time of the density matrix')
        elif self._t is None:
            self._t=0
        
        if U is not None:
            if n>=100:
                self()
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
                    
            else:
                for _ in range(n):
                    U*self()
        else:
            if seq.Dt%seq.taur<tol or -seq.Dt%seq.taur<tol:
                U=seq.U(t0=self.t,Dt=self.t+seq.Dt)  #Just generate the propagator and call with U
                self.DetProp(U=U,n=n)
                return self
            
            nsteps=np.round(seq.taur/seq.Dt,0).astype(int)
            assert np.abs(nsteps*seq.Dt-seq.taur)<tol,"Sequences shorter than a rotor period can only be propagated via DetProp if seq.Dt fits an integer number of times into the rotor period"
            
            seq.reset_prop_time(self.t)
            
            U=[seq.U() for _ in range(nsteps)]
            
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
            Which detection operator to plot. The default is 0.
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
                    Nuc,a=self.parseOp(detect)
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
                return r'<'+x+a+'$>' if Nuc=='e' else r'<$'+x+a+'$>'
            else:
                return r'<Op>'
                
        
        if det_num is None:
            for det_num in range(len(self._detect)):
                self.plot(det_num=det_num,ax=ax,FT=FT,mode=mode,apodize=apodize,axis=axis)
            if det_num:
                ax.set_ylabel(r'<Op>')
                ax.legend([det2label(detect) for detect in self.detect])
            return ax
        
        ap=self._apodize
        self._apodize=apodize
        
        
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
                if axis.lower()=='microseconds':
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
        self._apodize=ap
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
            
            A[k]=(rhod*det_d).real
            R[k]=-np.log(d).real/U.Dt
            f[k]=np.log(d).imag/U.Dt
            

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
        out+=f'Current time is {self.t*1e6:.3f} microseconds\n'
        out+=f'{len(self.t_axis)} time points have been recorded'
        if self.L is None:
            out+='\n[Currently uninitialized (L is None)]'
        out+='\n\n'+super().__repr__()
        return out
        
            
            
            

        
            
        