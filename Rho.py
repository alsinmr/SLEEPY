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
from pyRelaxSim import Defaults


dtype=Defaults['dtype']
tol=1e-6

class Rho():
    def __init__(self,rho0,detect,L):
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
        if not(isinstance(detect,list)):detect=[detect]  #Make sure a list
        self.detect=detect
        self._L=L
        self._Setup()
        
        

        
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
        return np.sort(self._taxis)
    
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
            i=np.argsort(self._taxis)
            return np.array(self._Ipwd).T[i].T
    
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
        v=1/(self.t_axis[1]-self.t_axis[0])/2*np.linspace(-1,1,len(self.t_axis))
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
        if np.diff(self.t_axis).max()-np.diff(self.t_axis).min()>1e-8:
            warnings.warn('Time points are not equally spaced. FT will be incorrect')
        return np.fft.fftshift(np.fft.fft(np.concatenate((self.I[:,:1]/2,self.I[:,1:]),axis=1),axis=1),axes=[1])
        
    
    def _Setup(self):
        """
        At initialization, we do not require Rho to know the spin-system yet. 
        However, for most functions, this is in fact required. Therefore, at
        the first operation with a propagator, we will run _setup to finalize
        the Rho setup.

        Returns
        -------
        None.

        """
        
        self._Ipwd=[[list() for _ in range(self.n_det)] for _ in range(self.pwdavg.N)]
        self._taxis=list()
        
        self._rho0=self.Op2vec(self.strOp2vec(self.rho0))
        self._detect=[self.Op2vec(self.strOp2vec(det),detect=True) for det in self.detect]
        self.reset()
        
    def prop(self,U):
        """
        Propagates the density matrix by the provided propagator

        Parameters
        ----------
        U : Propagator
            Propagator object.

        Returns
        -------
        None.

        """
            
        if U.L is not self.L:
            warnings.warn('Propagating using a system with a different Liouvillian than the initial Liouvillian')
            
        if np.abs((self.t-U.t0)%self.taur)>tol and np.abs((U.t0-self.t)%self.taur)>tol:
            warnings.warn('The initial time of the propagator is not equal to the current time of the density matrix')
            
        self._rho=[U0@rho for U0,rho in zip(U,self._rho)]
        self._t+=U.Dt
        return self
        
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
        return self.prop(U)
    
    def __ror__(self,U):
        return self.prop(U)
    
    def Detect(self):
        """
        Evaluates the density matrices at the current time point and stores
        the result

        Returns
        -------
        None.

        """
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
        return self.L.__len__()
    
    def DetProp(self,U,n:int=1):
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
        if n>=100:
            self()
            for k,(U0,rho) in enumerate(zip(U,self)):
                d,v=np.linalg.eig(U0)
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
        return self
    
    def strOp2vec(self,OpName:str):
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

        Returns
        -------
        Operator matrix

        """
        
        if not(isinstance(OpName,str)):return OpName #Just return if already a matrix
        
        if OpName[0]=='S':
            i=int(OpName[1])   #At the moment, I'm assuming this program won't work with 11 spins...
            return getattr(self.Op[0],OpName[2:])
        
        if OpName[-1] in ['x','y','z','p','m']:
            a=OpName[-1]
            Nuc=OpName[:-1]
        elif 'alpha' in OpName:
            a='alpha'
            Nuc=OpName[:-5]
        elif 'beta' in OpName:
            a='beta'
            Nuc=OpName[:-4]
        Op=np.zeros(self.Op.Mult.prod()*np.ones(2,dtype=int),dtype=dtype)
        i=self.expsys.Nucs==Nuc
        if not(np.any(i)):
            warnings.warn('Nucleus is not in the spin system or was not recognized')
        for i0 in np.argwhere(i)[:,0]:
            Op+=getattr(self.Op[i0],a)
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
        if detect:
            Op=Op.T.conj()
            Op/=np.trace(Op.T.conj()@Op)*nHam
        return np.tile(Op.reshape(Op.size),nHam)
            
    
    def reset(self):
        """
        Resets the density matrices back to rho0

        Returns
        -------
        None.

        """
        self._rho=[self._rho0 for _ in range(self.pwdavg.N)]
        self._t=0
    
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
        
        self._t=0
        
        self._Ipwd=[[list() for _ in range(self.n_det)] for _ in range(self.pwdavg.N)]
        self._taxis=list()
        self._rho=list() #Storage for numerical rho
        self._Setup()
        
    def plot(self,det_num:int=0,ax=None,FT:bool=False,imag:bool=True):
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
        imag : bool, optional
            Plot the imaginary components. The default is True

        Returns
        -------
        None.

        """
        if ax is None:ax=plt.figure().add_subplot(111)
        
        if FT:
            ax.plot(self.v_axis/1e3,self.FT[det_num].real)
            if imag:
                ax.plot(self.v_axis/1e3,self.FT[det_num].imag)
                ax.legend(('Re','Im'))
            ax.set_xlabel(r'$\nu$ / kHz')
            ax.set_ylabel('I / a.u.')
        else:
            ax.plot(self.t_axis*1e3,self.I[det_num].real)
            if not(isinstance(self.detect[det_num],str)) or self.detect[det_num][-1] in ['p','m'] and imag:
                ax.plot(self.t_axis*1e3,self.I[det_num].imag)
                ax.legend(('Re','Im'))
            ax.set_ylabel('<'+self.detect[det_num]+'>')
            ax.set_xlabel('t / ms')
            
    def extract_decay_rates(self,U,det_num:int=0,avg:bool=True,pwdavg:bool=False):
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
        
        

        R=np.zeros([U.L.pwdavg.N,U.shape[0]],dtype=float)
        f=np.zeros([U.L.pwdavg.N,U.shape[0]],dtype=float)
        A=np.zeros([U.L.pwdavg.N,U.shape[0]],dtype=float)
            
        for k,(rho0,U0) in enumerate(zip(self._rho,U)):
            d,v=np.linalg.eig(U0)
            rhod=np.linalg.pinv(v)@rho0
            det_d=self._detect[det_num]@v
            
            A[k]=(rhod*det_d).real
            R[k]=-np.log(np.abs(d))/U.Dt
            f[k]=(np.log(d/np.abs(d))/U.Dt).real
            

        if avg:
            R=(R*A).sum(-1)
            R/=A.sum(-1)
            A=A.sum(-1)
            if pwdavg:
                wt=U.L.pwdavg.weight*A
                wt/=wt.sum()
                Ravg=(R*wt).sum()
                return Ravg
                
            return R,A
        
        return R,v,A
    
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
        
        
        
            
            
            

        
            
        