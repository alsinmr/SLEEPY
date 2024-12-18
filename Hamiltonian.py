#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 15:47:19 2023

@author: albertsmith
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 11:49:00 2023

@author: albertsmith
"""

# import pyDIFRATE.HamTypes as HamTypes
from . import HamTypes
from copy import copy
from .Tools import Ham2Super,LeftSuper,RightSuper
import numpy as np
from . import Defaults
from scipy.linalg import expm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


class Hamiltonian():
    def __init__(self,expsys):
        
        self._expsys=expsys

        #Attach Hamiltonians for each interaction
        self.Hinter=list()
        isotropic=True
        for i in self.expsys:
            dct=i.copy()
            Ham=getattr(HamTypes,dct.pop('Type'))(expsys,**dct)
            isotropic&=Ham.isotropic
            self.Hinter.append(Ham)
        for k,b in enumerate(self.expsys.LF):
            if b:
                Ham=HamTypes._larmor(es=expsys,i=k)
                self.Hinter.append(Ham)

        if isotropic:
            self.expsys.pwdavg=expsys._iso_powder #set powder average to isotropic average if un-used
            self.expsys.n_gamma=1
            self._isotropic=True
        else:
            self._isotropic=False
        for Ham in self.Hinter:Ham.pwdavg=self.pwdavg #Share the powder average
            
            
        # self.sub=False
        self._index=-1
        
        self._initialized=True
    
    @property
    def _ctype(self):
        return Defaults['ctype']
    
    @property
    def sub(self):
        if self._index==-1:return False
        return True
    
    @property
    def rf(self):
        return self.expsys._rf
    
    @property
    def isotropic(self):
        return self._isotropic
    
    @property
    def static(self):
        return self.expsys.vr==0 or self.isotropic
    
    @property
    def expsys(self):
        return self._expsys
    
    @property
    def pwdavg(self):
        return self.expsys.pwdavg
    
    def Liouvillian(self):
        """
        Creates a Liouvillian from the Hamiltonian

        Returns
        -------
        Liouvillian
            Liouvillian for this Hamiltonian

        """
        return self.expsys.Liouvillian()

    
    def __setattr__(self,name,value):
        if hasattr(self,'_initialized') and self._initialized and \
            name not in ['_initialized','_index','sub','rf']:
            print('Hamiltonian cannot be edited after initialization!')
        else:
            super().__setattr__(name,value)
            
        
    def __getitem__(self,i:int):
        """
        Returns the ith element of the powder average, yielding terms Hn for the
        total Hamiltonian

        Parameters
        ----------
        i : int
            Element of the powder average.

        Returns
        -------
        None.

        """
        
        i%=len(self)
        out=copy(self)
        for k,H0 in enumerate(self.Hinter):
            out.Hinter[k]=H0[i]
        out._index=i
        # out.sub=True
        return out
    
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
    
    def Hn(self,n:int):
        """
        Returns the nth rotating component of the total Hamiltonian

        Parameters
        ----------
        n : int
            Component (-2,-1,0,1,2)

        Returns
        -------
        np.array

        """
        assert self.sub or self.isotropic,'Calling Hn requires indexing to a specific element of the powder average'
        
        
        out=np.zeros(self.shape,dtype=self._ctype)
        for Hinter in self.Hinter:
            out+=Hinter.Hn(n)
                
        # if n==0 and self.rf is not None:
        #     out+=self.rf()
                
        return out
    
    def H(self,step:int=0):
        """
        Constructs the Hamiltonian for the requested step of the rotor period.
        Not used for simulation- just provided for completeness

        Parameters
        ----------
        step : int, optional
            Step of the rotor period (0->n_gamma-1). The default is 0.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        ph=np.exp(1j*2*np.pi*step/self.expsys.n_gamma)
        return np.sum([self.Hn(m)*(ph**(-m)) for m in range(-2,3)],axis=0) 
    
    def eig2L(self,step:int):
        """
        Returns a matrix to diagonalize the Liouvillian corresponding to the 
        Hamiltonian, as well as the energies of the diagonalized states.

        The hamiltonian must be indexed and the rotor period step specified

        Parameters
        ----------
        step : int
            Step in the rotor period to diagonalize. The default is 0.

        Returns
        -------
        tuple
            (U,Ui,v)

        """
        H=self.H(step)
        for LF,v0,Op in zip(self.expsys.LF,self.expsys.v0,self.expsys.Op):
            if not(LF):
                H+=v0*Op.z 
        
        a,b=np.linalg.eigh(H)
        U=RightSuper(b)@LeftSuper(b.T.conj())
        Ui=RightSuper(b.T.conj())@LeftSuper(b)
        v=(np.tile(a,a.size)+np.repeat(a,a.size))/2
        return U,Ui,v
        
        
        
        
        
    @property
    def Energy(self):
        """
        Energy for each of the NxN states in the Hamiltonian, including 
        energy from the Larmor frequency (regardless of whether in lab frame).
        Neglects rotating terms, Hn, for n!=0

        Returns
        -------
        None.

        """

        H=self[0].Hn(0)

        expsys=self.expsys
        for LF,v0,Op in zip(expsys.LF,expsys.v0,expsys.Op):
            if not(LF):
                H+=v0*Op.z
        Hdiag=np.tile(np.atleast_2d(np.diag(H)).T,H.shape[0])
        
        energy=(Hdiag+Hdiag.T)/2
        
        # Below was to try to bring in off-diagonal terms of the Hamiltonian
        # I think where these terms are important, we probably shouldn't be
        # using this approach anyway
        # energy=(Hdiag+Hdiag.T)/2+(H-np.diag(np.diag(H)))
        
        return energy.reshape(energy.size).real*6.62607015e-34
    
    def Energy2(self,step:int):
        i=0 if self._index is None else self._index
        H=self[i].H(step)

        expsys=self.expsys
        for LF,v0,Op in zip(expsys.LF,expsys.v0,expsys.Op):
            if not(LF):
                H+=v0*Op.z
        Hdiag=np.tile(np.atleast_2d(np.diag(H)).T,H.shape[0])
        energy=(Hdiag+Hdiag.T)/2+(H-np.diag(np.diag(H)))
        return energy.reshape(energy.size).real*6.62607015e-34
    
    @property
    def shape(self):
        """
        Shape of the Hamiltonian to be returned

        Returns
        -------
        tuple

        """
        return self.expsys.Op.shape
        
    
    def Ln(self,n:int):
        """
        Returns the nth rotation component of the Liouvillian

        Parameters
        ----------
        n : int
            Component (-2,-1,0,1,2)

        Returns
        -------
        np.array

        """
        
        return Ham2Super(self.Hn(n))
    
    def L(self,step:int=0):
        """
        Constructs the Liouvillian for the requested step of the rotor period.
        Not used for simulation, just provided for completeness

        Parameters
        ----------
        step : int, optional
            Step of the rotor period (0->n_gamma-1). The default is 0.

        Returns
        -------
        None.

        """
        return -1j*2*np.pi*Ham2Super(self.H(step))
        # ph=np.exp(1j*2*np.pi*step/self.expsys.n_gamma)
        # return -1j*2*np.pi*np.sum([self.Ln(m)*(ph**(-m)) for m in range(-2,3)],axis=0)
        
        
        
    
    def rho_eq(self,pwdindex:int=0,step:int=None,sub1:bool=False):
        """
        Returns the equilibrium density operator for a given element of the
        powder average.
        

        Parameters
        ----------
        pwdindex : int, optional
            Index of the element of the powder average. Should not have an 
            influence unless the rotor is not at the magic angle or no 
            spinning is included (static, anisotropic). The default is 0.
        sub1 : bool, optional
            Subtracts the identity from the density matrix. Primarily for
            internal use.
            The default is False

        Returns
        -------
        None.

        """
        if self.static and not(self.isotropic): #Include all terms Hn
            H=np.sum([self[pwdindex].Hn(m) for m in range(-2,3)],axis=0)
        elif step is None:
            H=self[pwdindex].Hn(0)
        else:
            ph=np.exp(1j*2*np.pi*step/self.expsys.n_gamma)
            H=np.sum([self[pwdindex].Hn(k)*(ph**-k) for k in range(-2,3)],axis=0)
        for k,LF in enumerate(self.expsys.LF):
            if not(LF):
                H+=self.expsys.v0[k]*self.expsys.Op[k].z
            
        rho_eq=expm(6.62607015e-34*H/(1.380649e-23*self.expsys.T_K))
        # rho_eq/=np.trace(rho_eq)
        rho_eq/=np.sum(np.abs(rho_eq))
        if sub1:
            eye=np.eye(rho_eq.shape[0])
            rho_eq-=np.trace(rho_eq@eye)/rho_eq.shape[0]*eye
        
        
        return rho_eq
    
    def plot(self,what:str='H',cmap:str=None,mode:str='log',colorbar:bool=True,
             step:int=0,ax=None):
        """
        Visualizes the Liouvillian matrix. Options are what to view (what) and 
        how to display it (mode), as well as colormaps and one may optionally
        provide the axis.
        
        Note, one should index the Liouvillian before running. If this is not
        done, then we jump to the halfway point of the powder average
        
        what:
        'H' : Full Hamiltonian. Optionally specify time step
        'H0,H-1,H1,H-2,H2' : Component of Hamiltonian
        'rf' : Applied field matrix
         
        mode:
        'abs' : Colormap of the absolute value of the plot
        'log' : Similar to abs, but on a logarithmic scale
        're' : Real part of the Hamiltonian, where we indicate both
                    negative and positive values (imaginary part will be omitted)
        'im' : Imaginary part of the Hamiltonian, where we indicate both
                    negative and positive values (real part will be omitted)
        'spy' : Black/white for nonzero/zero (threshold applied at 1/1e6 of the max)



        Parameters
        ----------
        what : str, optional
            DESCRIPTION. The default is 'L'.
        cmap : str, optional
            DESCRIPTION. The default is 'YOrRd'.
        mode : str, optional
            DESCRIPTION. The default is 'abs'.
        colorbar : bool, optional
            DESCRIPTION. The default is True.
        step : int, optional
            DESCRIPTION. The default is 0.
        ax : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """

        return HamTypes.HamPlot(self,what=what,cmap=cmap,mode=mode,colorbar=colorbar,
                                step=step,ax=ax)
        
    
    def __repr__(self):
        out='Hamiltonian for the following experimental system:\n'
        out+=self.expsys.__repr__().rsplit('\n',1)[0]
        out+='\n'+super().__repr__()
        return out
    
    
class RF():
    def __init__(self,expsys=None):
        """
        Generates an RF Hamiltonian for a given expsys. Expsys can be provided
        after initialization, noting that RF will not be callable until it is
        provided.
        
        Fields is a dictionary and allows us to either define fields by channel
        ('1H','13C', etc)or by index (0,1,2) to apply to a specific spin. The
        channel or index is a dictionary key. The latter approach is 
        unphysical for a real experiment, but allows one to consider, for example,
        selective behavior without actually designing a selective pulse.
        
        fields={'13C':[50000,0,5000],'1H':[10000,0,0]}
        Applied fields on both 13C and 1H, with 50 and 10 kHz fields effectively
        Phases are 0 on both channels (radians), and offsets are 5 and 0 kHz,
        respectively.
        
        Alternatively:
        fields={0:[50000,0,5000],1:[10000,0,0]}
        If we just have the two spins (13C,1H), this would produce the same
        result as above. However, this could also allow us to apply different
        fields to the same type of spin.
        
        Note that one may just provide the field strength if desired, and the
        phase/offset will be set to zero
        
        

        Parameters
        ----------
        fields : dict
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        self.fields={}
        self.expsys=expsys
    
        self.fields={k:(float(0.),float(0.),float(0.)) for k in range(len(expsys.S))}
        
        
    @property
    def _ctype(self):
        return Defaults['ctype']
    
    def __call__(self):
        """
        Returns the Hamiltonian for the RF fields (non-rotating component)

        Returns
        -------
        None.

        """
        assert self.expsys is not None,"expsys must be defined before RF can be called"
        
        n=self.expsys.Op.Mult.prod()
        out=np.zeros([n,n],dtype=self._ctype)
        
        for name,value in self.fields.items():
            if not(hasattr(value,'__len__')):value=[value,0,0]  #Phase/offset default to zero
            if isinstance(name,str):
                for x,S in zip(self.expsys.Nucs==name,self.expsys.Op):
                    if x:
                        out+=(np.cos(value[1])*S.x+np.sin(value[1])*S.y)*value[0]-value[2]*S.z
            else:
                S=self.expsys.Op[name]
                out+=(np.cos(value[1])*S.x+np.sin(value[1])*S.y)*value[0]-value[2]*S.z
        return out
    
    def add_field(self,channel,v1:float=0,voff:float=0,phase:float=0):
        """
        Add a field by channel (1H,13C,etc.) or by index (0,1,etc).

        Parameters
        ----------
        channel : TYPE
            DESCRIPTION.
        v1 : float, optional
            Field strength. The default is 0.
        voff : float, optional
            Offset frequence. The default is 0.
        phase : float, optional
            Phase (in radians). The default is 0.

        Returns
        -------
        None.

        """
        if channel=='e':channel='e-'
        
        if isinstance(channel,int):
            self.fields.update({channel:(float(v1),float(phase),float(voff))})
        else:
            for key in self.fields:
                if self.expsys.Nucs[key]==channel:
                    self.fields[key]=float(v1),float(phase),float(voff)

