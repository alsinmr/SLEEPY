#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 11:49:36 2023

@author: albertsmith
"""

import numpy as np
import warnings
from .PowderAvg import RotInter
from copy import copy
from . import Defaults
from .Tools import NucInfo
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from . import Constants

class Ham1inter():
    def __init__(self,M=None,H=None,T=None,isotropic=False,delta=0,eta=0,euler=[0,0,0],avg=0,
                 rotor_angle=np.arccos(np.sqrt(1/3)),info={},es=None):
        
        self.M=M
        self.T=T
        self.H=H
        self.isotropic=isotropic
        self.delta=delta
        self.eta=eta
        self.euler=euler
        self.avg=avg
        self.pwdavg=None
        self.rotInter=None
        self.info=info
        self.rotor_angle=rotor_angle
        self.expsys=es
        
        self.A=None
        
    @property
    def _ctype(self):
        return Defaults['ctype']
        
    def __getitem__(self,i:int):
        """
        Get the ith element of the powder average

        Parameters
        ----------
        i : int
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if not(self.isotropic):
            assert self.pwdavg is not None,'pwdavg must be assigned before extracting components of anisotropic Hamiltonians'
            if self.rotInter is None:
                self.rotInter=RotInter(self.expsys,delta=self.delta,eta=self.eta,euler=self.euler,rotor_angle=self.rotor_angle)
            out=copy(self)
            if self.T is None:
                out.A=self.rotInter.Azz[i]
            else:
                out.A=self.rotInter.Afull[i]
            return out
        return self
    
    def __repr__(self):
        dct=copy(self.info)
        out='Hamiltonian for a single interaction\n'
        out+=f'Type: {dct.pop("Type")} '
        out+=f'on spin {dct.pop("i")}\n' if 'i' in dct else f'between spins {dct.pop("i0")} and {dct.pop("i1")}\n'
        
        def ef(euler):
            if hasattr(euler[0],'__iter__'):
                return ','.join([ef(e) for e in euler])
            else:
                return '['+','.join([f'{a*180/np.pi:.2f}' for a in euler])+']'
        
        if len(dct):
            out+='Arguments:\n\t'+\
                '\n\t'.join([f'{key}={ef(value) if key=="euler" else value}' for key,value in dct.items()])+'\n'
        out+='\n'+super().__repr__()    
        return out
    
    def __len__(self):
        if self.pwdavg is None:return 1
        return self.pwdavg.N
            
            
    def Hn(self,n=0,t=None):
        """
        Returns components of the Hamiltonian for a given orientation of the 
        powder average. Only works if the orientation has been set by indexing,
        i.e., use
        
        H[k].Hn(0)
        
        to extract these terms.
        
        Parameters
        ----------
        n : int
            Component rotating at n times the rotor frequency (-2,-1,0,1,2)

        Returns
        -------
        np.array
            Hamiltonian for the nth rotation component
        """

        assert n in [-2,-1,0,1,2],'n must be in [-2,-1,0,1,2]'

        if self.isotropic:
            if n==0:
                return self.H
            else:
                return np.zeros(self.H.shape,dtype=self._ctype)
        
        if self.T is None:
            out=self.M*self.A[n+2]
        else:
            out=np.sum([A*T*(-1)**q for T,A,q in zip(self.T[2],self.A[n+2],range(-2,3))],axis=0)
        
        if self.H is not None and n==0:
            out+=self.H

        return out
    
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

        return HamPlot(self,what=what,cmap=cmap,mode=mode,colorbar=colorbar,
                                step=step,ax=ax)
        # mode=mode.lower()
    
        # if ax is None:
        #     fig,ax=plt.subplots()
        # else:
        #     fig=None
        
        # if cmap is None:
        #     if mode == 'abs' or mode=='log':
        #         cmap='YlOrRd'
        #     elif mode == 'signed':
        #         cmap='BrBG'
        #     elif mode == 'spy':
        #         cmap= 'binary'
                
        # if what in ['H0','H1','H-1','H-2']:
        #     x=self.Hn(int(what[1:]))
        # elif what=='H':
        #     x=np.sum([self.Hn(k) for k in range(-2,3)],axis=0)
        # else:
        #     x=getattr(self[len(self)//2] if self._index==-1 else self,what)
        #     if hasattr(x,'__call__'):
        #         x=x(step)
        
        # sc0,sc1,sc=1,1,1
        # if mode=='abs':
        #     x=np.abs(x)
        #     sc=x.max()
        #     x/=sc
        # elif mode in ['re','im']:
        #     x=copy(x.real if mode=='re' else x.imag)
        #     sc=np.abs(x).max()
        #     x/=sc*2
        #     x+=.5
        # elif mode=='spy':
        #     cutoff=np.abs(x).max()*1e-6
        #     x=np.abs(x)>cutoff
        # elif mode=='log':
        #     x=np.abs(x)
        #     i=np.logical_not(x==0)
        #     if i.sum()!=0:
        #         if x[i].min()==x[i].max():
        #             sc0=sc1=np.log10(x[i].max())
        #             x[i]=1
        #         else:
        #             x[i]=np.log10(x[i])
        #             sc0=x[i].min()
        #             x[i]-=sc0
        #             x[i]+=x[i].max()*.2
        #             sc1=x[i].max()
        #             x[i]/=sc1
                    
        #             sc1=sc1/1.2+sc0
        # else:
        #     assert 0,'Unknown plotting mode (Try "abs", "re", "im", "spy", or "log")'
            
        # hdl=ax.imshow(x,cmap=cmap,vmin=0,vmax=1)
        
        # if colorbar and mode!='spy':
        #     hdl=plt.colorbar(hdl)
        #     if mode=='abs':
        #         hdl.set_ticks(np.linspace(0,1,6))
        #         hdl.set_ticklabels([f'{q:.2e}' for q in np.linspace(0,sc,6)])
        #         hdl.set_label(r'$|H_{n,n}|$')
        #     elif mode=='log':
        #         hdl.set_ticks(np.linspace(0,1,6))
        #         labels=['0',*[f'{10**q:.2e}' for q in np.linspace(sc0,sc1,5)]]
        #         hdl.set_ticklabels(labels)
        #         hdl.set_label(r'$|H_{n,n}|$')
        #     elif mode in ['re','im']:
        #         hdl.set_ticks(np.linspace(0,1,5))
        #         labels=[f'{q:.2e}' for q in np.linspace(-sc,sc,5)]
        #         hdl.set_ticklabels(labels)
        #         hdl.set_label(r'$H_{n,n}$')
            
        # labels=self.expsys.Op.Hlabels
        # if labels is not None:
        #     def format_func(value,tick_number):
        #         value=int(value)
        #         if value>=len(labels):return ''
        #         elif value<0:return ''
        #         return r'$\left|'+labels[value].replace('$','')+r'\right\rangle$'

            
        #     ax.set_xticklabels('',rotation=-90)
        #     ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
            
        #     def format_func(value,tick_number):
        #         value=int(value)
        #         if value>=len(labels):return ''
        #         elif value<0:return ''
        #         return r'$\left\langle'+labels[value].replace('$','')+r'\right|$'
        #     ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
            
        

        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        # if fig is not None:fig.tight_layout()
            
        # return ax
    

def _larmor(es,i:int):
    """
    Larmor frequency Hamiltonian (for Lab-frame simulation)

    Parameters
    ----------
    es : exp_sys
        Experimental system object.
    i : int
        index of the spin.

    Returns
    -------
    None.

    """
    info={'Type':'larmor','i':i}
    S=es.Op[i]
    return Ham1inter(H=es.v0[i]*S.z,isotropic=True,info=info,es=es)

def dipole(es,i0:int,i1:int,delta:float,eta:float=0,euler=[0,0,0]):
    """
    Dipole Hamiltonian

    Parameters
    ----------
    es : exp_sys
        Experimental system object.
    i0 : int
        index of the first spin.
    i1 : int
        index of the second spin.
    delta : float
        anisotropy of the dipole coupling
    eta   : float
        asymmetry of the dipole coupling (usually 0). Default is 0
    euler : list
        3 elements giving the euler angles for the dipole coupling.
        Default is [0,0,0]

    Returns
    -------
    Ham1inter

    """    
    info={'Type':'dipole','i0':i0,'i1':i1,'delta':delta,'eta':eta,'euler':euler}
    
    if es.LF[i0] or es.LF[i1]:  #Lab frame calculation
        T=es.Op[i0].T*es.Op[i1].T
        
        if es.LF[i0] and es.LF[i1]:
            T.set_mode('LF_LF')
        elif es.LF[i0]:
            T.set_mode('LF_RF')
        else:
            T.set_mode('RF_LF')
        
        return Ham1inter(T=T,isotropic=False,delta=delta,eta=eta,euler=euler,rotor_angle=es.rotor_angle,info=info,es=es)

    else:
        S,I=es.Op[i0],es.Op[i1]
        if es.Nucs[i0]==es.Nucs[i1]:
            M=np.sqrt(2/3)*(S.z*I.z-0.5*(S.x@I.x+S.y@I.y))     #Be careful. S.z*I.z is ok, but S.x*I.x is not (diag vs. non-diag)
        else:
            M=np.sqrt(2/3)*S.z*I.z
            
    return Ham1inter(M=M,isotropic=False,delta=delta,eta=eta,euler=euler,rotor_angle=es.rotor_angle,info=info,es=es)

def J(es,i0:int,i1:int,J:float):
    """
    J-coupling Hamiltonian

    Parameters
    ----------
    es : exp_sys
        Experimental system object.
    i0 : int
        index of the first spin.
    i1 : int
        index of the second spin.
    J : float
        Size of the J-coupled Hamiltonian.

    Returns
    -------
    Ham1inter

    """
    S,I=es.Op[i0],es.Op[i1]
    if es.Nucs[i0]==es.Nucs[i1] or (es.LF[i0] and es.LF[i1]):
        H=J*(S.x@I.x+S.y@I.y+S.z*I.z)
    else:
        H=J*S.z*I.z
        
    info={'Type':'J','i0':i0,'i1':i1,'J':J}
    
    return Ham1inter(H=H,avg=J,isotropic=True,info=info,es=es)

def CS(es,i:int,ppm:float=None,Hz:float=None):
    """
    Isotropic chemical shift. Provide shift in Hz or ppm.

    Parameters
    ----------
    es : exp_sys
        Experimental system object.
    i : int
        index of the spin.
    ppm : float
        Chemical shift offset in ppm.
    Hz : float
        Chemical shift offset in Hz.

    Returns
    -------
    Ham1inter

    """
    
    
    
    S=es.Op[i]
    if Hz is not None and ppm is not None:
        warnings.warn('Chemical shift provided in both Hz and ppm. Hz will be used')
        
    
    if Hz is None:
        assert ppm is not None,"ppm or Hz must be provided for CS"
        H=ppm*es.v0[i]/1e6*S.z
        avg=ppm*es.v0[i]/1e6
        info={'Type':'CS','i':i,'ppm':ppm}
    else:
        H=Hz*S.z
        avg=Hz
        info={'Type':'CS','i':i,'Hz':Hz}
        
    return Ham1inter(H=H,avg=avg,isotropic=True,info=info,es=es)
    
def CSA(es,i:int,delta:float=None,deltaHz:float=None,eta:float=0,euler=[0,0,0]):
    """
    Chemical shift anisotropy

    Parameters
    ----------
    es : exp_sys
        Experimental system object.
    i : int
        index of the spin.
    delta : float, optional
        anisotropy of the CSA (ppm)
    deltaHz : float, optional
        anisotropy of the CSA (Hz)
    eta   : float, optional
        asymmetry of the CSA. Default is 0
    euler : list, optional
        3 elements giving the euler angles for the CSA (or a list of 3 element euler angles)
        Default is [0,0,0]

    Returns
    -------
    Ham1inter

    """
    
    assert delta is not None or deltaHz is not None,"delta or deltaHz must be specified"
    if delta is not None:
        info={'Type':'CSA','i':i,'delta':delta,'eta':eta,'euler':euler}
        deltaHz=delta*es.v0[i]/1e6
    else:
        info={'Type':'CSA','i':i,'deltaHz':deltaHz,'eta':eta,'euler':euler}
    
    
    
    if es.LF[i]:  #Lab frame calculation
        T=es.Op[i].T
        T.set_mode('B0_LF')
       
        return Ham1inter(T=T,isotropic=False,delta=deltaHz,eta=eta,euler=euler,rotor_angle=es.rotor_angle,info=info,es=es)
    else:
        S=es.Op[i]
        M=np.sqrt(2/3)*S.z    
    
    return Ham1inter(M=M,isotropic=False,delta=deltaHz,eta=eta,euler=euler,rotor_angle=es.rotor_angle,info=info,es=es)
 

def hyperfine(es,i0:int,i1:int,Axx:float=0,Ayy:float=0,Azz:float=0,euler=[0,0,0]):
    """
    Hyperfine between electron and nucleus. Note that in this implementation,
    we are only including the secular term. This will allow transverse relaxation
    due to both electron T1 and hyperfine tensor reorientation, and also the
    pseudocontact shift. However, DNP will not be possible except for buildup
    in the transverse plane (no SE, CE, NOVEL, etc.)

    Parameters
    ----------
    es : exp_sys
        Experimental system object.
    i0 : int
        index of the first spin.
    i1 : int
        index of the second spin.
    Axx : float
        Axx component of the hyperfine.
    Ayy : float
        Ayy component of the hyperfine.
    Azz : flat
        Azz component of the hyperfine.
    euler : TYPE, optional
        DESCRIPTION. The default is [0,0,0].

    Returns
    -------
    Ham1inter

    """
    if es.Nucs[i0][0]!='e' and es.Nucs[i1][0]!='e':
        warnings.warn(f'Hyperfine coupling between two nuclei ({es.Nucs[i0]},{es.Nucs[i1]})')
    
    info={'Type':'hyperfine','i0':i0,'i1':i1,'Axx':Axx,'Ayy':Ayy,'Azz':Azz,'euler':euler}
    avg=(Axx+Ayy+Azz)/3
    
    Axx-=avg
    Ayy-=avg
    Azz-=avg
    
    q=np.argsort(np.abs([Axx,Ayy,Azz]))
    Ayy,Axx,Azz=np.array([Axx,Ayy,Azz])[q]
    if not(np.all(q==[1,0,2])):
        if np.all(q==[0,1,2]):
            euler0=[0,np.pi,np.pi/2]
        elif np.all(q==[1,2,0]):
            euler0=[np.pi,np.pi/2,0]
        elif np.all(q==[2,1,0]):
            euler0=[np.pi/2,np.pi/2,np.pi]
        elif np.all(q==[2,0,1]):
            euler0=[3*np.pi/2,np.pi/2,3*np.pi/2]
        elif np.all(q==[0,2,1]):
            euler0=[0,np.pi/2,np.pi/2]
        
        if hasattr(euler[0],'__len__'):
            euler=[euler0,*euler]
        else:
            euler=[euler0,euler]
    
    
    delta=Azz
    eta=(Ayy-Axx)/delta if delta else 0

    if es.LF[i0] or es.LF[i1]:  #Lab frame calculation
        T=es.Op[i0].T*es.Op[i1].T
        
        if es.LF[i0] and es.LF[i1]:
            T.set_mode('LF_LF')
        elif es.LF[i0]:
            T.set_mode('LF_RF')
        else:
            T.set_mode('RF_LF')
        
        H=-np.sqrt(3)*avg*T[0,0]   #Rank-0 contribution
        
        if delta:
            return Ham1inter(H=H,T=T,isotropic=False,delta=delta,eta=eta,euler=euler,avg=avg,rotor_angle=es.rotor_angle,info=info,es=es)
        else:
            return Ham1inter(H=H,isotropic=True,avg=avg,info=info,es=es)
    else:  #Rotating frame calculation
        S,I=es.Op[i0],es.Op[i1]
        M=np.sqrt(2/3)*S.z*I.z
        H=avg*S.z*I.z
        if delta:                        
            return Ham1inter(M=M,H=H,isotropic=False,delta=delta,eta=eta,euler=euler,avg=avg,rotor_angle=es.rotor_angle,info=info,es=es)
        else:
            return Ham1inter(H=H,avg=avg,isotropic=True,info=info,es=es)

def quadrupole(es,i:int,delta:float=None,DelPP:float=None,eta:float=0,euler=[0,0,0]):
    """
    Quadrupole coupling defined by its anisotropy (delta) and asymmetry (eta). 
    One may alternatively define the peak-to-peak separation(DelPP). For half integer
    spins, this is the distance between the central frequence and first peak,
    and for integer spins, the distance between the two central peaks.

    Parameters
    ----------
    es : exp_sys
        Experimental system object.
    i : int
        index of the spin.
    delta : float
        anisotropy of the quadrupole coupling. 
        Default is None (provide DelPP or delta)
    DelPP : float
        peak-to-peak distance for the quadrupole coupling. 
        Default is None (provide DelPP or delta)
    eta   : float
        asymmetry of the quadrupole coupling (usually 0). Default is 0
    euler : list
        3 elements giving the euler angles for the quadrupole coupling.
        Default is [0,0,0]

    Returns
    -------
    Ham1inter

    """
    
    assert delta is not None or DelPP is not None,"delta or DelPP must be provided"
    if delta is None:
        info={'Type':'quadrupole','i':i,'DelPP':DelPP,'eta':eta,'euler':euler}
        delta=DelPP*2/3
    else:
        info={'Type':'quadrupole','i':i,'delta':delta,'eta':eta,'euler':euler}
    
    S=es.Op[i]

    if es.LF[i]:
        T=S.T
        T=T*T
        return Ham1inter(T=T,isotropic=False,delta=delta,eta=eta,euler=euler,rotor_angle=es.rotor_angle,info=info,es=es)
    else:
        I=es.S[i]
        M=np.sqrt(2/3)*1/2*(3*S.z@S.z-I*(I+1)*S.eye)  
        
        return Ham1inter(M=M,isotropic=False,delta=delta,eta=eta,euler=euler,
                          rotor_angle=es.rotor_angle,info=info,es=es)

def g(es,i:int,gxx:float=2.0023193,gyy:float=2.0023193,gzz:float=2.0023193,euler=[0,0,0]):
    """
    electron g-tensor Hamiltonian. Note that the g-tensor values should be 
    typically positive.
    

    Parameters
    ----------
    es : exp_sys
        Experimental system object.
    i : int
        index of the spin.
    gxx : float, optional
        xx-component of the electron g-tensor. The default is 2.0023193.
    gyy : float, optional
        yy-component of the electron g-tensor. The default is 2.0023193.
    gzz : float, optional
        xx-component of the electron g-tensor. The default is 2.0023193.
    euler : TYPE, optional
        3 elements giving the euler angles for the g-tensor
        The default is [0,0,0].

    Returns
    -------
    Ham1inter

    """
    
    if es.Nucs[i][0]!='e':
        warnings.warn('g-tensor is being applied to a nucleus')
    
    
       
    info={'Type':'g','i':i,'gxx':gxx,'gyy':gyy,'gzz':gzz,'euler':euler}   
    
    avg=(gxx+gyy+gzz)/3
    if avg<0:
        warnings.warn('Expected a positive g-tensor')
    
    gxx-=avg
    gyy-=avg
    gzz-=avg
    
    
    q=np.argsort(np.abs([gxx,gyy,gzz]))
    gyy,gxx,gzz=np.array([gxx,gyy,gzz])[q]
    if not(np.all(q==[1,0,2])):
        if np.all(q==[0,1,2]):
            euler0=[0,np.pi,np.pi/2]
        elif np.all(q==[1,2,0]):
            euler0=[np.pi,np.pi/2,0]
        elif np.all(q==[2,1,0]):
            euler0=[np.pi/2,np.pi/2,np.pi]
        elif np.all(q==[2,0,1]):
            euler0=[3*np.pi/2,np.pi/2,3*np.pi/2]
        elif np.all(q==[0,2,1]):
            euler0=[0,np.pi/2,np.pi/2]
        
        if hasattr(euler[0],'__len__'):
            euler=[euler0,*euler]
        else:
            euler=[euler0,euler]
    
    
    mub=Constants['mub']
    # -9.2740100783e-24/6.62607015e-34  #Bohr magneton in Hz. Take positive g-values by convention
    
    avg1=mub*avg-NucInfo('e-')            #Values in Hz. Note that we take this in the rotating frame
    delta=gzz*mub*es.B0
    eta=(gyy-gxx)/gzz if delta else 0
    
    plt_avg=mub*avg*es.B0
    
    if es.LF[i]:  #Lab frame calculation
        T=es.Op[i].T
        T.set_mode('B0_LF')
        H=-np.sqrt(3)*avg1*T[0,0]*es.B0   #Rank-0 contribution
        if delta:
            return Ham1inter(H=H,T=T,isotropic=False,delta=delta,eta=eta,euler=euler,avg=plt_avg,rotor_angle=es.rotor_angle,info=info,es=es)
        else:
            return Ham1inter(H=H,avg=plt_avg,isotropic=True,info=info,es=es)
    else:  #Rotating frame calculation
        S=es.Op[i]
        M=np.sqrt(2/3)*S.z
        H=(avg1*es.B0)*S.z
        if delta:                        
            return Ham1inter(M=M,H=H,isotropic=False,delta=delta,eta=eta,euler=euler,avg=plt_avg,rotor_angle=es.rotor_angle,info=info,es=es)
        else:
            return Ham1inter(H=H,avg=plt_avg,isotropic=True,info=info,es=es)
        
        
def ZeroField(es,i:int,D:float,E:float=0,euler=[0,0,0]):
    """
    Electron zero-field splitting. This is equivalent to the quadrupole
    coupling, but uses the parameters D and E, as is commonly done in EPR. The
    principal components of the zero-field splitting are then given as
    
    D*[-1/3,-1/3,2/3]+E*[1,-1,0]
    

    Parameters
    ----------
    es : exp_sys
        Experimental system object.
    i : int
        index of the spin.
    D : float
        D parameter in Hz
    E : float, optional
        E parameter in Hz. The default is 0
    euler : TYPE, optional
        3 elements giving the euler angles for the g-tensor
        The default is [0,0,0].

    Returns
    -------
    Ham1inter

    """
    
    info={'Type':'ZeroField','i':i,'D':D,'E':E,'euler':euler}
    
    pc=D*np.array([-1,-1,2])/3+E*np.array([1,-1,0])
    
    q=np.argsort(np.abs(pc))
    
    y,x,z=pc[q]
    delta=z
    eta=(y-x)/delta
    
    S=es.Op[i]
    
    if es.LF[i]:
        T=S.T
        T=T*T
        return Ham1inter(T=T,isotropic=False,delta=delta,eta=eta,euler=euler,rotor_angle=es.rotor_angle,info=info,es=es)
    else:
        I=es.S[i]
    
        M=np.sqrt(2/3)*1/2*(3*S.z@S.z-I*(I+1)*S.eye)  
        
        return Ham1inter(M=M,isotropic=False,delta=delta,eta=eta,euler=euler,
                          rotor_angle=es.rotor_angle,info=info,es=es)
    
    
        
def HamPlot(H,what:str='H',cmap:str=None,mode:str='log',colorbar:bool=True,
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
    axis

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
    

    H=H[len(H)//2] if (hasattr(H,'_index') and H._index==-1) or (hasattr(H,'A') and H.A is None ) else H
    if what in ['H0','H1','H-1','H-2']:
        x=H.Hn(int(what[1:]))
    elif what=='H':
        if hasattr(H,'_index'):
            x=H.H(step)
        else:
            ph=np.exp(1j*2*np.pi*step/H.expsys.n_gamma)
            x=np.sum([H.Hn(m)*(ph**(-m)) for m in range(-2,3)],axis=0)
    else:
        x=getattr(H,what)
        if hasattr(x,'__call__'):
            if what=='rf':
                x=x()
            else:
                x=x(step)

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
        # This isn't always working if only one value present (??)
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
        
    hdl=ax.imshow(x,cmap=cmap,vmin=0,vmax=1)
    
    if colorbar and mode!='spy':
        hdl=plt.colorbar(hdl)
        if mode=='abs':
            hdl.set_ticks(np.linspace(0,1,6))
            hdl.set_ticklabels([f'{q:.2e}' for q in np.linspace(0,sc,6)])
            hdl.set_label(r'$|H_{n,n}|$')
        elif mode=='log':
            hdl.set_ticks(np.linspace(0,1,6))
            labels=['0',*[f'{10**q:.2e}' for q in np.linspace(sc0,sc1,5)]]
            hdl.set_ticklabels(labels)
            hdl.set_label(r'$|H_{n,n}|$')
        elif mode in ['re','im']:
            hdl.set_ticks(np.linspace(0,1,5))
            labels=[f'{q:.2e}' for q in np.linspace(-sc,sc,5)]
            hdl.set_ticklabels(labels)
            hdl.set_label(r'$H_{n,n}$')
        
    labels=H.expsys.Op.Hlabels
    if labels is not None:
        def format_func(value,tick_number):
            value=int(value)
            if value>=len(labels):return ''
            elif value<0:return ''
            return r'$\left|'+labels[value].replace('$','')+r'\right\rangle$'

        
        ax.set_xticklabels('',rotation=-90)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
        
        def format_func(value,tick_number):
            value=int(value)
            if value>=len(labels):return ''
            elif value<0:return ''
            return r'$\left\langle'+labels[value].replace('$','')+r'\right|$'
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
        
    

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if fig is not None:fig.tight_layout()
        
    return ax




                   