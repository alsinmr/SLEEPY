#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 14:34:23 2023

@author: albertsmith
"""

import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from .Propagator import Propagator

class Sequence():
    def __init__(self,L,cyclic=True,rho=None):
        """
        Generates a propagator for a specific pulse sequence. If the generated
        propagator is an integer number of rotor periods, then we can expect
        the fastest computational times. However, if an integer number of
        propagators fits into a rotor period, then we still can obtain a 
        significant speedup.
        
        If a propagator is generated from a sequence which is longer than the
        sequence's default length, then at the end of the defined sequence, the
        final state will be retained at the end of the sequence. Alternatively,
        if cyclic=True, then the sequence will be repeated for the duration of
        the propagator.
        
        If rho is defined for the sequence (optional), then the sequence will
        be converted to return reduced propagators. Note that propagators 
        resulting from this mode cannot be used together with other density
        matrices

        Parameters
        ----------
        L : Liouvillian
            Liouville operator object.    
        cyclic : bool
            Determines if the sequence repeats (True), or simply retains the
            final state in case a generated propagator is longer than the
            defined sequence. The default is True
        rho : Density matrix/detector object
            Include to run the sequence using reduced (block-diagonal) matrices
            Probably not used anymore....
        

        Returns
        -------
        None.

        """
        self.L=L
        
        
        ns=self.nspins
        self.fields.clear()
        self.fields.update({k:(0,0,0) for k in range(ns)}) #Current field values for each spin
        
        self.clear()

        
        self._spin_specific=False
        self.cyclic=cyclic
        # self._rho=None
        # self.rho=rho
        
        self.t0_seq=0
        
    def clear(self):
        ns=self.nspins
        self._t=np.array([0,np.inf])
        self._v1=np.zeros([ns,2])
        self._voff=np.zeros([ns,2])
        self._phase=np.zeros([ns,2])
        return self
    
    # @property
    # def rho(self):
    #     return self._rho
    
    # @rho.setter
    # def rho(self,rho):
    #     if rho is None:return
    #     assert self._rho is None,"rho cannot be changed for a sequence"
    #     self._rho=rho
        
    #     if rho.BlockDiagonal:
    #         rho.L=self.L
    #         blocks=rho.Blocks(self)
    #         block=np.sum(blocks,0).astype(bool)
    #         self.L=self.L.getBlock(block)
    #         self.block=block
    #         rho._reduce(self)
            
    @property
    def block(self):
        return self.L.block
    
    def getBlock(self,block):
        """
        Returns a sequence using the reduced Liouvillian

        Parameters
        ----------
        block : np.array (boolean)
            Defines what states are included in the reduced density matrix.

        Returns
        -------
        None.

        """
        seq=copy(self)
        seq.L=self.L.getBlock(block)
        return seq
    
    @property
    def reduced(self):
        return self.L.reduced
    
    @property
    def isotropic(self):
        return self.L.isotropic
    
    @property
    def rf(self):
        return self.expsys._rf
    
    @property
    def fields(self):
        return self.rf.fields
    
    @property
    def t(self):
        return self._t
    
    @property
    def Dt(self):
        if len(self._t)==2:return self.taur
        return self._t[-2]

    @property
    def v1(self):
        return self._v1
    
    @property
    def voff(self):
        return self._voff
        
    @property
    def phase(self):
        return self._phase
    
    @property
    def expsys(self):
        return self.L.expsys
    
    @property
    def nspins(self):
        return len(self.expsys.Op)
    
    @property
    def taur(self):
        return self.L.taur
        
    

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
    
    def add_channel(self,channel,t=[0,np.inf],v1=0,voff=0,phase=0):
        """
        Currently, we simply provide the pulse sequence as functions, either
        for each channel ('13C','1H',etc.) or alternatively for each spin 
        (0,1,2...). Not all channels/spins must be provided. 
        
        Note that providing a channel or spin twice will simply override the 
        previous entry. One can also combine both approaches, but if, say 
        spin 0 is carbon, and we specify this first, and then subsequently 
        specify channel 13C, then the latter will override the former.

        Each channel/spin may be provided with a different time axis, providing
        some flexibility in how the channels are specified. Note that for all times
        outside the given time axis, it is assumed that no field is applied to
        that channel. If the time axis is omitted, then it will default to start
        and end at 0 and np.inf, respectively, resulting in a constant field
        on that channel.
        

        Parameters
        ----------
        channel : str or int
            Specification of the channel as a string (1H,13C, etc.). May also 
            be provided as an integer, in which case the sequence will only be
            applied to the corresponding spin.
        t : 1d array
            Time axis for the pulse sequence. 
        v1 : 1d array or float
            Field strength (Hz). May be provided as a single value, in which case
            a constant field will be returned on the specified channel.
        voff : np.array or float, optional
            Offset of the applied field. May be omitted (defaults to zero), or
            may be provided as a single value (fixed offset). The default is None.
        phase : np.array or flaot, optional
            Offset of the applied field. May be omitted (defaults to zero), or
            may be provided as a single value (fixed offset). The default is None.

        Returns
        -------
        None.

        """
        if channel=='e':channel='e-'
        t=np.array(t)
        if t.ndim==0:
            t=np.array([0,t])
            
        assert np.all(t[1:]>=t[:-1]),'Time axis is not monotonically increasing.'
        self.new_t(t)
        
        
        for x,name in zip((v1,voff,phase),('_v1','_voff','_phase')):  #Loop over v1,voff,phase
            if not(hasattr(x,'__len__')):
                new=np.ones(self.t.shape)*x
            elif len(t)==len(self.t) and np.all(t==self.t):
                new=np.array(x)
            else:
                if len(x)==len(t)-1:
                    x=np.concatenate([x,[0]])
                assert len(x)==len(t),f"{name[1:]} has a different length than t"
                new=np.zeros(self.t.shape)
                for k,t0 in enumerate(self.t):
                    if t0<t[0] or t0>t[-1]:
                        new[k]=0
                    else:
                        i=np.argwhere(t<=t0)[-1,0]  #Last t less than or equal to current time point determines settin
                        new[k]=x[i]
            if isinstance(channel,int):
                getattr(self,name)[channel]=new
                self._spin_specific=True
            else:
                getattr(self,name)[channel==self.expsys.Nucs]=new
        return self
        
    def new_t(self,t):
        """
        Updates the time axis to allow for a new channel with potentially a
        different time axis specified

        Parameters
        ----------
        t : array
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        t_old=self.t
        self._t=np.unique(np.concatenate((self.t,t)))
    
        if t_old.size==self.t.size and np.all(t_old==self.t):   #Same t, so no update to fields necessary
            return
        
        for x,name in zip((self._v1,self._voff,self._phase),('_v1','_voff','_phase')):  #Loop over v1,voff,phase
            new=np.zeros([self.nspins,self.t.size])
            for k,t0 in enumerate(self.t):
                
                i=np.argwhere(t_old<=t0)[-1,0]
                new[:,k]=x[:,i]
                
            setattr(self,name,new)
            
    def plot(self,fig=None,ax=None,show_ph_off=True):
        """
        Plots the pulse sequence

        Parameters
        ----------

        Returns
        -------
        None.

        """
        
        if self._spin_specific:
            spins=np.arange(self.nspins)
        else:
            spins=[np.argwhere(chn==self.expsys.Nucs)[0,0] for chn in np.unique(self.expsys.Nucs)]
        
        if ax is None:
            if fig is None:
                fig=plt.figure()
            ax=[fig.add_subplot(len(spins),1,k+1) for k in range(len(spins))]
        
        # if len(self.t)==2 and self.isotropic:
        #     assert 0,"For isotropic systems, one must specify tf"
        # elif len(self.t)==2:
        #     tf=self.taur
        # elif self.isotropic:
        #     tf=self.t[-2]
        # elif self.t[-2]%self.taur<1e-10:
        #     tf=self.t[-2]
        # else:
        #     tf=(self.t[-2]//self.taur+1)*self.taur  #Plot until end of next rotor period
        
        tf=self.Dt
        
        t=np.concatenate(([0],self.t[1:-1].repeat(2),[tf]))
        
        cmap=plt.get_cmap('tab10')
        for a,s in zip(ax,spins):
            a.sharex(ax[0])
            # v1=np.concatenate((self.v1[s,:-2].repeat(2)))
            v1=self.v1[s,:-1].repeat(2)/1e3
            a.plot(t*1e6,v1,color=cmap(3))
            a.plot(t*1e6,np.zeros(t.shape),color='black',linewidth=1.5)
            a.text(0,0.95*self.v1.max()/1e3,s if self._spin_specific else self.expsys.Nucs[s])
            a.set_ylabel(r'$v_1$ / kHz')
            a.set_ylim([0,self.v1.max()*1.1/1e3 if self.v1.max()>0 else 100])
            
            for k,t0 in enumerate(self.t[:-1]):
                
                ch=False
                if show_ph_off:
                    if k==0:
                        ch=True
                        a.text(t0*1e6,.3*a.get_ylim()[1],f'{self.phase[s,k]*180/np.pi:.0f}'+r'$^\circ$')
                        a.text(t0*1e6,.1*a.get_ylim()[1],f'{self.voff[s,k]/1e3:.0f} kHz')
                    else:
                        if self.phase[s,k]!=self.phase[s,k-1]:
                            ch=True
                            a.text(t0*1e6,.3*a.get_ylim()[1],f'{self.phase[s,k]*180/np.pi:.0f}'+r'$^\circ$')
                        if self.voff[s,k]!=self.voff[s,k-1]:
                            ch=True
                            a.text(t0*1e6,.1*a.get_ylim()[1],f'{self.voff[s,k]/1e3:.0f} kHz')
                    if ch:
                        a.plot([t0*1e6,t0*1e6],a.get_ylim(),linestyle=':',color='grey')
                            
                
        ax[-1].set_xlabel(r't / $\mu$s')
        return ax
            
        
    def U(self,Dt:float=None,t0:float=None,t0_seq:float=None):
        """
        Returns the propagator corresponding to the stored sequence. If Dt is
        not specified, then Dt will extend to the last specified point in the
        sequence. If no time has been specified, then it will revert to one
        rotor period, unless the system is isotropic, in which case an error 
        will occur.
        
        If t0 is not specified, then the stored pulse sequence will begin at
        the end of the last calculated propagator. Otherwise, t0 specifies
        the time in the rotor period to start calculating the propagator of
        the pulse sequence.
        
        The stored time axis in the sequence is used as a relative time axis, 
        to t0, i.e. we will not cut off the beginning of the sequence when t0 
        is not 0.

        Parameters
        ----------
        Dt : float, optional
            Length of the propagator. The default is None, which sets it to
            return the full pulse sequence
        t0 : float, optional
            Initial time for the propagator. The default is None, which sets t0
            to the end of the last calculated propagator
            
        t0_seq : float, optional
            Initial time relative to the sequence start. That is, t0 tells where
            in the rotor period to start, and t0_seq tells where in the sequence
            to start. Note that if t0_seq is greater than seq.Dt, then the
            modulo of t0_seq%seq.Dt is used. Note that if sequence is cyclic,
            then an unspecified t0_seq will default to t0. 

        Returns
        -------
        U : Propagator
            DESCRIPTION.

        """
        
        

        if Dt is None:Dt=self.Dt
        
        assert Dt is not None,"For static systems, Dt needs to either be set in the sequence, or when generating U"
        
        if self.isotropic:t0=0
        if t0 is None:
            t0=0 if self.L.static else self.expsys._tprop%self.taur
            
 
            
        tf=t0+Dt
        
        if t0_seq is None:
            if self.Dt is None:
                t0_seq=0
            else:
                t0_seq=self.t0_seq%self.Dt if self.cyclic else 0
        
        
        if self.cyclic and len(self.t)>2:
            t0_seq%=self.Dt
            nreps=int((Dt+t0_seq)/self.Dt)+1
            t=copy(self.t)
            t=np.tile(self.t[:-2],nreps)+(np.arange(nreps)*self.Dt).repeat(len(t)-2)
            t=np.concatenate((t,self.t[-2:-1]+self.Dt*(nreps-1)))
            v1=np.tile(self.v1[:,:-2],nreps)
            v1=np.concatenate((v1,self.v1[:,-2:-1]),axis=-1)
            phase=np.tile(self.phase[:,:-2],nreps)
            phase=np.concatenate((phase,self.phase[:,-2:-1]),axis=-1)
            voff=np.tile(self.voff[:,:-2],nreps)
            voff=np.concatenate((voff,self.voff[:,-2:-1]),axis=-1)
            
            
            self.t0_seq=(t0_seq+Dt)%self.Dt
        else:
            t=copy(self.t)
            v1=copy(self.v1)
            phase=copy(self.phase)
            voff=copy(self.voff)
        

        i=np.argmax(t>t0_seq)-1
        t=t[i:]-t0_seq
        t[0]=0
        
        
        #WHY DO THESE SOMETIMES ONLY GET ONE ELEMENT????
        t=t+t0 #Absolute time axis (relative to rotor period)
        
        
        

        
        # ini_fields=copy(self.fields)

        # U=self.L.Ueye(t[0])
        i1=np.argmax(t>=tf) #First time after tf
        t=np.concatenate((t[:i1],[tf]))
        
        ph_acc=(np.diff(t)*voff[:,i:i+i1]).sum(-1)*2*np.pi
        
        # for m,(ta,tb) in enumerate(zip(t[:-1],t[1:])):
        #     for k,(v1,phase,voff) in enumerate(zip(self.v1,self.phase,self.voff)):
        #         self.fields[k]=(v1[m],phase[m],voff[m])
        #     U0=self.L.U(t0=ta,Dt=tb-ta)
        #     U=U0*U
         
        # self.L.rf.fields=ini_fields        
        
        # dct={'t':t,'v1':copy(self.v1)[:,i:],'phase':copy(self.phase)[:,i:],'voff':copy(self.voff)[:,i:]}
        dct={'t':t,'v1':v1[:,i:i+i1+1],'phase':phase[:,i:i+i1+1],'voff':voff[:,i:i+i1+1]}
        self.expsys._tprop=0 if self.taur is None else tf%self.taur
        
        out=Propagator(U=dct,t0=t0,tf=tf,taur=self.taur,L=self.L,isotropic=self.isotropic,phase_accum=ph_acc)
        
        
        return out
    
    def __pow__(self,n:int):
        """
        Power of the sequence (valid for integer rotor period sequences)

        Returns
        -------
        None.

        """
        
        return self.U()**n
    
    def __repr__(self):
        out='Sequence for the following Liouvillian:\n\t'
        out+=self.L.__repr__().rsplit('\n',maxsplit=2)[0].replace('\n','\n\t')[:-2]
        if self.Dt is None:
            out+='\n\nSequence length unassigned'
        else:
            out+=f'\n\nDefault sequence length is {self.Dt*1e6:.2f} microseconds'
        out+='\n\n'+super().__repr__()
        return out
    
    
    
    
    
                