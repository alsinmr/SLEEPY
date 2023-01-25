#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 14:34:23 2023

@author: albertsmith
"""

import numpy as np
import matplotlib.pyplot as plt

class Sequence():
    def __init__(self,L):
        """
        Generates a propagator for a specific pulse sequence. If the generated
        propagator is an integer number of rotor periods, then we can expect
        the fastest computational times. However, if an integer number of
        propagators fits into a rotor period, then we still can obtain a 
        significant speedup.

        Parameters
        ----------
        L : Liouvillian
            Liouville matrix.

        Returns
        -------
        None.

        """
        self.L=L
        
        
        ns=self.nspins
        self.fields.clear()
        self.fields.update({k:(0,0,0) for k in range(ns)}) #Current field values for each spin
        
        self._t=np.array([0,np.inf])
        self._v1=np.zeros([ns,2])
        self._voff=np.zeros([ns,2])
        self._phase=np.zeros([ns,2])
        
        self._spin_specific=False
    
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
        t=np.array(t)
        self.new_t(t)
        
        
        for x,name in zip((v1,voff,phase),('_v1','_voff','_phase')):  #Loop over v1,voff,phase
            if not(hasattr(x,'__len__')):
                new=np.ones(self.t.shape)*x
            elif np.all(t==self.t):
                new=np.array(x)
            else:
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
            
    def plot(self,fig=None):
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
        
        if fig is None:fig=plt.figure()
        ax=[fig.add_subplot(2,1,k+1) for k in range(2)]
        
        if len(self.t)==2 and self.isotropic:
            assert 0,"For isotropic systems, one must specify tf"
        elif len(self.t)==2:
            tf=self.taur
        elif self.isotropic:
            tf=self.t[-2]
        elif self.t[-2]%self.taur<1e-10:
            tf=self.t[-2]
        else:
            tf=(self.t[-2]//self.taur+1)*self.taur  #Plot until end of next rotor period
        
        t=np.concatenate(([0],self.t[1:-1].repeat(2),[tf]))
        
        cmap=plt.get_cmap('tab10')
        for a,s in zip(ax,spins):
            # v1=np.concatenate((self.v1[s,:-2].repeat(2)))
            v1=self.v1[s,:-1].repeat(2)/1e3
            a.plot(t*1e6,v1,color=cmap(3))
            a.plot(t*1e6,np.zeros(t.shape),color='black',linewidth=1.5)
            a.text(0,0.5*self.v1.max()/1e3,s if self._spin_specific else self.expsys.Nucs[s])
            a.set_ylabel(r'$v_1$ / kHz')
            a.set_ylim([0,self.v1.max()*1.1/1e3])
        ax[-1].set_xlabel(r't / $\mu$s')
            
        
    def U(self,t0:float=None,Dt:float=None):
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
        t0 : float, optional
            Initial time for the propagator. The default is None, which sets t0
            to the end of the last calculated propagator
        Dt : float, optional
            Length of the propagator. The default is None, which sets it to
            return the full pulse sequence

        Returns
        -------
        U : Propagator
            DESCRIPTION.

        """
        

        if Dt is None:Dt=self.t[-2]
        if self.isotropic:t0=0
        if t0 is None:t0=self.expsys._tprop%self.taur
            
        tf=t0+Dt
        
        # if tf is None:
        #     if len(self.t)==2 and self.isotropic:
        #         assert 0,"For isotropic systems, one must specify tf"
        #     elif len(self.t)==2:
        #         tf=self.taur
        #     elif self.isotropic:
        #         tf=self.t[-2]
        #     elif self.t[-2]%self.taur<1e-10:
        #         tf=self.t[-2]
        #     else:
        #         tf=(self.t[-2]//self.taur+1)*self.taur
        
        t=self.t+t0 #Absolute time axis
        
        # i0=np.argwhere(self.t<=t0)[-1,0]  #Last time point before t0
        # i1=np.argwhere(self.t>=tf)[0,0]  #First time after tf
        
        # t=np.concatenate(([t0],self.t[i0+1:i1],[tf]))
        
        # U=None
        
        # for m,(ta,tb) in enumerate(zip(t[:-1],t[1:])):
        #     for k,(v1,phase,voff) in enumerate(zip(self.v1,self.phase,self.voff)):
        #         self.fields[k]=(v1[i0+m],phase[i0+m],voff[i0+m])
        #     U0=self.L.U(t0=ta,tf=tb)
            
        #     if U is None:
        #         U=U0
        #     else:
        #         U=U0*U
        
        U=None
        i1=np.argwhere(t>=tf)[0,0] #First time after tf
        t=np.concatenate((t[:i1],[tf]))
        for m,(ta,tb) in enumerate(zip(t[:-1],t[1:])):
            for k,(v1,phase,voff) in enumerate(zip(self.v1,self.phase,self.voff)):
                self.fields[k]=(v1[m],phase[m],voff[m])
            U0=self.L.U(t0=ta,Dt=tb-ta)
            U=U0 if U is None else U0*U
        
        ns=self.nspins
        self.fields.update({k:(0,0,0) for k in range(ns)}) #Turn off all fields  
        
        return U
                
            