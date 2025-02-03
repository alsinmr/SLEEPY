 # -*- coding: utf-8 -*-

import numpy as np
import warnings
import matplotlib.pyplot as plt

class LFrf():
    def __init__(self,seq,min_steps:int=2):
        """
        Allows application of an RF field in the lab frame, by explicitely including
        the oscillation. 
        
        Currently set up for constant fields. Returns a propagator for one 
        rotor period (or of arbitrary length for static experiments).
        
        Fields to be applied are specified via a sequence object. Just specify
        constant fields:
            
            e.g. (with MAS)
            seq=L.Sequence()
            seq.add_field('1H',v1=100)
            
            (without MAS)
            seq=L.Sequence(Dt=1e-3)
            seq.add_field('1H',v1=100)

        Parameters
        ----------
        seq : Sequence
            Sequence to specify field strengths on 1 (later maybe 2) channels
        min_steps : int, optional
            Minimum number of steps per oscillation of the applied field. 
            The default is 4.
        Returns
        -------
        None.

        """
        
        self.min_steps=min_steps
        
        self.seq0=seq
        
        assert len(np.unique(self.expsys.Nucs[self.v_index]))==1,"Currently, only one Lab Frame rf field supported"
        if not(self.L.static):
            assert self.Dt==self.taur,"Currently, only implemented for one rotor period (seq.Dt should equal taur)"
        
    
    #%% Initialize the sequence
    @property
    def seq0(self):
        return self._seq0
    @seq0.setter
    def seq0(self,seq0):
        self._seq=None
        self._U=None
        self._seq0=seq0
    
    #%% Properties extracted from seq0
    @property
    def v1(self):
        return self._seq0.v1[:,0]
    @property
    def voff(self):
        return self._seq0.voff[:,0]
    @property
    def phase(self):
        return self._seq0.phase[:,0]
    @property
    def Dt(self):
        return self._seq0.Dt
    @property
    def L(self):
        return self._seq0.L
    
    
    #%% Various properties from expsys
    @property
    def expsys(self):
        return self.L.expsys
    
    @property
    def taur(self):
        return self.expsys.taur
    
    @property
    def v0(self):
        return self.expsys.v0
    
    @property
    def LF(self):
        return self.expsys.LF
    
    @property
    def n_gamma(self):
        return self.expsys.n_gamma
        
#%% Other properties
    @property
    def v(self):
        return self.v0+self.voff
    
    @property
    def v_index(self):
        return np.logical_and(self.v1>0,self.LF)
    
    @property
    def n_steps(self):
        """
        This may get updated eventually, to accomodate two fields
        """
        return self.min_steps
    
    @property
    def Dt0(self):
        """
        Length of steps to change the RF amplitude

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return np.abs(1/(self.v[self.v_index][0]*self.n_steps))

#%% Functions to generate short time step propagators
    @property
    def seq(self):
        if self._seq is None:
            seq=self.L.Sequence()
            t=np.arange(self.n_steps+1)*self.Dt0
            for k,v_index in enumerate(self.v_index):
                if v_index:
                    if self.n_steps==2 and self.phase[k]==0:
                        v1=2*self.v1[k]*np.cos(2*np.pi*t*self.v[k])
                        v1*=np.pi/4
                        phase=np.pi*(v1<0)+np.pi/2
                        v1=np.abs(v1)
                        seq.add_channel(k,t=t,v1=v1,phase=phase)
                    else:
                        v1=2*self.v1[k]*np.cos(2*np.pi*t*self.v[k])
                        FT=self.FT(v1=v1)[1][0]
                        i=np.argmax(np.abs(FT[:len(FT)//2]))
                        sc=self.v1[k]/np.abs(FT[i]).max()
                        v1*=sc
                        phase=np.pi*(v1<0)+np.arctan2(FT[i].imag,FT[i].real)+self.phase[k]
                        v1=np.abs(v1)
                        seq.add_channel(k,t=t,v1=v1,phase=phase)
                else:
                    seq.add_channel(k,t=t,v1=self.v1[k],phase=self.phase[k],voff=self.voff[k])
            
                    
            self._seq=seq
            
        return self._seq
        
        
    def Ustep(self,step:int=None):
        
        if not(self.L.static):
            assert step is not None,"step required except for static measurements"

            t0=step*self.taur/self.n_gamma
            U0=self.seq.U(t0=t0)
        else:
            U0=self.seq.U()
            
        p=self.Dt/U0.Dt if self.L.static else self.taur/self.n_gamma/U0.Dt
        
        warnings.filterwarnings("ignore", 
            message="Power of a propagator should only be used if the propagator length is an integer multiple of rotor periods")
        warnings.filterwarnings("ignore", 
            message="Warning: non-integer powers may not accurately reflect state of propagator in the middle of a rotor period")
        
        U=U0**p
        
        warnings.filterwarnings("default", 
            message="Power of a propagator should only be used if the propagator length is an integer multiple of rotor periods")
        warnings.filterwarnings("default", 
            message="Warning: non-integer powers may not accurately reflect state of propagator in the middle of a rotor period")
        
        return U
    
    def U(self,progress:bool=True):
        
        if self._U is None:
            if progress:
                ProgressBar(0,self.n_gamma,"LF calculation:",suffix="complete",decimals=0,length=30)
            
            U=self.L.Ueye()
            for k in range(self.n_gamma):
                U=self.Ustep(k)*U
                if progress:
                    ProgressBar(k+1,self.n_gamma,"LF calculation:",suffix="complete",decimals=0,length=30)
            self._U=U
        return self._U
    
    
    def FT(self,v1=None,nreps=100):
        if v1 is None:
            I=self.seq.v1[self.v_index,:-2]
            I[self.seq.phase[self.v_index,:-2]>3]*=-1
        else:
            I=np.atleast_2d(v1[:-1])
        I=np.tile(np.repeat(I,nreps,axis=-1),reps=nreps)
        
        I[:,0]/=2
        S=np.fft.fftshift(np.fft.fft(I,2*I.shape[1],axis=-1),axes=[-1])
        S/=I.shape[1]
        f=1/(2*self.Dt0/nreps)*np.linspace(-1,1,S.shape[1])
        f-=(f[1]-f[0])/2
        
        return f,S
        
    
    def plot(self,ax=None,nreps=100):
        """
        Plots the spectrum of the applied field. Field is repeated nreps time
        to minimize distortions from the end of the sequence

        Parameters
        ----------
        ax : TYPE, optional
            DESCRIPTION. The default is ax.
        nreps : TYPE, optional
            DESCRIPTION. The default is 100.

        Returns
        -------
        None.

        """
        f,S=self.FT(nreps=nreps)
        
        if ax is None:
            ax=plt.subplots()[1]
        ax.plot(f/1e6,np.abs((S).T),color='black',linewidth=2)
        ax.plot(f/1e6,((S).real.T))
        ax.plot(f/1e6,((S).imag.T))
        
        ax.set_xlabel('f / MHz')
        ax.legend(('Abs','Real','Imag'))
        
        return ax



def ProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█',last=[0]):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """

    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        filledLength = int(length)
        bar = fill * filledLength + '-' * (length - filledLength)
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
        print('\nCompleted')        