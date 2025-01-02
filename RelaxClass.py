# -*- coding: utf-8 -*-

from warnings import warn
from . import Defaults
from copy import copy
import numpy as np
from .Tools import Ham2Super,BlockDiagonal
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from . import Constants

class RelaxClass():
    """
    Class for implementing relaxation processes that are dependent on the
    orientation of the sample. Specifically, deals with changes in the eigenbasis
    of the Hamiltonian as a function of position in the rotor period and in the
    powder average, allowing relaxation to consistently occur in the eigenbasis.
    Also adjusts the thermal equilibrium throughout the rotor period and through
    the powder average.
    
    All relaxation methods should have "step:int=None" as the last argument.
    
    Defining step will call the method and return a relaxation matrix. If step
    is omitted, then arguments for the method will be stored, but the method
    will not be called. Calling RelaxClass() will run all stored methods, or
    access the cache if methods have already been run for that orientation.
    """
    
    h=Constants['h']
    # h=6.62607015e-34
    
    def __init__(self,L):
        """
        Initializes the Relaxation Class. This 

        Parameters
        ----------
        L : TYPE
            DESCRIPTION.

        Returns
        -------
        self

        """
        
        
        self.L=L
        
        self.methods=[]
        self.clear_cache()
        self.Peq=False
        self._deactivate=False
        self.sc=1
    
    def __call__(self,step:int):
        if not(self.active):return 0
        
        assert self.L.sub,"Calling LrelaxOS (RelaxClass) requires indexing to a specific element of the power average"
        
        
        if Defaults['cache']:
            cache=self._cache[self.L._index][step%self.L.expsys.n_gamma]
            if cache is not None:
                if cache.shape[0]!=self._L.shape[0]:
                    self.clear_cache()
                else:
                    return cache
        
        out=np.zeros(self._L.shape,dtype=Defaults['ctype'])
        for method in self.methods:
            fun=getattr(self,method['method'])
            kwargs=copy(method)
            kwargs.pop('method')
            
            out+=fun(step=step,**kwargs)[self.block][:,self.block]
            
        if Defaults['cache']:
            self._cache[self.L._index][step%self.L.expsys.n_gamma]=out
            
        return out
            
    @property        
    def L(self):
        if self._L.reduced:
            # self._L._L=self._L._L[self._L._index]
            # self._L._L._index=self._L._index
            self._L._L=self._L._L[self._L._index]
            return self._L._L
        else:
            return self._L
        
    @L.setter
    def L(self,L):
        self._L=L
    
    
    
    @property
    def block(self):
        return self._L.block
    
    @property
    def reduced(self):
        return self._L.reduced
    
    @property
    def active(self):
        return bool(len(self.methods))
    
    @property
    def v0(self):
        return self.L.expsys.v0
    
    @property
    def Op(self):
        return self.L.expsys.Op
    
    @property
    def T_K(self):
        return self.L.expsys.T_K
    
    def clear(self):
        """
        Clears all relaxation methods from the RelaxClass

        Returns
        -------
        self

        """
        self.methods=[]
        self.Peq=False
        self.clear_cache()
    
    def clear_cache(self):
        """
        Clears the cache of relaxation matrices from the RelaxClass. Should
        be run for addition of new relaxation processes and edits to the
        expsys.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        self._cache=[[None for _ in range(self.L.expsys.n_gamma)] for _ in range(len(self.L))]
        return self
    
    @property
    def list_methods(self):
        out=[]
        for x in dir(self):
            if x=='list_methods' or '__' in x:continue
            attr=getattr(self,x)
            if hasattr(attr,'__func__') and attr.__func__.__code__.co_argcount>1 \
                and attr.__func__.__code__.co_varnames[attr.__func__.__code__.co_argcount-1]=='step':
                    out.append(x)
        return out
    
    def Lindblad(self,M,E):
        """
        Provides the Lindblad thermalization matrix (I think) for matrix M and
        frequencies, v. Ideally, M would be in the eigenbasis of the Hamiltonian,
        and v the appropriately sorted eigenfrequencies.

        Parameters
        ----------
        M : np.array
            Adiabatic (symmetric) matrix to be thermalized
        E : np.array
            Vector containing the energy (Joules) for each state of M. Note
            that this is NOT the diagonal of the Liouvillian, where alpha/beta
            states have 0 energy.

        Returns
        -------
        np.array

        """
        
        out=np.zeros(M.shape,dtype=M.dtype)
        index=np.argwhere(M-np.diag(np.diag(M)))
        index.sort(-1)
        index=np.unique(index,axis=0)
        
        for i0,i1 in index:
            DelE=E[i0]-E[i1]
            rat=np.exp(DelE/(Constants['kB']*self.T_K))
            
            # Del=M[i0,i1]*(1-rat)/(1+rat)*np.sign(M[i0,i1])
            
            #Suppose Del should b positive. 
            #Del is corrected above
            #Case 1: i0 has switched sign
                
                
            # out[i0,i1]=-Del*M[i0,i1]/np.abs(M[i0,i1])
            # out[i1,i1]+=Del
            # out[i1,i0]=Del*M[i0,i1]/np.abs(M[i0,i1])
            # out[i0,i0]+=-Del
            
            
            # TODO I don't see why out[i0,i0] should always be reduced
            # Shouldn't it depend on which has higher energy?
            Del=M[i0,i1]*(1-rat)/(1+rat)
            
            out[i0,i0]-=np.abs(Del)
            out[i1,i1]+=np.abs(Del)
            out[i0,i1]=-Del
            out[i1,i0]=Del.conj()

            # print(M[i0,i1])

            #This one seems to break DynamicThermal
            # out[i0,i0]+=M[i0,i0]*(1-rat)/(1+rat)
            # out[i1,i1]-=M[i1,i1]*(1-rat)/(1+rat)
            # out[i0,i1]=-M[i0,i1]*(1-rat)/(1+rat)
            # out[i1,i0]=M[i1,i0]*(1-rat)/(1+rat)
            
            # out[i0,i0]-=Del
            # out[i1,i1]+=Del
            # out[i0,i1]=-Del
            # out[i1,i0]=Del
            
            
        return out
        
    
    def recovery(self,step:int=None):
        """
        Provides thermalization for the Lrelax matrix of L (not part of RelaxClass)
        as a function of rotor angle

        Parameters
        ----------
        step : int, optional
            The default is None, which runs setup for this method, and returns 
            self. Providing a value for step will return a relaxation matrix

        Returns
        -------
        self/np.array

        """
        # Just setup
        if step is None:
            # Check to see if method is already here (don't put in twice)
            for m in self.methods:
                if m['method']=='Thermal':return self
            
            self.methods.append({'method':'recovery'})
            self.Peq=True
            return self.clear_cache()
        
        
        # Run the function
        L=self.L
        
        return self.Lindblad(L.Lrelax,L.Energy2(step))
    
    def T1(self,i:int,T1:float,Thermal:bool=False,state:int=None,step:int=None):
        """
        Introduces T1 relaxation on a single spin. Relaxation is introduced in
        the eigenbasis. Transitions in the eigenbasis will be included based
        on having the largest contributions from an isotropic relaxation operator
        (Lx^2+Ly^2+Lz^2) for that spin in the eigenbasis.

        Parameters
        ----------
        i : int
            DESCRIPTION.
        T1 : float
            T1 relaxation time in seconds
        Thermal : bool, optional
           Flag to thermalize the system. The default is False.
        step : int, optional
            The default is None, which runs setup for this method, and returns 
            self. Providing a value for step will return a relaxation matrix
        state : int, optional
            For states in exchange, this index allows us to specify relaxation 
            for each state separately. The default is None.

        Returns
        -------
        self/np.array

        """
        if step is None:
            # Check to see if method is already here for this spin
            for k,m in enumerate(self.methods):
                if m['method']=='T1' and m['i']==i and m['state']==state:
                    self.methods.pop(k)
                    break
                
            self.methods.append({'method':'T1','i':i,'T1':T1,'Thermal':Thermal,'state':state})
            if Thermal:self.Peq=True
            return self.clear_cache()
        
        L=self.L
        
        Lx,Ly,Lz=[Ham2Super(getattr(self.Op[i],q)) for q in ['x','y','z']]
        
        M=Lx@Lx+Ly@Ly+0*Lz@Lz #This is isotropic (will not transform for 1 spin)
        
        N=len(L.H)      #Number of Hamiltonians
        n=L.H[0].shape[0]  #Dimension of Hamiltonians
        
        Lrelax=np.zeros([n**2*N,n**2*N],dtype=Defaults['ctype'])
        
        nt=(np.prod(self.Op.Mult)//(self.Op.Mult[i]))**2*(self.Op.Mult[i]-1)  #Number of T1 transitions
        
        loop=[(k,H) for k,H in enumerate(L.H)] if state is None else [(state,L.H[state])]
        
        for k,H in loop:
            U,Ui,v=H.eig2L(step)
            Mp=U@M@Ui
            
            x=np.abs(Mp-np.diag(np.diag(Mp)))
            index=np.unravel_index(np.argsort(x.reshape(x.size))[-1:-(2*nt+1):-1],Mp.shape)
            index=np.unique(np.sort(np.concatenate([index],axis=0).T,axis=-1),axis=0)

            out=np.zeros(Mp.shape,dtype=Mp.dtype)
            for i0,i1 in index:
                out[i0,i1]=-0.5/T1*Mp[i0,i1]/np.abs(Mp[i0,i1])
                out[i1,i0]=-0.5/T1*Mp[i1,i0]/np.abs(Mp[i1,i0])
                out[i0,i0]-=0.5/T1
                out[i1,i1]-=0.5/T1
            
            
            
            if Thermal:
                out+=self.Lindblad(out, v*self.h)
                
            out=Ui@out@U
            
            
            # Is this really ok?
            # out-=np.diag(np.diag(out))
            # out=np.abs(out)
            # out-=np.diag(out.sum(0))
            # TODO : The action on coherences is somehow wrong (in some/all cases?)
                
        
            Lrelax[k*n**2:(k+1)*n**2][:,k*n**2:(k+1)*n**2]=out
        
        return Lrelax
    
    
    def T2(self,i:int,T2:float,state:int=None,step:int=None):
        """
        Introduces T2 relaxation on a single spin. Relaxation is introduced in
        the eigenbasis. Transitions in the eigenbasis will be included based
        on having frequencies near the Larmor frequency of the selected spin.
        This function should be used with caution where a spin might resonate 
        far away from its Larmor frequency

        Parameters
        ----------
        i : int
            DESCRIPTION.
        T2 : float
            T2 relaxation time in seconds
        step : int, optional
            The default is None, which runs setup for this method, and returns 
            self. Providing a value for step will return a relaxation matrix
        state : int, optional
            For states in exchange, this index allows us to specify relaxation 
            for each state separately. The default is None.

        Returns
        -------
        self/np.array

        """
        if step is None:
            # Check to see if method is already here for this spin
            for k,m in enumerate(self.methods):
                if m['method']=='T2' and m['i']==i and m['state']==state:
                    self.methods.pop(k)
                    break
                
            self.methods.append({'method':'T2','i':i,'T2':T2,'state':state})
            return self.clear_cache()
        
        L=self.L
        
        
        Lz=Ham2Super(self.Op[i].z)
        
        Lx,Ly,Lz=[Ham2Super(getattr(self.Op[i],q)) for q in ['x','y','z']]
        
        # TODO why does this change behavior?
        M=Lx@Lx+Ly@Ly+Lz@Lz #This is isotropic (will not transform for 1 spin)
        # M=Lz@Lz
        
        N=len(L.H)      #Number of Hamiltonians
        n=L.H[0].shape[0]  #Dimension of Hamiltonians
        
        Lrelax=np.zeros([n**2*N,n**2*N],dtype=Defaults['ctype'])
        
        #nt=(np.prod(self.Op.Mult)//(self.Op.Mult[i]))**2*(self.Op.Mult[i]**2-self.Op.Mult[i])  #Number of T2 terms
        
        nt=(np.prod(self.Op.Mult)//(self.Op.Mult[i]))**2*(self.Op.Mult[i]-1)  #Number of T1 transitions
        
        loop=[(k,H) for k,H in enumerate(L.H)] if state is None else [(state,L.H[state])]
        for k,H in loop:
            U,Ui,v=H.eig2L(step)
            Mp=U@M@Ui
            
            x=np.abs(Mp-np.diag(np.diag(Mp)))
            index0=np.unravel_index(np.argsort(x.reshape(x.size))[-1:-(2*nt+1):-1],Mp.shape)[0]
            
            
            
            index=np.argsort(np.diag(Mp))
            out=np.zeros(Mp.shape,dtype=Mp.dtype)
            for i0 in index:
                if not(i0 in index0):
                    out[i0,i0]=-1/T2*np.sign(Mp[i0,i0])

            # out=np.zeros(Mp.shape,dtype=Mp.dtype)
            # for i0 in index:
            #     out[i0,i0]=-1/T2
                            
            out=Ui@out@U
                
            Lrelax[k*n**2:(k+1)*n**2][:,k*n**2:(k+1)*n**2]=out
        
        return Lrelax
    
    def DynamicThermal(self,step:int=None):
        """
        Attempts to thermalize dynamic processes.

        Parameters
        ----------
        step : int, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        if step is None:
            # Check to see if method is already here for this spin
            for k,m in enumerate(self.methods):
                if m['method']=='DynamicThermal':
                    self.methods.pop(k)
                    break
            self.Peq=True
                
            self.methods.append({'method':'DynamicThermal'})
            
            # Scaling factor for numerical stability
            sc=1/np.abs(self.L.expsys.Peq).max()
        
            self.sc=sc
            return self.clear_cache()
        
        
        L=self.L

        # Lex correction
        
        nH=len(L.H)
        n=L.shape[0]//nH
        rho_eq=L.rho_eq(step=step)
        rho_eq=[rho_eq[k*n:(k+1)*n] for k in range(nH)]
        
        for k in range(len(rho_eq)):
            rho_eq[k][np.abs(rho_eq[k])<1e-8]=0
            
        # Lex0=np.zeros(L.Lex.shape,dtype=Defaults['ctype'])
        # for k in range(nH):
        #     for j in range(k+1,nH):
        #         DelK=np.zeros(n,dtype=Defaults['ctype'])
        #         d=rho_eq[k]+rho_eq[j]
        #         i=d.astype(bool)
        #         DelK[i]=(L.kex[k,j]*rho_eq[j]-L.kex[j,k]*rho_eq[k])[i]\
        #             /(d[i])
        #         Lex0[j*n:(j+1)*n][:,k*n:(k+1)*n]+=np.diag(DelK)
        #         Lex0[k*n:(k+1)*n][:,j*n:(j+1)*n]-=np.diag(DelK)

        # Lex0-=np.diag(np.sum(Lex0,axis=0))
        # # Lex0=0
        # L0=L.Lcoh(step)+L.Lex+L.Lrelax+Lex0
        
        L0=L.Lcoh(step)+L.Lex+L.Lrelax
        
        
        # a,b=np.linalg.eig(L.kex)
        # i=np.argmin(np.abs(a))
        # ex_eq=b[:,i]/b[:,i].sum()
        
        # rho_eq=np.sum([L.rho_eq(step=step,Hindex=k)*eq for k,eq in enumerate(ex_eq)],axis=0)
        # rho_eq=np.tile(rho_eq,len(ex_eq))
        # recovery=-L0@rho_eq
        

        recovery=-L0@L.rho_eq(step=step)
        
        n=L.H[0].shape[0]
        N=len(L.H)
        one=np.concatenate([np.eye(n).reshape(n**2) for _ in range(N)])
        out=np.array([one*r for r in recovery])
        

        return out*self.sc
        # return (out*self.sc+Lex0)
            
        
    def plot(self,what:str='L',cmap:str=None,mode:str='log',colorbar:bool=True,
             step:int=0,block:int=None,ax=None) -> plt.axes:
        """
        Visualizes the Liouvillian matrix. Options are what to view (what) and 
        how to display it (mode), as well as colormaps and one may optionally
        provide the axis.
        
        Note, one should index the Liouvillian before running. If this is not
        done, then we jump to the halfway point of the powder average
        
        what:
        'L' : Full Relaxation matrix. Optionally specify time step
        'T1','T2','recovery', or any other defined relaxation matrices
        
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
                


        if what=='L':
            x=(self[len(self)//2] if self.L._index==-1 else self)(step)
        else:
            i=np.argwhere([m['method']==what for m in self.methods])[:,0]
            x=np.zeros(self._L.shape,dtype=Defaults['ctype'])
            for i0 in i:
                kwargs=copy(self.methods[i0])
                kwargs.pop('method')
                
                x+=getattr(self,what)(step=step,**kwargs)[self.block][:,self.block]
        
                
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
            # i=np.logical_not(x==0)
            i=np.abs(x)>np.abs(x).max()/1e8
            x[np.logical_not(i)]=0
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
        
        if block is not None:
            assert isinstance(block,int),'block must be an integer'
            bi=BlockDiagonal(self[len(self)//2].L(0))
            assert block<len(bi),f"block must be less than the number of independent blocks in the Liouville matrix ({len(bi)})"
            bi=bi[block]
            x=x[bi][:,bi]
        elif hasattr(self,'block'):
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
            
        labels=self.L.expsys.Op.Llabels
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
        
        
        
        
        
        
        
        
        
        
        
        