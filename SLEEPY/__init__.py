# __init__.py


Defaults={}
from numpy import float64 as _rtype       #Not much gain if we reduced precision.
from numpy import complex128 as _ctype    #Also, overflow errors become common at lower precision
Defaults.update({'rtype':_rtype,'ctype':_ctype,'parallel':False,'cache':True,'MaxPropCache':10,
                 'ncores':None,'verbose':True,'zoom':False,
                 'Colab':False,'Binder':False,'parallel_chunk_size':None})

_h=6.62607015e-34
Constants={'h':_h,  #Planck constant, Js
           'kB':1.380649e-23, #Boltzmann constant, J/K
           'mub':-9.2740100783e-24/_h, # Bohr Magneton Hz/T
           'ge':2.0023193043609236, #g factor of free electron, unitless
           'mun':5.05078369931e-27/6.62607015e-34, #Nuclear magneton, Hz/T
           'mu0':1.256637e-6  #Permeability of vacuum [T^2m^3/J]
           }


from . import Tools
from .PowderAvg import PowderAvg
from .SpinOp import SpinOp
from .ExpSys import ExpSys
from .Hamiltonian import Hamiltonian
from .Liouvillian import Liouvillian
from .Sequence import Sequence
from .Rho import Rho
from .LFrf import LFrf

from .plot_tools import set_dark


from matplotlib.axes import Subplot as _Subplot
from matplotlib.gridspec import SubplotSpec as _SubplotSpec
if hasattr(_SubplotSpec,'is_first_col'):
    def _fun(self):
        return self.get_subplotspec().is_first_col()
    _Subplot.is_first_col=_fun
    def _fun(self):
        return self.get_subplotspec().is_first_row()
    _Subplot.is_first_row=_fun
    def _fun(self):
        return self.get_subplotspec().is_last_col()
    _Subplot.is_last_col=_fun
    def _fun(self):
        return self.get_subplotspec().is_last_row()
    _Subplot.is_last_row=_fun

import sys as _sys
if 'google.colab' in _sys.modules:
    Defaults['Colab']=True
    Defaults['zoom']=True
    from google.colab import output
    is_dark = output.eval_js('document.documentElement.matches("[theme=dark]")')
    if is_dark:set_dark()

        
import os as _os
if 'USER' in _os.environ and _os.environ['USER']=='jovyan':
    Defaults['Binder']=True
    Defaults['zoom']=True
    