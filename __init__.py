# __init__.py


Defaults={}
from numpy import float64 as _rtype       #Not much gain if we reduced precision.
from numpy import complex128 as _ctype    #Also, overflow errors become common at lower precision
Defaults.update({'rtype':_rtype,'ctype':_ctype,'parallel':False,'cache':True,'ncores':None,'verbose':True})

Constants={'h':6.62607015e-34,'kB':1.380649e-23,'mub':-9.2740100783e-24/6.62607015e-34,
           'ge':2.0023193043609236,'muen':5.05078369931e-27}

from . import Tools
from .PowderAvg import PowderAvg
from .SpinOp import SpinOp
from .ExpSys import ExpSys
from .Hamiltonian import Hamiltonian
from .Liouvillian import Liouvillian
from .Sequence import Sequence
from .Rho import Rho
from .LFrf import LFrf



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
    from google.colab import output
    is_dark = output.eval_js('document.documentElement.matches("[theme=dark]")')
    if is_dark:
        import matplotlib.pyplot as plt
        x=56
        plt.rcParams["figure.facecolor"]=(x/256,x/256,x/256)
        plt.rcParams["axes.facecolor"]=(x/256,x/256,x/256)
        plt.rcParams["axes.edgecolor"]=(1,1,1)
        plt.rcParams["axes.labelcolor"]=(1,1,1)
        plt.rcParams["xtick.color"]=(1,1,1)
        plt.rcParams["ytick.color"]=(1,1,1)
        plt.rcParams["text.color"]=(1,1,1)
    