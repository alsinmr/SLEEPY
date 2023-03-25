# __init__.py


Defaults={}
from numpy import float64 as _rtype       #Not much gain if we reduced precision.
from numpy import complex128 as _ctype    #Also, overflow errors become common at lower precision
Defaults.update({'rtype':_rtype,'ctype':_ctype,'parallel':True})

from . import Tools
from .PowderAvg import PowderAvg
from .SpinOp import SpinOp
from .ExpSys import ExpSys
from .Hamiltonian import Hamiltonian
# from .Hamiltonian import Hamiltonian
from .Liouvillian import Liouvillian
from .Sequence import Sequence
from .Rho import Rho


from matplotlib.axes import Subplot as _Subplot
from matplotlib.axes import SubplotSpec as _SubplotSpec
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