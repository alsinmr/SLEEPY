# __init__.py


Defaults={}
from numpy import float64 as _rtype       #Not much gain if we reduced precision.
from numpy import complex128 as _ctype    #Also, overflow errors become common at lower precision
Defaults.update({'rtype':_rtype,'ctype':_ctype,'parallel':True})

from . import Tools
from .PowderAvg import PowderAvg as PowderAvg
from .SpinOp import SpinOp as Spinop
from .ExpSys import ExpSys as ExpSys
from .Hamiltonian import Hamiltonian as Hamiltonian
# from .Hamiltonian import Hamiltonian
from .Liouvillian import Liouvillian as Liouvillian
from .Sequence import Sequence as Sequence
from .Rho import Rho as Rho


from matplotlib.axes import Subplot as _Subplot
if not(hasattr(_Subplot,'is_first_col')):
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

