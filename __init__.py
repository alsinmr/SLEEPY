# __init__.py


Defaults={}
from numpy import float64 as _rtype       #Not much gain if we reduced precision.
from numpy import complex128 as _ctype    #Also, overflow errors become common at lower precision
Defaults.update({'rtype':_rtype,'ctype':_ctype,'parallel':True})

# from . import Tools
from .PowderAvg import PowderAvg
from .SpinOp import SpinOp
from .ExpSys import ExpSys
from .Hamiltonian import Hamiltonian
from .Liouvillian import Liouvillian
from .Sequence import Sequence
from .Rho import Rho


import matplotlib as _matplotlib
if not(hasattr(_matplotlib.axes._subplots.Subplot,'is_first_row')):
    def is_first_row(self):
        return self.get_subplotspec().is_first_row()
    _matplotlib.axes._subplots.Subplot.is_first_row=is_first_row
    def is_first_col(self):
        return self.get_subplotspec().is_first_col()
    _matplotlib.axes._subplots.Subplot.is_first_col=is_first_col
    def is_last_row(self):
        return self.get_subplotspec().is_last_row()
    _matplotlib.axes._subplots.Subplot.is_last_row=is_last_row
    def is_last_col(self):
        return self.get_subplotspec().is_last_col()
    _matplotlib.axes._subplots.Subplot.is_last_col=is_last_col