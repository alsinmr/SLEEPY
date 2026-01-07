# __init__.py


Defaults={}
from numpy import float64 as _rtype       #Not much gain if we reduced precision.
from numpy import complex128 as _ctype    #Also, overflow errors become common at lower precision
Defaults.update({'rtype':_rtype,'ctype':_ctype,'parallel':False,'cache':True,'MaxPropCache':10,
                 'ncores':None,'verbose':True,'zoom':False,
                 'Colab':False,'Binder':False,'parallel_chunk_size':None,
                 'Hz_gyro_sign_depend':True})

import os as _os

_h=6.62607015e-34
Constants={'h':_h,  #Planck constant, Js
           'kB':1.380649e-23, #Boltzmann constant, J/K
           'mub':-9.2740100783e-24/_h, # Bohr Magneton Hz/T
           'ge':2.0023193043609236, #g factor of free electron, unitless
           'mun':5.05078369931e-27/6.62607015e-34, #Nuclear magneton, Hz/T
           'mu0':1.256637e-6  #Permeability of vacuum [T^2m^3/J]
           }

#%% Load version info
version_info=''
previous_version_info={}
__version__=None
key=None

file=_os.path.join(_os.path.split(__file__)[0],'version_info.txt')
if _os.path.exists(file):
    with open(file,'r') as f:
        for line in f:
            if 'VERSION' in line:
                if __version__ is None:
                    __version__=line.strip().split('VERSION ')[1]
                    continue
                else:
                    key=line.strip().split('VERSION ')[1]
                    previous_version_info[key]=''
            if key is None and __version__ is not None:
                version_info=version_info+'\n'+line.strip()
            elif __version__ is not None:
                previous_version_info[key]=previous_version_info[key]+'\n'+line.strip()
    

#%% Print version info

def print_version_info():
    file=_os.path.join(_os.path.split(__file__)[0],'first_run.txt')
    if _os.path.exists(file):
        with open(file,'r') as f:
            if __version__ == f.readline().strip():
                return
            
    print(f'You are running SLEEPY version {__version__}\n')
    print('Please read the version notes (also found in SLEEPY.version_info):')
    print(version_info)
    print('\nPrevious version info is found in SLEEPY.previous_version_info')
    with open(file,'w') as f:
        f.write(__version__)
        
print_version_info()
    



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

        

if 'USER' in _os.environ and _os.environ['USER']=='jovyan':
    Defaults['Binder']=True
    Defaults['zoom']=True
    

def saveDefaults():
    file=_os.path.join(_os.path.split(__file__)[0],'Defaults.txt')
    with open(file,'w') as f:
        for key,value in Defaults.items():
            if key in ['rtype','ctype']:continue  #Don't save the data types
            f.write(f'{key} : {str(value)}\n')
            
def loadDefaults():
    file=_os.path.join(_os.path.split(__file__)[0],'Defaults.txt')
    if not(_os.path.exists(file)):return
    
    print('Loading Defaults from file')
    
    with open(file,'r') as f:
        for line in f:
            if ' : ' in line:
                key,value=line.strip().split(' : ')
                if value.isdigit():value=int(value)
                if value=='True':value=True
                if value=='False':value=False
                if value=='None':value=None
                Defaults[key]=value
                
loadDefaults()

del(f,line,file,print_version_info)

    
    