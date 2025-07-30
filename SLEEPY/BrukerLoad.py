import os
import re
import numpy as np
from typing import Union, List, Dict


class BrukerSpec():
    def __init__(self,location=None,Range=None,proc_no=1,spec=None):
        if spec is None:
            opt={'range':Range,'proc_no':proc_no}
            spec=load_bruker_spec(location,opt=opt)
            
        self._spec=spec
        
        setattr(self,'par',spec['par'])
        
        for key,value in spec.items():
            setattr(self,key,value)
            if key[0]=='f':
                dim=key[1]
                setattr(self,f'f{dim}Hz',value*self.par[f'd{dim}']['SF'])
        self.S=self.S.T

            
    def plot(self,axis='ppm',par=None,ax=None,norm:bool=False,**kwargs):
        """
        Plots 1D and 2D spectra

        Parameters
        ----------
        axis : TYPE, optional
            DESCRIPTION. The default is 'ppm'.
        par : TYPE, optional
            DESCRIPTION. The default is None.
        norm: bool, optional
            Normalize the spectrum. The default is False
        ax : TYPE, optional
            DESCRIPTION. The default is None.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        ax : TYPE
            DESCRIPTION.

        """
        if self.ndim==1:
            ax=plot1D(self,axis=axis,ax=ax,norm=norm,**kwargs)
        elif self.ndim==2:
            ax=quik_2Dplot(self,par=par,ax=ax,norm=norm,**kwargs)
        return ax
    
    @property
    def shape(self):
        return self.S.shape
    
    
    def peaks(self,cutoff:float=.1,unit='ppm',mode='+'):
        """
        Finds peaks in the spectrum. 

        Parameters
        ----------
        cutoff : float, optional
            Minimum peak height relative to the spectrum max. The default is .1.
        unit : str, optional
            Unit to return peak positions: 'Hz' or 'ppm' (default)
        mode : str, optional
            Specify "+", "-", or "both" for positive, negative, or both types
            of peaks


        Returns
        -------
        list of tuples
        Peak positions in ppm (u)

        """
        
        S0=self.S.real
        
        peaks=np.ones(S0.shape,dtype=bool)
        for k in range(self.ndim):
            S=np.swapaxes(S0,0,k)
            if mode=='+':
                peaks0=np.logical_and(np.diff(S[:-1],axis=0)>0,np.diff(S[1:],axis=0)<0)
            elif mode=='-':
                peaks0=np.logical_and(np.diff(S[:-1],axis=0)<0,np.diff(S[1:],axis=0)>0)
            else:
                peaks0=np.logical_or(np.logical_and(np.diff(S[:-1],axis=0)>0,np.diff(S[1:],axis=0)<0),
                                     np.logical_and(np.diff(S[:-1],axis=0)<0,np.diff(S[1:],axis=0)>0))
            peaks0=np.concatenate((np.zeros([1,*S.shape[1:]]), peaks0, np.zeros([1,*S.shape[1:]])),axis=0)
            
            peaks=np.logical_and(peaks,np.swapaxes(peaks0,0,k))
        
        maxS=max(S0.max(),np.abs(S0.min()))
        
        if mode=='-':
            peaks[S0>-maxS*cutoff]=False
        elif mode=='+':
            peaks[S0<maxS*cutoff]=False
        else:
            peaks[np.abs(S0)<maxS]=False
            
        axes=[getattr(self,f'f{k+1}Hz' if unit=='Hz' else f'f{k+1}') for k in range(self.ndim)]
        
        i=np.argwhere(peaks).T
        
        peaks=np.array([a[i0] for a,i0 in zip(axes,i)])
            
        return peaks
        

def get_param_bruker(filename: str, params: Union[str, List[str]]) -> Union[str, float, List[float], Dict[str, Union[str, float, List[float]]]]:
    """
    Read one or more parameters from a Bruker TopSpin processed parameter file (e.g., procs).
    """
    with open(filename, 'r', encoding='latin-1') as f:
        file_contents = f.read()

    def extract_param(param_name: str):
        match = re.search(rf'##\${re.escape(param_name)}=([^\n]*)', file_contents)
        if not match:
            return None
        value_line = match.group(1).strip()
        if value_line.startswith('('):
            pattern = rf'##\${re.escape(param_name)}=.*?\n(.*?)(?=\n##|\Z)'
            vector_match = re.search(pattern, file_contents, re.DOTALL)
            if vector_match:
                vector_str = vector_match.group(1).replace('\n', ' ')
                try:
                    return [float(v) for v in vector_str.split()]
                except ValueError:
                    return vector_str.strip()
        else:
            try:
                return float(value_line)
            except ValueError:
                return value_line.strip()

    if isinstance(params, list):
        return {param: extract_param(param) for param in params}
    else:
        return extract_param(params)

def load_bruker_spec(location: str, opt: Union[Dict, List[float], None] = None):
    """
    Load processed Bruker TopSpin NMR spectra (1D–3D) with optional range/phase control.
    Returns a dictionary with spectrum data, metadata, and axes.
    """
    # Parse input options
    if isinstance(opt, dict):
        range_ = opt.get("range", None)
        proc_no = opt.get("proc_no", 1)
        phase = str(opt.get("phase", "n")).lower().startswith("y")
    elif isinstance(opt, list) or isinstance(opt, np.ndarray):
        range_ = opt
        proc_no = 1
        phase = False
    else:
        range_ = None
        proc_no = 1
        phase = False

    # Resolve processing folder
    if os.path.isfile(os.path.join(location, 'procs')):
        folder = os.path.join(location, '')
    elif os.path.isfile(os.path.join(location, f'pdata/{proc_no}/procs')):
        folder = os.path.join(location, f'pdata/{proc_no}/')
    else:
        raise FileNotFoundError("Location incorrect or required Bruker files are missing.")

    # Determine dimensionality
    files = os.listdir(folder)
    ndim = 0
    for f in files:
        if re.match(r'proc\d*s$', f):
            dim = 1 if f == 'procs' else int(re.findall(r'\d+', f)[0])
            ndim = max(ndim, dim)
    if ndim == 0:
        raise ValueError("No proc files found to determine dimensionality.")

    data_file = f"{ndim}{'r'*ndim}"
    if not os.path.isfile(os.path.join(folder, data_file)):
        raise FileNotFoundError(f"Processed data file '{data_file}' not found.")

    # Load parameter data
    params = ['SI', 'SF', 'SW_p', 'AXNUC', 'OFFSET', 'XDIM']
    par_proc = ['SF', 'STSI', 'SW_p', 'TDeff', 'FTSIZE', 'WDW', 'SSB', 'GB', 'LB', 'PHC1', 'SI']
    blk = []
    Nuc = []
    td = []
    f_axes = []
    for k in range(ndim):
        pfile = 'procs' if k == ndim - 1 else f'proc{ndim - k}s'
        out = get_param_bruker(os.path.join(folder, pfile), params)
        blk.append(int(out.get('XDIM', out['SI'])))
        Nuc.append(out['AXNUC'].strip('<>'))
        td.append(int(out['SI']))
        axis = np.linspace(out['OFFSET'] - out['SW_p'] / out['SF'], out['OFFSET'], int(out['SI']), endpoint=False)
        f_axes.append(axis + (axis[1] - axis[0]))

    blk = [b if b > 0 else t for b, t in zip(blk, td)]

    spec = {'S': None}
    for i, (f, n) in enumerate(zip(f_axes, Nuc), 1):
        spec[f"f{i}"] = f
        spec[f"Nuc{i}"] = n

    # Read processing info
    spec['par'] = {}
    for k in range(ndim):
        pfile = 'procs' if k == ndim - 1 else f'proc{ndim - k}s'
        out = get_param_bruker(os.path.join(folder, pfile), par_proc)
        dkey = f'd{k+1}'
        spec['par'][dkey] = {
            'WDW': out['WDW'], 'SSB': out['SSB'], 'GB': out['GB'], 'LB': out['LB'],
            'AQ': 1/(out['SW_p']*2*(out['FTSIZE']-1)/(out['STSI']-1))*(out['TDeff']-2),
            'SI': out['FTSIZE'], 'TD': out['TDeff'], 'SF': out['SF'], 'PHC1': out['PHC1'],
            'SWH0': out['SW_p'] * (out['FTSIZE']-1)/out['STSI']
        }

    # Load spectrum
    byte_order = '<' if get_param_bruker(os.path.join(folder, 'procs'), 'BYTORDP') == 0 else '>'
    data_path = os.path.join(folder, data_file)
    data = np.fromfile(data_path, dtype=byte_order + 'i4')
    data = data.reshape(td[::-1], order='F')
    spec['S'] = np.flip(data, axis=tuple(range(ndim)))

    # Load title if exists
    title_path = os.path.join(folder, 'title')
    if os.path.isfile(title_path):
        with open(title_path, 'r', encoding='latin-1') as f:
            spec['title'] = f.read().strip()

    # Truncate if range is given
    if range_ is not None:
        for k in range(ndim):
            fkey = f"f{k+1}"
            lo = np.abs(spec[fkey] - range_[2*k]).argmin()
            hi = np.abs(spec[fkey] - range_[2*k+1]).argmin()
            idx = slice(min(lo, hi), max(lo, hi)+1)
            spec[fkey] = spec[fkey][idx]
            spec['S'] = np.take(spec['S'], indices=range(idx.start, idx.stop), axis=k)

    # Calculate SW and SWH
    for k in range(ndim):
        fkey = f"f{k+1}"
        dkey = f'd{k+1}'
        spec['par'][dkey]['SW'] = np.abs(spec[fkey][-1] - spec[fkey][0])
        spec['par'][dkey]['SWH'] = spec['par'][dkey]['SW'] * spec['par'][dkey]['SF']
        spec['par'][dkey]['SIp'] = len(spec[fkey])

    # Apply scaling
    scale = 2 ** get_param_bruker(os.path.join(folder, 'procs'), 'NC_proc')
    spec['S'] = spec['S'] * scale

    # (Optional) Load phase data (not included here for brevity but can be added)
    spec['ndim']=ndim

    return spec


# Converted from MATLAB: quik_2Dplot
import numpy as np
import matplotlib.pyplot as plt


def labels(Nuc,axis='ppm'):
    mass0="".join(re.findall('[0-9]',Nuc))
    mass=fr'$^{{{mass0}}}$'
    nuc="".join(re.findall('[a-z]|[A-Z]',Nuc))
    
    return fr'$\delta$({mass}{nuc}) / {axis}'

def plot1D(spec,axis='ppm', norm:bool=False,ax=None,**kwargs):
    """
    Plots a 1D spectrum

    Parameters
    ----------
    spec : TYPE
        DESCRIPTION.
    axis : str
        Type of axis (ppm, Hz, kHz)
    ax : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    if ax is None:
        ax=plt.subplots()[1]
        
    
    
    assert axis in ['ppm','Hz','kHz'],"axis must be ppm, Hz, or kHz"
    
    sc=1/np.abs(spec.S).max() if norm else 1
    
    if axis=='ppm':
        ax.plot(spec.f1,spec.S.real*sc,**kwargs)
        ax.set_xlabel(labels(spec.Nuc1,axis))
    elif axis=='Hz':
        ax.plot(spec.f1Hz,spec.S.real*sc,**kwargs)
        ax.set_xlabel(labels(spec.Nuc1,axis))
    elif axis=='kHz':
        ax.plot(spec.f1Hz/1000,spec.S.real*sc,**kwargs)
        ax.set_xlabel(labels(spec.Nuc1,axis))
    
        
    ax.set_xlim(ax.get_xlim()[::-1])
    
    return ax
    
    
    

def quik_2Dplot(spec, par=None, norm:bool=False, ax=None, **kwargs):
    """
    Plot 2D NMR spectrum from 'spec' structure and optional 'par' plotting parameters.

    Parameters:
        spec: dict with keys 'S', 'f1', 'f2', and optionally 'Nuc1', 'Nuc2', 'title'
        par: dict with optional keys:
            - n_contours: Number of contour levels
            - cutoff: Minimum intensity threshold (float or [low, high])
            - mode: 'log' or 'linear' (default 'log')
            - colormap: Custom colormap as (n, 3) array
            - range: [f1min, f1max, f2min, f2max] to restrict plot axes
            - dim: '12' to flip axes
            - scaled: 'y' if contours should match previous plot
    Returns:
        Contour handle(s)
    """
    if ax is None:
        ax=plt.subplots()[1]
    
    if isinstance(spec.S,dict):
        S = np.array(spec['S'])
        f1 = np.array(spec['f1'])
        f2 = np.array(spec['f2'])
        Nuc1 = spec.get('Nuc1', '')
        Nuc2 = spec.get('Nuc2', '')
        title_str = spec.get('title', '')
    else:
        S = np.array(spec.S)
        f1 = np.array(spec.f1)
        f2 = np.array(spec.f2)
        Nuc1 = spec.Nuc1
        Nuc2 = spec.Nuc2
        title_str = spec.title
        
    sc=1/np.abs(spec.S).max() if norm else 1

    # --- Parameter defaults --- #
    n_contours = par.get('n_contours', None) if par else None
    cutoff = par.get('cutoff', None) if par else None
    mode = par.get('mode', 'log') if par else 'log'
    colormap = par.get('colormap', None) if par else None
    axis_range = None
    scaled = par.get('scaled', 'n') if par else 'n'
    dim = par.get('dim', '21') if par else '21'

    if dim == '12':
        S = S.T
        f1, f2 = f2, f1
        Nuc1, Nuc2 = Nuc2, Nuc1

    # --- Axis Range Clipping --- #
    if par and 'range' in par:
        f1min, f1max, f2min, f2max = par['range']
        idx1 = (f1 >= f1min) & (f1 <= f1max)
        idx2 = (f2 >= f2min) & (f2 <= f2max)
        S = S[np.ix_(idx1, idx2)]
        f1 = f1[idx1]
        f2 = f2[idx2]
        axis_range = [f2[0], f2[-1], f1[0], f1[-1]]
    else:
        axis_range = [f2[0], f2[-1], f1[0], f1[-1]]

    # --- Cutoff threshold --- #
    max_S = np.max(np.abs(S))
    if cutoff is None:
        temp = S.flatten()
        rmsp = np.sqrt(np.mean(temp[temp > 0] ** 2))
        rmsn = np.sqrt(np.mean(temp[temp < 0] ** 2))
        cutoff = min(rmsp, rmsn) * 4 / max_S

    # --- Contour levels --- #
    if mode.lower() == 'log':
        nc = n_contours if n_contours else 15
        levellistp = np.logspace(np.log10(cutoff * max_S), np.log10(max_S), nc)
        levellistm = -levellistp[::-1]
    else:
        nc = n_contours if n_contours else 50
        if isinstance(cutoff, (list, tuple)) and len(cutoff) == 2:
            low, high = sorted(cutoff)
            S[S > high * max_S] = high * max_S
            S[S < -high * max_S] = -high * max_S
            S[np.abs(S) < low * max_S] = 0
        else:
            S[np.abs(S) < cutoff * max_S] = 0

    # --- Plotting --- #
    fig, ax = plt.gca(), plt.gca()
    if mode.lower() == 'log':
        cs1 = ax.contour(f2, f1, S*sc, levels=levellistp, colors='r',**kwargs)
        cs2 = ax.contour(f2, f1, S*sc, levels=levellistm, colors='b',**kwargs)
        handle = [cs1, cs2]
    else:
        if colormap is not None:
            plt.set_cmap(colormap)
        cs = ax.contour(f2, f1, S*sc, levels=nc,**kwargs)
        handle = cs

    ax.set_xlim(axis_range[0], axis_range[1])
    ax.set_ylim(axis_range[2], axis_range[3])
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    # --- Axis labels --- #
    def format_nucleus(nuc):
        num = ''.join([c for c in nuc if c.isdigit()])
        lett = ''.join([c for c in nuc if not c.isdigit()])
        return f"$^{{{num}}}${lett}"

    ax.set_xlabel(f"$\\delta$ ({format_nucleus(Nuc2)}) / ppm")
    ax.set_ylabel(f"$\\delta$ ({format_nucleus(Nuc1)}) / ppm")

    if title_str:
        ax.set_title(title_str[:40])

    plt.tight_layout()
    return ax


def clip_spec_nD(spec0, range_vals):
    """
    Truncates an n-dimensional spectrum given a range.

    Parameters:
        spec0 (dict): Spectrum with keys 'S', 'f1', 'f2', etc., and optionally 'par'
        range_vals (list): Flat list of lower and upper bounds [LB1, UB1, LB2, UB2, ...]

    Returns:
        spec (dict): Clipped spectrum
    """
    ndim = len([dim for dim in spec0['S'].shape if dim > 1])
    
    if len(range_vals) != ndim * 2:
        raise ValueError("range has wrong number of elements")

    index = [0] * (2 * ndim)
    spec = {k: v for k, v in spec0.items() if k != 'S'}  # copy without 'S'

    for k in range(1, ndim + 1):
        fk = np.array(spec0[f"f{k}"])
        lb = range_vals[2 * (k - 1)]
        ub = range_vals[2 * (k - 1) + 1]

        index[2 * (k - 1)] = np.argmin(np.abs(fk - lb))
        index[2 * (k - 1) + 1] = np.argmin(np.abs(fk - ub))

        start = index[2 * (k - 1)]
        end = index[2 * (k - 1) + 1]
        if start > end:
            start, end = end, start  # ensure correct order

        spec[f"f{k}"] = fk[start:end+1]

        if 'par' in spec and f"d{k}" in spec['par']:
            d = spec['par'][f"d{k}"]
            d['SIp'] = len(spec[f"f{k}"])

            if 'SF' in d:
                d['SWH'] = abs(spec[f"f{k}"][-1] - spec[f"f{k}"][0]) * d['SF']
            elif 'SWH' in d:
                d.pop('SWH', None)

            d['SW'] = abs(spec[f"f{k}"][-1] - spec[f"f{k}"][0])

            if 'user' in d:
                t0 = np.linspace(0, d['AQ'], len(d['user']))
                t = np.linspace(0, d['AQ'], d['SIp'])
                d['user'] = np.interp(t, t0, d['user'])

    # Truncate S using slices
    slices = []
    for i in range(ndim):
        start = index[2 * i]
        end = index[2 * i + 1] + 1
        slices.append(slice(min(start, end-1), max(start, end)))

    spec['S'] = spec0['S'][tuple(slices)]
    return spec
