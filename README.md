<img src="https://raw.githubusercontent.com/alsinmr/SLEEPY_tutorial/refs/heads/main/JupyterBook/logo_dark.png" alt="drawing" width="600"/>

# SLEEPY
Spins in Liouville space for rElaxation and Exchange in PYthon

SLEEPY is software intended for the simulation of NMR spin-systems including exchange processes and relaxation. Includes easy setup of the spin system, and allows application of arbitrary pulse sequences in both the rotating- and lab-frames.

Requires standard python packages plus numpy/scipy/matplotlib. Multiprocess is recommended, and ipywidgets are used for creating zoomable plots, e.g. Google Colab, although is not required. Note that we highly recommend using the Intel MKL libraries with numpy/scipy at least if you're using an Intel CPU. Anaconda installs these by default on Intel machines (Miniconda does not!). Note that we have not been able to run MKL with Python 12 or 13.

Benchmarked version
* Python: 3.11.13
* Numpy: 1.24.3
* Scipy: 1.10.1
* Multiprocessing: 0.70.15

Other tested versions (speed may vary):
* Python: 3.8.8, 3.9.3, 3.11.11, 3.13.2
* numpy: 1.20.1, 2.0.2, 2.2.3
* scipy: 1.6.2, 1.14.1, 1.15.2
* matplotlib: 3.3.4, 3.10.0

Testing has been performed on the various notebooks found at [https://github.com/alsinmr/SLEEPY_tutoria/](https://github.com/alsinmr/SLEEPY_tutoria/) using Jupyter Notebooks on MacOS and on myBinder.org, and also on Google Colab.

Installation is possible via pip. Please run
```
pip install sleepy-nmr
```
This will create the "sleepy" module (not sleepy-nmr!). Typical install time is ~5 seconds (install time will be significantly longer if numpy, scipy, and matplotlib are not already installed, since these will be installed as dependencies).

The SLEEPY tutorial provides extensive examples of applications of SLEEPY and is available online at 
[http://sleepy-nmr.org](http://sleepy-nmr.org)
or
[https://alsinmr.github.io/SLEEPY](https://alsinmr.github.io/SLEEPY)



These examples may be downloaded as Jupyter notebooks or run online via Google Colab (demo). 

We also provide example scripts here, in the folder 'examples', which may be run as Python3 scripts. To run these, please navigate to the examples folder and then run either

```
python3 ShortExamples.py
```
or
```
python3 Figure3.py
```
Both scripts will run a number of examples and create a folder of the corresponding figures. The former calculation takes on the order of 1 minute for 6 examples, and the latter on the order of 10 minutes for 9 examples (these correspond to Figure 3 of the corresponding manuscript).


Copyright 2025 Albert Smith-Penzel, Kai Zumpfe

All files are copyrighted under the GNU General Public License. A copy of the license has been provided in the file LICENSE

Funding for this project provided by:

Deutsche Forschungsgemeinschaft (DFG) grant 450148812
