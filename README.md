![logo](https://github.com/alsinmr/SLEEPY/blob/master/logo.png?raw=true)
# SLEEPY
Spin-simulation in Liouville space for rElaxation and Exchange in PYthon

Currently under development. Requires standard python packages plus numpy/scipy/matplotlib.

Under development should be taken seriously. Please contact me if you want to publish something using SLEEPY sims, so I can at least check that everything is valid for the simulations.

Known issues:
Quadrupole relaxation is not included

Relaxation towards thermal equilibrium occurs with a fixed value. For a spin that the equilibrium polarization varies throughout the rotor period, this will not be correct (relevant for DNP, and effects such as pseudocontact shift in solid-state NMR).

Frequency offsets are not handled correctly in the case that the offset is changed during the simulation

Copyright 2023 Albert Smith-Penzel

All files are copyrighted under the GNU General Public License. A copy of the license has been provided in the file LICENSE

Funding for this project provided by:

Deutsche Forschungsgemeinschaft (DFG) grant 450148812
