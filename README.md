# SLEEP
Spin-simulation in Liouville space for rElaxation and Exchange in PYthon

Currently under development. Requires standard python packages plus numpy/scipy/matplotlib.

Next steps:
1) Deal with powder averaging. Currently, only the JCP59 powder average is really valid because the others do not include gamma angles.
Therefore, we need to add Gamma angles to these averages, and implement some kind of recycling scheme for propagators.
2) Move propagator calculation machinery from the Liouvillian/sequence to the propagators themselves. This could allow one to skip 
calculating the propagators when being multiplied by the density matrix. Then, we would not calculate the matrix exponentials but only 
their influence on the density matrix. This should be faster if the propagators do not get recycled.



Copyright 2023 Albert Smith-Penzel

All files are copyrighted under the GNU General Public License. A copy of the license has been provided in the file LICENSE

Funding for this project provided by:

Deutsche Forschungsgemeinschaft (DFG) grant SM 576/1-1
