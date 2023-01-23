{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "e9dff832",
   "metadata": {},
   "source": [
    "# Example 1: 2-spin exchange"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "366e1c47",
   "metadata": {},
   "source": [
    "In this example, we look at a simple case of a single spin with its chemical shift modulated by an exchange process. Then, we will see how its behavior changes as a function of correlation time and population"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0f9b2793",
   "metadata": {},
   "source": [
    "### Installs and Imports"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "23de63f5",
   "metadata": {},
   "outputs": [],
   "source": [
    "#!!pip install git+https://github.com/alsinmr/pyRelaxSim.git\n",
    "import sys\n",
    "sys.path.append('/Users/albertsmith/Documents/GitHub')\n",
    "import pyRelaxSim as RS\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "587c4095",
   "metadata": {},
   "source": [
    "## Build the spin system"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2ab6d368",
   "metadata": {},
   "source": [
    "For relaxation induced by exchange, we always build the spin system with at least two different sets of interactions. Not all interactions must change, but at least one interaction should be different– otherwise no relaxation will occure. Note that best-practice is to build the first spin-system, and copy it and only edit the parameters that are changed in the second spin-system.\n",
    "\n",
    "pyRelaxSim takes the main experimental parameters (excepting rf fields) upon initialization of a spin-system, and then interactions are added afterwards."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "cc1b913c",
   "metadata": {},
   "outputs": [],
   "source": [
    "ex0=RS.ExpSys(v0H=600,Nucs='13C')     #1-spin system at 600 MHz (14.1 T)\n",
    "ex0.set_inter(Type='CS',i=0,ppm=0)    #Chemical shift for spin 0 at 0 ppm\n",
    "ex1=ex0.copy()   #Copies the spin-system\n",
    "ex1.set_inter(Type='CS',i=0,ppm=10)   #Change of chemical shift by 10 ppm (~1500 Hz)  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "e6519897",
   "metadata": {},
   "outputs": [],
   "source": [
    "H0=RS.Hamiltonian(ex0)   #Hamiltonian with CS=0 ppm\n",
    "H1=RS.Hamiltonian(ex1)   #Hamiltonian with CS=10 ppm\n",
    "L=RS.Liouvillian((H0,H1)) #Liouvillian with both Hamiltonians"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "85ca68bf",
   "metadata": {},
   "source": [
    "## Define initial density operator and detection operator"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "29a66b13",
   "metadata": {},
   "outputs": [],
   "source": [
    "rho=RS.Rho(rho0='13Cx',detect='13Cp',L=L)   #Specify by Nucleus type and operator type"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a3970cd8",
   "metadata": {},
   "source": [
    "## Calculate Spectrum as Function of Correlation Time"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "64b06bd6",
   "metadata": {},
   "source": [
    "In this case, there isn't really a sequence. We just start the magnetization on Sx, propagate it, and observe at each propagation step."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "4f7d4e66",
   "metadata": {},
   "outputs": [],
   "source": [
    "dt=1/1500/100\n",
    "tc0=np.logspace(-5,-2,4)\n",
    "tc=tc0[2]\n",
    "L.kex=np.array([[-1/(2*tc),1/(2*tc)],[1/(2*tc),-1/(2*tc)]])\n",
    "U=L.U(t0=0,tf=dt)  #Propagator for a time step dt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "41a683d6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "6.666666666666667e-06"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "U.Dt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1a15c480",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
