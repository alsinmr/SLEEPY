#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 22:32:33 2023

@author: albertsmith
"""

class DetRho0():
    def __init__(self,rho0,detect):
        """
        Creates an object that contains both the initial density matrix and
        the detector matrix. One may then apply propagators to the density
        matrix and detect the magnetization.

        Parameters
        ----------
        rho0 : Spinop or, str, optional
            Initial density matrix, specify by string or the operator itself. 
            Operators may be found in expsys.Op.
        detect : Detection matrix, specify by string or the operator itself. 
            Operators may be found in expsys.Op. Multiple detection matrices
            may be specified by providing a list of operators.

        Returns
        -------
        None.

        """
        
        self.rho0=rho0
        self.detect=detect
        