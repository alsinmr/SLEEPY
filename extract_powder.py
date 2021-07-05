#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 09:57:55 2021

@author: albertsmith
"""

import os

reading=False
with open('powder.txt','r') as f:
    for line in f:
        if reading:
            if ';' in line:
                reading=False
            else:
                line=line.replace('}','').replace('{','').replace(',','')
                fr.write(line)
        elif 'CRYSTALLITE' in line:
            reading=True
            fr=open(os.path.join('PowderFiles',line.split(' ')[1][:-8]+'.txt'),'w')
            
            
                