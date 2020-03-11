#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 14:51:36 2020

@author: phurbc
"""

import numpy as np
from scipy import special
import pickle
import time
#from mayavi import mlab
from ase.io import read
import quippy
import matplotlib.pyplot as plt



#Constants that can be changed. 
#L = length of box (periodic boundary conditions)
#rec = 1/L
#l = orbital quantum number, d = cutoff distance for nearest neighbour atoms
#c = threshold for 2 particles to be considered 'connected'
#t = threshold value for number of connections for solid-like particle
rho = 1
L = 7.93701/(rho**(1/3))
d = 4.25/(rho**(1/3))


#function to find list of parameters for this configuration of coordinates
#takes in coords, l quantum number, L (length of box),rec = 1/L
#and cut-off distance for nearest-neighbour bonds d
def find_descriptor(file):    
    #read file into ASE atoms object
    my_atoms=read(file)
    #define descriptor
    desc=quippy.descriptors.Descriptor("soap cutoff=4.25 cutoff_transition_width=0.0 atom_sigma=0.7 n_max =8 l_max=6 n_species=2 species_Z={32 52}")
    descriptors=desc.calc(my_atoms)['data']

    #write parameters to seperate file to be used for machine learning
    with open('quenched_Ge+Te_soap_n=8_l=6_sigma=0.7_run0-19.txt','ab') as file:
        pickle.dump(descriptors,file)
    return descriptors



#function which takes in files and some parameters,
#and outputs number of solid-like particles in each configuration
def read_file(file,L,d):
    start = time.time()
    total_descriptor_time =0
    j=1
    xyz = open(file)
    #loop over all 101 configurations
    while j<=1:
        s1 = time.time()
        desc = find_descriptor(file)
        e1 = time.time()
        t1 = e1-s1
        total_descriptor_time+=t1
        print('Processed configuration {}'.format(j))
        j+=1
    xyz.close()
    end = time.time()
    total_time = end-start
    print(total_time)
    print('Total time to constructs descriptors = {} s.'.format(total_descriptor_time))
    return 


print(read_file("quenched_0.xyz",L,d))