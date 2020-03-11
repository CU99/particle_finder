#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 22:52:41 2020

@author: phurbc
"""

import numpy as np
from scipy import special
import pickle
import time
#from mayavi import mlab
import ase
import quippy



#Constants that can be changed. 
#L = length of box (periodic boundary conditions)
#rec = 1/L
#l = orbital quantum number, d = cutoff distance for nearest neighbour atoms
#c = threshold for 2 particles to be considered 'connected'
#t = threshold value for number of connections for solid-like particle
rho = 1
L = 7.93701/(rho**(1/3))
d = 1.5/(rho**(1/3))


#function to find list of parameters for this configuration of coordinates
#takes in coords, l quantum number, L (length of box),rec = 1/L
#and cut-off distance for nearest-neighbour bonds d
def find_descriptor(coords):    
    #put coordinates into ASE array
    coords_array=np.stack(coords)
    labels=['Ar' for x in range(0,500)]
    my_atoms=ase.Atoms(labels, positions=coords_array)
    #set size of box
    my_atoms.set_cell([L, L, L])
    #Periodic boundary conditions
    my_atoms.pbc=True
    #define descriptor
    desc=quippy.descriptors.Descriptor("soap cutoff={} cutoff_transition_width=0.0 atom_sigma=0.25 n_max =4 l_max=6".format(d))
    descriptors=desc.calc(my_atoms)['data']
    #write parameters to seperate file to be used for machine learning
    with open('liquid1.8_soap_n=4.txt','ab') as file:
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
    while True:
        atoms = []
        coordinates = []
        try:
            n_atoms = int(xyz.readline())
        except:
            print("End of file reached")
            break
        
        comment = xyz.readline()
        #read through files and puts coordinates into array
        for line in xyz:
            atom,x,y,z = line.split()
            atoms.append(atom)
            coordinates.append(np.array([float(x), float(y), float(z)]))
            if len(coordinates)==n_atoms:
                break
        s1 = time.time()
        descriptors = find_descriptor(coordinates)
        e1 = time.time()
        t1 = e1-s1
        total_descriptor_time+=t1
        print('Processed configuration {}. Timestamp = {}'.format(j, comment))
        j+=1
    xyz.close()
    end = time.time()
    total_time = end-start
    print(total_time)
    print('Total time to constructs descriptors = {} s.'.format(total_descriptor_time))
    return 


print(read_file("liquidT1.8_1.xyz",L,d))