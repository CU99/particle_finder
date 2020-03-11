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
import ase.neighborlist as nl



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
    symbols =my_atoms.get_chemical_symbols()
    #find nearest neighbour vectors
    FirstAtom, SecondAtom, vects = nl.neighbor_list(['i','j','D'], my_atoms, 4.25, self_interaction=False)
    #ensure periodic boundary conditions are kept
    cell = my_atoms.get_cell()
    newvects = nl.mic(vects, cell, pbc=[True, True, True])
    #work out symmetry functions for each particle
    descriptors=[]
    for i in range(len(symbols)):
        #find indices of neighburs of i
        indices = [a for a, x in enumerate(FirstAtom) if x == i]
        #find vectors between i and its neighbours
        neigh_vec=np.array([newvects[b] for b in indices])       
        #sum each function over all the neighbours
        f1=0
        f2=0
        s1=0
        s2=0
        s3=0
        t1=0
        t2=0
        t3=0
        t4=0
        fo1=0
        fo2=0
        fo3=0
        fo4=0
        fo5=0
        ff1=0
        ff2=0
        ff3=0
        ff4=0
        ff5=0
        ff6=0
        bfo=0
        #for each neighbouring particle to particle i: 
        for j, n in enumerate(neigh_vec):
            #normalise the vector connecting i to j
            vector = n/np.linalg.norm(n)
            #split into x, y and z
            x,y,z=vector[0],vector[1],vector[2]
            #Work out all symmetry functions
            first_1 = 0.5*x +0.866025*y 
            first_2 = z
            f1+=first_1
            f2+=first_2
            second_1 = 0.540062*x**2 -0.801784*x*y +0.0771517*y**2 -0.617213*z**2 
            second_2 = 0.92582*x*y + 0.534522*y**2 -0.534522*z**2 
            second_3 = 0.707107*x*z + 1.22474*y*z
            s1+=second_1
            s2+=second_2
            s3+=second_3
            third_1 = 0.53619*x**3 + 0.121136*x**2*y -1.32882*x*y**2 + 0.121136*y**3 -0.279751*x*z**2 -0.484544*y*z**2
            third_2 = 0.312772*x**2*y + 0.722315*x*y**2 + 0.312772*y**3 -0.722315*x*z**2 -1.25109*y*z**2
            third_3 = 1.12916*x**2*z -1.15045*x*y*z + 0.464948*y**2*z -0.531369*z**3
            third_4 = 1.78227*x*y*z + 1.02899*y**2*z -0.342997*z**3
            t1+=third_1
            t2+=third_2
            t3+=third_3
            t4+=third_4
            fourth_1 = 0.285044*x**4 + 0.542539*x**3*y -0.432264*x**2*y**2 -0.97657*x*y**3 + 0.15975*y**4 -1.278*x**2*z**2 + 1.30209*x*y*z**2 -0.526235*y**2*z**2 + 0.300706*z**4
            fourth_2 = 1.19161*x**3*y -0.893343*x**2*y**2 -0.63434*x*y**3 + 0.16087*y**4 + 0.893343*x**2*z**2 -1.67181*x*y*z**2 -0.0718782*y**2*z**2 -0.136911*z**4
            fourth_3 = 1.14953*x**3*z + 0.48431*x**2*y*z -2.33014*x*y**2*z + 0.48431*y**3*z -0.372822*x*z**3 -0.645746*y*z**3
            fourth_4 = 0.518321*x**2*y**2 + 0.598506*x*y**3 + 0.172774*y**4 -0.518321*x**2*z**2 -1.79552*x*y*z**2 -1.55496*y**2*z**2 + 0.345547*z**4
            fourth_5 = 0.854242*x**2*y*z + 1.97279*x*y**2*z + 0.854242*y**3*z -0.657596*x*z**3 -1.13899*y*z**3
            fo1+=fourth_1
            fo2+=fourth_2
            fo3+=fourth_3
            fo4+=fourth_4
            fo5+=fourth_5
            fifth_1 = 0.240391*x**5 -0.509292*x**4*y -0.876962*x**3*y**2 + 1.23302*x**2*y**3 -0.077379*x*y**4 -0.0589707*y**5 -1.52695*x**3*z**2 -0.643317*x**2*y*z**2 + 3.09516*x*y**2*z**2 -0.643317*y**3*z**2 + 0.247613*x*z**4 + 0.428878*y*z**4
            fifth_2 = 0.96686*x**4*y + 0.964265*x**3*y**2 -1.72842*x**2*y**3 -0.727203*x*y**4 + 0.234432*y**5 -0.964265*x**3*z**2 -0.615905*x**2*y*z**2 + 1.47042*x*y**2*z**2 -0.615905*y**3*z**2 + 0.237062*x*z**4 + 0.410603*y*z**4
            fifth_3 = 0.900562*x**4*z + 0.400687*x**3*y*z -0.0495722*x**2*y**2*z -2.00344*x*y**3*z + 0.437888*y**4*z -1.7846*x**2*z**3 + 1.60275*x*y*z**3 -0.859252*y**2*z**3 + 0.264385*z**5
            fifth_4 = 0.17967*x**3*y**2 + 0.518662*x**2*y**3 + 0.419229*x*y**4 + 0.103732*y**5 -0.17967*x**3*z**2 -1.55599*x**2*y*z**2 -3.05439*x*y**2*z**2 -1.55599*y**3*z**2 + 0.598899*x*z**4 + 1.03732*y*z**4
            fifth_5 = 3.13679*x**3*y*z -2.06432*x**2*y**2*z -1.33807*x*y**3*z + 0.519245*y**4*z + 0.688106*x**2*z**3 -1.79872*x*y*z**3 -0.350385*y**2*z**3 -0.0337721*z**5
            fifth_6 = 1.77394*x**2*y**2*z + 2.04837*x*y**3*z + 0.591312*y**4*z -0.591312*x**2*z**3 -2.04837*x*y*z**3 -1.77394*y**2*z**3 + 0.236525*z**5
            ff1+=fifth_1
            ff2+=fifth_2
            ff3+=fifth_3
            ff4+=fifth_4
            ff5+=fifth_5
            ff6+=fifth_6
            beta_fourth = 0.365148*x**4 -1.09545*x**2*y**2 + 0.365148*y**4 -1.09545*x**2*z**2 -1.09545*y**2*z**2 + 0.365148*z**4
            bfo+=beta_fourth
        #arrange all functions at each order into vector, and find magnitude of each vector    
        first_order=np.linalg.norm(np.array([f1,f2]))
        second_order=np.linalg.norm(np.array([s1,s2,s3]))
        third_order=np.linalg.norm(np.array([t1,t2,t3,t4]))
        fourth_order=np.linalg.norm(np.array([fo1,fo2,fo3,fo4,fo5]))
        fifth_order=np.linalg.norm(np.array([ff1,ff2,ff3,ff4,ff5,ff6]))
        beta_fourth_order=np.linalg.norm(np.array([bfo]))
        #arrange descriptors into list of 6 components
        desc = [first_order,second_order,third_order,fourth_order,fifth_order,beta_fourth_order]
        #append descriptor for particle i into overall descriptor list
        descriptors.append(desc)   
    #write parameters to seperate file to be used for machine learning
    with open('quenched_sym_run0-19.txt','ab') as file:
        pickle.dump(descriptors,file)
    return descriptors

#function which takes in files and some parameters,
#and outputs number of solid-like particles in each configuration
def sol_or_liq(file,L,d):
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


print(sol_or_liq("quenched_0.xyz",L,d))