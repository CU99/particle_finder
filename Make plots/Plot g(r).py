# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 18:03:03 2020

@author: chidd
"""

import numpy as np
from scipy import special
import pickle
import time
from mayavi import mlab
import matplotlib.pyplot as plt



#Constants that can be changed. 
#L = length of box (periodic boundary conditions)
#rec = 1/L
#l = orbital quantum number, d = cutoff distance for nearest neighbour atoms
rho = 1
L = 7.93701/(rho**(1/3))
l = 6
d = 1.5/(rho**(1/3))


#function to find vector between 2 co-ordinates a and b, length of box L,rec 1/L
def find_vector(a,b,L):
    c=b-a
        
    d=c/L

    #ensure each component of reduced vector <=0.5
    d=d-np.floor(d+0.5)

    r = d*L
    return r


#function to find list of parameters for this configuration of coordinates
#takes in coords, l quantum number, L (length of box),rec = 1/L
#and cut-off distance for nearest-neighbour bonds d
def find_param(coords,l,L,d):    
    distances = []
    #find parameter for each particle a
    for i, a in enumerate(coords):
        #loop over all particles b for each particle a
        for j, b in enumerate(coords):
            #if a==b, don't use particle b 
            if i!=j:
                r=find_vector(a,b,L)
                dist = np.linalg.norm(r)
                distances.append(dist)
    return distances


def sol_or_liq(file,l,L,d):
    start = time.time()
    total_class_time =0
    j=1
    xyz = open(file)
    #loop over all 101 configurations
    while j<=1:
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
        distances = find_param(coordinates,l,L,d)
        #plot histogram (radial distribution function)
        plt.hist(distances, bins=50, range=(1,3.5))
        plt.xticks(np.arange(1, 4, step=0.5), fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('r*', fontsize=20)
        plt.ylabel('g(r)',fontsize=20)
        plt.show()
        print('Processed configuration {}. Timestamp = {}'.format(j, comment))
        j+=1
    xyz.close()
    end = time.time()
    total_time = end-start
    print(total_time)
    print(total_class_time)
    return len(distances)

print(sol_or_liq("liquid.xyz",l,L,d))
