# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 01:12:58 2020

@author: chidd
"""

import numpy as np
from scipy import special
import pickle
import time
from mayavi import mlab



#Constants that can be changed. 
#L = length of box (periodic boundary conditions)
#rec = 1/L
#l = orbital quantum number, d = cutoff distance for nearest neighbour atoms
#c = threshold for 2 particles to be considered 'connected'
#t = threshold value for number of connections for solid-like particle
rho = 1
L = 7.93701/(rho**(1/3))
d = 1.8/(rho**(1/3))


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
def find_param(coords,L,d):    
    param = []
    #find parameter for each particle a
    for i, a in enumerate(coords):
        vectors=[]
        mag = []
        nearest_neighbours=[]
        #loop over all particles b for each particle a
        for j, b in enumerate(coords):
            #if a==b, don't use particle b 
            if i!=j:
                    
                r=find_vector(a,b,L)

                #check if particle b is nearest neighbour to a
                m=np.dot(r,r)
                if m<=d**2:
                    #append vector to new list and carry on loop
                    vectors.append(r)
                    mag.append(m)
                
        #find parameter for particle a and append to parameters list
        neighbour_indices=np.argpartition(mag,12)[0:12]
        for w in neighbour_indices:
            nearest_neighbours.append(vectors[w])
            
        param.append(nearest_neighbours)

    #write parameters to seperate file to be used for machine learning
    with open('liquid1.8_vectors.txt','ab') as file:
        pickle.dump(param,file)
    return param



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
        parameters = find_param(coordinates,L,d)
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
