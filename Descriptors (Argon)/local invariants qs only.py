# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 11:07:50 2020

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
l = 6
l2=4
A1=((4*np.pi)/(2*l+1))**(1/2)
A2=((4*np.pi)/(2*l2+1))**(1/2)
d = 1.5/(rho**(1/3))
c = 0.5
t = 7


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
def find_param(coords,l,l2,L,d):    
    myrange=range(-l,l+1)
    myrange2=range(-l2,l2+1)
    param = []
    #find parameter for each particle a
    for i, a in enumerate(coords):
        sph_harm=[]
        sph_harm2=[]
        #loop over all particles b for each particle a
        for j, b in enumerate(coords):
            #if a==b, don't use particle b 
            if i!=j:
                    
                r=find_vector(a,b,L)

                #check if particle b is nearest neighbour to a
                if np.dot(r,r)<=d**2:
                    #construct spherical harmonic
                    theta = np.arctan2(r[1],r[0])
                    phi = np.arccos(r[2]/np.linalg.norm(r))               
                    new_vec = np.array([special.sph_harm(m,l,theta,phi) for m in myrange]) 
                    new_vec2 = np.array([special.sph_harm(m,l2,theta,phi) for m in myrange2])
                    #append spherical harmonic to new list and carry on loop
                    sph_harm.append(new_vec)
                    sph_harm2.append(new_vec2)
        #find parameters for particle a and append to parameters list
        
        total = sum(sph_harm)
        N=len(sph_harm)
        p1 = total/N
        P1=A1*np.linalg.norm(p1)
        
        total2=sum(sph_harm2)
        N2=len(sph_harm2)
        p2=total2/N2
        P2=A2*np.linalg.norm(p2)
        
        P=[P1,P2]
        param.append(P)

    #write parameters to seperate file to be used for machine learning
    with open('liquid1.8_locinv.txt','ab') as file:
        pickle.dump(param,file)
    return param


#function which takes in files and some parameters,
#and outputs number of solid-like particles in each configuration
def read_file(file,l,l2,L,d):
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
        parameters = find_param(coordinates,l,l2,L,d)
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


print(read_file("liquidT1.8_1.xyz",l,l2,L,d))
