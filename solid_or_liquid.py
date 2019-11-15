# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 14:08:49 2019

@author: chidd
"""
import numpy as np
from scipy import special
import pickle

#Constants that can be changed. 
#L = length of box (periodic boundary conditions)
#rec = 1/L
#l = orbital quantum number, d = cutoff distance for nearest neighbour atoms
#c = threshold for 2 particles to be considered 'connected'
#t = threshold value for number of connections for solid-like particle
L = 7.93701
l = 6
d = 1.5
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
def find_param(coords,l,L,d):    
    myrange=range(-l,l+1)
    param = []
    #find parameter for each particle a
    for i, a in enumerate(coords):
        sph_harm=[]
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
                    
                    #append spherical harmonic to new list and carry on loop
                    sph_harm.append(new_vec)
                
        #find parameter for particle a and append to parameters list
        total = sum(sph_harm)
        N=len(sph_harm)
        p = total/N
        P=p.tolist()
        param.append(P)

    #write parameters to seperate file to be used for machine learning
    with open('sp.txt','ab') as file:
        pickle.dump(param,file)
    return param


#function to normalise parameters and returns as array of arrays
def norm_param(param):
    n_param = [[i/np.linalg.norm(j) for i in j] for j in param]
    norm_array = np.array([np.array(n) for n in n_param])
    return norm_array


#function to count number of solid-like particles in each configuration
#takes in normalised parameters, coordinates, length of box L
#cutoff distance d, connection threshold c and threshold value t
def state_classify(norm_par,coords,L,d,c,t):
    solid_count = 0
    for i, a in enumerate(norm_par):
        coorda = coords[i]
        j=0
        con_count = 0
        for j, b in enumerate(norm_par):
            coordb = coords[j]
            if i!=j:
                #apply 2 conditions - on dot product of parameters 
                # and on distance between particles a and b
                r= find_vector(coorda, coordb,L)
                dist = np.dot(r,r)
                #if a and b are both connected and nearest neighbours
                #add 1 to connection count
                if dist<=d**2:
                    dotp = np.vdot(a,b)
                    if dotp>c:
                        con_count +=1
        #if >7 connections, add 1 to no. of solid-like particles in config
        if con_count>t:
            solid_count +=1
    return solid_count


#function which takes in files and some parameters,
#and outputs number of solid-like particles in each configuration
def sol_or_liq(file,l,L,d,c,t):
    j=1
    a =[]
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
        for line in xyz:
            atom,x,y,z = line.split()
            atoms.append(atom)
            coordinates.append(np.array([float(x), float(y), float(z)]))
            if len(coordinates)==n_atoms:
                break
        parameters = find_param(coordinates,l,L,d)
        normal = norm_param(parameters)
        sl = state_classify(normal,coordinates,L,d,c,t)
        print('Processed configuration {}. Timestamp = {}'.format(j, comment))
        a.append(sl)
        j+=1
    xyz.close()
    return a

print(sol_or_liq("solid.xyz",l,L,d,c,t))

    
   
        
        
    
        


