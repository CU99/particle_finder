# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 14:08:49 2019

@author: chidd
"""
import numpy as np
from scipy import special

#Constants that can be changed. 
#L = length of box (periodic boundary conditions)
#l = orbital quantum number, d = cutoff distance for nearest neighbour atoms
#c = threshold for 2 particles to be considered 'connected'
#t = threshold value for number of connections for solid-like particle
L = 7.93701
l = 6
d = 1.5
c = 0.5
t = 7


#function to find vector between 2 co-ordinates a and b, length of box L
def find_vector(a,b,L):
    vec1 = np.array(a)
    vec2 = np.array(b)
    c=(vec2-vec1)
        
    d=c/L

    #ensure each component of reduced vector <=0.5
    d[0]=d[0]-np.floor(d[0]+0.5)
    d[1]=d[1]-np.floor(d[1]+0.5)
    d[2]=d[2]-np.floor(d[2]+0.5)
    
    r = d*L
    return r


#function to find list of parameters for this configuration of coordinates
#takes in coords, l quantum number, L (length of box)
#and cut-off distance for nearest-neighbour bonds d
def find_param(coords,l,L,d):    
    i=0
    param = []
    #find parameter for each particle a
    while i < len(coords):    
        a = coords[i]
        j=0
        sph_harm=[]
        #loop over all particles b for each particle a
        while j < len(coords):
            b = coords[j]
            #if a==b, don't use particle b 
            if a!=b:
                    
                r=find_vector(a,b,L)
        
                #construct spherical harmonic
                theta = np.arctan2(r[1],r[0])
                phi = np.arccos(r[2]/np.linalg.norm(r))
                new_vec = np.empty(2*l+1,dtype=np.complex128)
                k=0
                for m in range (-l, l+1):
                    new_vec[k] = special.sph_harm(m,l,theta,phi)
                    k+=1
                    
    
                #check if particle b is nearest neighbour to a
                if np.linalg.norm(r)<=d:
                    sph_harm.append(new_vec)
                    j+=1
                    if j>= len(coords):
                        break
                    b=coords[j]
        
                else:
                    #append spherical harmonic to new list and carry on loop
                    j+=1
                    if j>= len(coords):
                        break
                    b=coords[j]
            else:
                j+=1
                if j>= len(coords):
                    break
                b=coords[j]
                
        #find parameter for particle a
        total = sum(sph_harm)
        N=len(sph_harm)
        p = total/N
        param.append(p)
        i+=1
    return param


#function to normalise parameters and returns as array of arrays
def norm_param(param):
    n_param = [[i/np.linalg.norm(j) for i in j] for j in param]
    norm_array = np.array([np.array(n) for n in n_param])
    return norm_array


#function to count number of solid-like particles in each configuration
#takes in normalised parameters, coordinates, length of box L,
#cutoff distance d, connection threshold c and threshold value t
def state_classify(norm_par,coords,L,d,c,t):
    i=0
    solid_count = 0
    while i<len(norm_par):
        a = norm_par[i]
        coorda = coords[i]
        j=0
        con_count = 0
        while j<len(norm_par):
            b = norm_par[j]
            coordb = coords[j]
            if i!=j:
                #apply 2 conditions - on dot product of parameters 
                # and on distance between particles a and b
                dotp = np.vdot(a,b)
                dist = np.linalg.norm(find_vector(coorda, coordb, L))
                #if a and b are both connected and nearest neighbours
                #add 1 to connection count
                if dotp>c and dist<=d:
                    con_count +=1
                    j+=1
                    if j>=len(norm_par):
                        break
                    b = norm_par[j]
                    coordb = coords[j]
                else:
                    j+=1
                    if j>=len(norm_par):
                        break
                    b = norm_par[j]
                    coordb = coords[j]
            else:
                j+=1
                if j>=len(norm_par):
                        break
                b = norm_par[j]
                coordb = coords[j]
        #if >7 connections, add 1 to no. of solid-like particles in config
        if con_count>t:
            solid_count +=1
            i+=1
        else:
            i+=1
    return solid_count


#function which takes in files and some parameters,
#and outputs number of solid-like particles in each configuration
def sol_or_liq(file,l,L,d,c,t):
    j=1
    a =[]
    b=[]
    xyz = open(file)
    #loop over all 101 configurations
    while j<=3:
        atoms = []
        coordinates = []
        i = 0
        n_atoms = int(xyz.readline())
        comment = xyz.readline()
        b.append(comment)
        for line in xyz:
            atom,x,y,z = line.split()
            atoms.append(atom)
            coordinates.append([float(x), float(y), float(z)])
            i+=1
            if i==n_atoms:
                break
        parameters = find_param(coordinates,l,L,d)
        normal = norm_param(parameters)
        sl = state_classify(normal,coordinates,L,d,c,t)
        a.append(sl)
        j+=1
    xyz.close()
    return a,b


print(sol_or_liq("liquid.xyz",l,L,d,c,t))

    
   
        
        
    
        


