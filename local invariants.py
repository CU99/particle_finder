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
    neighbours = []
    #find parameter for each particle a
    for i, a in enumerate(coords):
        sph_harm=[]
        sph_harm2=[]
        n=[]
        #loop over all particles b for each particle a
        for j, b in enumerate(coords):
            #if a==b, don't use particle b 
            if i!=j:
                    
                r=find_vector(a,b,L)

                #check if particle b is nearest neighbour to a
                if np.dot(r,r)<=d**2:
                    #append nearest neighbours to new list
                    n.append(j)
                    #construct spherical harmonic
                    theta = np.arctan2(r[1],r[0])
                    phi = np.arccos(r[2]/np.linalg.norm(r))               
                    new_vec = np.array([special.sph_harm(m,l,theta,phi) for m in myrange]) 
                    new_vec2 = np.array([special.sph_harm(m,l2,theta,phi) for m in myrange2])
                    #append spherical harmonic to new list and carry on loop
                    sph_harm.append(new_vec)
                    sph_harm2.append(new_vec2)
        #find parameters for particle a and append to parameters list
        #append all neighbours to new list
        neighbours.append(n)
        
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
    with open('liquid2.0_locinv.txt','ab') as file:
        pickle.dump(param,file)
    return param, neighbours


#function to normalise parameters and returns as array of arrays
def norm_param(param):
    n_param = [[i/np.linalg.norm(j) for i in j] for j in param]
    norm_array = np.array([np.array(n) for n in n_param])
    return norm_array


#function to count number of solid-like particles in each configuration
#takes in normalised parameters, coordinates, length of box L
#cutoff distance d, connection threshold c and threshold value t
def state_classify(norm_par,coordinates,neighbours,L,d,c,t):
    solid_count = 0
    solid_coords=[]
    liquid_coords=[]
    for i, a in enumerate(norm_par):
        n = neighbours[i]
        con_count = 0
        #only loop over neighbouring particles
        for f, g in enumerate(n):
            b = norm_par[g]
            dotp = np.vdot(a,b)
            if dotp>c:
                con_count +=1
        #if >7 connections, add 1 to no. of solid-like particles in config
        if con_count>t:
            solid_count +=1
            solid_coords.append(coordinates[i])
        else:
            liquid_coords.append(coordinates[i])
    return solid_count,solid_coords,liquid_coords


#function which takes in files and some parameters,
#and outputs number of solid-like particles in each configuration
def sol_or_liq(file,l,l2,L,d,c,t):
    start = time.time()
    total_class_time =0
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
        #read through files and puts coordinates into array
        for line in xyz:
            atom,x,y,z = line.split()
            atoms.append(atom)
            coordinates.append(np.array([float(x), float(y), float(z)]))
            if len(coordinates)==n_atoms:
                break
        parameters, neighbours = find_param(coordinates,l,l2,L,d)
        s1 = time.time()
        normal = norm_param(parameters)
        sl, solid_c, liquid_c = state_classify(normal,coordinates,neighbours,L,d,c,t)
        e1 = time.time()
        t1 = e1-s1
        total_class_time+=t1
        #split coordinates array into solid x,y,z and liquid x,y,z lists
        solid_x=[]
        solid_y=[]
        solid_z=[]
        for e,f,g in solid_c:
            solid_x.append(e)
            solid_y.append(f)
            solid_z.append(g)
        liquid_x=[]
        liquid_y=[]
        liquid_z=[]
        for m,n,o in liquid_c:
            liquid_x.append(m)
            liquid_y.append(n)
            liquid_z.append(o)
        #plot cubes, solid in 1 colour, liquid in another colour,with a cube of length L around it to mark boundaries
        #mlab.points3d([L/2],[L/2],[L/2],mode='cube',scale_factor=L,opacity=0.2,color=(1,1,1))    
        #mlab.points3d(solid_x, solid_y, solid_z, scale_factor=1.0,color=(1,0,0), mode='sphere',resolution=12,opacity=1.0)
        #mlab.points3d(liquid_x, liquid_y, liquid_z, scale_factor=1.0,color=(0,0,1), mode='sphere',resolution=12,opacity=1.0)
        #save snapshot of scene
        #mlab.savefig('snapshotquench{}.png'.format(j),magnification=5)
        print('Processed configuration {}. Timestamp = {}'.format(j, comment))
        #mlab.show()
        a.append(sl)
        j+=1
    xyz.close()
    end = time.time()
    total_time = end-start
    print(total_time)
    print(total_class_time)
    return a


print(sol_or_liq("liquidT2.0_1.xyz",l,l2,L,d,c,t))
