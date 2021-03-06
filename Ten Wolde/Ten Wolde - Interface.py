# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 12:14:19 2020

@author: chidd
"""

import numpy as np
from scipy import special
import pickle
import time
from mayavi import mlab



#Constants that can be changed. 
#L = size of box (periodic boundary conditions)
#rec = 1/L
#l = orbital quantum number, d = cutoff distance for nearest neighbour atoms
#c = threshold for 2 particles to be considered 'connected'
#t = threshold value for number of connections for solid-like particle
#rho = density

rho = 0.965
L=np.array([7.8559306,7.8559306,67.14331])
Rec_L=1/L
l = 6
cub_x = [0,L[0],L[0],0,0,0,L[0],L[0],L[0],L[0],0,0,0,0,L[0],L[0]]
cub_y = [0,0,L[1],L[1],0,0,0,0,L[1],L[1],L[1],L[1],L[1],0,0,L[1]]
cub_z = [0,0,0,0,0,L[2],L[2],0,0,L[2],L[2],0,L[2],L[2],L[2],L[2]]
d = 1.5/(rho**(1/3))
c = 0.5
t = 7


#function to find vector between 2 co-ordinates a and b, length of box L,rec 1/L
def find_vector(a,b,L,Rec_L):
    c=b-a    
    
    d=c*Rec_L
    
    d=d-np.floor(d+0.5)

    r=d*L
    return r


#function to find list of parameters for this configuration of coordinates
#takes in coords, l quantum number, L (length of box),rec = 1/L
#and cut-off distance for nearest-neighbour bonds d
def find_param(coords,l,L,Rec_L,d):    
    myrange=range(-l,l+1)
    param = []
    neighbours = []
    #find parameter for each particle a
    for i, a in enumerate(coords):
        sph_harm=[]
        n=[]
        #loop over all particles b for each particle a
        for j, b in enumerate(coords):
            #if a==b, don't use particle b 
            if i!=j:
                    
                r=find_vector(a,b,L,Rec_L)

                #check if particle b is nearest neighbour to a
                if np.dot(r,r)<=d**2:
                    #append nearest neighbours to new list
                    n.append(j)
                    #construct spherical harmonic
                    theta = np.arctan2(r[1],r[0])
                    phi = np.arccos(r[2]/np.linalg.norm(r))               
                    new_vec = np.array([special.sph_harm(m,l,theta,phi) for m in myrange]) 
                    
                    #append spherical harmonic to new list and carry on loop
                    sph_harm.append(new_vec)
                
        #find parameter for particle a and append to parameters list
        #append all neighbours to new list
        neighbours.append(n)
        total = sum(sph_harm)
        N=len(sph_harm)
        p = total/N
        P=p.tolist()
        param.append(P)

    #write parameters to seperate file to be used for machine learning
    #with open('traj.txt','ab') as file:
    #    pickle.dump(param,file)
    return param, neighbours


#function to normalise parameters and returns as array of arrays
def norm_param(param):
    n_param = [[i/np.linalg.norm(j) for i in j] for j in param]
    norm_array = np.array([np.array(n) for n in n_param])
    return norm_array


#function to count number of solid-like particles in each configuration
#takes in normalised parameters, coordinates, length of box L
#cutoff distance d, connection threshold c and threshold value t
def state_classify(norm_par,coordinates,neighbours,d,c,t):
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
def sol_or_liq(file,l,L,Rec_L,d,c,t):
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
        parameters, neighbours = find_param(coordinates,l,L,Rec_L,d)
        s1 = time.time()
        normal = norm_param(parameters)
        sl, solid_c, liquid_c = state_classify(normal,coordinates,neighbours,d,c,t)
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
        #mlab.plot3d(cub_x,cub_y,cub_z,opacity=1,color=(0,0,0))    
        #mlab.points3d(solid_x, solid_y, solid_z, scale_factor=1.0,color=(1,0,0), mode='sphere',resolution=12,opacity=1.0)
        #mlab.points3d(liquid_x, liquid_y, liquid_z, scale_factor=1.0,color=(0,0,1), mode='sphere',resolution=12,opacity=1.0)
        #save snapshot of scene
        #mlab.savefig('snapshotinterface{}.png'.format(j),magnification=5)
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


print(sol_or_liq("traj.xyz",l,L,Rec_L,d,c,t))

    
   
        
        
    
        


