# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 20:08:19 2019

@author: chidd
"""

import numpy as np
from scipy import special
import pickle
import time
from mayavi import mlab
import os


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
    neighbours = []
    #find parameter for each particle a
    for i, a in enumerate(coords):
        sph_harm=[]
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
    with open('l.txt','ab') as file:
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
def sol_or_liq(file,l,L,d,c,t):
    j=1
    a =[]
    plot_coords=[]
    xyz = open(file)
    #loop over all 101 configurations
    while j<=5:
        atoms = []
        coordinates = []
        plot_c = []
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
        parameters, neighbours = find_param(coordinates,l,L,d)
        normal = norm_param(parameters)
        sl, solid_c, liquid_c = state_classify(normal,coordinates,neighbours,L,d,c,t)
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
        plot_c.extend([solid_x,solid_y,solid_z,liquid_x,liquid_y,liquid_z])
        plot_coords.append(plot_c)
        print('Processed configuration {}. Timestamp = {}'.format(j, comment))
        a.append(sl)
        j+=1
    xyz.close()
    return a,plot_coords

classification, coordinates = sol_or_liq("liquid.xyz",l,L,d,c,t)
print(classification)

#function to produce an animation of Mayavi plots
@mlab.animate(delay=1000)
def anim():
    splot = mlab.points3d(np.ones(500),np.ones(500),np.ones(500),scale_factor=1.0,color=(1,0,0), mode='sphere',resolution=12,opacity=1.0)    
    lplot=mlab.points3d(np.ones(500),np.ones(500),np.ones(500),scale_factor=1.0,color=(0,0,1), mode='sphere',resolution=12,opacity=1.0)
    cube=mlab.points3d([L/2],[L/2],[L/2],mode='cube',scale_factor=L,opacity=0.3,color=(0,0,0))
    h=1
    while True:
        for s_x,s_y,s_z,l_x,l_y,l_z in coordinates:
            splot.mlab_source.reset(x=s_x, y=s_y, z=s_z)
            lplot.mlab_source.reset(x=l_x, y=l_y, z=l_z)
            h+=1
            yield
anim()        
mlab.show()



    
