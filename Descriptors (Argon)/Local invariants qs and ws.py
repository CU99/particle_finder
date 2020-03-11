# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from scipy import special
import pickle
import time
#from mayavi import mlab
from itertools import combinations
import py3nj

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
                    #construct spherical harmonics
                    theta = np.arctan2(r[1],r[0])
                    phi = np.arccos(r[2]/np.linalg.norm(r))               
                    new_vec = np.array([special.sph_harm(m,l,theta,phi) for m in myrange]) 
                    new_vec2 = np.array([special.sph_harm(m,l2,theta,phi) for m in myrange2])
                    #append spherical harmonics to new list and carry on loop
                    sph_harm.append(new_vec)
                    sph_harm2.append(new_vec2)
        #find parameter for particle a and append to parameters list
        #append all neighbours to new list
        neighbours.append(n)
        
        total = sum(sph_harm)
        N=len(sph_harm)
        p1 = total/N
        
        #make local invariants for l=6
        q6=A1*np.linalg.norm(p1)
        combo6=[seq for seq in combinations(myrange, 3) if sum(seq) == 0]
        combo6_list=[list(elem) for elem in combo6]
        W6=[]
        for u in combo6_list:
            wig=py3nj.wigner3j(2*l,2*l,2*l,2*u[0],2*u[1],2*u[2])
            w_l=np.real(wig*p1[u[0]+6]*p1[u[1]+6]*p1[u[2]+6])
            W6.append(w_l)
        w6=sum(W6)/(np.linalg.norm(p1)**3)
            
        
        #make local invariants for l=4
        total2=sum(sph_harm2)
        N2=len(sph_harm2)
        p2=total2/N2
        
        q4=A2*np.linalg.norm(p2)
        combo4=[seq for seq in combinations(myrange2, 3) if sum(seq) == 0]
        combo4_list=[list(elem) for elem in combo4]
        W4=[]
        for v in combo4_list:
            wig2=py3nj.wigner3j(2*l2,2*l2,2*l2,2*v[0],2*v[1],2*v[2])
            w_l2=np.real(wig2*p2[v[0]+4]*p2[v[1]+4]*p2[v[2]+4])
            W4.append(w_l2)
        w4=sum(W4)/(np.linalg.norm(p2)**3)
        
        P=[q6,q4,w6,w4]
        param.append(P)

    #write parameters to seperate file to be used for machine learning
    with open('liquid1.8_q6q4w6w4.txt','ab') as file:
        pickle.dump(param,file)
    return param, neighbours



#function which takes in files and some parameters,
#and outputs number of solid-like particles in each configuration
def read_file(file,l,l2,L,d):
    start = time.time()
    j=1
    total_descriptor_time=0
    xyz = open(file)
    #loop over all 101 configurations
    while j<=101:
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
        parameters, neighbours = find_param(coordinates,l,l2,L,d)
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

