# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 13:49:31 2020

@author: chidd
"""
import numpy as np
from mayavi import mlab
import pickle

Ge_coords=[]
Te_coords=[]
with open('quenched_Ge_coords.txt','rb') as file:
    Ge_coords.append(pickle.load(file))
    
with open('quenched_Te_coords.txt','rb') as file2:
    Te_coords.append(pickle.load(file2))    
    
    
Ge=[item for sublist in Ge_coords for item in sublist]
Te=[item for sublist in Te_coords for item in sublist]

Ge_x=[]
Ge_y=[]
Ge_z=[]
for e,f,g in Ge:
    Ge_x.append(e)
    Ge_y.append(f)
    Ge_z.append(g)
Te_x=[]
Te_y=[]
Te_z=[]
for m,n,o in Te:
    Te_x.append(m)
    Te_y.append(n)
    Te_z.append(o)

#plot Germanium atoms in yellow, Tellurium in purple
mlab.points3d(Ge_x, Ge_y, Ge_z, scale_factor=2.5,color=(1,1,0), mode='sphere',resolution=12,opacity=1.0)
mlab.points3d(Te_x, Te_y, Te_z, scale_factor=2.5,color=(0.8,0,1), mode='sphere',resolution=12,opacity=1.0) 
mlab.savefig('quenched_GeTe.png',magnification=5)
mlab.show()  