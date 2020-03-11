# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 18:47:01 2020

@author: chidd
"""
import numpy as np
import pickle

c=[]

with open('quench.txt','rb') as file:
    i=1
    j=101
    try:
        while True:
            if i<=j:
                c.append(pickle.load(file))
                i+=1
            else:
                pickle.load(file)
    except EOFError:
        pass
    
C = [item for sublist in c for item in sublist]

test=[]

for item in C:
    new_g = []
    for k in item:
        new_g.append(np.real(k))
#        new_g.append(np.imag(k))
    test.append(new_g)
    
test_data = np.array(test)
np.save('quench',test_data)   

print(test_data.shape)