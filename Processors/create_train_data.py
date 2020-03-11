# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 15:40:02 2020

@author: chidd
"""

import numpy as np
import pickle

c=[]

with open('solid_soap_n=9_sigma=0.4.txt','rb') as file:
    i=1
    j=10
    try:
        while True:
            if i<=j:
                c.append(pickle.load(file))
                i+=1
            else:
                pickle.load(file)
    except EOFError:
        pass

with open('quench_soap_n=9_sigma=0.4_run1-20.txt','rb') as file:
    i=1
    j=20
    try:
        while True:
            if i<=j:
                c.append(pickle.load(file))
                i+=1
            else:
                pickle.load(file)
    except EOFError:
        pass
    
with open('quench_soap_n=9_sigma=0.4_run21-40.txt','rb') as file:
    i=1
    j=20
    try:
        while True:
            if i<=j:
                c.append(pickle.load(file))
                i+=1
            else:
                pickle.load(file)
    except EOFError:
        pass    

with open('quench_soap_n=9_sigma=0.4_run41-60.txt','rb') as file:
    i=1
    j=20
    try:
        while True:
            if i<=j:
                c.append(pickle.load(file))
                i+=1
            else:
                pickle.load(file)
    except EOFError:
        pass

with open('quench_soap_n=9_sigma=0.4_run61-80.txt','rb') as file:
    i=1
    j=20
    try:
        while True:
            if i<=j:
                c.append(pickle.load(file))
                i+=1
            else:
                pickle.load(file)
    except EOFError:
        pass

with open('quench_soap_n=9_sigma=0.4_run81-100.txt','rb') as file:
    i=1
    j=20
    try:
        while True:
            if i<=j:
                c.append(pickle.load(file))
                i+=1
            else:
                pickle.load(file)
    except EOFError:
        pass

with open('liquid_soap_n=9_sigma=0.4.txt','rb') as file:
    i=1
    j=30
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

train=[]

for item in C:
    new_g = []
    for k in item:
        new_g.append(np.real(k))
#        new_g.append(np.imag(k))
    train.append(new_g)
    
train_data = np.array(train)
np.save('train_soap_n=9_sigma=0.4',train_data)

print(train_data.shape)

m = np.ones(55000)
n = np.zeros(15000)
train_labels=np.append(m,n)
np.save('train_soap_labels',train_labels)
print(len(train_labels))



