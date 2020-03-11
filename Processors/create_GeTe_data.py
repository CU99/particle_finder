#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 22:27:19 2020

@author: phurbc
"""

import numpy as np
import pickle
import time

d=[]
e=[]
f=[]


start=time.time()

with open('alpha_soap+sym_run0-19.txt','rb') as file:
    i=1
    j=20
    try:
        while True:
            if i<=j:
                d.append(pickle.load(file))
                i+=1
            else:
                pickle.load(file)
    except EOFError:
        pass

with open('alpha_soap+sym_run20-39.txt','rb') as file:
    i=1
    j=20
    try:
        while True:
            if i<=j:
                d.append(pickle.load(file))
                i+=1
            else:
                pickle.load(file)
    except EOFError:
        pass

with open('alpha_soap+sym_run40-59.txt','rb') as file:
    i=1
    j=20
    try:
        while True:
            if i<=j:
                d.append(pickle.load(file))
                i+=1
            else:
                pickle.load(file)
    except EOFError:
        pass
    
with open('alpha_soap+sym_run60-79.txt','rb') as file:
    i=1
    j=20
    try:
        while True:
            if i<=j:
                d.append(pickle.load(file))
                i+=1
            else:
                pickle.load(file)
    except EOFError:
        pass    
    
with open('alpha_soap+sym_run80-99.txt','rb') as file:
    i=1
    j=20
    try:
        while True:
            if i<=j:
                d.append(pickle.load(file))
                i+=1
            else:
                pickle.load(file)
    except EOFError:
        pass    
    
    
    
with open('beta_soap+sym_run0-19.txt','rb') as file:
    i=1
    j=20
    try:
        while True:
            if i<=j:
                e.append(pickle.load(file))
                i+=1
            else:
                pickle.load(file)
    except EOFError:
        pass

with open('beta_soap+sym_run20-39.txt','rb') as file:
    i=1
    j=20
    try:
        while True:
            if i<=j:
                e.append(pickle.load(file))
                i+=1
            else:
                pickle.load(file)
    except EOFError:
        pass

with open('beta_soap+sym_run40-59.txt','rb') as file:
    i=1
    j=20
    try:
        while True:
            if i<=j:
                e.append(pickle.load(file))
                i+=1
            else:
                pickle.load(file)
    except EOFError:
        pass
    
with open('beta_soap+sym_run60-79.txt','rb') as file:
    i=1
    j=20
    try:
        while True:
            if i<=j:
                e.append(pickle.load(file))
                i+=1
            else:
                pickle.load(file)
    except EOFError:
        pass    
    
with open('beta_soap+sym_run80-99.txt','rb') as file:
    i=1
    j=20
    try:
        while True:
            if i<=j:
                e.append(pickle.load(file))
                i+=1
            else:
                pickle.load(file)
    except EOFError:
        pass
    

with open('quenched_soap+sym_run0-19.txt','rb') as file:
    i=1
    j=20
    try:
        while True:
            if i<=j:
                f.append(pickle.load(file))
                i+=1
            else:
                pickle.load(file)
    except EOFError:
        pass

with open('quenched_soap+sym_run20-39.txt','rb') as file:
    i=1
    j=20
    try:
        while True:
            if i<=j:
                f.append(pickle.load(file))
                i+=1
            else:
                pickle.load(file)
    except EOFError:
        pass

with open('quenched_soap+sym_run40-59.txt','rb') as file:
    i=1
    j=20
    try:
        while True:
            if i<=j:
                f.append(pickle.load(file))
                i+=1
            else:
                pickle.load(file)
    except EOFError:
        pass
    
with open('quenched_soap+sym_run60-79.txt','rb') as file:
    i=1
    j=20
    try:
        while True:
            if i<=j:
                f.append(pickle.load(file))
                i+=1
            else:
                pickle.load(file)
    except EOFError:
        pass    
    
with open('quenched_soap+sym_run80-99.txt','rb') as file:
    i=1
    j=20
    try:
        while True:
            if i<=j:
                f.append(pickle.load(file))
                i+=1
            else:
                pickle.load(file)
    except EOFError:
        pass

#create long lists of alpha, beta and quenched particles    
D = [item for sublist in d for item in sublist]   
E = [item for sublist in e for item in sublist]    
F = [item for sublist in f for item in sublist]

train=[]
alpha=[]
beta=[]
quenched=[]

#create train and test data
train.extend(D[0:10800])
train.extend(E[0:10800])
train.extend(F[0:10800])

alpha.extend(D[10800:21600])
beta.extend(E[10800:21600])
quenched.extend(F[10800:21600])

t_data=[]
a_data=[]
b_data=[]
q_data=[]

for item in train:
    new_e = []
    for i in item:
        new_e.append(np.real(i))
#        new_g.append(np.imag(k))
    t_data.append(new_e)
    
for item in alpha:
    new_f = []
    for j in item:
        new_f.append(np.real(j))
#        new_g.append(np.imag(k))
    a_data.append(new_f)
    
for item in beta:
    new_g = []
    for k in item:
        new_g.append(np.real(k))
#        new_g.append(np.imag(k))
    b_data.append(new_g)

for item in quenched:
    new_h = []
    for l in item:
        new_h.append(np.real(l))
#        new_g.append(np.imag(k))
    q_data.append(new_h)

train_data = np.array(t_data)
alpha_data = np.array(a_data)
beta_data = np.array(b_data)
quenched_data = np.array(q_data)

end=time.time()

print(end-start)

np.save('train_soap+sym',train_data)
np.save('alpha_soap+sym',alpha_data)
np.save('beta_soap+sym',beta_data)
np.save('quenched_soap+sym',quenched_data)

print(train_data.shape)
print(alpha_data.shape)
print(beta_data.shape)
print(quenched_data.shape)

#create train and test labels
zeros=np.full(10800,0)
ones=np.full(10800,1)
twos=np.full(10800,2)

train_labels=np.concatenate((zeros,ones,twos))
alpha_labels=zeros
beta_labels=ones
quenched_labels=twos

np.save('train_GeTe_labels',train_labels)
np.save('alpha_GeTe_labels',alpha_labels)
np.save('beta_GeTe_labels',beta_labels)
np.save('quenched_GeTe_labels',quenched_labels)

print(train_labels[21599:21601])
print(alpha_labels.shape)

