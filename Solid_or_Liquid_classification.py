#!/usr/bin/env python
# coding: utf-8

# ##### Copyright 2018 The TensorFlow Authors.

# In[1]:


#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# In[2]:


#@title MIT License
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


# # Basic classification: Classify particles as solid or liquid

# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://www.tensorflow.org/tutorials/keras/classification"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
#   </td>
#   <td>
#     <a href="https://storage.googleapis.com/tensorflow_docs/docs/site/en/tutorials/keras/classification.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
#   </td>
# </table>

# This guide trains a neural network model to classify particles as solid and liquid.
# 
# This guide uses [tf.keras](https://www.tensorflow.org/guide/keras), a high-level API to build and train models in TensorFlow.

# In[23]:


try:
  # %tensorflow_version only exists in Colab.
  get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
  pass


# In[24]:


from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import time

print(tf.__version__)


# ## Import the parameter data

# This guide uses parameters generated using the Lennard-Jones method, based on spherical harmonics:
# 
# Here, 51,000 parameters are used to train the network and 50,000 parameters to evaluate how accurately the network learned to classify particles.

# In[25]:


train_data = np.load('train_data.npy')
train_labels = np.load('train_labels.npy')
test_data = np.load('test_data.npy')
test_labels = np.load('test_labels.npy')


# Loading the dataset returns four NumPy arrays:
# 
# * The `train_data` and `train_labels` arrays are the *training set*—the data the model uses to learn.
# * The model is tested against the *test set*, the `test_data`, and `test_labels` arrays.
# 
# The data are 1D NumPy arrays with 26 components (l=6, 2l+1=13, 13*2 = 26 as real and imaginary parts were split up). The *labels* are an array of 2 integers, 1 for solid, 0 for liquid.
# 
# <table>
#   <tr>
#     <th>Label</th>
#     <th>Class</th>
#   </tr>
#   <tr>
#     <td>0</td>
#     <td>Liquid</td>
#   </tr>
#   <tr>
#     <td>1</td>
#     <td>Solid</td>
#   </tr>
# </table>
# 
# Each particle is mapped to a single label. Since the *class names* are not included with the dataset, store them here to use later when making plots later:

# In[26]:


classification = ['Liquid','Solid']


# ## Explore the data
# 
# Let's explore the format of the dataset before training the model. The following shows there are 51000 parameters, each with 26 components:

# In[27]:


train_data.shape


# Likewise, there are 51,000 labels in the training set:

# In[28]:


len(train_labels)


# Each label is an integer, 0 or 1:

# In[29]:


train_labels


# There are 50,000 parameters in the test set. Again, each parameter has 26 components

# In[30]:


test_data.shape


# And the test set contains 50,000 labels:

# In[31]:


len(test_labels)


# ## Preprocess the data
# No pre-processing was required for this classification.
# 

# ## Build the model
# 
# Building the neural network requires configuring the layers of the model, then compiling the model.

# ### Set up the layers
# 
# The basic building block of a neural network is the *layer*. Layers extract representations from the data fed into them. Hopefully, these representations are meaningful for the problem at hand.
# 
# Most of deep learning consists of chaining together simple layers. Most layers, such as `tf.keras.layers.Dense`, have parameters that are learned during training.

# In[32]:


model = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])


# There was no need for flattening as the data was already a 1D array.
# 
# After the pixels are flattened, the network consists of a sequence of two `tf.keras.layers.Dense` layers. These are densely connected, or fully connected, neural layers. The first `Dense` layer has 128 nodes (or neurons). The second (and last) layer is a 10-node *softmax* layer that returns an array of 2 probability scores that sum to 1. Each node contains a score that indicates the probability that the current parameter belongs to one of the 2 classes.
# 
# ### Compile the model
# 
# Before the model is ready for training, it needs a few more settings. These are added during the model's *compile* step:
# 
# * *Loss function* —This measures how accurate the model is during training. You want to minimize this function to "steer" the model in the right direction.
# * *Optimizer* —This is how the model is updated based on the data it sees and its loss function.
# * *Metrics* —Used to monitor the training and testing steps. The following example uses *accuracy*, the fraction of the images that are correctly classified.

# In[33]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# ## Train the model
# 
# Training the neural network model requires the following steps:
# 
# 1. Feed the training data to the model. In this example, the training data is in the `train_data` and `train_labels` arrays.
# 2. The model learns to associate parameters and labels.
# 3. You ask the model to make predictions about a test set—in this example, the `test_data` array. Verify that the predictions match the labels from the `test_labels` array.
# 
# To start training,  call the `model.fit` method—so called because it "fits" the model to the training data:

# In[34]:


start = time.time()
model.fit(train_data, train_labels, epochs=10)
end = time.time()
print(end-start)


# As the model trains, the loss and accuracy metrics are displayed. This model reaches an accuracy of 100% on the training data after 2 epochs.

# ## Evaluate accuracy
# 
# Next, compare how the model performs on the test dataset:

# In[35]:


start = time.time()

test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

end = time.time()
print(end-start)


# It turns out that the accuracy on the test dataset is the same as the accuracy on the training dataset. If there was a gap between training accuracy and test accuracy, this would represent *overfitting*. Overfitting is when a machine learning model performs worse on new, previously unseen inputs than on the training data.

# ## Make predictions
# 
# With the model trained, you can use it to make predictions about some data.

# In[36]:


start=time.time()
predictions = model.predict(test_data)
end = time.time()
print(end-start)


# Here, the model has predicted the label for each image in the testing set. Let's take a look at the first prediction:

# In[37]:


predictions[0]


# A prediction is an array of 2 numbers. They represent the model's "confidence" that the parameter corresponds to each of the 2 states (liquid or solid). You can see which label has the highest confidence value:

# In[38]:


np.argmax(predictions[0])


# So, the model is most confident that this particle is a solid, or `classification[1]`. Examining the test label shows that this classification is correct:

# In[40]:


test_labels[0]

