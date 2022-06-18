#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 16:36:59 2022

@author: patricknormile
"""

import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
X,y = make_blobs(n_samples=28, n_features=2,centers=3)

plt.scatter(X[:,0], X[:,1], c=y)
plt.show()

# In[p]
from sklearn.cluster import DBSCAN, OPTICS, Birch, KMeans

# cluster centers
p0 = X[y==0,:].mean(axis=0)
p1 = X[y==1,:].mean(axis=0)
p2 = X[y==2,:].mean(axis=0)

plt.scatter(X[:,0], X[:,1], c=y)

plt.scatter(p0[0], p0[1], c='red')

plt.scatter(p1[0], p1[1], c='blue')
plt.scatter(p2[0], p2[1], c='green')

km = KMeans(n_clusters=3)
km.fit(X)
cc = km.cluster_centers_
plt.scatter(cc[:,0], cc[:,1], c='grey')

# In[resize]
import os, sys
cwd_path = os.path.abspath('.')
sys.path.append(cwd_path)
from image_processor import ImageClusterer
a  = ImageClusterer('./raw_images/drytool.jpg')
a.resize_image(8)

def upscale_image(image, factor) : 
    in_shape = np.array(image.shape)
    new_shape = in_shape * factor
    new_shape[-1] /= factor
    new_image = np.zeros(new_shape)
    for i in range(new_shape[0]) :
        for j in range(new_shape[1]) : 
            new_image[i//factor,j//factor,:] = image[i,j,:]
    return new_image
upscale_image(a.image, 4)


