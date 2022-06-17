#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 21:31:48 2022

@author: patricknormile
"""

import os, sys
from pathlib import Path
from image_processor import ImageClusterer as IC
from sklearn.cluster import KMeans, SpectralClustering, Birch, BisectingKMeans,\
    MiniBatchKMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from itertools import product

if __name__ == '__main__' :
    cwd_path = os.path.abspath('.')
    sys.path.append(cwd_path)

    raw_path = os.path.join(cwd_path , 'raw_images')
    raw_image_path = lambda x : os.path.join(raw_path, x)
    Path(os.path.join(cwd_path, 'altered_images')).mkdir(exist_ok=True)
    
    all_imgs = [*os.walk(raw_path)][0][2]
    all_k = [*range(2,25)]
    for i,k in product(all_imgs, all_k) : 
        fname = i[:i.find('.jpg')]
        ic = IC(raw_image_path(i))
        ic.cluster_image(MiniBatchKMeans,
                         save_name=f"{fname}_mbkm_{k}",
                         n_clusters=k,
                         random_state=923829)
        print(f"ran {i} with {k} clusters")