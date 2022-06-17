#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 18:47:51 2022

@author: patricknormile
"""

import os, sys
from pathlib import Path
from image_processor import ImageClusterer as IC
from sklearn.cluster import KMeans, SpectralClustering, Birch, BisectingKMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


if __name__ == '__main__' : 
    cwd_path = os.path.abspath('.')
    sys.path.append(cwd_path)

    raw_path = os.path.join(cwd_path , 'raw_images')
    raw_image_path = lambda x : os.path.join(raw_path, x)
    
    Path(os.path.join(cwd_path, 'altered_images')).mkdir(exist_ok=True)
    
    ex_file = 'moto2.jpg'
    ex = IC(raw_image_path(ex_file))
    ex.cluster_image(BisectingKMeans,save_name='moto2_bisect_16.jpg', 
                     n_clusters=16,
                     random_state=9234,
                     )
    ex.show_image(ex.color_clustered)


