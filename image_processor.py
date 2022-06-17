#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 18:59:21 2022

@author: patricknormile
"""
import numpy as np
from PIL import Image
from pathlib import Path
from skimage import io
from sklearn.cluster import KMeans, SpectralClustering, Birch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os, sys

class ImageProcessor() : 
    
    """
    preprocess an image for better speed in clustering algs
    """
    
    def __init__(self, image_path) : 
        self.image_path = image_path
        self.image = io.imread(self.image_path)
        self.arrfit = None
        #self.altered_image_path = self.image_path.replace('raw_images',
        #                                                  'altered_images')
    
    def resize_image(self, img, factor) : 
        """
        resize an image (np array) by a factor ?
        """
        pass
    
    def unscale_image(self, img, factor) :
        pass
    
    def preprocess_color(self, image=None) : 
        """
        general preprocessor for color (resize to matrix)
        """
        if image is None :
            image = self.image
        self.original_shape = image.shape
        self.arrfit = image.reshape((-1, 3))
        return self.arrfit
    
    def show_image(self, image=None) : 
        if image is None : 
            image = self.image
        io.imshow(image)
        

class ImageClusterer(ImageProcessor) : 
    

    
    def cluster_image(self,alg,save_name=None,**kwargs) : 
        if self.arrfit is None : 
            self.preprocess_color(self.image)
        
        self.alg = alg(**kwargs)
        self.alg.fit(self.arrfit)
        labels = self.alg.labels_
        centers = self.alg.cluster_centers_
        self.color_clustered = centers[labels].reshape(self.original_shape)\
            .astype('uint8')
        
        pthobj = Path(self.image_path)
        if save_name is None :     
            save_name = pthobj.stem + '.' + pthobj.suffix
        
        parent = pthobj.parent.parent
        self.altered_image_path = os.path.join(parent,
                                               'altered_images',
                                               save_name)
        io.imsave(self.altered_image_path,
                  self.color_clustered)
        
        



   