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
from sklearn.cluster import KMeans, SpectralClustering, Birch, MiniBatchKMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os, sys


"""

To do - put in a space/color preprocessor
put in a space only processor & divide region into clusters & take avg color 
by region
scale/unscale images for speed & granularity
make more flexible with other clustering algs that don't have labels & centers

"""

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
    
    def resize_image(self, factor) : 
        """
        resize an image (np array) by a factor ?
        """
        in_img = Image.open(self.image_path) # could you convert np array to right type?
        self.orig_shape = np.array(in_img.size)
        resized_image = in_img.resize(self.orig_shape // factor)
        self.new_shape = np.array(resized_image.size)
        self.image = np.array(resized_image)
        
    
    def unscale_image(self, image=None,factor=8) :
        """
        unscale color-clustered
        """
        if image is None : 
            image = Image.open(self.altered_image_path)
        image.resize(self.orig_shape)
        image.show()
        
    
    def preprocess_color(self, image=None) : 
        """
        general preprocessor for color (resize to matrix)
        """
        if image is None :
            image = self.image
        self.original_shape = image.shape
        self.arrfit = image.reshape((-1, 3))
        return self.arrfit
    
    def preprocess_color_space(self, image=None) : 
        if image is None : 
            image=self.image
        self.original_shape = image.shape
        rng = image.shape[0:2]
        x,y = np.meshgrid(range(rng[0]), range(rng[1]))
        scl = MinMaxScaler()
        coordinate = np.concatenate([image.reshape(-1,3), 
                                     x.reshape(-1, 1), 
                                     y.reshape(-1, 1)], 
                                    axis=1)
        self.arrfit = coordinate
        
    
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
        if 'cluster_centers_' in dir(self.alg) :
            centers = self.alg.cluster_centers_
        else : 
            centers = [self.arrfit[labels==i,:3].mean(axis=0) 
                       for i in np.unique(labels)]
        self.color_clustered = centers[labels][:,:3].reshape(self.original_shape)\
            .astype('uint8')
        
        pthobj = Path(self.image_path)
        if save_name is None :     
            save_name = pthobj.stem + pthobj.suffix
        
        parent = pthobj.parent.parent
        self.altered_image_path = os.path.join(parent,
                                               'altered_images',
                                               save_name)
        io.imsave(self.altered_image_path,
                  self.color_clustered)
        
        



   