#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 18:59:21 2022

@author: patricknormile
"""
import numpy as np
from PIL import Image
from skimage import io

class ImageProcessor() : 
    
    """
    preprocess an image for better speed in clustering algs
    """
    
    def __init__(self, image_path) : 
        self.image_path = image_path
        
    
    def resize_image(self, img, factor) : 
        """
        resize an image (np array) by a factor ?
        """
        pass
    
    def unscale_image(self, img, factor) :