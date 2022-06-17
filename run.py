#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 18:47:51 2022

@author: patricknormile
"""

import os, sys
from pathlib import Path
sys.path.append(cwd_path)
import cluster_func
cwd_path = os.path.abspath('.')

raw_path = os.path.join(cwd_path , 'raw_images')
raw_image_path = lambda x : os.path.join(raw_path, x)

Path(os.path.join(cwd_path, 'altered_images')).mkdir(exist_ok=True)


