#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 17:08:50 2019

@author: claran
"""
import os
import cv2
import numpy as np
from PIL import Image

thisdir = '../women_headshots'
picture_names = []
success = 0
fail = 0

for r, d, f in os.walk(thisdir):
    for file in f:
        if ".jpg" in file:
            picture_names.append(os.path.join(r,file))


for img_path in picture_names:
    try:
        #load image
        #img_path = '/Users/claran/Downloads/DiandraForrestAlbino.jpg'
        #img_path = '../women_headshots/59. ivana_4x3_cl.jpg'
        image = cv2.imread(img_path)
        image = image[:,:,::-1]
        out = Image.fromarray(image)
        success += 1
        out.save('/Users/claran/Documents/conex/cosmetic_recommender/images/image'+str(success)+'.jpg')
    except:
        print('Fail')        