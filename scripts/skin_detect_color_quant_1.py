#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 21:43:48 2019

@author: claran
"""

import numpy as np
import cv2
import dlib
from PIL.ImageDraw import Draw
from PIL import Image, ImageStat 
from skimage import io
import matplotlib.pyplot as plt
import os


thisdir = '../women_headshots'
picture_names = []
success = 0
fail = 0


#function to detect the faces
def detect_faces(image):

    # Create a face detector
    face_detector = dlib.get_frontal_face_detector()

    # Run detector and get bounding boxes of the faces on image.
    detected_faces = face_detector(image, 1)
    face_frames = [(x.left(), x.top(),
                    x.right(), x.bottom()) for x in detected_faces]

    return face_frames


#function to perform colour quantization
def color_quant(img):
    #converting opencv image type to an array with 3 features/columns for k means clustering
    img = img.convert('RGB')
    img = np.array(img) 
    # Convert RGB to BGR 
    img = img[:, :, ::-1].copy() 
    Z = img.reshape((-1,3))
    
    # convert to np.float32
    Z = np.float32(Z)
    
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 5
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    
    
    
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    
    return res2


#list all the file names of the images in the folder
# r=root, d=directories, f = files
for r, d, f in os.walk(thisdir):
    for file in f:
        if ".jpg" in file:
            picture_names.append(os.path.join(r,file))
            
            
            
for img_path in picture_names:
    try:
        #load image
        image = cv2.imread(img_path)
        #make a copy of the image for processing
        img = image.copy()
        #face detection
        detected_faces = detect_faces(img)
        #crop the detected face
        img = Image.fromarray(img).crop(detected_faces[0])
        
        
        #color quantization
        quantized_image = color_quant(img)
        
        
        face = Image.fromarray(quantized_image)
        median = ImageStat.Stat(face).median
        
        #to convert BGR to RGB
        org_img = image[:, :, ::-1].copy()
        #draw an ellipse with fill color as the detected skin color
        out = Image.fromarray(org_img)
        d = Draw(out)
        d.ellipse(((0,0),(0.2*image.shape[0],0.2*image.shape[1])), fill = tuple(median))
        
        #output the results
        #plt.imshow(res2)
        #plt.show()
        #plt.imshow(out)
        #plt.axis('off')
        #plt.show()
        success += 1
        out.save('../results/out3/out_file_'+str(success)+'.jpg')
        print('Success ' + str(success))
        
    except:
        fail += 1
        print('Fail ' + str(fail))
        
print('Success: ' + str(success) + '\tFail: ' + str(fail))
