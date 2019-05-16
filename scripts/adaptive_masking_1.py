#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 15:20:13 2019

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
kernel = np.ones((5,5), np.uint8)


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
    K = 4
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    
    
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    
    return res2

#img_path = '../picture/pic.jpg'
for r, d, f in os.walk(thisdir):
    for file in f:
        if ".jpg" in file:
            picture_names.append(os.path.join(r,file))


for img_path in picture_names[:1]:
    try:
        #load image
        image = cv2.imread(img_path)
        #make a copy of the image for processing
        img = image.copy()
        
        '''img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        # equalize the histogram of the Y channel
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        
        # convert the YUV image back to RGB format
        img_hist_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        '''
        #face detection
        detected_faces = detect_faces(img)
        #crop the detected face
        img = Image.fromarray(img).crop(detected_faces[0])
        face = img.copy()       #saving a copy of color image of cropped face
        
        #convert to a numpy array for opencv processing functions
        img = img.convert('RGB')
        img = np.array(img)
        img = img[:, :, ::-1].copy()
        #face = img.copy()
        #img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        
        
        #creating a mask
        face_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        face_gray = cv2.medianBlur(face_gray,5)
        face_th = cv2.adaptiveThreshold(face_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,11,2)
        
    
        #increasing the length of the mask for RGB planes. 
        mask = cv2.merge((face_th,face_th,face_th))
        face_and = cv2.bitwise_and(img, mask)
        #plt.imshow(face_and)
        #plt.show()


        #color quantization
        quantized_image = color_quant(face)
        #plt.imshow(quantized_image)
        #plt.show()
        
        face = Image.fromarray(quantized_image)
        face_np = face.convert('RGB')
        face_np = np.array(face_np)
        face_np = face_np[:, :, ::-1].copy()
        
        #splitting colors
        R = face_and[:, :, 0]      
        l = len(R[np.nonzero(R)])
        
        mean = [int(x/l) for x in ImageStat.Stat(face).sum]
        
        #to convert BGR to RGB
        org_img = image[:, :, ::-1].copy()
        #draw an ellipse with fill color as the detected skin color
        out = Image.fromarray(org_img)
        d = Draw(out)
        d.ellipse(((0,0),(0.2*image.shape[0],0.2*image.shape[1])), fill = tuple(mean))
        success += 1
        #out.save('../results/out6/out_file_'+str(success)+'.jpg')
        print('Success ' + str(success))
        plt.imshow(out)
        plt.show()
    except:
        fail += 1
        print('Fail ' + str(fail))
        
print('Success: ' + str(success) + '\tFail: ' + str(fail))