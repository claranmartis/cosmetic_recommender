#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 14:17:10 2019

@author: claran
"""

import cv2
import numpy as np
import dlib
from PIL.ImageDraw import Draw
from PIL import Image, ImageStat 
from skimage import io
import matplotlib.pyplot as plt
import os
from imutils import face_utils
import imutils


thisdir = '../women_headshots'
picture_names = []
success = 0
fail = 0

kernel = np.ones((5,5), np.uint8)


 
def strokeEdge(src, dst, blurKSize = 7, edgeKSize = 5):
    # medianFilter with kernelsize == 7 is expensive
    if blurKSize >= 3:
        # first blur image to cancel noise
        # then convert to grayscale image
        blurredSrc = cv2.medianBlur(src, blurKSize)
        graySrc = cv2.cvtColor(blurredSrc, cv2.COLOR_BGR2GRAY)
    else:
        # scrip blurring image
        graySrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # we have to convert to grayscale since Laplacian only works on grayscale images
    # then we can apply laplacian edge-finding filter
    # cv2.CV_8U means the channel depth is first 8 bits
    cv2.Laplacian(graySrc, cv2.CV_8U, graySrc, ksize=edgeKSize)
    # normalize and inverse color
    # making the edges have black color and background has white color
    normalizedInverseAlpha = (1.0 / 255) * (255 - graySrc)
    channels = cv2.split(src)
    # multiply normalized grayscale image with source image
    # to darken the edge
    for channel in channels:
        channel[:] = channel * normalizedInverseAlpha
    cv2.merge(channels, dst)
    


#function to detect the faces
def detect_faces(image):

    # Create a face detector
    face_detector = dlib.get_frontal_face_detector()

    # Run detector and get bounding boxes of the faces on image.
    detected_faces = face_detector(image, 1)
    face_frames = [(x.left(), x.top(),
                    x.right(), x.bottom()) for x in detected_faces]

    return face_frames

def facial_features(image):
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    (rbStart, rbEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
    (lbStart, lbEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
    
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    
    shape = predictor(image, rects[0])
    shape = face_utils.shape_to_np(shape)
    
    # determine the facial landmarks for the face region, then
	
    # convert the landmark (x, y)-coordinates to a NumPy array
    	
    shape = predictor(gray, rects[0])
    shape = face_utils.shape_to_np(shape)
    
    # extract the left and right eye coordinates, then use the
    # coordinates to compute the eye aspect ratio for both eyes
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    mouth = shape[mStart:mEnd]
    l_eyebrow = shape[lbStart:lbEnd]
    r_eyebrow = shape[rbStart:rbEnd]
    
    clone = image.copy()
    
    
    cv2.fillPoly(clone, pts=[leftEye], color=(0,0,0))
    cv2.fillPoly(clone, pts=[rightEye], color=(0,0,0))
    cv2.fillPoly(clone, pts=[mouth], color=(0,0,0))
    cv2.fillPoly(clone, pts=[l_eyebrow], color=(0,0,0))
    cv2.fillPoly(clone, pts=[r_eyebrow], color=(0,0,0))
    
    '''
    clone = clone[:, :, ::-1]
    		
    plt.imshow(clone)
    plt.title("Image")
    plt.show()
    '''
    return clone
    
    

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


for img_path in picture_names:
    try:
        #load image
        image = cv2.imread(img_path)
        #make a copy of the image for processing
        img = image.copy()
        
        #mask facial features like eyes, eyebrows, and lips
        img = facial_features(img)
        
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
        #face_gray = cv2.medianBlur(face_gray,5)
        
        #face_gray = cv2.blur(face_gray,(7,7))
        face_th = cv2.adaptiveThreshold(face_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,11,2)
        
        #face_th = cv2.erode(face_th, kernel, iterations=1)
        #plt.imshow(face_th, 'gray')
        #plt.show()
        
    
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
        '''
        face_np = face.convert('RGB')
        face_np = np.array(face_np)
        face_np = face_np[:, :, ::-1].copy()
        '''
        
        #splitting colors
        R = face_and[:,:,2]
        '''
        R = int(np.median(face_and[:, :, 2]))
        G = int(np.median(face_and[:, :, 1]))
        B = int(np.median(face_and[:, :, 0]))
        
        
        median = [B,R,G]'''
        
        l = len(R[np.nonzero(R)])
        
        mean = [int(x/l) for x in ImageStat.Stat(face).sum]
        
        #to convert BGR to RGB
        org_img = image[:, :, ::-1].copy()
        #draw an ellipse with fill color as the detected skin color
        out = Image.fromarray(org_img)
        d = Draw(out)
        d.ellipse(((0,0),(0.2*image.shape[0],0.2*image.shape[1])), fill = tuple(mean))
        success += 1
        out.save('../results/out7/out_file_'+str(success)+'.jpg')
        print('Success ' + str(success))
        #plt.imshow(out)
        #plt.show()
    except:
        fail += 1
        print('Fail ' + str(fail))
        
print('Success: ' + str(success) + '\tFail: ' + str(fail))