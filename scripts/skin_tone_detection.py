#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 17:09:11 2019

@author: claran
"""

import dlib
from PIL.ImageDraw import Draw
from PIL import Image, ImageStat 
from skimage import io
import matplotlib.pyplot as plt



def detect_faces(image):

    # Create a face detector
    face_detector = dlib.get_frontal_face_detector()

    # Run detector and get bounding boxes of the faces on image.
    detected_faces = face_detector(image, 1)
    face_frames = [(x.left(), x.top(),
                    x.right(), x.bottom()) for x in detected_faces]

    return face_frames

# Load image
img_path = '../picture/pic.jpg'
image = io.imread(img_path)

# Detect faces
detected_faces = detect_faces(image)

# Crop faces and plot
#for n, face_rect in enumerate(detected_faces):    #use loop for multiple faces in the image
face = Image.fromarray(image).crop(detected_faces[0])
median = ImageStat.Stat(face).median

#printing the RGB value of the skin tone
print('RGB value of skin tone is: ')
print(median)

#showing the image with the skin tone on the top left
out = Image.fromarray(image)
d = Draw(out)
d.ellipse(((0,0),(0.2*image.shape[0],0.2*image.shape[1])), fill = tuple(median))
plt.subplot(1, len(detected_faces), n+1)
plt.axis('off')
plt.imshow(out)
