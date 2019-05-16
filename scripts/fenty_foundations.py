#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 14:17:10 2019

@author: claran

The algorithm used is as follows:
    1. Load image
    2. detect and crop face
    3. mask facial features
    4. Convert to BW image with adaptive threshold. This BW image will be used as a mask
    5. convert the mask from 2D to 3D and apply to color image using AND operation
    5. remove 20% from all sides.
    6. find the skin colour using median while ignoring the masked region.
    7. match skin color to the fenty foundation colors
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

#foundation colors obstained from fenty website
fenty = {130:(237,212,181), 
         120:(249,228,207),
         110:(250,232,220),
         105:(246,220,179),
         100:(250,234,223),
         140:(248,226,201),
         145:(221,193,152),
         150:(236,211,187),
         160:(238,205,173),
         200:(225,289,158),
         190:(228,195,165),
         185:(226,197,163),
         180:(244,214,187),
         170:(226,192,167),
         210:(218,174,138),
         220:(232,189,153),
         230:(217,176,144),
         235:(216,173,125),
         270:(204,158,116),
         260:(214,170,132),
         255:(210,171,124),
         250:(220,174,132),
         240:(211,168,132),
         280:(233,176,138),
         290:(213,177,133),
         300:(194,148,106),
         310:(201,145,97),
         350:(175,122,80),
         345:(184,144,104),
         340:(164,120,88),
         330:(202,147,97),
         320:(202,146,106),
         360:(181,127,80),
         370:(192,133,94),
         380:(194,138,98),
         385:(165,125,83),
         430:(131,82,49),
         420:(162,107,65),
         410:(149,95,54),
         400:(160,101,50),
         390:(160,109,71),
         440:(146,93,59),
         445:(130,92,57),
         450:(123,81,54),
         460:(118,71,49),
         498:(56,45,40),
         495:(74,42,26),
         490:(84,51,29),
         480:(98,60,37),
         470:(100,58,34)
         }

 
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
    K = 3
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    
    
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    
    return res2


#function to recommend the top 4 foundations
def match_foundation(skin_color):
    color_log = {}
    options = [0,0,0,0]
    
    for k,v in fenty.items():
        p1 = np.array(v)
        p2 = np.array(skin_color)
        p2 = p2[::-1]   #converting skin color from BGR to RGB
        dist = np.linalg.norm(p2-p1)
        c = {k:dist}
        color_log.update(c)
        '''if(dist < min_dist):
            #print(dist,min_dist)
            min_dist = dist
            color_code = k'''
    
    for i in range(4):
        options[i] = min(color_log,key=color_log.get)
        color_log.pop(min(color_log,key=color_log.get))
        
    
    return options
        
        



#img_path = '../picture/pic.jpg'
for r, d, f in os.walk(thisdir):
    for file in f:
        if ".jpg" in file:
            picture_names.append(os.path.join(r,file))

'''
os.chdir('../results')
directory = 'out11'
if not os.path.exists(directory):
    os.makedirs(directory)
    print('new directory_created and changed')
    print(os.getcwd())
    
#os.chdir(directory)
            
'''

for img_path in picture_names[:5]:
    try:
        #load image
        #img_path = '/Users/claran/Downloads/DiandraForrestAlbino.jpg'
        #img_path = '../women_headshots/59. ivana_4x3_cl.jpg'
        image = cv2.imread(img_path)
        #make a copy of the image for processing
        img = image.copy()
        
        #correction for uneven lighting
        lab_img = cv2.cvtColor(image,cv2.COLOR_BGR2Lab)
        l,a,b = cv2.split(lab_img)
        plt.imshow(l,cmap='gray')
        #plt.show()
        plt.imshow(a,cmap='gray')
        #plt.show()
        plt.imshow(b,cmap='gray')
        #plt.show()
        #plt.imshow(lab_img)
        #plt.show()
        #L,A,B = cv2.split(lab_img)
        #clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        #cl = clahe.apply(L)
        #limg = cv2.merge((cl,A,B))
        #plt.imshow(limg)
        #lab_img[: ,:, 0] = cv2.equalizeHist(lab_img[: ,:, 0])
        #lab_img[: ,:, 0] = cv2.medianBlur(lab_img[: ,:, 0],5)
        #lab_img[: ,:, 0] = np.full(lab_img[: ,:, 0].shape,np.median(lab_img[: ,:, 0]))
        img = cv2.cvtColor(lab_img,cv2.COLOR_Lab2RGB)
        #plt.imshow(img)
        plt.axis('off')
        #plt.title('Correction for uneven lighting')
        #plt.show()
        img = cv2.cvtColor(lab_img,cv2.COLOR_Lab2BGR)
        
        
        #mask facial features like eyes, eyebrows, and lips
        img = facial_features(img)
        
        #displaying initial mask
        plt.imshow(img[:,:,::-1])
        plt.show()
        
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

        #displaying cropped face
        plt.imshow(img)
        plt.show()
        cropped_face = img
        
        
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
        
        #displaying image after both masks
        plt.imshow(face_and)
        plt.show()
        both_mask = face_and


        #face_and = face_and[:, :, ::-1]
        face = Image.fromarray(face_and)
        
        #color quantization
        quantized_image = color_quant(face)
        quantized_image = quantized_image[:, :, ::-1]
        
        #displaying image after color quantization
        plt.imshow(quantized_image)
        plt.show()
        #This section is not being used in the final code, but even if used it gives similar result
        #uncomment the below line to use color quantization in the code
        #face = Image.fromarray(quantized_image)
        

        
        #face = Image.fromarray(quantized_image)
        
        face_np = face.convert('RGB')
        face_np = np.array(face_np)
        face_np = face_np[:, :, ::-1].copy()

        
        #removing 20% from all sides in the image. This removes the effect of hair and maximizes the area of the skin
        face_np = face_np[int(.2*face_and.shape[0]):int(.8*face_and.shape[0]), int(.2*face_and.shape[1]):int(.8*face_and.shape[1])]
        
        
        
        #splitting colors
        #R = face_and[:,:,2]
        
        #R = int(np.median(face_np[:, :, 2]))
        #G = int(np.median(face_np[:, :, 1]))
        #B = int(np.median(face_np[:, :, 0]))
        
        R,G,B = cv2.split(face_np)
        
        median = [0,0,0]
        median[2] = int(np.median(B[np.nonzero(B)]))
        median[1] = int(np.median(G[np.nonzero(G)]))
        median[0] = int(np.median(R[np.nonzero(R)]))
        
        
        #l = len(R[np.nonzero(R)])
        #mean = [int(x/l) for x in ImageStat.Stat(face).sum]
        #median = ImageStat.Stat(face).median
        
        
        #recommending top 4 foundations
        k = match_foundation(median)
        #print(k)
        
        foundation_best = fenty[k[0]]
        foundation_best = foundation_best[::-1]
        
        foundation_1 = fenty[k[1]]
        foundation_1 = foundation_1[::-1]
        
        foundation_2 = fenty[k[2]]
        foundation_2 = foundation_2[::-1]
        
        foundation_3 = fenty[k[3]]
        foundation_3 = foundation_3[::-1]
        
        
        #adding recommendation patches
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.circle(image,(int(0.1*image.shape[0]),int(0.15*image.shape[0])),int(0.1*image.shape[0]),(median),-1)
        cv2.circle(image,(int(0.1*image.shape[0]),2*int(0.15*image.shape[0])),int(0.1*image.shape[0]),(foundation_best),-1)
        cv2.putText(image,'Skin',(int(0.1*image.shape[0]),int(0.15*image.shape[0])), font, 0.002*image.shape[0],(255,255,255),int(0.01*image.shape[1]),cv2.LINE_AA)
        cv2.putText(image,'Foundation'+str(k[0]),(int(0.1*image.shape[0]),2*int(0.15*image.shape[0])), font, 0.002*image.shape[0],(255,255,255),int(0.01*image.shape[1]),cv2.LINE_AA)
        cv2.putText(image,'Other options',(int(0.1*image.shape[1]),int(0.8*image.shape[0])), font, 0.002*image.shape[0],(255,255,255),int(0.01*image.shape[1]),cv2.LINE_AA)
        cv2.circle(image,(int(0.2*image.shape[1]),int(0.9*image.shape[0])),int(0.1*image.shape[0]),(foundation_1),-1)
        cv2.circle(image,(int(0.5*image.shape[1]),int(0.9*image.shape[0])),int(0.1*image.shape[0]),(foundation_2),-1)
        cv2.circle(image,(int(0.8*image.shape[1]),int(0.9*image.shape[0])),int(0.1*image.shape[0]),(foundation_3),-1)
        cv2.putText(image,str(k[1]),(int(0.2*image.shape[1]),int(0.9*image.shape[0])), font, 0.002*image.shape[0],(255,255,255),int(0.01*image.shape[1]),cv2.LINE_AA)
        cv2.putText(image,str(k[2]),(int(0.5*image.shape[1]),int(0.9*image.shape[0])), font, 0.002*image.shape[0],(255,255,255),int(0.01*image.shape[1]),cv2.LINE_AA)
        cv2.putText(image,str(k[3]),(int(0.8*image.shape[1]),int(0.9*image.shape[0])), font, 0.002*image.shape[0],(255,255,255),int(0.01*image.shape[1]),cv2.LINE_AA)
        org_img = image[:, :, ::-1].copy()
        #draw an ellipse with fill color as the detected skin color
        out = Image.fromarray(org_img)
        success += 1
        #create a folder for each image and store all the interemediate files in the new folder
        #out.save('../results/out10/out_file_'+str(success)+'.jpg')
        print('Success ' + str(success))
        #displaying the final output
        plt.imshow(out)
        plt.axis('off')
        #plt.show()
        
        '''
        directory = 'out_file_' + str(success)
        if not os.path.exists(directory):
            os.makedirs(directory)
            os.chdir(directory)
            print('directory changed to' + str(directory))
        
        out.save('out_file_'+str(success)+'/final.jpg')
        cropped_face.save('out_file_'+str(success)+'/masked_features.jpg')
        both_mask.save('out_file_'+str(success)+'/adaptive_BW_mask.jpg')
        #quantized_image.save('out_file_'+str(success)+'/quantized.jpg')
        
        '''
        
        cropped_face = Image.fromarray(cropped_face)
        both_mask = Image.fromarray(both_mask)
        
        out.save('/Users/claran/Documents/conex/cosmetic_recommender/results/out11/out_file_'+str(success)+'final.jpg')
        cropped_face.save('/Users/claran/Documents/conex/cosmetic_recommender/results/out11/out_file_'+str(success)+'masked_features.jpg')
        both_mask.save('/Users/claran/Documents/conex/cosmetic_recommender/results/out11/out_file_'+str(success)+'adaptive_BW_mask.jpg')
        #quantized_image.save('/Users/claran/Documents/conex/cosmetic_recommender/results/out11/out_file_'+str(success)+'quantized.jpg')
        

    except:
        fail += 1
        print('Fail ' + str(fail))
        
print('Success: ' + str(success) + '\tFail: ' + str(fail))