#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#1.OpenCV-Python is a library of Python
#bindings designed to solve computer vision problems.
##2.Computer Vision, often abbreviated as CV, is defined as a
##field of study that seeks to develop techniques to help computers “see” and
##understand the content of digital images such as photographs and videos. The
##problem of computer vision appears simple because it is trivially solved by
##people, even very young children.
##
##3.Computer vision, on the other hand, focuses on making sense of
##what a machine sees. A computer vision system inputs an image and outputs
##task-specific knowledge, such as object labels and coordinates. Computer
##vision and image processing work together in many cases
import cv2

##The os module is a part of the standard library, or stdlib, within Python 3.
##This means that it comes with your Python installation, but you still must
##import it. Sample code using os: import os.

##The OS module in python provides functions for interacting with the
##operating system. OS, comes under Python's standard utility modules. This
##module provides a portable way of using operating system dependent functionality.
##The *os* and *os.path* modules include many functions to interact with the
##file system.
import os

#loads the load_model from keras model into memory
from keras.models import load_model

##NumPy is a general-purpose array-processing package. It provides a
##high-performance multidimensional array object, and tools for working with
##these arrays.
##It is the fundamental package for scientific computing with Python.
##It contains various features including these important ones
import numpy as np


##This module contains classes for loading Sound objects and controlling playback.
##The mixer module is optional and depends on SDL_mixer. Your program should
##test that pygame.mixerpygame module for loading and playing sounds is available
##and initialized before using it.
##The mixer module has a limited number of channels for playback of sounds.
##Usually programs tell pygame to start playing audio and it selects an available
##channel automatically
from pygame import mixer

#PYTHON HAS NO DATATYPE TIME SO  WE HAVE TO IMPORT IT EXPLICITLY
#import time

##The pygame.mixe           r.init() function takes several optional arguments to control the
##playback rate and sample size.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
mixer.init()
#create a new sound object from file 'alarm.wav'
#sound = mixer.Sound(r'E:\DummyData\Project Eye Detection\TestProject\beep.wav')

#load haar cascade files containing features
#pretrained classifiers used for face detection

face = cv2.CascadeClassifier('E:\DummyData\Project Eye Detection\TestProject\haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('E:\DummyData\Project Eye Detection\TestProject\haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('E:\DummyData\Project Eye Detection\TestProject\haar cascade files\haarcascade_righteye_2splits.xml')
print(face)

lbl=['Close','Open']

model = load_model(r'E:\DummyData\Project Eye Detection\TestProject\models\cnncat2.h5')
##os.getcwd() returns the absolute path of the working directory where Python is
##currently running as a string
##get current working directory
path = os.getcwd()
# capture a video from the camera we have to create VideoCapture object
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    height,width = frame.shape[:2] 
    #changing colorspaces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )
#the x and y location of the rectangle, and the rectangle’s width and height (w , h).
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w]
        count=count+1
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))
        r_eye= r_eye/255
        #Gives a new shape to an array without changing its data.
        r_eye=  r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        rpred = model.predict_classes(r_eye)
        if(rpred[0]==1):
            lbl='Open' 
        if(rpred[0]==0):
            lbl='Closed'
        break

    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]
        count=count+1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred = model.predict_classes(l_eye)
        if(lpred[0]==1):
            lbl='Open'   
        if(lpred[0]==0):
            lbl='Closed'
        break

    if(rpred[0]==0 and lpred[0]==0):
        score=score+1
        #gives anti-aliased line which looks great for curves.
        cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):
    else:
        score=score-1
        cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    #cv2.putText()::used to draw a text string on any image
        
    if(score<0):
        score=0   
    cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    if(score>15):
        #person is feeling sleepy so we beep the alarm
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)# used to save an image to any 
        #storage device. This will save the image according to the specified format in 
        #current working directory.
        try:
            sound.play()
            
        except:  # isplaying = False
            pass
        if(thicc<16):#thicc:Convert characters to their fullwidth representation.
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
    cv2.imshow('frame',frame)#display an image in a window. The window automatically 
    #fits to the image size
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # introduce a delay of n milliseconds while rendering images to windows. When used as 
        #cv::waitKey(0) it returns the key pressed by the 
        #user on the active window. This is typically used for keyboard input from user in 
        #OpenCV programs
        break
cap.release()#release the camera device resource
cv2.destroyAllWindows()
#When everything is done, release the capture video_capture.release()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




