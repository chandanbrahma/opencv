#!/usr/bin/env python
# coding: utf-8

# In[35]:


import os


# In[36]:


path = 'C:\datascience\opencv'
os.chdir(path)


# In[17]:


import cv2
import numpy as np
import time


# In[34]:


classify_car = cv2.CascadeClassifier('haarcascade_car.xml')
vid_capture = cv2.VideoCapture('Video.mp4')
while vid_capture.isOpened():    
    ret,frame=vid_capture.read()    
    greyscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
    cars_detected = classify_car.detectMultiScale(grayscale_img, 1.4 ,2)    
    for (x,y,w,h) in cars_detected:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,255), 2)
        cv2.imshow('here are the cars',frame)   
    if cv2.waitkey(1) & 0xFF == ord('q'):            
        break
            
vid_capture.release()
cv2.destroyAllWindows()     
            
            
        
        
    
        


# In[ ]:





# In[2]:


import cv2
import time
import numpy as np

# Create our body classifier
car_classifier = cv2.CascadeClassifier('haarcascade_car.xml')

# Initiate video capture for video file
cap = cv2.VideoCapture('Video.mp4')


# Loop once video is successfully loaded
while cap.isOpened():
    
    time.sleep(.0005)
    # Read first frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    # Pass frame to our car classifier
    cars = car_classifier.detectMultiScale(gray, 1.4, 2)
    
    # Extract bounding boxes for any bodies identified
    for (x,y,w,h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.imshow('Cars', frame)

    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:




