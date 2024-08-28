# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 20:24:22 2024

@author: Alfredo
"""

import streamlit as st
import numpy as np
import cv2
from datetime import datetime

import tensorflow as tf # Importing TensorFlow


st.title('🎈 Safety Gear Identifier ') #title of app


st.info('This app uses a conv model to identify if the person on camera is or is not wearing safety gear') #adding info text to the app
st.write('Hello world!')


logtxtbox = st.empty() #empty textbox used to print class


model = tf.keras.models.load_model('Notebooks/safety_gear_detect_V4.keras') #load the model from memory




#function to use the model to predict if image is positive or negative and print the result
def im_assesment(img):
    #preprocess image
    preprocessed_image = cv2.resize(img, (400, 400))
    preprocessed_image = np.array(preprocessed_image) / 255.0

    #predict the class using the image
    predic = model(np.expand_dims(preprocessed_image, axis = 0))
    
    #given the labels, predict the class as max value     
    labels = ['pos', 'neg']
    preds_class = np.argmax(predic)
    preds_label = labels[preds_class]
    now = datetime
    
    # Get the current timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    logtxtbox.header(f"Predicted Class: {preds_label} | Timestamp: {timestamp}")
    
    if preds_class == 1:
        cl_pred_im.image('banners/warning.png')
    else:
        cl_pred_im.empty()



cap = cv2.VideoCapture(1) # using CV2 to capture images using the webcam n in the system (on my laptop 0 is back 1 is front cam)

col1, col2 = st.columns([4,1]) #2 columns to display image and prediction

with col1:
    #create frame for viewing image
    frame_placeholder = st.empty()

with col2:
    cl_pred_im = st.empty() #image to display if positive or negative


capture_button_pressed = st.button("Capture", key = 'c') #add stop button
stop_button_pressed = st.button("Stop", key = 'q', type= 'primary') #add stop button


#capture image, calculate prediction and display until Stop button is pressed
while cap.isOpened() and not stop_button_pressed:
    
    ret, frame = cap.read() #get the frame from video capture device (actual image), ret --> true or False if the image was captured
    
    if not ret: #break the loop if capture fails
        st.write("The Video capture has ended")
        break
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #transform from BGR to RGB
    frame_placeholder.image(frame, channels = 'RGB') #display in the frame
    
    im_assesment(frame) #predict image class
    
    if cv2.waitKey(1) & 0xFF == ord('q') or stop_button_pressed: # wait 1ms to see if we pressed 'q' to exit the while loop
        break

cap.release() #release cameras to be used 
cv2.destroyAllWindows() #close all 