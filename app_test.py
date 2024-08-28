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


st.title('🎈 Safety Gear Identifier ')


st.info('This app uses a conv model to identify if the person on camera is or is not wearing safety gear')
st.write('Hello world!')


logtxtbox = st.empty()
#preds_label = 'pos'


#st.subheader(f'\nPredicted Class: {preds_label}')

model = tf.keras.models.load_model('Notebooks/safety_gear_detect_V4.keras')

def im_assesment(img): #use the model to predict if image is positive or negative and print the result
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
    
    logtxtbox.text(f"Predicted Class: {preds_label} | Timestamp: {timestamp}")


cap = cv2.VideoCapture(1) # using CV2 to capture images using the webcam n in the system (on my laptop 0 is back 1 is front cam

frame_placeholder = st.empty()
stop_button_pressed = st.button("Stop")

while cap.isOpened() and not stop_button_pressed: #dispay until we escape the loop
    ret, frame = cap.read() #get the frame from video capture device (actual image), ret --> true or False if the image was captured
    
    if not ret: #break the loop if capture fails
        st.write("The Video capture has ended")
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #transform from BGR to RGB
    frame_placeholder.image(frame, channels = 'RGB') #display in the frame
    
    im_assesment(frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q') or stop_button_pressed: # wait 1ms to see if we pressed 'q' to exit the while loop
        break

cap.release() #release cameras to be used 
cv2.destroyAllWindows() #close all 