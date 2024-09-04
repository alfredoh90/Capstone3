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


st.title('SafeCheck AI') #title of app

st.logo("banners/logo.png") #adding the logo to top corner
st.image("banners/logo.png") #adding the logo to the starting image

st.info('SafeCheck AI: Automating Safety, Protecting Lives.') #adding info text to the app

# Sidebar
st.sidebar.header(("The App"))
st.sidebar.markdown((
    "SafeCheck AI is a cutting-edge safety solution designed for warehouses and industrial sites, leveraging the power of advanced convolutional neural networks to ensure worker compliance with safety protocols. With real-time image recognition, SafeCheck AI detects if personnel are wearing essential protective gear—such as hard hats and vests—before they enter the operational area. The app helps prevent accidents, ensures compliance with safety regulations, and fosters a culture of responsibility in high-risk environments. SafeCheck AI is your go-to tool for automating safety checks and promoting a safer workplace."
))

st.sidebar.header(("About the team:"))
st.sidebar.markdown((
    "At SafeCheck AI, our team combines expertise in data science, supply chain operations, and technology to create innovative solutions for workplace safety. Led by :orange[Alfredo Hernandez], our Senior Data Specialist with a background in demand planning, supply chain management, and data analytics, we leverage cutting-edge machine learning models like convolutional neural networks to deliver accurate safety gear detection. Alfredo’s experience at companies like Shopify, SFN, and Flexport, along with his dual degrees in Industrial and Computer Science Engineering and a Master’s in Innovation, ensures our app is both effective and scalable for industrial environments."
    """
- [Linkedin](https://www.linkedin.com/in/a-hernandez-h/)
- [Github](https://github.com/alfredoh90/Capstone3)
"""
))


logtxtbox = st.empty() #empty textbox used to print class

with st.spinner('Model is being loaded..'):
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


#capture_button_pressed = st.button("Capture", key = 'c') #add stop button
stop_button_pressed = st.button("Stop", type= 'primary') #add stop button

j = 0
#capture image, calculate prediction and display until Stop button is pressed
while cap.isOpened() and not stop_button_pressed:
    try:
        ret, frame = cap.read() #get the frame from video capture device (actual image), ret --> true or False if the image was captured
        
        if not ret: #break the loop if capture fails
            st.write("The Video capture has ended")
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #transform from BGR to RGB
        frame_placeholder.image(frame, channels = 'RGB') #display in the frame
        
        if j % 50 == 0: #Calculate prediction every 50 frames
            im_assesment(frame) #predict image class
        j+=1
    except:
        print('Exception generated')
        break
print('Close')
cap.release() #release cameras to be used 
cv2.destroyAllWindows() #close all 