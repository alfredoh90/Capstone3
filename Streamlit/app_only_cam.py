# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 15:03:01 2024

@author: Alfredo
"""

import streamlit as st
import numpy as np
import cv2
from datetime import datetime
import io
import zipfile
from PIL import Image


st.title('🎈 Safety Gear Identifier ') #title of app


st.info('This app uses a conv model to identify if the person on camera is or is not wearing safety gear') #adding info text to the app
st.write('Hello world!')

logtxtbox = st.empty() #empty textbox used to print class



#function to use the model to predict if image is positive or negative and print the result
def im_assesment(img):
    #preprocess image
    preprocessed_image = cv2.resize(img, (400, 400))
    preprocessed_image = np.array(preprocessed_image) / 255.0

    #predict the class using the image
    #predic = model(np.expand_dims(preprocessed_image, axis = 0))
    predic = [0, 1]
    #given the labels, predict the class as max value     
    labels = ['pos', 'neg']
    preds_class = np.argmax(predic)
    preds_label = labels[preds_class]
    
    # Get the current timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    timestamp_log = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    
    
    logtxtbox.header(f"Predicted Class: {preds_label} | Timestamp: {timestamp}")
    
    # Store the current pred in session state
    st.session_state.pred_image = f"{preds_label}_{timestamp_log}"
    
    if preds_class == 1:
        cl_pred_im.image('banners/warning.png')
    else:
        cl_pred_im.empty()
    
    

# Initialize Streamlit session state
if 'pred_image' not in st.session_state:
    st.session_state.pred_image = ''

if 'captured_images' not in st.session_state:
    st.session_state.captured_images = []

if 'captured_images_captions' not in st.session_state:
    st.session_state.captured_images_captions = []

# Define a callback function for the capture button
def capture_button_callback():
    # Capture the current frame
    if 'frame' in st.session_state:
        st.session_state.captured_images.append(st.session_state.frame)
        st.session_state.captured_images_captions.append(st.session_state.pred_image)
        

# Define action to create a ZIP file from captured images
def create_zip_from_images():
    # Create an in-memory bytes buffer
    buffer = io.BytesIO()
    # Create a ZIP file
    with zipfile.ZipFile(buffer, 'w') as zip_file:
        for i, (image, caption) in enumerate(zip(st.session_state.captured_images, st.session_state.captured_images_captions)):
            # Convert image to PIL format and save to bytes
            pil_image = Image.fromarray(image)
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            # Add the image to the ZIP file with a caption as the file name
            zip_file.writestr(f'{caption}.png', img_byte_arr.getvalue())
    # Move to the beginning of the buffer
    buffer.seek(0)
    return buffer.getvalue()

zip_data = create_zip_from_images() # create empty zip_data

cap = cv2.VideoCapture(1) # using CV2 to capture images using the webcam n in the system (on my laptop 0 is back 1 is front cam)

col1, col2 = st.columns([4,1]) #2 columns to display image and prediction

with col1:
    #create frame for viewing image
    frame_placeholder = st.empty()

with col2:
    cl_pred_im = st.empty() #image to display if positive or negative

capt_button = st.button("Capture") #add capture button

# Display download button
st.download_button(
    label="Download All Images",
    data=zip_data,
    file_name="captured_images.zip"
    #mime="application/zip"
    )

    
stop_button_pressed = st.button("Stop", type= 'primary') #add stop button

cont_images_saved = st.container()



#capture image, calculate prediction and display until Stop button is pressed
while cap.isOpened() and not stop_button_pressed:
    
    ret, frame = cap.read() #get the frame from video capture device (actual image), ret --> true or False if the image was captured
    
    if not ret: #break the loop if capture fails
        st.write("The Video capture has ended")
        break
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #transform from BGR to RGB
    frame_placeholder.image(frame, channels = 'RGB') #display in the frame
    
    im_assesment(frame) #predict image class
    
    # Store the current frame in session state
    st.session_state.frame = frame
    
    if capt_button:
        capture_button_callback() #add capture button
        capt_button = False #reset the button
        cont_images_saved.image(st.session_state.captured_images, caption = st.session_state.captured_images_captions, channels='RGB') #display images captured
        zip_data = create_zip_from_images()


st.text('Goodbye')   
cap.release() #release cameras to be used 
cv2.destroyAllWindows() #close all 