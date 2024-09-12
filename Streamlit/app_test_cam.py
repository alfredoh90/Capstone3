# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 20:24:22 2024

@author: Alfredo
"""

import streamlit as st
import numpy as np
import cv2
from datetime import datetime
import io
import zipfile
from PIL import Image

import pyautogui

import tensorflow as tf # Importing TensorFlow


st.title('🎈 Safety Gear Identifier ') #title of app
st.logo("banners/logo.png") #adding the logo to top corner

print('starting')

left_co, cent_co,last_co = st.columns(3) #center the logo
with cent_co:
      st.image("banners/logo.png", width = 200) #adding the logo to the starting image


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

print('loading model')
if "model" not in st.session_state.keys():
    with st.spinner('Model is being loaded..'):
        st.session_state["model"] = tf.keras.models.load_model('Notebooks/safety_gear_detect_V4.keras') #load the model from memory
model = st.session_state["model"]
    
print('model loaded')



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
    timestamp_log = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    
    logtxtbox.header(f"Predicted Class: {preds_label} | Timestamp: {timestamp}") #Display results
    
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

print('starting loop')

cap = cv2.VideoCapture(1) # using CV2 to capture images using the webcam n in the system (on my laptop 0 is back 1 is front cam)

col1, col2 = st.columns([4,1]) #2 columns to display image and prediction

with col1:
    #create frame for viewing image
    frame_placeholder = st.empty()

with col2:
    cl_pred_im = st.empty() #image to display if positive or negative

capt_button = st.button("Capture") #add capture button
stop_button_pressed = st.button("Stop", type= 'primary') #add stop button
zip_data = create_zip_from_images()
# Display download button with the latest ZIP data
st.download_button(
    label="Download All Images",
    data=zip_data,
    file_name="captured_images.zip",
    mime="application/zip"
    )
cont_images_saved = st.container() #container to display images taken


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
        
        if j % 50 == 0: #predict every 50 frames
            im_assesment(frame) #predict image class
            
        
        # Store the current frame in session state
        st.session_state.frame = frame
    
        if capt_button:
            capture_button_callback() #add capture button
            capt_button = False #reset the button
            cont_images_saved.image(st.session_state.captured_images, caption = st.session_state.captured_images_captions, channels='RGB') #display images captured
            zip_data = create_zip_from_images()
            

        
        print(j)
        j+=1
    except Exception as e:
        print('Exception generated')
        print(e)
        break
print('Close')
cap.release() #release cameras to be used 
cv2.destroyAllWindows() #close all 

if st.button("Reset", type = 'primary'): #create reset button by reloading the page
    pyautogui.hotkey("ctrl","F5")
