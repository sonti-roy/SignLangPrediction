import streamlit as st
import numpy as np
from PIL import Image
import cv2



# from skimage.io import imread
# from skimage.transform import resize

img_file_buffer = st.camera_input("Take a picture")

# Saves
img = Image.open(img_file_buffer)
img = img.save("img.jpg")

# OpenCv Read
img = cv2.imread("img.jpg")

# Display
st.image(img, channels="BGR", use_column_width=True)



    