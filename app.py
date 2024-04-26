import streamlit as st
import numpy as np
from PIL import Image


# from skimage.io import imread
# from skimage.transform import resize

img_file_buffer = st.camera_input("Take a picture")

# read the image and convert to array

image = Image.open(img_file_buffer)
image = np.array(image)

st.image(image, caption="Captured Image", use_column_width=True)



    