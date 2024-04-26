import streamlit as st
import numpy as np
# from skimage.io import imread
# from skimage.transform import resize

img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer:
    st.image(img_file_buffer)
    
# flat_data_arr=[] #input array 

# if img_file_buffer is not None:
#     # To read image file buffer with OpenCV:
#     bytes_data = img_file_buffer.getvalue()
#     # 
#     img_array=imread(img_file_buffer) 
#     img_resized=resize(img_array,(64,64,3)) 
#     flat_data_arr.append(img_resized.flatten()) 
#     flat_data=np.array(flat_data_arr) 
#     print(img_array.shape)
    
#     st.write(img_array)
#     st.write(flat_data)
    