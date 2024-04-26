import streamlit as st
import cv2
import numpy as np
from skimage.transform import resize
from skimage.io import imread

# Importing Required Modules 
from rembg import remove 
from PIL import Image 


img_file_buffer = st.camera_input("Take a picture")






if img_file_buffer is not None:
    # Processing the image 
    input = Image.open(img_file_buffer) 

    # Removing the background from the given Image 
    img_file_buffer = remove(input) 
    # To read image file buffer with OpenCV:
    # bytes_data = img_file_buffer.getvalue()
    # cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    # resize the image
    img_file_buffer=imread(img_file_buffer) 
    cv2_img = resize(img_file_buffer, (64, 64, 3))
    ## flatten the image
    flat_data = cv2_img.flatten()
    flat_data=np.array(flat_data) 
    # reshape to 2D array
    flat_data = flat_data.reshape(1,-1)
    
    # select only 10 features from 0 to 10
    flat_data = flat_data[:,0:10]

    # # transfor the array using pca to select 10 features
    # from sklearn.decomposition import PCA
    # num_features = 10  # Number of features to extract
    # pca = PCA(n_components=num_features)  # PCA for feature extraction
    # # Perform PCA to extract 10 features
    # pca.fit(flat_data)
    # flat_data_pca = pca.transform(flat_data)
    
    # load the model
    import pickle
    # load the model from disk
    loaded_model = pickle.load(open('svm_model.sav', 'rb'))
    # make a prediction
    prediction = loaded_model.predict(flat_data)
    st.write(prediction)
    



    