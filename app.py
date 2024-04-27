import streamlit as st
import cv2
import numpy as np
from skimage.transform import resize
from skimage.io import imread

# Importing Required Modules 
from rembg import remove 
from PIL import Image 
import pickle


img_file_buffer = st.camera_input("Take a picture")



Categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']


if img_file_buffer is not None:
    
    img = Image.open(img_file_buffer)
    
    img = remove(img) 
    img_array = np.array(img)
    
    target_img_resized=resize(img_array,(96,96,3))
    target_img_flatten = target_img_resized.flatten()

    # Load saved PCA model
    saved_pca = pickle.load(open("pca_model.pkl", 'rb'))

    # Subtract mean and project onto eigenvectors
    mean_vector = saved_pca.mean_
    eigenvectors = saved_pca.components_
    centered_image = target_img_flatten - mean_vector
    pca_transformed = np.dot(centered_image, eigenvectors.T)

    # reshape the pca_transformed array
    pca_transformed = pca_transformed.reshape(1, -1)
    
    model = pickle.load(open("svm_model.pkl", 'rb'))

    probability=model.predict_proba(pca_transformed)
    for ind,val in enumerate(Categories):
        print(f'{val} = {probability[0][ind]*100}%')
        print("The predicted image is : "+Categories[model.predict(pca_transformed)[0]])




    