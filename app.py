import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
# import mediapipe as mp
from skimage.transform import resize
from sklearn.cluster import KMeans
import pickle
from skimage.feature import ORB
from skimage.color import rgb2gray
from skimage.io import imread


# # import cv2
# # import numpy as np
# # from skimage.transform import resize
# # from skimage.io import imread

# # # Importing Required Modules 
# # from rembg import remove 
# # from PIL import Image 
# # import pickle



img_file_buffer = st.camera_input("Take a picture")

# # Initialize holistic model and drawing utils
# mp_holistic = mp.solutions.holistic
# mp_drawing = mp.solutions.drawing_utils

# # Function to process the image and detect hand
# def detect_hand(image):
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # Initialize holistic model with min detection confidence
#     holistic_model = mp_holistic.Holistic(
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5
#     )

#     # Make prediction on the image
#     results = holistic_model.process(image_rgb)

#     # Extract hand landmarks
#     if results.right_hand_landmarks:
#         # Get landmarks for right hand
#         hand_landmarks = results.right_hand_landmarks

#         # Extract bounding box coordinates for the hand
#         bbox_min_x = min(hand_landmarks.landmark, key=lambda x: x.x).x
#         bbox_min_y = min(hand_landmarks.landmark, key=lambda x: x.y).y
#         bbox_max_x = max(hand_landmarks.landmark, key=lambda x: x.x).x
#         bbox_max_y = max(hand_landmarks.landmark, key=lambda x: x.y).y

#         # Convert bounding box coordinates to pixel values
#         img_height, img_width, _ = image.shape
#         bbox_min_x = int(bbox_min_x * img_width)
#         bbox_min_y = int(bbox_min_y * img_height)
#         bbox_max_x = int(bbox_max_x * img_width)
#         bbox_max_y = int(bbox_max_y * img_height)

#         # Add extra space around the bounding box (e.g., 20 pixels)
#         extra_space = 20
#         bbox_min_x -= extra_space
#         bbox_min_y -= extra_space
#         bbox_max_x += extra_space
#         bbox_max_y += extra_space

#         # Ensure the coordinates are within the image boundaries
#         bbox_min_x = max(0, bbox_min_x)
#         bbox_min_y = max(0, bbox_min_y)
#         bbox_max_x = min(img_width, bbox_max_x)
#         bbox_max_y = min(img_height, bbox_max_y)

#         # Crop hand region from the image
#         hand_cropped = image[bbox_min_y:bbox_max_y, bbox_min_x:bbox_max_x]

#         return hand_cropped

#     else:
#         return None

# # Display title
# st.title("Hand Detection and Cropping")


# if img_file_buffer is not None:
#     # Read the uploaded image
#     image = cv2.imdecode(np.frombuffer(img_file_buffer.read(), np.uint8), 1)

#     # Display the uploaded image
#     st.subheader("Uploaded Image")
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # Detect hand and crop if detected
#     hand_cropped = detect_hand(image)
if img_file_buffer is not None:
    # Display the cropped hand region
    # st.subheader("Cropped Hand Region")
    # plt.imshow(cv2.cvtColor(hand_cropped, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # st.pyplot()
    
    Categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    
    
    ##########################################################
    # load the model from disk
    loaded_model = pickle.load(open("svm_model.pkl", 'rb'))
    
    # Function to extract ORB features from a single image
    def extract_sift_features(image):
        # Convert the image to grayscale if it's not already
        if len(image.shape) == 3:
            image = rgb2gray(image)

        # Initialize ORB detector
        orb = ORB(n_keypoints=1000)  # You can adjust the number of keypoints as needed

        # Detect ORB keypoints and descriptors
        orb.detect_and_extract(image)
        keypoints = orb.keypoints
        descriptors = orb.descriptors

        return descriptors



    # # Function to extract SIFT features from a single image
    # def extract_sift_features(image):
    #     # Convert the image to grayscale if it's not already
    #     if len(image.shape) == 3:
    #         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #     # Initialize SIFT detector
    #     sift = cv2.SIFT_create()

    #     # Detect SIFT keypoints and descriptors
    #     keypoints, descriptors = sift.detectAndCompute(image, None)

    #     return descriptors

    # Load the image
    # image_path = 'output.png'
    image = cv2.imread(img_file_buffer)
    # image = cv2.cvtColor(hand_cropped, cv2.COLOR_BGR2GRAY)

    # Extract SIFT descriptors for the image
    descriptors = extract_sift_features(image)

    if descriptors is not None:
        # Apply k-means clustering to create bins/clusters
        num_clusters = 100  # Number of clusters (you can adjust this)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(descriptors)

        # Function to generate histogram of features using k-means clusters
        def generate_histogram(image_descriptors):
            # Predict cluster indices for each descriptor
            cluster_indices = kmeans.predict(image_descriptors)

            # Create histogram of features
            hist, _ = np.histogram(cluster_indices, bins=num_clusters, range=(0, num_clusters))

            # Normalize the histogram to sum to 1
            hist = hist.astype(float)
            hist /= hist.sum()

            return hist

        # Generate histogram of features using k-means clusters
        hist = generate_histogram(descriptors)

        # Print the shape of the histogram
        print("Histogram shape:", hist.shape)
    else:
        print("No SIFT descriptors found in the image.")
        
        
    # predict the label of the image
    # Predict the label of the image using the trained SVM model
    predicted_label = loaded_model.predict(hist.reshape(1, -1))
    
    # PRINT THE PREDICTED LABEL
    st.write("The predicted character is: ", predicted_label)
            
        


# Categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']


# if img_file_buffer is not None:
    
#     # Initialize holistic model and drawing utils
#     mp_holistic = mp.solutions.holistic
#     mp_drawing = mp.solutions.drawing_utils

#     # Load the image
#     # image_path = '/kaggle/input/download/download.jpg'
#     image = cv2.imread(img_file_buffer)

#     # Convert the image from BGR to RGB
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # Initialize holistic model with min detection confidence
#     holistic_model = mp_holistic.Holistic(
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5
#     )

#     # Make prediction on the image
#     results = holistic_model.process(image_rgb)

#     # Extract hand landmarks
#     if results.right_hand_landmarks:
#         # Get landmarks for right hand
#         hand_landmarks = results.right_hand_landmarks

#         # Extract bounding box coordinates for the hand
#         bbox_min_x = min(hand_landmarks.landmark, key=lambda x: x.x).x
#         bbox_min_y = min(hand_landmarks.landmark, key=lambda x: x.y).y
#         bbox_max_x = max(hand_landmarks.landmark, key=lambda x: x.x).x
#         bbox_max_y = max(hand_landmarks.landmark, key=lambda x: x.y).y

#         # Convert bounding box coordinates to pixel values
#         img_height, img_width, _ = image.shape
#         bbox_min_x = int(bbox_min_x * img_width)
#         bbox_min_y = int(bbox_min_y * img_height)
#         bbox_max_x = int(bbox_max_x * img_width)
#         bbox_max_y = int(bbox_max_y * img_height)

#         # Add extra space around the bounding box (e.g., 20 pixels)
#         extra_space = 20
#         bbox_min_x -= extra_space
#         bbox_min_y -= extra_space
#         bbox_max_x += extra_space
#         bbox_max_y += extra_space

#         # Ensure the coordinates are within the image boundaries
#         bbox_min_x = max(0, bbox_min_x)
#         bbox_min_y = max(0, bbox_min_y)
#         bbox_max_x = min(img_width, bbox_max_x)
#         bbox_max_y = min(img_height, bbox_max_y)

#         # Crop hand region from the image
#         hand_cropped = image[bbox_min_y:bbox_max_y, bbox_min_x:bbox_max_x]

#         # Display the cropped hand region
#         plt.imshow(cv2.cvtColor(hand_cropped, cv2.COLOR_BGR2RGB))
#         plt.axis('off')  # Turn off axis
#         plt.show()
        
#     else:
#         print('No hand detected in the image.')

#     # Release resources
#     holistic_model.close()



# import streamlit as st
# import cv2
# import numpy as np
# from skimage.transform import resize
# from skimage.io import imread

# # Importing Required Modules 
# from rembg import remove 
# from PIL import Image 
# import pickle
# import mediapipe as mp

# # Initialize Mediapipe Hands model
# mpHands = mp.solutions.hands
# hands = mpHands.Hands(
#     static_image_mode=False,
#     model_complexity=1,
#     min_detection_confidence=0.75,
#     min_tracking_confidence=0.75,
#     max_num_hands=2
# )

# # Function to crop hand from the image
# def crop_hand(image, hand_landmarks):
#     x_min, y_min = np.min(hand_landmarks, axis=0)
#     x_max, y_max = np.max(hand_landmarks, axis=0)
#     cropped_hand = image[int(y_min):int(y_max), int(x_min):int(x_max)]
#     return cropped_hand

# Categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# # Streamlit app


# # Camera input to capture a picture
# img_file_buffer = st.camera_input("Capture a picture")

# if img_file_buffer is not None:
#     img = Image.open(img_file_buffer)

#     # Convert to RGB and process with Mediapipe Hands
#     img_rgb = np.array(img.convert('RGB'))
#     results = hands.process(img_rgb)

#     # Check if hands are present in the image
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             # Crop the hand region
#             cropped_hand = crop_hand(img_rgb, np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark]))
#             # Remove background using rembg
#             cropped_hand = remove(cropped_hand)
#             # Resize and flatten the image for classification
#             target_img_resized = resize(cropped_hand, (96, 96, 3))
#             target_img_flatten = target_img_resized.flatten()

#             # Load saved PCA model
#             saved_pca = pickle.load(open("pca_model.pkl", 'rb'))

#             # Subtract mean and project onto eigenvectors
#             mean_vector = saved_pca.mean_
#             eigenvectors = saved_pca.components_
#             centered_image = target_img_flatten - mean_vector
#             pca_transformed = np.dot(centered_image, eigenvectors.T)

#             # Reshape the pca_transformed array
#             pca_transformed = pca_transformed.reshape(1, -1)

#             # Load SVM model for classification
#             model = pickle.load(open("svm_model.pkl", 'rb'))

#             # Predict probabilities and show results
#             probability = model.predict_proba(pca_transformed)
#             for ind, val in enumerate(Categories):
#                 st.write(f'{val} : {probability[0][ind]}')
#             st.write("The predicted character is: ", Categories[np.argmax(probability)])


    