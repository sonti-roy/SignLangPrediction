import streamlit as st
import cv2

# st.title('Hello World!')

# st.write('This is a simple Streamlit app.')

# Access the camera
cap = cv2.VideoCapture(0)

# Check if camera is opened successfully
if not cap.isOpened():
    st.error("Unable to access camera")
else:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the captured frame
    st.image(frame, channels="BGR")

    # Button to click live picture
    if st.button("Click Picture"):
        # Save the frame as an image file
        cv2.imwrite("live_picture.jpg", frame)
        st.success("Picture saved successfully")

# Release the camera
cap.release()