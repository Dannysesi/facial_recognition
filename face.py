import streamlit as st
from deepface import DeepFace
import cv2
from PIL import Image
import numpy as np
import os
import tempfile

# Title of the Streamlit app
st.title("Face Recognition App using DeepFace (Facenet)")

# Specify the folder containing reference images
REFERENCE_FOLDER = "known_faces"  # Replace this with the actual folder path

# Function to capture a frame from the webcam
def capture_image_from_webcam():
    st.write("Starting webcam...")

    # Start the video capture
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Could not access the webcam.")
        return None

    # Capture a single frame
    ret, frame = cap.read()

    # Release the webcam
    cap.release()

    if not ret:
        st.error("Error: Could not capture image from the webcam.")
        return None

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame_rgb

# Display the folder where reference images are stored
st.write(f"Checking faces against images in folder: {REFERENCE_FOLDER}")

# Capture an image from the webcam when the button is clicked
if st.button("Capture Image from Webcam"):
    # Capture the image from the webcam
    img2 = capture_image_from_webcam()

    if img2 is not None:
        # Display the captured webcam image
        st.image(img2, caption="Webcam Image", width=300)

        # Use `tempfile` to create a temporary file for the captured webcam image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as webcam_img_tempfile:
            img2_pil = Image.fromarray(img2)
            img2_pil.save(webcam_img_tempfile.name)

        # Perform face recognition against all images in the folder using DeepFace (Facenet model)
        try:
            st.write("Performing face recognition...")

            # Find matching faces in the folder
            result = DeepFace.find(img_path=webcam_img_tempfile.name, db_path=REFERENCE_FOLDER, model_name="Facenet")

            # If matches are found, display the best match
            if len(result) > 0 and not result[0].empty:
                # Get the best match
                best_match = result[0].iloc[0]
                best_match_image = best_match['identity']
                match_name = os.path.basename(best_match_image).split('.')[0]  # Extract name from filename

                st.success(f"Face identified: {match_name} ✅")
                st.image(best_match_image, caption=f"Best Match: {match_name}", width=300)
            else:
                st.error("Face detected, but no match found. ❌")

            # Clean up temporary webcam image
            os.remove(webcam_img_tempfile.name)

        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.info("Click the button to capture an image from the webcam for face recognition.")
