import streamlit as st
from deepface import DeepFace
import cv2
from PIL import Image
import os
import tempfile
import time
import json

# Title of the Streamlit app
st.title("AI-Based Facial Recognition for Airport Security")

# Specify the folder containing authorized passenger images
PASSENGER_DB = "known_faces"  # Replace with actual folder path
THREAT_DB = "known_threats"   # Replace with the folder containing flagged individuals

# Path to the JSON file containing passenger metadata
PASSENGER_DATA_FILE = "passenger_data.json"

# Load passenger data from the JSON file
def load_passenger_data():
    if os.path.exists(PASSENGER_DATA_FILE):
        with open(PASSENGER_DATA_FILE, 'r') as f:
            return json.load(f)
    return {}

# Initialize session state for controlling the video stream
if 'stop_recording' not in st.session_state:
    st.session_state['stop_recording'] = False

last_passenger_name = None  # Variable to store the last identified passenger
last_threat_name = None     # Variable to store the last identified threat

# Function to capture real-time video from the webcam
def capture_video_without_display():
    global last_passenger_name, last_threat_name
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Error: Could not access the webcam.")
        return
    
    st.write("Starting live video stream for facial recognition...")

    # Create a placeholder for the stop button so it can be updated
    stop_button_placeholder = st.empty()

    if stop_button_placeholder.button("Stop Recording"):
        st.session_state['stop_recording'] = True

    # Process the video stream
    while cap.isOpened():
        # Check if the stop button was pressed
        if st.session_state['stop_recording']:
            st.write("Video recording stopped.")
            break

        ret, frame = cap.read()
        if not ret:
            st.error("Error: Could not read frame from the webcam.")
            break
        
        # Convert BGR to RGB (DeepFace expects RGB images)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Save the current frame to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_img:
            img_pil = Image.fromarray(frame_rgb)
            img_pil.save(temp_img.name)

        # Perform face recognition for both passenger verification and threat detection
        try:
            # Check if the person is in the passenger database
            passenger_result = DeepFace.find(img_path=temp_img.name, db_path=PASSENGER_DB, model_name="Facenet")

            # Check if the person is in the threat database
            threat_result = DeepFace.find(img_path=temp_img.name, db_path=THREAT_DB, model_name="Facenet")

            # Load passenger metadata
            passenger_data = load_passenger_data()

            # Handle passenger identification
            if len(passenger_result) > 0 and not passenger_result[0].empty:
                best_match_passenger = passenger_result[0].iloc[0]
                passenger_name = os.path.basename(best_match_passenger['identity']).split('.')[0]

                # Display the result only if the passenger is different from the last identified one
                if passenger_name != last_passenger_name:
                    st.success(f"Passenger Identified: {passenger_name} ✅")
                    last_passenger_name = passenger_name  # Update the last passenger name
                    
                    # Retrieve and display the contact information if available
                    if passenger_name in passenger_data:
                        contact_info = passenger_data[passenger_name].get("contact", "N/A")
                        from_location = passenger_data[passenger_name].get("from", "N/A")
                        to_location = passenger_data[passenger_name].get("to", "N/A")
                        email = passenger_data[passenger_name].get("email", "N/A")
                        
                        info_message = f"""
                            Contact Info: {contact_info}\n
                            Traveling From: {from_location}\n
                            Traveling To: {to_location}\n
                            Email: {email}
                            """
                        st.info(info_message)
                    else:
                        st.warning("No additional information available for this passenger.")

            else:
                if last_passenger_name is not None:
                    st.error("Passenger not recognized. ❌")
                    last_passenger_name = None  # Reset last passenger name if no match

            # Handle threat detection
            if len(threat_result) > 0 and not threat_result[0].empty:
                best_match_threat = threat_result[0].iloc[0]
                threat_name = os.path.basename(best_match_threat['identity']).split('.')[0]

                # Display the result only if the threat is different from the last identified one
                if threat_name != last_threat_name:
                    st.error(f"Threat Detected: {threat_name} ⚠️")
                    last_threat_name = threat_name  # Update the last threat name

            else:
                if last_threat_name is not None:
                    st.write("No security threats detected.")
                    last_threat_name = None  # Reset last threat name if no match

        except Exception as e:
            st.error("Passenger not found in the database.")

        # Temporary delay for smoother processing
        time.sleep(0.1)

    # Release the webcam
    cap.release()

# Button to start the video recording (without displaying the video)
if st.button("Start Live Face Recognition"):
    st.session_state['stop_recording'] = False
    capture_video_without_display()
