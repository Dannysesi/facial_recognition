import streamlit as st
import os
import json
from PIL import Image
from deepface import DeepFace
import cv2
import tempfile
import time

# Specify the folder where known faces are saved
KNOWN_FACES_DIR = "known_faces"
PASSENGER_DB = "known_faces"
THREAT_DB = "known_threats"
PASSENGER_DATA_FILE = "passenger_data.json"

# Create the folder if it doesn't exist
if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

# Function to save the image with the passenger's name
def save_passenger_image(passenger_name, img_file):
    clean_name = "".join([c if c.isalnum() else "_" for c in passenger_name])
    img_path = os.path.join(KNOWN_FACES_DIR, f"{clean_name}.jpg")
    
    with open(img_path, "wb") as f:
        f.write(img_file.getbuffer())

    return img_path

# Function to save passenger metadata to a JSON file
def save_passenger_data(passenger_name, from_location, to_location, contact_info, email):
    data_file = PASSENGER_DATA_FILE

    if os.path.exists(data_file):
        with open(data_file, 'r') as f:
            passenger_data = json.load(f)
    else:
        passenger_data = {}

    passenger_data[passenger_name] = {
        "image": f"{passenger_name}.jpg",
        "from": from_location,
        "to": to_location,
        "contact": contact_info,
        "email": email
    }

    with open(data_file, 'w') as f:
        json.dump(passenger_data, f, indent=4)

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

    stop_button_placeholder = st.empty()
    if stop_button_placeholder.button("Stop Recording"):
        st.session_state['stop_recording'] = True

    while cap.isOpened():
        if st.session_state['stop_recording']:
            st.write("Video recording stopped.")
            break

        ret, frame = cap.read()
        if not ret:
            st.error("Error: Could not read frame from the webcam.")
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_img:
            img_pil = Image.fromarray(frame_rgb)
            img_pil.save(temp_img.name)

        try:
            passenger_result = DeepFace.find(img_path=temp_img.name, db_path=PASSENGER_DB, model_name="Facenet")
            threat_result = DeepFace.find(img_path=temp_img.name, db_path=THREAT_DB, model_name="Facenet")

            passenger_data = load_passenger_data()

            if len(passenger_result) > 0 and not passenger_result[0].empty:
                best_match_passenger = passenger_result[0].iloc[0]
                passenger_name = os.path.basename(best_match_passenger['identity']).split('.')[0]

                if passenger_name != last_passenger_name:
                    st.success(f"Passenger Identified: {passenger_name} ✅")
                    last_passenger_name = passenger_name
                    
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
                    last_passenger_name = None

            if len(threat_result) > 0 and not threat_result[0].empty:
                best_match_threat = threat_result[0].iloc[0]
                threat_name = os.path.basename(best_match_threat['identity']).split('.')[0]

                if threat_name != last_threat_name:
                    st.error(f"Threat Detected: {threat_name} ⚠️")
                    last_threat_name = threat_name

            else:
                if last_threat_name is not None:
                    st.write("No security threats detected.")
                    last_threat_name = None

        except Exception as e:
            st.error("Passenger not found in the database.")

        time.sleep(0.1)

    cap.release()


st.title("AI-Based Facial Recognition for Airport Security")

# Layout the Streamlit app with tabs
tab1, tab2 = st.tabs(["Passenger Registration", "Facial Recognition"])

# Tab 1: Passenger Registration
with tab1:
    st.header("Register New Passenger")

    with st.form("passenger_registration"):
        passenger_name = st.text_input("Enter Passenger Name")
        from_location = st.text_input("Traveling From")
        to_location = st.text_input("Traveling To")
        contact_info = st.text_input("Contact Information")
        email = st.text_input("Email Address")
        img_file = st.file_uploader("Upload Passenger Image", type=["jpg", "jpeg", "png"])

        submit_button = st.form_submit_button("Register Passenger")

    if submit_button:
        if passenger_name and img_file and from_location and to_location and contact_info and email:
            saved_image_path = save_passenger_image(passenger_name, img_file)
            save_passenger_data(passenger_name, from_location, to_location, contact_info, email)

            st.success(f"Passenger {passenger_name} registered successfully!")
            st.image(saved_image_path, caption=f"Image saved as {passenger_name}.jpg")
        else:
            st.error("Please provide all the required information (name, image, and details).")

# Tab 2: Facial Recognition
with tab2:
    st.header("Verify Passenger")

    if st.button("Start Live Face Recognition"):
        st.session_state['stop_recording'] = False
        capture_video_without_display()
