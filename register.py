import streamlit as st
import os
import json
from PIL import Image

# Specify the folder where known faces are saved
KNOWN_FACES_DIR = "known_faces"

# Create the folder if it doesn't exist
if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

# Function to save the image with the passenger's name
def save_passenger_image(passenger_name, img_file):
    # Clean the passenger's name to avoid illegal characters in filenames
    clean_name = "".join([c if c.isalnum() else "_" for c in passenger_name])
    # Create the full path for the image
    img_path = os.path.join(KNOWN_FACES_DIR, f"{clean_name}.jpg")
    
    # Save the image to the folder
    with open(img_path, "wb") as f:
        f.write(img_file.getbuffer())

    return img_path

# Function to save passenger metadata to a JSON file
def save_passenger_data(passenger_name, from_location, to_location, contact_info, email):
    data_file = 'passenger_data.json'

    # Load existing passenger data if available
    if os.path.exists(data_file):
        with open(data_file, 'r') as f:
            passenger_data = json.load(f)
    else:
        passenger_data = {}

    # Add the new passenger details
    passenger_data[passenger_name] = {
        "image": f"{passenger_name}.jpg",
        "from": from_location,
        "to": to_location,
        "contact": contact_info,
        "email": email
    }

    # Save the updated data back to the JSON file
    with open(data_file, 'w') as f:
        json.dump(passenger_data, f, indent=4)

# Form for registering a passenger
st.header("Register New Passenger")

with st.form("passenger_registration"):
    passenger_name = st.text_input("Enter Passenger Name")
    from_location = st.text_input("Traveling From")
    to_location = st.text_input("Traveling To")
    contact_info = st.text_input("Contact Information")
    email = st.text_input("Email Address")
    img_file = st.file_uploader("Upload Passenger Image", type=["jpg", "jpeg", "png"])

    # Submit button for the form
    submit_button = st.form_submit_button("Register Passenger")

if submit_button:
    if passenger_name and img_file and from_location and to_location and contact_info and email:
        # Save the passenger image to the known_faces directory
        saved_image_path = save_passenger_image(passenger_name, img_file)
        
        # Save the passenger metadata to the JSON file
        save_passenger_data(passenger_name, from_location, to_location, contact_info, email)

        st.success(f"Passenger {passenger_name} registered successfully!")
        st.image(saved_image_path, caption=f"Image saved as {passenger_name}.jpg")
    else:
        st.error("Please provide all the required information (name, image, and details).")
