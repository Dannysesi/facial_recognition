# Passenger Registration and Facial Recognition System

This project is an AI-powered application that combines a **Passenger Registration System** and **Facial Recognition System** for security purposes, such as airport or train station security. The app registers passenger details and then identifies passengers in real-time through facial recognition, helping to ensure that only authorized individuals pass through secure areas. It also includes a **Threat Detection System** to identify flagged individuals or potential threats.


## Features

**Passenger Registration:**
- Register passengers by entering details such as name, contact information, travel details, and an image.
- Saves the passenger’s photo in a known_faces directory and stores metadata (name, contact, travel info, email) in a JSON file for future identification.
  
**Facial Recognition:**
- Real-time facial recognition using a live video feed to identify registered passengers.
- Displays the name, contact info, and travel details of identified passengers.
- Threat detection by comparing captured faces to a separate Threat Database (known_threats), raising an alert if a flagged individual is detected.
  
**Security Features:**
- Passenger identification and threat detection based on DeepFace facial recognition with the Facenet model.
- Alerts for unknown passengers or flagged individuals, ensuring a secure environment.
  
**User Interface:**
- Simple form-based registration process with image upload.
- Start and stop real-time face recognition with a single click in the Facial Recognition tab.

## Technologies Used
- Python 3.11.1
- Streamlit: A framework for creating interactive web apps in Python.
- DeepFace: A deep learning-based facial recognition library.
- OpenCV: For accessing and processing real-time video streams.
- Pillow: For image processing (handling uploaded images).
- JSON: For storing passenger metadata.

## Credits
Special thanks to the open-source community for the amazing tools like Streamlit, DeepFace, and OpenCV that made this project possible.
## License

[MIT](https://choosealicense.com/licenses/mit/)

