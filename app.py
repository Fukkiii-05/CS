import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Vision AI", layout="wide")

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

st.title("📸 AI Face Detector")
st.write("Upload an image to detect faces using Computer Vision.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original")
        st.image(image, use_container_width=True)
        
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        
        annotated_image = img_array.copy()
        
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(annotated_image, detection)
            
            with col2:
                st.subheader("AI Result")
                st.image(annotated_image, use_container_width=True)
                st.success(f"Detected {len(results.detections)} face(s)!")
        else:
            st.warning("No faces detected.")
else:
    st.info("Please upload an image to start.")
