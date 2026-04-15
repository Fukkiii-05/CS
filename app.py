import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="AI Vision Pro", layout="wide")

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

st.title("🤖 AI Computer Vision App")
st.write("Apni image upload krein aur AI model automatically chehry k features detect kry ga.")

st.sidebar.header("Settings")
detection_confidence = st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.5)

uploaded_file = st.file_uploader("Image choose krein...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=2,
        min_detection_confidence=detection_confidence) as face_mesh:

        results = face_mesh.process(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))

        output_image = image_np.copy()

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=output_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)

            with col2:
                st.subheader("AI Processed")
                st.image(output_image, use_container_width=True)
                st.success(f"AI ne {len(results.multi_face_landmarks)} face(s) detect kiye hain!")
        else:
            st.warning("Koi face detect nahi hua. Confidence level kam kr k dekhein.")

else:
    st.info("App start krny k liye image upload krein.")
