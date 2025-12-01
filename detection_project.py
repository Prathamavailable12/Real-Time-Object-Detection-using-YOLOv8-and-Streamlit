import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os

st.set_page_config(page_title="Real-Time Object Detection", layout="wide")

st.title("🎯 Real-Time Object Detection using YOLOv8")
st.markdown("Upload a video or use your webcam to detect objects in real-time.")

# Load YOLOv8 model
@st.cache_resource
def load_model():
    return YOLO("yolov8m.pt")

model = load_model()

# Sidebar options
option = st.sidebar.radio("Select Input Source", ("Webcam", "Upload Video","Upload Image"))

if option == "Webcam":
    st.write("Click 'Start Detection' to begin using your webcam.")

    start = st.button("Start Detection")
    stop = st.button("Stop Detection")

    if start:
        cap = cv2.VideoCapture(0)
        frame_window = st.image([])

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or stop:
                break

            results = model(frame)
            annotated_frame = results[0].plot()

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_window.image(frame_rgb)

        cap.release()

elif option == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

    if uploaded_file:
        temp_video = tempfile.NamedTemporaryFile(delete=False)
        temp_video.write(uploaded_file.read())
        video_path = temp_video.name

        cap = cv2.VideoCapture(video_path)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            annotated_frame = results[0].plot()
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb)

        cap.release()
        os.remove(video_path)

elif option == "Upload Image":
    uploaded_file = st.file_uploader("Upload a Image file", type=["jpeg", "png", "jpg"])

    if uploaded_file:
        temp_image = tempfile.NamedTemporaryFile(delete=False)
        temp_image.write(uploaded_file.read())
        image_path = temp_image.name

        cap = cv2.VideoCapture(image_path)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            annotated_frame = results[0].plot()
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb)

        cap.release()
        os.remove(image_path)

st.markdown("---")
st.caption("Developed by Janmesh Patel & Pratham Trivedi — Introduction to AI Project")
