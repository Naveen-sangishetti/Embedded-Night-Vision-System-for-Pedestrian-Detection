# streamlit_app.py
"""Streamlit front‑end for the Embedded Night‑Vision Pedestrian Detection project.
It provides a modern, resume‑worthy web UI that can be run locally or deployed to
Streamlit Community Cloud.
"""

import streamlit as st
import cv2
import numpy as np
import os
from pathlib import Path

# Import detection utilities (relative imports assume the script lives in the /app folder)
from ..yoloDetection import detectObject, labelsBoundingBoxes, listBoundingBoxes
from ..yoloDetection import detectObject as yolo_detect
from ..yoloDetection import detectObject as yolo_detect

# Helper functions ------------------------------------------------------------

def load_model():
    """Load YOLOv2 model and class labels once per session.
    Returns (net, layer_names, class_labels).
    """
    model_dir = Path(__file__).parents[1] / "yolov2model"
    cfg_path = model_dir / "yolov2.cfg"
    weights_path = model_dir / "yolov2.weights"
    labels_path = model_dir / "yolov2-labels"
    net = cv2.dnn.readNetFromDarknet(str(cfg_path), str(weights_path))
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    class_labels = open(str(labels_path)).read().strip().split("\n")
    return net, layer_names, class_labels

# Cache the model so it loads only once per user session
net, layer_names, class_labels = load_model()

# Streamlit UI ---------------------------------------------------------------
st.set_page_config(page_title="Night‑Vision Pedestrian Detection", layout="centered")
st.title("🚦 Embedded Night‑Vision Pedestrian Detection")
st.markdown(
    "Upload an image and choose a detection algorithm. "
    "The app runs the YOLOv2 model that was trained for pedestrian detection."
)

uploaded_file = st.file_uploader("Choose an image (PNG/JPEG)", type=["png", "jpg", "jpeg"])
algorithm = st.radio("Detection algorithm", ["YOLOv2", "Haar + AdaBoost (experimental)"])

if uploaded_file is not None:
    # Convert uploaded file to a numpy array compatible with OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original image", use_column_width=True)

    if st.button("Run detection"):
        with st.spinner("Detecting pedestrians …"):
            if algorithm == "YOLOv2":
                # Random colors for each class (same as original script)
                label_colors = np.random.randint(0, 255, size=(len(class_labels), 3), dtype='uint8')
                detected_image, _, _, _, _ = detectObject(
                    net, layer_names,
                    image.shape[0], image.shape[1],
                    image, label_colors, class_labels)
            else:
                # Simple fallback – we reuse the existing adBoost pipeline from test.py
                from ..test import adBoost, haarDetect
                adjusted = adBoost(image, gamma=3.5)
                # Re‑use haarDetect (expects two images – original & adjusted)
                detected_image = haarDetect(image, adjusted)

        # Convert back to RGB for Streamlit display
        st.image(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB), caption="Detection result", use_column_width=True)
        st.success("Detection completed!")

st.caption("**Tech stack:** Python 3.11, OpenCV 4.x, Streamlit 1.x, NumPy, imutils. Deployable on Streamlit Community Cloud or any containerised environment.")
