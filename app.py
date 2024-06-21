import streamlit as st
from PIL import Image
import numpy as np
from settings import DEFAULT_CONFIDENCE_THRESHOLD
from faster_rcnn import faster_rcnn_detect
from yolo import yolo_detect, LABELS as LABELS_YOLO

st.set_page_config(
    page_title="Object Detection",
    page_icon="👁‍🗨",
    initial_sidebar_state="expanded"
)

# Streamlit UI
st.title("Object Detection")
st.markdown('This is an application for object detection using Faster R-CNN and YOLO')
st.caption("Upload an image and click the 'Analyse Image' button to detect components in the image.")
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

# Sidebar
st.sidebar.header("Machine Learning Model")
model_type = st.sidebar.radio("Select Model", ("Faster R-CNN", "YOLO"))
confidence_threshold = st.sidebar.slider(
    "Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Analyse Image"):
        if model_type == "Faster R-CNN":
            image_with_boxes, detected_components = faster_rcnn_detect(image, confidence_threshold)
            st.image(image_with_boxes, caption="Detected Components - Faster R-CNN", use_column_width=True)
        
        elif model_type == "YOLO":
            image_np = np.array(image)
            image_with_boxes, class_ids, indices = yolo_detect(image_np, confidence_threshold)

            st.image(image_with_boxes, caption="Detected Components - YOLO", use_column_width=True)
        
        st.write("Detected components:")
        if model_type == "Faster R-CNN":
            for component in detected_components:
                st.write(f"- {component}")
        elif model_type == "YOLO":
            unique_components = set(LABELS_YOLO[class_ids[i]] for i in indices.flatten())
            for component in unique_components:
                st.write(f"- {component}")
