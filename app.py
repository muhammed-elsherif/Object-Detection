import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import models, transforms
from settings import DEFAULT_CONFIDENCE_THRESHOLD
from labels import LABELS
from colors import COLORS

# Load a pre-trained object detection model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.ToTensor()
])

st.set_page_config(
    page_title="Object Detection",
    page_icon="👁‍🗨",
    initial_sidebar_state="expanded"
)

# Streamlit UI
st.title("Object Detection")
st.markdown('This is an application for object detection using R-CNN and YOLO')
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
        if model_type == 'YOLO':
            st.error("Not implemented yet!!")
        image_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(image_tensor)

        # Convert to PIL image
        image_with_boxes = image.copy()
        draw = ImageDraw.Draw(image_with_boxes)
        font = ImageFont.load_default()

        detected_components = set()
        for box, label, score in zip(outputs[0]['boxes'], outputs[0]['labels'], outputs[0]['scores']):
            if score >= confidence_threshold:
                detected_components.add(LABELS[label.item()])
                color = COLORS[LABELS[label.item()]]
                # Draw bounding box
                draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline=color, width=2)
                # Draw label with confidence score
                text = f"{LABELS[label.item()]}: {score:.2f}"
                draw.rectangle(((box[0], box[1] - 10), (box[0] + len(text) * 6, box[1])), fill=color)
                draw.text((box[0], box[1] - 10), text, fill="white", font=font)
    
        
        st.image(image_with_boxes, caption="Detected Components", use_column_width=True)
        
        st.write("Detected components:")
        for component in detected_components:
            st.write(f"- {component}")
