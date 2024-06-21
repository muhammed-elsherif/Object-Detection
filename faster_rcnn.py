import torch
from torchvision import models, transforms
from PIL import ImageDraw, ImageFont
from labels import LABELS, COLORS

# Load a pre-trained Faster R-CNN model
frcnn_model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
frcnn_model.eval()

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.ToTensor()
])

# Define the COCO labels
LABELS = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

import random
random.seed(42)
COLORS = {label: "#{:06x}".format(random.randint(0, 0xFFFFFF)) for label in LABELS}

def faster_rcnn_detect(image, confidence_threshold):
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = frcnn_model(image_tensor)

    # Convert to PIL image
    image_with_boxes = image.copy()
    draw = ImageDraw.Draw(image_with_boxes)
    font = ImageFont.load_default()

    detected_components = set()
    for box, label, score in zip(outputs[0]['boxes'], outputs[0]['labels'], outputs[0]['scores']):
        if score >= confidence_threshold:
            label_str = LABELS[label.item()-1]
            detected_components.add(label_str)
            color = COLORS[LABELS[label.item()]]

            # Draw bounding box
            draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline=color, width=2)
            
            # Draw label with confidence score
            text = f"{LABELS[label.item()]}: {score:.2f}"
            draw.rectangle(((box[0], box[1] - 10), (box[0] + len(text) * 6, box[1])), fill=color)
            draw.text((box[0], box[1] - 10), text, fill="white", font=font)

    return image_with_boxes, detected_components
