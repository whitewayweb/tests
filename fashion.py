import streamlit as st
import cv2
import numpy as np
import requests

# Load the YOLO object detection model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Define the class labels and corresponding colors
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Define a function to perform object detection
def detect_objects(image):
    # Get the image height and width
    (H, W) = image.shape[:2]
    
    # Get the output layer names from the YOLO model
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Construct a blob from the input image and perform a forward pass through the YOLO model
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(layer_names)

    # Initialize lists to store the bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    # Loop over each of the layer outputs
    for output in layer_outputs:
        # Loop over each of the detections
        for detection in output:
            # Extract the class ID and confidence of the current detection
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter out weak detections
            if confidence > 0.5:
                # Scale the bounding box coordinates to the size of the image
                box = detection[0:4] * np.array([W, H, W, H])
                (center_x, center_y, width, height) = box.astype("int")

                # Calculate the top-left corner of the bounding box
                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))

                # Add the bounding box, confidence, and class ID to their respective lists
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maxima suppression to remove overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Loop over the remaining bounding boxes and draw them on the image
    for i in indices:
        i = i[0]
        box = boxes[i]
        (x, y, w, h) = box
        color = [int(c) for c in colors[class_ids[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

# Define the Streamlit app
st.title("Object Detection using YOLO")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
