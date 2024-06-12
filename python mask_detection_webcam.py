import os 
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.utils import CustomObjectScope

# Define a custom function to replace DepthwiseConv2D layer during model loading
def custom_depthwise_conv2d(*args, **kwargs):
    kwargs.pop('groups', None)  # Remove the 'groups' argument if present
    return DepthwiseConv2D(*args, **kwargs)

# Load the model with a custom object scope and the custom DepthwiseConv2D function
with CustomObjectScope({'DepthwiseConv2D': custom_depthwise_conv2d}):
    model = load_model("mask_detection_best.h5")

# Define a function to capture video from webcam and perform mask detection
def mask_detection_webcam(model):
    # Load the Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Start capturing video from webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Iterate over detected faces
        for (x, y, w, h) in faces:
            # Extract the face ROI
            face_roi = frame[y:y+h, x:x+w]

            # Resize the face ROI to match the input size of the model
            face_roi_resized = cv2.resize(face_roi, (224, 224))

            # Preprocess the face ROI
            face_roi_preprocessed = np.expand_dims(face_roi_resized / 255.0, axis=0)

            # Perform mask detection using the loaded model
            prediction = model.predict(face_roi_preprocessed)

            # Determine the class label based on the prediction
            label = "Mask" if prediction[0][0] > 0.5 else "No Mask"

            # Draw a rectangle around the detected face
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            # Display the label above the rectangle
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        # Display the resulting frame
        cv2.imshow('Mask Detection (Press q to Quit)', frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()

# Call the mask detection function with the loaded model
mask_detection_webcam(model)
