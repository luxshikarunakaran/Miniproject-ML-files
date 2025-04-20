import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model('lung_cancer_classifier.h5')

# Define class labels
class_labels = ['adenocarcinoma', 'large_cell_carcinoma', 'normal', 'squamous_cell_carcinoma']

# Function to preprocess the image for prediction
def preprocess_for_prediction(image):
    img = cv2.resize(image, (128, 128))  # Resize to the input size of the model
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
    img = img.astype('float32') / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Streamlit application
st.title("Lung Cancer Detection from CT Scan Images")

# Upload image
uploaded_file = st.file_uploader("Choose a CT scan image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = np.array(image)

    # Preprocess the image (e.g., Gaussian blur)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Thresholding to create a binary image
    _, binary = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image and predict
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for color drawing
    predictions = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Filter out small areas
            # Get the bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box

            # Crop the detected area
            cropped_image = image[y:y + h, x:x + w]

            # Preprocess the cropped image for prediction
            preprocessed_image = preprocess_for_prediction(cropped_image)

            # Make prediction
            prediction = model.predict(preprocessed_image)
            predicted_class = class_labels[np.argmax(prediction)]
            predictions.append(predicted_class)

            # Put the predicted class label on the image
            cv2.putText(output_image, predicted_class, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the results
    st.subheader("Original CT Scan")
    st.image(image, caption='Original Image', use_column_width=True)

    st.subheader("Detected Areas with Predictions")
    st.image(output_image, caption='Detected Areas', use_column_width=True)

    # Print predictions for detected areas
    for i, pred in enumerate(predictions):
        st.write(f'Detected area {i + 1}: {pred}')