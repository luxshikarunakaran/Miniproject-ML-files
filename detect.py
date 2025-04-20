# EfficientNetB0 model
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load the trained EfficientNetB0 model
model = load_model('best_model_effinetb0.h5')

# Define class labels
class_labels = ['adenocarcinoma', 'large_cell_carcinoma', 'normal', 'squamous_cell_carcinoma']


# Function to preprocess the image for prediction (based on EfficientNetB0 requirements)
def preprocess_for_prediction(image):
    img = cv2.resize(image, (350, 350))  # Resize to the input size of EfficientNetB0 (350x350)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB (if needed)
    img = img.astype('float32')  # Convert image to float32
    img = preprocess_input(img)  # EfficientNetB0-specific preprocessing
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


# Load the CT scan image
image_path = 'Data/test/squamous.cell.carcinoma/000111.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if image is None:
    print("Error: Unable to load the image. Please check the file format and path.")
else:
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
            predicted_class_idx = np.argmax(prediction)  # Index of the class with the highest probability
            predicted_class = class_labels[predicted_class_idx]
            predicted_confidence = np.max(prediction)  # Confidence score for the predicted class

            predictions.append((predicted_class, predicted_confidence))

            # Put the predicted class label with confidence on the image
            cv2.putText(output_image, f'{predicted_class} ({predicted_confidence:.2f})',
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original CT Scan')
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Detected Areas with Predictions')
    plt.imshow(output_image)
    plt.axis('off')

    plt.show()

    # Print predictions for detected areas with confidence
    for i, (pred_class, pred_confidence) in enumerate(predictions):
        print(f'Detected area {i + 1}: {pred_class} with confidence {pred_confidence:.2f}')

    # Display final result message
    if predictions:
        final_pred_class, final_pred_confidence = predictions[0]  # Taking the first prediction
        print(f'Final prediction result: {final_pred_class} with confidence {final_pred_confidence:.2f}')
    else:
        print("No significant areas detected.")
