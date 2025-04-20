import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import Rescaling, RandomFlip, RandomRotation, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

# Define paths
train_path = "Data/train"
valid_path = "Data/valid"
test_path = "Data/test"

# Parameters
IMAGE_SIZE = 224
N_CLASSES = 4
BATCH_SIZE = 32
CHANNELS = 1

# Function to convert grayscale images to RGB
def grayscale_to_rgb(image):
    return tf.image.grayscale_to_rgb(image)

# Data augmentation and preprocessing
train_datagen = tf.keras.Sequential([
    Rescaling(1./255.),
    RandomFlip("horizontal_and_vertical"),
    RandomRotation(0.2),
    tf.keras.layers.Lambda(grayscale_to_rgb)  # Convert grayscale to RGB
])

valid_datagen = tf.keras.Sequential([
    Rescaling(1./255.),
    tf.keras.layers.Lambda(grayscale_to_rgb)  # Convert grayscale to RGB
])

# Load datasets
train_generator = image_dataset_from_directory(
    train_path,
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    color_mode="grayscale"
).map(lambda x, y: (train_datagen(x), y))

valid_generator = image_dataset_from_directory(
    valid_path,
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    color_mode="grayscale"
).map(lambda x, y: (valid_datagen(x), y))

test_generator = image_dataset_from_directory(
    test_path,
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    color_mode="grayscale"
).map(lambda x, y: (valid_datagen(x), y))

# Define the model
input_tensor = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))  # RGB input
base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(N_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
checkpointer = ModelCheckpoint('chestmodel_resnet.keras', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=15)

# Train the model
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=valid_generator,
    callbacks=[checkpointer, early_stopping]
)

# Evaluate the model
result = model.evaluate(test_generator)

# Plot training history
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.legend(loc='right')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()


# Get class names
test_class_names = test_generator.class_names
# Prediction function
def predict(model, img):
    img_array = tf.expand_dims(img, 0)  # Create a batch
    predictions = model.predict(img_array)
    predicted_class = test_class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

# Display predictions
plt.figure(figsize=(15, 15))
for images, labels in test_generator.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8").squeeze(), cmap='gray')
        predicted_class, confidence = predict(model, images[i].numpy())
        actual_class = test_class_names[labels[i]]
        plt.title(f"Actual: {actual_class},\nPredicted: {predicted_class}\nConfidence: {confidence}%")
        plt.axis("off")
plt.show()
