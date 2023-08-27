import tensorflow as tf
import cv2
import numpy as np
import json

# Load the trained model
model = tf.keras.models.load_model('/home/firo/Github/Img-Recon/my_model.keras')

# Initialize training_data list (load from file if available)
try:
    with open('/home/firo/Github/Img-Recon/training_data.json', 'r') as file:
        training_data = json.load(file)
        print("Training data loaded from file.")
except FileNotFoundError:
    training_data = []
    print("Training data file not found. Initializing an empty list.")

# Interactive Feedback Loop
while True:
    # Prompt user to enter the image path
    image_path = input("Enter the path of the input image: ")

    # Capture and preprocess input image
    input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if input_image is None:
        print("Invalid image path. Please try again.")
        continue

    input_image = cv2.resize(input_image, (28, 28))
    input_image = input_image / 255.0

    # Get model prediction
    prediction = model.predict(np.expand_dims(input_image, axis=0))[0]
    predicted_digit = np.argmax(prediction)

    # Display prediction
    print(f"Model predicted: {predicted_digit}")

    # Get user's corrected label
    correct_label = int(input("Enter the correct digit label (0-9): "))

    # Update training data with corrected label
    corrected_data_point = (input_image.tolist(), correct_label)
    training_data.append(corrected_data_point)

    # After loop: Save training data to file
    with open('/home/firo/Github/Img-Recon/training_data.json', 'w') as file:
        json.dump(training_data, file)
        print("Training data saved to file.")
    # Reset the session and model to ensure a fresh start
    tf.keras.backend.clear_session()

    # Rebuild the model architecture
    model = tf.keras.models.load_model('/home/firo/Github/Img-Recon/my_model.keras')

    # Retrain the model using updated training data
    if training_data:
        updated_train_labels = [label for _, label in training_data]
        updated_train_labels = tf.keras.utils.to_categorical(updated_train_labels, num_classes=10)
        updated_train_images = np.array([np.array(image) for image, _ in training_data])

        model.fit(updated_train_images, updated_train_labels, epochs=10)
