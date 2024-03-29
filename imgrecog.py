import tensorflow as tf
import cv2
import numpy as np
import json

model = tf.keras.models.load_model('/home/firo/Github/Img-Recon/my_model.keras')

# Initialize training_data list (load from file if available)
try:
    with open('/home/firo/Github/Img-Recon/training_data.json', 'r') as file:
        training_data = json.load(file)
        print("Training data loaded from file.")
except FileNotFoundError:
    training_data = []
    print("Training data file not found. Initializing an empty list.")

while True:
    choice = input("Enter the path of the input image or type 'q' to quit: ")
    
    if choice.lower() == 'q':
        break
    
    # Capture and preprocess input image with cv2 
    image_path = choice
    input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Get model prediction
    prediction = model.predict(np.expand_dims(input_image, axis=0))[0]
    predicted_digit = np.argmax(prediction)

    print(f"Model predicted: {predicted_digit}")

    # Get user's corrected label
    correct_label = int(input("Enter the correct digit label (0-9): "))

    # Checking the model's prediction
    if correct_label != predicted_digit:
        # Append the corrected data point to training data
        corrected_data_point = (input_image.tolist(), correct_label)
        training_data.append(corrected_data_point)
        print("Data point added to training data.")

# Save the final training data before exiting the script
with open('/home/firo/Github/Img-Recon/training_data.json', 'w') as file:
    json.dump(training_data, file)
    print("Training data saved to file.")
