from imports import *
import tensorflow as tf
import numpy as np

# Function to load and preprocess the image
# def load_and_preprocess_image(image_path, img_size=(128, 128)):
#     img = image.load_img(image_path, target_size=img_size)
#     img_array = image.img_to_array(img)
#     img_array = tf.image.convert_image_dtype(img_array, dtype=tf.float32)
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#     return img_array


def load_and_preprocess_image(img_path, target_size=(128, 128)):

    """Takes an image and returns img_array which is the processed image array. This image array can be used for predictions"""

    img = image.load_img(img_path, target_size=target_size)  # Resize image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize to [0, 1]
    return img_array

def predict(img_array, model):
    
    """Takes the processed img_array and the model for prediction as parameters. Returns the result in text (malignant or benign)."""
    # Make a prediction
    prediction = model.predict(img_array)
    print(prediction[0])
    # Interpret the prediction
    if prediction[0] > 0.5:
        return 'malignant'
    else:
        return 'benign'