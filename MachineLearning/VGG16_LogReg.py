import os

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Initialize VGG16 model without the top classification layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# Create a model that outputs the features from the last convolutional layer
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

# Function to get the embedding of an image
def get_image_embedding(img):
    img = img.resize((224, 224))  # Resize image to 224x224
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess image
    features = feature_extractor.predict(img_array)  # Extract features
    features_flattened = features.flatten()  # Flatten the features
    return features_flattened



# Collect class names
def pre_process(path):
    X = []
    Y = []
    class_names = []
    for class_name in os.listdir(path):
        class_dir = os.path.join(path, class_name)
        if os.path.isdir(class_dir):
            class_names.append(class_name)

    # Encode class names
    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)

    # Collect image vectors and labels
    for class_name in class_names:
        class_dir = os.path.join(path, class_name)
        if os.path.isdir(class_dir):
            for filename in os.listdir(class_dir):
                file_path = os.path.join(class_dir, filename)
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    try:
                        embedding = get_image_embedding(file_path)
                        X.append(embedding)  # Add the image vector to X
                        Y.append(label_encoder.transform([class_name])[0])  # Add the encoded label to Y
                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")
    return X, Y