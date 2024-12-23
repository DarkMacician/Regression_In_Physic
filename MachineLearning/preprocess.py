import os

import cv2
import face_recognition
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from PIL import Image
import joblib

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

def get_image_embedding(img_path):
    img = Image.open(img_path).resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = feature_extractor.predict(img_array)
    features_flattened = features.flatten()
    return features_flattened

def label(path):
    class_names = []
    for class_name in os.listdir(path):
        class_dir = os.path.join(path, class_name)
        if os.path.isdir(class_dir):
            class_names.append(class_name)

    # Encode class names
    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)
    return label_encoder


def apply_face_recognition_and_save(image_path):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for i, (top, right, bottom, left) in enumerate(face_locations):
        face_image = image[top:bottom, left:right]
        face_image_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite("face.jpg", face_image_bgr)
