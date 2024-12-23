import os
import joblib
import cv2
import requests
from VGG16_LogReg import feature_extractor
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from preprocess import get_image_embedding, label, apply_face_recognition_and_save

#label_encoder = joblib.load("D:/MachineLearning/label_encoder.pkl")
model = joblib.load("D:/MachineLearning/weight_faces.pkl")
path1 = "D:\MachineLearning\Faces"
path = "D:\MachineLearning\captured_frame_0.jpg"

label_encoder = label(path1)
r = label_encoder.inverse_transform([model.predict([get_image_embedding(path)])])[0]
#print(model.predict_proba(get_image_embedding(path)))
print(r)