from keras.applications.resnet50 import preprocess_input, ResNet50
import numpy as np
from utils.Utils import path_to_tensor
import cv2


def load_keras_model():
    return ResNet50(weights='imagenet')


def ResNet50_predict_labels(img_path, model):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(model.predict(img))


def dog_detector(img_path):
    model = load_keras_model()
    prediction = ResNet50_predict_labels(img_path, model)
    return ((prediction <= 268) & (prediction >= 151))


face_cascade = cv2.CascadeClassifier('../models/faces.xml')


def face_detector(img_path, face_cascade=face_cascade):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0