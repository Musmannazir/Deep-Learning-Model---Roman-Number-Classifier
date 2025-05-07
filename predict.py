import cv2
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore

model = load_model('model/roman_model.h5')
classes = np.load('model/classes.npy')

def predict_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  
    if img is None:
        return "Invalid image"
    
    img = cv2.resize(img, (64, 64))
    img = img / 255.0  

    img = img.reshape(1, 64, 64, 1)  

    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    
    return classes[class_index]

