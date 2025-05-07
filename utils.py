import os
import cv2
import numpy as np

def load_dataset(root_folder, image_size=(64, 64)):
    images = []
    labels = []
    for split_folder in os.listdir(root_folder): 
        split_path = os.path.join(root_folder, split_folder)
        if not os.path.isdir(split_path):
            continue
        for label in os.listdir(split_path):  
            label_folder = os.path.join(split_path, label)
            if not os.path.isdir(label_folder):
                continue
            for filename in os.listdir(label_folder):
                filepath = os.path.join(label_folder, filename)
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, image_size)
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)
