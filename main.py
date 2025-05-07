from utils import load_dataset
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense # type: ignore
from tensorflow.keras.models import save_model # type: ignore
import os

os.makedirs('model', exist_ok=True)

# Load dataset
images, labels = load_dataset('Dataset/Dataset')

# Preprocessing
images = images / 255.0  # Normalize
images = images.reshape(-1, 64, 64, 1)

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(labels)

# Save encoder classes for later use
np.save('model/classes.npy', encoder.classes_)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(images, y_encoded, test_size=0.2, random_state=42)

# Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(encoder.classes_), activation='softmax')  
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save the trained model
save_model(model, 'model/roman_model.h5')
