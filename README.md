# 🧠 Roman Numeral Classifier (Handwritten)

A deep learning-based application to classify handwritten Roman numerals using a Convolutional Neural Network (CNN) with a user-friendly GUI built using **Tkinter**.

## 🔍 Features

- 🖼️ **GUI Interface**: Easily upload and classify images via Tkinter-based GUI  
- 🤖 **CNN Model**: Trained on grayscale images of handwritten Roman numerals  
- 📊 **Evaluation Metrics**:  
  - Confusion Matrix  
  - Class-wise Accuracy Bar Chart  
  - Precision-Recall Curve  
- 🔄 **Live Prediction**: Classifies new uploaded images using the trained model  
- 📁 Model and classes are saved in the `model/` directory for reuse


## 🚀 How to Run

1. Clone the repo  
2. Place your dataset in `Dataset/Dataset/` folder (split into `train/` and `val/`)  
3. Run `train_model.py` to train the model  
4. Run `gui_app.py` to launch the GUI  

## 🛠️ Tech Stack

- **Python**
- **TensorFlow / Keras**
- **OpenCV**
- **Tkinter**
- **Matplotlib**
- **scikit-learn**
- **PIL (Pillow)**

## 📌 Sample GUI
![image](https://github.com/user-attachments/assets/e9fff948-a2b4-49d9-b391-297ea306aaee)
![image](https://github.com/user-attachments/assets/27bb8edf-75b2-46ac-b413-123f7885ed83)
![image](https://github.com/user-attachments/assets/b6e39319-11fe-4e68-9c48-b456cb55ab83)
![image](https://github.com/user-attachments/assets/05d78567-a912-4e5d-a65b-e2c470a7d105)

## 📈 Sample Evaluation Plots

- ✅ Confusion Matrix  
- 📊 Class-wise Accuracy  
- 📉 Precision-Recall Curve

