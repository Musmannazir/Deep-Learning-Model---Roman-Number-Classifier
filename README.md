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

## 🗂️ Folder Structure

├── model/
│ ├── roman_model.h5
│ └── classes.npy
├── Dataset/
│ └── Dataset/
│ ├── train/
│ └── val/
├── utils.py
├── train_model.py
├── predict.py
├── gui_app.py


## 🚀 How to Run

1. Clone the repo  
2. Place your dataset in `Dataset/Dataset/` folder (split into `train/` and `val/`)  
3. Run `train_model.py` to train the model  
4. Run `gui_app.py` to launch the GUI  

```bash
python train_model.py
python gui_app.py
🛠️ Tech Stack
Python

TensorFlow / Keras

OpenCV

Tkinter

Matplotlib

scikit-learn

PIL (Pillow)

📌 Sample GUI
(Add screenshot of the GUI window here if available)

📈 Sample Evaluation Plots
Confusion Matrix

Class-wise Accuracy

Precision-Recall Curve

🙌 Contributions
Feel free to open issues or pull requests to enhance this project!

yaml
Copy
Edit

---

Let me know if you also want a `requirements.txt` file or license section added!
