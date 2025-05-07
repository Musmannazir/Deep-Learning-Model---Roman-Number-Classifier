import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
from PIL import Image, ImageTk
from predict import predict_image

import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model # type: ignore
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load model and classes
model = load_model('model/roman_model.h5')
class_names = np.load('model/classes.npy')
val_dir = 'Dataset/Dataset/val'

def load_validation_data():
    X_val, y_val = [], []
    for label in os.listdir(val_dir):
        label_path = os.path.join(val_dir, label)
        if not os.path.isdir(label_path):
            continue
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (64, 64))
            img = img / 255.0
            X_val.append(img)
            y_val.append(label)
    X_val = np.array(X_val).reshape(-1, 64, 64, 1)
    y_val = np.array([np.where(class_names == label)[0][0] for label in y_val])
    return X_val, y_val

class RomanNumeralApp:
    def __init__(self, master):
        self.master = master
        master.title("Roman Numeral Classifier")
        master.geometry("600x700")
        master.configure(bg="#e8f0f2")

        self.frame = Frame(master, bg="#e8f0f2")
        self.frame.pack(pady=10, padx=20, fill="both", expand=True)

        self.title_label = Label(self.frame, text="Roman Numeral Classifier", bg="#e8f0f2", font=("Helvetica", 18, "bold"))
        self.title_label.pack(pady=10)

        self.instructions_label = Label(self.frame, text="Upload a Handwritten Roman Numeral Image", bg="#e8f0f2", font=("Arial", 12))
        self.instructions_label.pack(pady=5)

        self.upload_btn = Button(self.frame, text="Browse Image", command=self.upload_image, width=20, height=2, font=("Arial", 12, "bold"), bg="#5c6bc0", fg="white", relief="flat")
        self.upload_btn.pack(pady=10)

        self.image_label = Label(self.frame, bg="#e8f0f2")
        self.image_label.pack(pady=10)

        self.result_label = Label(self.frame, text="", font=("Arial", 16, "bold"), bg="#e8f0f2", fg="#00796b")
        self.result_label.pack(pady=10)

        self.check_plot_btn = Button(self.frame, text="Check Plots", command=self.show_plot, width=20, height=2, font=("Arial", 12, "bold"), bg="#388e3c", fg="white", relief="flat")
        self.check_plot_btn.pack(pady=10)

        self.current_plot = 0
        self.plots = []

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        img = Image.open(file_path).resize((200, 200))
        img_tk = ImageTk.PhotoImage(img)
        self.image_label.configure(image=img_tk)
        self.image_label.image = img_tk

        result = predict_image(file_path)
        self.result_label.config(text=f"Prediction: {result}")

    def show_plot(self):
        self.current_plot = 0
        self.generate_plots()
        self.display_plot(self.current_plot)

    def show_next_plot(self):
        self.current_plot = (self.current_plot + 1) % len(self.plots)
        self.display_plot(self.current_plot)

    def generate_plots(self):
        X_val, y_val = load_validation_data()
        pred = model.predict(X_val)
        pred_indices = np.argmax(pred, axis=1)

        # Confusion Matrix
        cm = confusion_matrix(y_val, pred_indices)
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(ax=ax1, cmap='Blues')
        ax1.set_title("Confusion Matrix")
        fig1.tight_layout()

        # Class Accuracy Bar Graph
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.bar(class_names, class_accuracy, color='skyblue')
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Validation Accuracy per Class")
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y')
        fig2.tight_layout()

        self.plots = [fig1, fig2]

    def display_plot(self, index):
        fig = self.plots[index]

        # Create new pop-up window
        plot_window = tk.Toplevel(self.master)
        plot_window.title(f"Plot {index + 1}")
        plot_window.geometry("700x600")

        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10, fill="both", expand=True)

        # Next button inside pop-up window
        next_btn = Button(plot_window, text="Next", command=lambda: [plot_window.destroy(), self.show_next_plot()],
                          font=("Arial", 10, "bold"), bg="#0288d1", fg="white", relief="flat")
        next_btn.pack(pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    app = RomanNumeralApp(root)
    root.mainloop()
