import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf

# Load the emotion detection model
model = tf.keras.models.load_model('emotion_detection_cnn_model.h5')

# Define emotions (adjust as per your dataset)
emotions = ['Angry', 'Happy', 'Sad', 'Surprised', 'Neutral', 'Fearful', 'Disgusted']

# Function to classify emotion from an image
def classify_emotion(image):
    img = cv2.resize(image, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    result = model.predict(img)
    emotion_index = np.argmax(result)
    return emotions[emotion_index]

# Function to update the image and classify emotion in real-time
def update_frame():
    global cap, label_img, label_result
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        label_img.imgtk = imgtk
        label_img.configure(image=imgtk)

        # Classify emotion
        emotion = classify_emotion(frame_rgb)
        label_result.config(text=f"Predicted Emotion: {emotion}")
        print(f"Predicted Emotion: {emotion}")  # Print predicted emotion for debugging

    window.after(10, update_frame)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Create a GUI window
window = tk.Tk()
window.title("Real-Time Emotion Detection")

# Add components to the GUI
label_title = tk.Label(window, text="Emotion Detection using CNN", font=("Helvetica", 16))
label_title.pack(pady=10)

label_img = Label(window)
label_img.pack()

label_result = tk.Label(window, text="", font=("Helvetica", 14))
label_result.pack(pady=10)

# Start updating frames
update_frame()

# Run the GUI
window.mainloop()

# Release the webcam when done
cap.release()
cv2.destroyAllWindows()
