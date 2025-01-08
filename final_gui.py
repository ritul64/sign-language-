import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from collections import deque
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk

# Initialize webcam and modules
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("C:/Users/ritul/OneDrive/Desktop/projects/sign language/Model/keras_model.h5",   
                        "C:/Users/ritul/OneDrive/Desktop/projects/sign language/Model/labels.txt")

# Constants
offset = 20
imgSize = 300
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
          "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

# Sliding window for smoothing predictions
smooth_predictions = deque(maxlen=10)

# Tkinter GUI
root = tk.Tk()
root.title("Sign Language Recognition")
root.geometry("800x600")

text_display = tk.Text(root, height=10, width=50, font=("Helvetica", 16))
text_display.pack(pady=10)

frame = tk.Frame(root)
frame.pack(pady=10)

btn_add = tk.Button(frame, text="Add Character", bg="blue", fg="white", font=("Helvetica", 12),
                    command=lambda: add_character(final_prediction))
btn_add.grid(row=0, column=0, padx=5)

btn_clear = tk.Button(frame, text="Clear All", bg="yellow", font=("Helvetica", 12), command=lambda: text_display.delete(1.0, tk.END))
btn_clear.grid(row=0, column=1, padx=5)

btn_save = tk.Button(frame, text="Save to File", bg="green", fg="white", font=("Helvetica", 12), command=lambda: save_to_file())
btn_save.grid(row=0, column=2, padx=5)

btn_space = tk.Button(frame, text="Space", bg="purple", fg="white", font=("Helvetica", 12), command=lambda: text_display.insert(tk.END, " "))
btn_space.grid(row=0, column=3, padx=5)

btn_exit = tk.Button(frame, text="Exit", bg="red", fg="white", font=("Helvetica", 12), command=root.destroy)
btn_exit.grid(row=0, column=4, padx=5)


# Functionality
def add_character(character):
    if character:
        text_display.insert(tk.END, character)


def save_to_file():
    text = text_display.get(1.0, tk.END).strip()
    if text:
        file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                 filetypes=[("Text Files", ".txt"), ("All Files", ".*")])
        if file_path:
            with open(file_path, "w") as file:
                file.write(text)
            messagebox.showinfo("Save Successful", f"Text saved to {file_path}")
    else:
        messagebox.showwarning("No Text", "There is no text to save!")


# Main loop
def capture_frame():
    global final_prediction
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        return

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a white image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop hand region
        try:
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        except:
            return

        imgCropShape = imgCrop.shape
        aspectRatio = h / w

        # Resize and center the cropped hand image
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Get prediction
        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        confidence = max(prediction)

        # Apply confidence threshold
        if confidence > 0.5:
            smooth_predictions.append(labels[index])
        else:
            smooth_predictions.append("")

        # Get the most common prediction in the window
        final_prediction = max(set(smooth_predictions), key=smooth_predictions.count)

        # Display prediction on the frame
        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 120, y - offset - 10), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, final_prediction, (x, y - 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)

    # Convert image for Tkinter
    imgRGB = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
    imgPIL = Image.fromarray(imgRGB)
    imgTK = ImageTk.PhotoImage(imgPIL)

    # Display the image in a Label widget
    if not hasattr(capture_frame, "label"):
        capture_frame.label = tk.Label(root)
        capture_frame.label.pack()

    capture_frame.label.configure(image=imgTK)
    capture_frame.label.image = imgTK

    root.after(100, capture_frame)  # Adjust the interval for smoother updates


# Start the main GUI loop
capture_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
