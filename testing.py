import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Text
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk
import mediapipe as mp

# Load pre-trained model
model = load_model("C:/Users/ritul/OneDrive/Desktop/projects/sign language/Model/sign_language_model.keras")

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Initialize the main Tkinter window
root = tk.Tk()
root.title("Sign Language to Text")

# Create a frame for video feed and prediction display
video_frame = tk.Frame(root)
video_frame.pack(side=tk.LEFT, padx=10, pady=10)

# Initialize webcam feed
cap = cv2.VideoCapture(0)

# Tkinter Label to display the video frame
label_video = tk.Label(video_frame)
label_video.pack()

# Initialize the text output
text_display = tk.Text(root, height=5, width=50)
text_display.pack(side=tk.BOTTOM, padx=10, pady=10)

# Functions for buttons
def clear_text():
    text_display.delete("1.0", tk.END)

def save_to_file():
    file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
    if file_path:
        with open(file_path, "w") as f:
            f.write(text_display.get("1.0", tk.END))

def quit_app():
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

# Button controls
button_frame = tk.Frame(root)
button_frame.pack(side=tk.BOTTOM, pady=10)

btn_clear = tk.Button(button_frame, text="Clear All", command=clear_text, width=20, height=2, bg="khaki")
btn_clear.grid(row=0, column=0, padx=5)

btn_save = tk.Button(button_frame, text="Save to a Text File", command=save_to_file, width=20, height=2, bg="light green")
btn_save.grid(row=0, column=1, padx=5)

btn_quit = tk.Button(button_frame, text="Quit", command=quit_app, width=20, height=2, bg="indian red")
btn_quit.grid(row=0, column=2, padx=5)

# Load label list for alphabet
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Space', 'Delete']

def process_frame():
    ret, frame = cap.read()
    if ret:
        # Flip the frame for a mirrored effect
        frame = cv2.flip(frame, 1)

        # Process the image and detect hands
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # If hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get bounding box for the hand
                h, w, _ = frame.shape
                x_min = int(min([landmark.x for landmark in hand_landmarks.landmark]) * w)
                x_max = int(max([landmark.x for landmark in hand_landmarks.landmark]) * w)
                y_min = int(min([landmark.y for landmark in hand_landmarks.landmark]) * h)
                y_max = int(max([landmark.y for landmark in hand_landmarks.landmark]) * h)

                # Extract the region of interest (ROI) for prediction
                roi = frame[y_min:y_max, x_min:x_max]
                if roi.size > 0:
                    # Preprocess the ROI for model prediction
                    resized_frame = cv2.resize(roi, (160, 160))  # Resize frame to match model input size
                    normalized_frame = resized_frame / 255.0  # Normalize frame
                    input_frame = np.expand_dims(normalized_frame, axis=0)  # Expand dimensions

                    # Make prediction
                    prediction = model.predict(input_frame)
                    predicted_class = np.argmax(prediction)
                    confidence = np.max(prediction)

                    # Print predictions for debugging
                    print(f"Predicted class: {predicted_class}, Confidence: {confidence}")

                    if confidence > 0.5:  # Adjust the threshold as needed
                        predicted_label = labels[predicted_class]
                        print(f"Detected label: {predicted_label}")

                        # Display prediction on the text box
                        if predicted_label == "Space":
                            text_display.insert(tk.END, " ")
                        elif predicted_label == "Delete":
                            current_text = text_display.get("1.0", tk.END)[:-2]
                            text_display.delete("1.0", tk.END)
                            text_display.insert(tk.END, current_text)
                        else:
                            text_display.insert(tk.END, predicted_label)

        # Convert the frame to RGB and display in the Tkinter label
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (300, 300))
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
        label_video.imgtk = img_tk
        label_video.configure(image=img_tk)

    # Call this function again after 10 ms
    root.after(10, process_frame)

# Start video processing
process_frame()

# Run the Tkinter main loop
root.mainloop()
