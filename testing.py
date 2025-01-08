import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from collections import deque

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

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break
    
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
            continue

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

        # Debugging: print prediction and confidence
        print(f"Prediction: {prediction}, Confidence: {confidence}")

        # Apply confidence threshold (optional: you can adjust this)
        if confidence > 0.5:
            smooth_predictions.append(labels[index])
        else:
            smooth_predictions.append("")

        # Get the most common prediction in the window
        final_prediction = max(set(smooth_predictions), key=smooth_predictions.count)

        # Display prediction
        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 120, y - offset - 10), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, final_prediction, (x, y - 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)

        # Show cropped and processed images
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    # Display the output
    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)
    if key == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
