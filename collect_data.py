import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import os
import time

# Initialize video capture with higher resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Increase resolution (e.g., 1280x720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Initialize the hand detector with higher confidence threshold
detector = HandDetector(maxHands=1, detectionCon=0.8)  # Increase detection confidence

# Parameters for image capture
offset = 20
imgSize = 300
folder = "Data/Z"
max_images = 1000 # Maximum number of images per dataset
counter = 0

# Ensure the data folder exists
if not os.path.exists(folder):
    os.makedirs(folder)  # Create the directory if it does not exist

def process_hand_crop(hand, img):
    """Process the hand crop and return the resized image."""
    x, y, w, h = hand['bbox']

    # Ensure the coordinates are within the image boundaries
    imgHeight, imgWidth, _ = img.shape
    y_min = max(0, y - offset)
    y_max = min(imgHeight, y + h + offset)
    x_min = max(0, x - offset)
    x_max = min(imgWidth, x + w + offset)

    imgCrop = img[y_min:y_max, x_min:x_max]

    # Create a white canvas for the resized image
    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

    # Resize according to the aspect ratio
    aspectRatio = h / w
    if aspectRatio > 1:  # Taller than wide
        k = imgSize / h
        wCal = math.ceil(k * w)
        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
        wGap = (imgSize - wCal) // 2
        imgWhite[:, wGap:wCal + wGap] = imgResize
    else:  # Wider than tall
        k = imgSize / w
        hCal = math.ceil(k * h)
        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
        hGap = (imgSize - hCal) // 2
        imgWhite[hGap:hCal + hGap, :] = imgResize

    return imgCrop, imgWhite

# Create a resizable window for displaying the image
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image.")
        break

    # Detect hands
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        imgCrop, imgWhite = process_hand_crop(hand, img)

        # Show the cropped hand and the white canvas
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)
    else:
        cv2.imshow("ImageCrop", np.zeros((imgSize, imgSize, 3), np.uint8))
        cv2.imshow("ImageWhite", np.ones((imgSize, imgSize, 3), np.uint8) * 255)

    # Display the original image in a resizable window
    cv2.imshow("Image", img)

    # Check for key press and window close event
    key = cv2.waitKey(1)  # Reduced wait time for faster response

    # Start capturing images when 'C' is pressed
    if key == ord("c"):
        if counter < max_images:
            counter += 1
            cv2.imwrite(f'{folder}/Image_{counter}_{int(time.time())}.jpg', imgWhite)
            print(f"Captured {counter}/{max_images} images")
        else:
            print("Maximum image limit reached.")

    elif key == ord("q") or cv2.getWindowProperty('Image', cv2.WND_PROP_VISIBLE) < 1:
        print("Exiting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
