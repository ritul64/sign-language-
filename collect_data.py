import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
folder = "Data/C"
counter = 0

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image.")
        break

    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Ensure the coordinates are within the image boundaries
        imgHeight, imgWidth, _ = img.shape
        y_min = max(0, y - offset)
        y_max = min(imgHeight, y + h + offset)
        x_min = max(0, x - offset)
        x_max = min(imgWidth, x + w + offset)

        imgCrop = img[y_min:y_max, x_min:x_max]

        # Check if imgCrop is empty
        if imgCrop.size == 0:
            print("Cropped image is empty.")
            continue

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Resize according to the aspect ratio
        imgCropShape = imgCrop.shape
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)

    # Check for key press and window close event
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
    elif key == ord("q") or cv2.getWindowProperty('Image', cv2.WND_PROP_VISIBLE) < 1:
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
