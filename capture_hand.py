import os
import cv2
import time

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 36  # 26 letters + 10 digits
dataset_size = 50  # Number of images per class

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream or file")
    exit()

alphabet_digits = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    print(f'Capturing hand signs for {alphabet_digits[j]} ({j})')
    print('Press "R" to start capturing images for this class.')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        cv2.putText(frame, 'Ready? Press "Q" to quit or "R" to start capturing!', (100, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        key = cv2.waitKey(25)
        if key == ord('q') or cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1:
            print("Exiting hand signs capture...")
            cap.release()
            cv2.destroyAllWindows()
            exit()
        elif key == ord('r'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        cv2.imshow('frame', frame)

        if cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed by user. Exiting hand signs capture...")
            break

        key = cv2.waitKey(25)
        if key == ord('q'):
            print("Exiting hand signs capture...")
            cap.release()
            cv2.destroyAllWindows()
            exit()

        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)
        counter += 1
        
        time.sleep(1)

cap.release()
cv2.destroyAllWindows()
