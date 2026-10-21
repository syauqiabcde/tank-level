import cv2
import os

cap = cv2.VideoCapture(0)
os.makedirs("dataset/images", exist_ok=True)

count = 89

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Capture", frame)

    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("s"):
        filename = f"dataset/images/tank_{count:04d}.jpg"
        cv2.imwrite(filename, frame)
        print("Saved:", filename)
        count += 1

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()