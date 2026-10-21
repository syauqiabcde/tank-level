import cv2
import numpy as np

# -------- Tank calibration --------
# Define tank rectangle manually: x, y, width, height
TANK_X = 150
TANK_Y = 20
TANK_W = 480
TANK_H = 380

# Real tank height in cm (optional)
REAL_TANK_HEIGHT_CM = 28.5

def detect_liquid_level(frame, tank_x, tank_y, tank_w, tank_h):
    # Crop tank region
    roi = frame[tank_y:tank_y+tank_h, tank_x:tank_x+tank_w]

    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Compute vertical gradient to find horizontal boundary
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    grad_y = np.abs(grad_y)

    # Average edge strength across each row
    row_strength = np.mean(grad_y, axis=1)

    # Ignore top and bottom margins to avoid tank borders
    margin = 10
    valid_rows = row_strength[margin:tank_h-margin]

    if len(valid_rows) == 0:
        return None

    # Liquid level is assumed to be strongest horizontal edge
    level_row = np.argmax(valid_rows) + margin

    return level_row

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot grab frame")
        break

    level_row = detect_liquid_level(frame, TANK_X, TANK_Y, TANK_W, TANK_H)

    # Draw tank box
    cv2.rectangle(frame, (TANK_X, TANK_Y), (TANK_X + TANK_W, TANK_Y + TANK_H), (0, 255, 0), 2)

    if level_row is not None:
        y_level_global = TANK_Y + level_row

        # Draw detected liquid level
        cv2.line(frame, (TANK_X, y_level_global), (TANK_X + TANK_W, y_level_global), (0, 0, 255), 2)

        # Calculate fill percentage
        filled_fraction = (TANK_H - level_row) / TANK_H
        filled_fraction = max(0.0, min(1.0, filled_fraction))
        percentage = filled_fraction * 100.0

        # Calculate liquid height in cm
        liquid_height_cm = filled_fraction * REAL_TANK_HEIGHT_CM

        cv2.putText(frame, f"Level: {percentage:.1f}%", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Tank Level Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()