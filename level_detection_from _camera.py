import numpy as np
import cv2

def detect_liquid_level(frame, tank_x, tank_y, tank_w, tank_h, REAL_TANK_HEIGHT_CM):
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
    
    if level_row is not None:
        y_level_global = tank_y + level_row

        # Calculate fill percentage
        filled_fraction = (tank_h - level_row) / tank_h
        filled_fraction = max(0.0, min(1.0, filled_fraction))
        percentage = filled_fraction * 100.0

        # Calculate liquid height in cm
        liquid_height_cm = filled_fraction * REAL_TANK_HEIGHT_CM

    return percentage, liquid_height_cm