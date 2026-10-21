import cv2
from ultralytics import YOLO
import numpy as np


class TankDetection:
    def __init__(
        self,
        tank_height,
        detect_interval=5,
        camera_width=640,
        camera_height=480,
        camera_index=0,
        model_path="best_ncnn_model",
        conf_threshold=0.9,
    ):
        """
        tank_height: Real tank height in cm.
        detect_interval: Run the YOLO/NCNN tank model once every N frames.
                         The previous tank box is reused between detections.
        camera_width, camera_height: Requested camera resolution.
        camera_index: Camera device index, usually 0.
        model_path: Path to exported YOLO NCNN model folder.
        conf_threshold: Minimum confidence for tank detection.
        """
        if detect_interval < 1:
            raise ValueError("detect_interval must be >= 1")

        self.model = YOLO(model_path)
        self.cam = cv2.VideoCapture(camera_index)

        # Lower camera resolution to reduce frame-processing cost on Raspberry Pi.
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

        # This is not supported by all camera backends, but helps reduce old-frame lag when available.
        self.cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.tank_height = tank_height
        self.detect_interval = detect_interval
        self.conf_threshold = conf_threshold

        self.frame_count = 0
        self.last_tank_box = None  # (x1, y1, x2, y2)
        self.last_conf = None

    def _clamp_box_to_frame(self, frame, x1, y1, x2, y2):
        h, w = frame.shape[:2]

        x1 = max(0, min(int(x1), w - 1))
        x2 = max(0, min(int(x2), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        y2 = max(0, min(int(y2), h - 1))

        if x2 <= x1 or y2 <= y1:
            return None

        return x1, y1, x2, y2

    def _draw_tank_box(self, frame, x1, y1, x2, y2, conf=None, cached=False):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = "tank"
        if conf is not None:
            label += f" {conf:.2f}"
        if cached:
            label += " cached"

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    def _run_tank_model(self, frame):
        results = self.model(frame, conf=self.conf_threshold, max_det=1, verbose=False)

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            # Because max_det=1, take the first box only.
            box = result.boxes[0]

            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # Safety check: class 0 must be your tank class.
            if cls_id != 0:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            clamped_box = self._clamp_box_to_frame(frame, x1, y1, x2, y2)

            if clamped_box is None:
                continue

            x1, y1, x2, y2 = clamped_box
            tank_crop = frame[y1:y2, x1:x2]

            if tank_crop.size == 0:
                continue

            return (x1, y1, x2, y2), conf

        return None, None

    def detect_tank(self):
        ret, frame = self.cam.read()

        if not ret or frame is None:
            return None, None, None, None, None, False

        self.frame_count += 1

        should_run_model = (
            self.last_tank_box is None
            or self.frame_count % self.detect_interval == 0
        )

        if should_run_model:
            tank_box, conf = self._run_tank_model(frame)

            if tank_box is not None:
                self.last_tank_box = tank_box
                self.last_conf = conf

        if self.last_tank_box is None:
            return frame, None, None, None, None, False

        x1, y1, x2, y2 = self.last_tank_box
        self._draw_tank_box(
            frame,
            x1,
            y1,
            x2,
            y2,
            conf=self.last_conf,
            cached=not should_run_model,
        )

        return frame, x1, y1, x2, y2, True

    def detect_level(self, frame, x1, y1, x2, y2, REAL_TANK_HEIGHT_CM):
        # Make sure coordinates are valid.
        h_img, w_img = frame.shape[:2]

        x1 = max(0, min(x1, w_img - 1))
        x2 = max(0, min(x2, w_img - 1))
        y1 = max(0, min(y1, h_img - 1))
        y2 = max(0, min(y2, h_img - 1))

        if x2 <= x1 or y2 <= y1:
            return None, None

        # Use inner crop to avoid detecting the tank border.
        x_margin = 5
        y_margin = 10

        inner_x1 = x1 + x_margin
        inner_x2 = x2 - x_margin
        inner_y1 = y1 + y_margin
        inner_y2 = y2 - y_margin

        if inner_x2 <= inner_x1 or inner_y2 <= inner_y1:
            return None, None

        roi = frame[inner_y1:inner_y2, inner_x1:inner_x2]

        if roi.size == 0:
            return None, None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Compute vertical gradient to find horizontal boundary.
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        grad_y = np.abs(grad_y)

        # Average edge strength across each row.
        row_strength = np.mean(grad_y, axis=1)

        if len(row_strength) == 0:
            return None, None

        # Find strongest horizontal edge inside the inner ROI.
        level_row = np.argmax(row_strength)

        # Convert ROI row back to full-frame coordinate.
        y_level_global = inner_y1 + level_row

        # Calculate fill percentage based on the full detected tank box.
        tank_height_pixels = y2 - y1
        filled_fraction = (y2 - y_level_global) / tank_height_pixels
        filled_fraction = max(0.0, min(1.0, filled_fraction))

        percentage = filled_fraction * 100.0
        liquid_height_cm = filled_fraction * REAL_TANK_HEIGHT_CM

        # Draw liquid level line inside the inner tank area, not over the detected border.
        cv2.line(
            frame,
            (inner_x1, y_level_global),
            (inner_x2, y_level_global),
            (0, 0, 255),
            2,
        )

        return percentage, liquid_height_cm

    def detect(self):
        frame, x1, y1, x2, y2, status_tank = self.detect_tank()

        if frame is None:
            return None, None, False

        if status_tank:
            percentage, liquid_height_cm = self.detect_level(
                frame,
                x1,
                y1,
                x2,
                y2,
                self.tank_height,
            )
            cv2.imshow("Tank Detection", frame)
            return percentage, liquid_height_cm, True

        cv2.putText(
            frame,
            "Tank not detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        cv2.imshow("Tank Detection", frame)
        return None, None, False

    def release(self):
        self.cam.release()
        cv2.destroyAllWindows()
