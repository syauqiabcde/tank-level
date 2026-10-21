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
                record_video=False,
                video_filename="tank_recording.avi",
                video_fps=20
                ):
        self.model = YOLO("best_ncnn_model")
        self.cam = cv2.VideoCapture(0)

        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
        self.cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.tank_height = tank_height
        self.detect_interval = detect_interval
        self.frame_count = 0
        self.last_tank_box = None
        self.conf_threshold = 0.9
        self.last_level_y = None
        self.level_smoothing_alpha = 0.35
        self.level_search_window_ratio = 0.25

        self.record_video = record_video
        self.video_writer = None
        self.video_filename = video_filename
        self.video_fps = video_fps
        self.camera_width = camera_width
        self.camera_height = camera_height

        if self.record_video:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self.video_writer = cv2.VideoWriter(
                self.video_filename,
                fourcc,
                self.video_fps,
                (self.camera_width, self.camera_height)
            )

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
        h_img, w_img = frame.shape[:2]

        x1 = max(0, min(x1, w_img - 1))
        x2 = max(0, min(x2, w_img - 1))
        y1 = max(0, min(y1, h_img - 1))
        y2 = max(0, min(y2, h_img - 1))

        if x2 <= x1 or y2 <= y1:
            return None, None

        tank_w = x2 - x1
        tank_h = y2 - y1

        # Use only the center part of the tank.
        # This avoids side walls, tank borders, and corner reflections.
        x_margin = int(0.18 * tank_w)
        y_margin_top = int(0.08 * tank_h)
        y_margin_bottom = int(0.08 * tank_h)

        inner_x1 = x1 + x_margin
        inner_x2 = x2 - x_margin
        inner_y1 = y1 + y_margin_top
        inner_y2 = y2 - y_margin_bottom

        if inner_x2 <= inner_x1 or inner_y2 <= inner_y1:
            return None, None

        roi = frame[inner_y1:inner_y2, inner_x1:inner_x2]

        if roi.size == 0:
            return None, None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Improve local contrast. Useful for transparent tanks or weak waterlines.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        roi_h, roi_w = blurred.shape[:2]

        # 1. Horizontal edge score
        grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        grad_y = np.abs(grad_y)

        # Use percentile instead of mean.
        # This is more robust against small bright reflections.
        edge_score = np.percentile(grad_y, 85, axis=1)

        # 2. Contrast score above vs below each candidate row
        window = max(4, int(0.03 * roi_h))
        contrast_score = np.zeros(roi_h, dtype=np.float32)

        for r in range(window, roi_h - window):
            above = blurred[r - window:r, :].mean()
            below = blurred[r:r + window, :].mean()
            contrast_score[r] = abs(float(above) - float(below))

        def normalize(arr):
            arr = arr.astype(np.float32)
            min_val = np.min(arr)
            max_val = np.max(arr)
            if max_val - min_val < 1e-6:
                return np.zeros_like(arr)
            return (arr - min_val) / (max_val - min_val)

        edge_score_norm = normalize(edge_score)
        contrast_score_norm = normalize(contrast_score)

        # Combine both signals
        score = 0.65 * edge_score_norm + 0.35 * contrast_score_norm

        # Ignore very top and bottom of ROI
        score[:window] = 0
        score[-window:] = 0

        original_score = score.copy()

        # 3. Temporal tracking: prefer level close to previous detected level
        if self.last_level_y is not None:
            previous_local_y = self.last_level_y - inner_y1

            if 0 <= previous_local_y < roi_h:
                search_window = max(20, int(self.level_search_window_ratio * roi_h))

                lower = max(0, int(previous_local_y - search_window))
                upper = min(roi_h, int(previous_local_y + search_window))

                masked_score = np.zeros_like(score)
                masked_score[lower:upper] = score[lower:upper]

                # Use masked score only if it still contains useful information
                if np.max(masked_score) > 1e-6:
                    score = masked_score

        level_row = int(np.argmax(score))

        # Simple confidence check
        nonzero_score = score[score > 0]
        if len(nonzero_score) == 0:
            return None, None

        peak_score = float(score[level_row])
        mean_score = float(np.mean(nonzero_score))
        confidence = peak_score / (mean_score + 1e-6)

        # If confidence is low, reuse previous level instead of jumping randomly
        if confidence < 1.4 and self.last_level_y is not None:
            y_level_global = self.last_level_y
        else:
            y_level_global = inner_y1 + level_row

            # 4. Smooth the level to avoid jitter
            if self.last_level_y is not None:
                y_level_global = int(
                    self.level_smoothing_alpha * y_level_global
                    + (1.0 - self.level_smoothing_alpha) * self.last_level_y
                )

        self.last_level_y = y_level_global

        # Calculate fill percentage based on full detected tank box
        filled_fraction = (y2 - y_level_global) / tank_h
        filled_fraction = max(0.0, min(1.0, filled_fraction))

        percentage = filled_fraction * 100.0
        liquid_height_cm = filled_fraction * REAL_TANK_HEIGHT_CM

        # Draw result
        cv2.line(
            frame,
            (x1, y_level_global),
            (x2, y_level_global),
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"Level: {percentage:.1f}%",
            (x1, min(y2 + 25, h_img - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
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
            if self.record_video and self.video_writer is not None:
                self.video_writer.write(frame)
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
        if self.record_video and self.video_writer is not None:
            self.video_writer.write(frame)
        return None, None, False

    def release(self):
        self.cam.release()
        if self.video_writer is not None:
            self.video_writer.release()
        cv2.destroyAllWindows()

