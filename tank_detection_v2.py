import cv2
from ultralytics import YOLO
import numpy as np


class TankDetection:
    """
    Tank & level detector using best_ncnn_model_v2.

    The model is expected to output two classes:
        Class 0 → tank
        Class 1 → level  (the liquid surface bounding box)

    Fill percentage is derived purely from bounding-box geometry —
    no frame/edge analysis is performed.
    """

    CLASS_TANK  = 0
    CLASS_LEVEL = 1

    def __init__(
        self,
        tank_height,
        detect_interval=5,
        camera_width=640,
        camera_height=480,
        conf_threshold=0.5,
        record_video=False,
        video_filename="tank_recording_v2.avi",
        video_fps=20,
    ):
        self.model = YOLO("best_ncnn_model_v2", task="detect")
        self.cam   = cv2.VideoCapture(0)

        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH,  camera_width)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
        self.cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.tank_height      = tank_height
        self.detect_interval  = detect_interval
        self.conf_threshold   = conf_threshold
        self.frame_count      = 0

        # Cache last known boxes so detection still draws on skipped frames
        self.last_tank_box    = None   # (x1, y1, x2, y2)
        self.last_level_box   = None   # (x1, y1, x2, y2)
        self.last_tank_conf   = None
        self.last_level_conf  = None

        # Smoothing for fill percentage (EMA)
        self.smoothing_alpha  = 0.35
        self.last_percentage  = None

        self.camera_width  = camera_width
        self.camera_height = camera_height

        self.record_video  = record_video
        self.video_writer  = None
        self.video_filename = video_filename
        self.video_fps     = video_fps

        if self.record_video:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self.video_writer = cv2.VideoWriter(
                self.video_filename,
                fourcc,
                self.video_fps,
                (self.camera_width, self.camera_height),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clamp_box(self, frame, x1, y1, x2, y2):
        h, w = frame.shape[:2]
        x1, x2 = max(0, int(x1)), min(int(x2), w - 1)
        y1, y2 = max(0, int(y1)), min(int(y2), h - 1)
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _run_model(self, frame):
        """
        Run inference and return the best tank box and best level box.
        Returns (tank_result, level_result) where each is ((x1,y1,x2,y2), conf) or (None, None).
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)

        best_tank  = None
        best_level = None

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                coords = box.xyxy[0].cpu().numpy().astype(int)
                clamped = self._clamp_box(frame, *coords)

                if clamped is None:
                    continue

                if cls_id == self.CLASS_TANK:
                    if best_tank is None or conf > best_tank[1]:
                        best_tank = (clamped, conf)

                elif cls_id == self.CLASS_LEVEL:
                    if best_level is None or conf > best_level[1]:
                        best_level = (clamped, conf)

        tank_box,  tank_conf  = best_tank  if best_tank  else (None, None)
        level_box, level_conf = best_level if best_level else (None, None)

        return (tank_box, tank_conf), (level_box, level_conf)

    def _draw_tank_box(self, frame, x1, y1, x2, y2, conf, cached=False):
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"tank {conf:.2f}" + (" [cached]" if cached else "")
        cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def _draw_level_line(self, frame, tank_box, level_box, conf, percentage, cached=False):
        """
        Draw a horizontal line at the TOP edge of the level bounding box
        (top of the liquid surface) spanning the full tank width.
        """
        tx1, ty1, tx2, ty2 = tank_box
        lx1, ly1, lx2, ly2 = level_box

        # Top of the level bbox = liquid surface
        y_surface = int((ly2 - ly1)/2 + ly1)
        h_img = frame.shape[0]

        color = (0, 0, 255)
        cv2.line(frame, (tx1, y_surface), (tx2, y_surface), color, 2)

        label = f"Level: {percentage:.1f}%" + (" [cached]" if cached else "")
        cv2.putText(frame, label, (tx1, min(ty2 + 25, h_img - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # ------------------------------------------------------------------
    # Percentage calculation
    # ------------------------------------------------------------------

    def _calculate_percentage(self, tank_box, level_box):
        """
        Derive fill percentage from bounding boxes.

        The top of the level bbox marks the liquid surface.
        Fraction = (tank_bottom − surface_y) / tank_height_px
        """
        tx1, ty1, tx2, ty2 = tank_box
        lx1, ly1, lx2, ly2 = level_box

        tank_h_px = ty2 - ty1
        if tank_h_px <= 0:
            return None

        y_surface = ly1
        filled_px = ty2 - y_surface
        fraction  = max(0.0, min(1.0, filled_px / tank_h_px))

        return fraction * 100.0

    # ------------------------------------------------------------------
    # Public API  (same signature as original TankDetection)
    # ------------------------------------------------------------------

    def detect(self):
        """
        Read one frame, run detection, draw annotations, show with cv2.imshow.

        Returns:
            percentage      (float | None)  – 0–100 fill level
            liquid_height_cm (float | None) – actual height in cm
            status          (bool)          – True if tank was found
        """
        ret, frame = self.cam.read()
        if not ret or frame is None:
            return None, None, False

        self.frame_count += 1
        should_run = (
            self.last_tank_box is None
            or self.frame_count % self.detect_interval == 0
        )

        if should_run:
            (tank_box, tank_conf), (level_box, level_conf) = self._run_model(frame)

            if tank_box is not None:
                self.last_tank_box   = tank_box
                self.last_tank_conf  = tank_conf

            if level_box is not None:
                self.last_level_box  = level_box
                self.last_level_conf = level_conf

        # ── No tank detected at all ─────────────────────────────────────
        if self.last_tank_box is None:
            cv2.putText(frame, "Tank not detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Tank Detection v2", frame)
            if self.record_video and self.video_writer:
                self.video_writer.write(frame)
            return None, None, False

        # ── Draw tank box ───────────────────────────────────────────────
        self._draw_tank_box(
            frame, *self.last_tank_box,
            conf=self.last_tank_conf,
            cached=not should_run,
        )

        # ── Level calculation ───────────────────────────────────────────
        percentage       = None
        liquid_height_cm = None

        if self.last_level_box is not None:
            raw_pct = self._calculate_percentage(self.last_tank_box, self.last_level_box)

            if raw_pct is not None:
                # EMA smoothing
                if self.last_percentage is not None:
                    raw_pct = (self.smoothing_alpha * raw_pct
                               + (1.0 - self.smoothing_alpha) * self.last_percentage)

                self.last_percentage = raw_pct
                percentage           = raw_pct
                liquid_height_cm     = (raw_pct / 100.0) * self.tank_height

                self._draw_level_line(
                    frame,
                    self.last_tank_box,
                    self.last_level_box,
                    self.last_level_conf,
                    percentage,
                    cached=not should_run,
                )
        else:
            cv2.putText(frame, "Level not detected", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            return None, None, False

        cv2.imshow("Tank Detection v2", frame)
        if self.record_video and self.video_writer:
            self.video_writer.write(frame)

        return percentage, liquid_height_cm, True

    def release(self):
        self.cam.release()
        if self.video_writer is not None:
            self.video_writer.release()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Quick test – run this file directly
# ---------------------------------------------------------------------------
