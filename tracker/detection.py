import warnings
import numpy as np
import torch
import supervision as sv
from ultralytics import YOLO

warnings.filterwarnings("ignore", category=FutureWarning, module="supervision")

PERSON_CLASS = 0
BALL_CLASS = 32


class FootballDetector:
    """YOLOv8 detection + ByteTrack player tracking."""

    def __init__(self, model_name: str = "yolov8n.pt", conf: float = 0.3, imgsz: int = 1280):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Detector] Using device: {self.device.upper()}")
        self.model = YOLO(model_name)
        self.conf = conf
        self.imgsz = imgsz
        self.tracker = sv.ByteTrack(minimum_matching_threshold=0.8)

        # Ball state — persist last known position when detection drops
        self.last_ball_pos: np.ndarray | None = None
        self.ball_lost_frames = 0
        self.BALL_LOST_THRESHOLD = 20  # processed frames before we give up on last position

    def process_frame(self, frame: np.ndarray):
        """
        Returns:
            players      – sv.Detections with tracker_id assigned
            ball_center  – np.array([x, y]) or None
            ball_xyxy    – np.array([x1,y1,x2,y2]) or None
        """
        results = self.model(
            frame,
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False,
            device=self.device,
        )[0]

        detections = sv.Detections.from_ultralytics(results)

        # Split into players and ball
        if len(detections) == 0:
            self.ball_lost_frames += 1
            ball_center = self.last_ball_pos if self.ball_lost_frames < self.BALL_LOST_THRESHOLD else None
            return sv.Detections.empty(), ball_center, None

        person_mask = detections.class_id == PERSON_CLASS
        ball_mask = detections.class_id == BALL_CLASS

        players = detections[person_mask]
        balls = detections[ball_mask]

        # Track players across frames
        if len(players) > 0:
            players = self.tracker.update_with_detections(players)

        # Best ball detection (highest confidence)
        ball_center = None
        ball_xyxy = None
        if len(balls) > 0:
            idx = int(np.argmax(balls.confidence))
            x1, y1, x2, y2 = balls.xyxy[idx]
            ball_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            ball_xyxy = balls.xyxy[idx]
            self.last_ball_pos = ball_center
            self.ball_lost_frames = 0
        else:
            self.ball_lost_frames += 1
            if self.ball_lost_frames < self.BALL_LOST_THRESHOLD:
                ball_center = self.last_ball_pos  # use last known

        return players, ball_center, ball_xyxy
