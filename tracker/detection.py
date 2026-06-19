import os
import warnings
import numpy as np
import torch
import supervision as sv
from ultralytics import YOLO

warnings.filterwarnings("ignore", category=FutureWarning, module="supervision")

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Default detection model. Using the STOCK model: round-1 fine-tuning (finetuned.pt)
# didn't improve the ball and over-detected sideline/background people, cluttering
# the gallery. To use the fine-tuned weights again, set:
#     DEFAULT_MODEL = os.path.join(_PROJECT, "finetuned.pt")
DEFAULT_MODEL = "yolo11n.pt"

# BoT-SORT config shipped with this project (ReID + camera-motion compensation)
_TRACKER_CFG = os.path.join(_PROJECT, "botsort_football.yaml")


def _find_class(names: dict, candidates: set, default: int) -> int:
    """Find a class id by name (works for stock COCO and our fine-tuned model)."""
    for i, n in names.items():
        if str(n).lower() in candidates:
            return int(i)
    return default


class FootballDetector:
    """YOLO detection + BoT-SORT tracking (ReID + camera-motion compensation).

    BoT-SORT replaces the old motion-only ByteTrack so that player IDs survive
    the panning Veo camera (camera-motion compensation) and brief occlusions
    (appearance ReID). Tracking is done by Ultralytics' native model.track(),
    which keeps state internally between calls via persist=True.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, conf: float = 0.25, imgsz: int = 1280,
                 min_player_frac: float = 0.04):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Detector] Using device: {self.device.upper()} | model: {os.path.basename(model_name)}")
        self.model = YOLO(model_name)
        # Separate model instance for the ball pass, so its predict() calls can't
        # disturb the player tracker's persistent state (they'd otherwise share
        # one predictor). Same weights, negligible extra VRAM.
        self.ball_model = YOLO(model_name)
        # Class ids read from the model, so this works for the stock model
        # (person=0, sports ball=32) and our fine-tuned one (player=0, ball=1).
        self.person_class = _find_class(self.model.names, {"player", "person"}, 0)
        self.ball_class = _find_class(self.model.names, {"ball", "sports ball"}, 32)
        self.conf = conf
        self.imgsz = imgsz
        self.tracker_cfg = _TRACKER_CFG
        # The ball is small and motion-blurred, so it scores low confidence. The
        # tracker's new-track threshold throws it away (only ~10% of frames keep
        # it), so we detect the ball in a SEPARATE low-confidence pass (~90%).
        self.ball_conf = 0.10
        # Drop players shorter than this fraction of the frame height. Players on
        # an adjacent pitch in the background are far away and therefore tiny, so
        # a size threshold removes them — and unlike a fixed pixel box it follows
        # the panning camera (a far player stays small whichever way we pan).
        self.min_player_frac = min_player_frac

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
        # --- Players: track at normal confidence (BoT-SORT: ReID + GMC) ---
        # classes pinned to players only; the shared predictor means any leaked
        # `classes=` would otherwise change what gets tracked.
        results = self.model.track(
            frame,
            persist=True,             # keep track identities between calls
            tracker=self.tracker_cfg,
            conf=self.conf,
            imgsz=self.imgsz,
            classes=[self.person_class],
            verbose=False,
            device=self.device,
        )[0]
        players = sv.Detections.from_ultralytics(results)

        # Drop far-away (tiny) players — typically people on an adjacent pitch.
        if len(players) > 0:
            min_h = self.min_player_frac * frame.shape[0]
            heights = players.xyxy[:, 3] - players.xyxy[:, 1]
            players = players[heights >= min_h]

        # --- Ball: separate low-confidence detection pass (no tracking) ---
        ball_res = self.ball_model.predict(
            frame, conf=self.ball_conf, imgsz=self.imgsz,
            classes=[self.ball_class], verbose=False, device=self.device,
        )[0]
        balls = sv.Detections.from_ultralytics(ball_res)

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
