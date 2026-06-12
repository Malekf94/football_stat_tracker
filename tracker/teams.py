import cv2
import numpy as np
from sklearn.cluster import KMeans


class TeamClassifier:
    """
    Separates players into two teams using jersey colour (HSV K-Means).

    Approach:
    - Extract the torso region of each player bounding box
    - Compute mean HSV colour, filtering out shadows and grass reflections
    - After collecting enough samples, fit K-Means (k=2)
    - Cache per-tracker_id assignments to stay consistent across frames
    """

    TEAM_COLORS_BGR = {
        0: (255, 100, 0),    # Blue
        1: (0, 80, 255),     # Red
        -1: (160, 160, 160), # Unknown (grey)
    }

    def __init__(self):
        self.kmeans: KMeans | None = None
        self.calibrated = False
        self.track_team_map: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_jersey_feature(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray | None:
        """Return a 3-float HSV feature vector for the jersey region, or None."""
        x1, y1, x2, y2 = map(int, bbox)
        h = y2 - y1
        w = x2 - x1

        if h < 16 or w < 6:
            return None

        # Torso band: skip top 20% (head/neck) and bottom 35% (shorts/legs)
        jy1 = y1 + int(h * 0.20)
        jy2 = y1 + int(h * 0.65)
        crop = frame[jy1:jy2, x1:x2]

        if crop.size == 0:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Filter out near-black (shadows) and near-white/low-saturation (grass reflections)
        mask = (
            (hsv[:, :, 2] > 40)   &   # not too dark
            (hsv[:, :, 2] < 240)  &   # not blown out
            (hsv[:, :, 1] > 25)        # has actual colour
        )

        if mask.sum() < 15:
            return None

        mean_h = float(hsv[:, :, 0][mask].mean())
        mean_s = float(hsv[:, :, 1][mask].mean())
        mean_v = float(hsv[:, :, 2][mask].mean())

        return np.array([mean_h, mean_s, mean_v], dtype=np.float64)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, features: list[np.ndarray]) -> None:
        """Fit K-Means on collected jersey-colour samples."""
        X = np.array(features)
        self.kmeans = KMeans(n_clusters=2, random_state=42, n_init=15)
        self.kmeans.fit(X)
        self.calibrated = True

        c0 = self.kmeans.cluster_centers_[0]
        c1 = self.kmeans.cluster_centers_[1]
        print(f"[Teams] Calibrated — Team 0 H={c0[0]:.0f} S={c0[1]:.0f} | Team 1 H={c1[0]:.0f} S={c1[1]:.0f}")

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify_player(self, tracker_id: int, frame: np.ndarray, bbox: np.ndarray) -> int:
        """Classify player into team 0 or 1. Result is cached per tracker_id."""
        if tracker_id in self.track_team_map:
            return self.track_team_map[tracker_id]

        if not self.calibrated:
            return -1

        feat = self.extract_jersey_feature(frame, bbox)
        if feat is None:
            return -1

        team = int(self.kmeans.predict([feat])[0])
        self.track_team_map[tracker_id] = team
        return team

    def get_color(self, team: int) -> tuple[int, int, int]:
        return self.TEAM_COLORS_BGR.get(team, self.TEAM_COLORS_BGR[-1])
