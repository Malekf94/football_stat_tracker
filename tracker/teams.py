import cv2
import numpy as np
from sklearn.cluster import KMeans


class TeamClassifier:
    """
    Separates players into two teams by BIB COLOUR, clustered per match.

    Two design choices that measurably beat the alternatives on our footage:
      * **Tight bib-centre crop** (centre of the chest, not the whole torso) — avoids
        arms/skin/shorts/background that otherwise contaminate the colour and pull
        e.g. a blue player toward "red". (Raised colour separability ~79% -> ~87%.)
      * **Chromaticity feature** R/(R+G+B), G/(R+G+B) — brightness-independent, so a
        bib in shadow still reads as its true colour. Generalises to ANY two bib
        colours (it clusters whatever two are present), so it works across matches.

    A track's team is a majority vote over its CLEAR (close/large) frames; tiny
    distant crops don't vote and instead inherit the team via the auto-stitch
    chain. Low-confidence tracks stay "unknown" (-1) so they can't create phantom
    pass/turnover events.

    Public interface (unchanged for analyze.py / replay.py):
        crop_player(frame, bbox)           -> bib crop (np.ndarray) or None
        calibrate(crops)                   -> fit 2-colour KMeans on crops
        classify_player(tid, frame, bbox)  -> team 0/1/-1 (per-track majority)
        track_team_map                     -> dict[tracker_id, team]
        get_color(team)
    """

    TEAM_COLORS_BGR = {
        0: (255, 100, 0),    # Blue
        1: (0, 80, 255),     # Red
        -1: (160, 160, 160),  # Unknown (grey)
    }

    MAX_VOTES = 25
    MIN_CONFIDENT_VOTES = 4
    CONFIDENT_AGREEMENT = 0.7
    CLEAR_MIN_FRAC = 0.07     # only vote from frames where the player is this tall

    def __init__(self, device: str | None = None):   # device kept for call-compat
        self.kmeans: KMeans | None = None
        self.calibrated = False
        self.track_team_map: dict[int, int] = {}
        self._votes: dict[int, list[int]] = {}

    def crop_player(self, frame: np.ndarray, bbox: np.ndarray, min_h: int = 32) -> np.ndarray | None:
        """Return the tight centre-of-bib crop (BGR), or None if too small."""
        x1, y1, x2, y2 = map(int, bbox)
        h, w = y2 - y1, x2 - x1
        if h < min_h or w < 10:
            return None
        cy1, cy2 = y1 + int(0.18 * h), y1 + int(0.48 * h)        # upper chest
        cx1, cx2 = x1 + int(0.25 * w), x2 - int(0.25 * w)        # centre, skip arms
        crop = frame[max(0, cy1):cy2, max(0, cx1):cx2]
        return crop if crop.size else None

    @staticmethod
    def _feature(crop: np.ndarray) -> np.ndarray | None:
        """Brightness-independent chromaticity of the saturated bib pixels."""
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        s, v = hsv[:, :, 1], hsv[:, :, 2]
        mask = (s > 60) & (v > 40) & (v < 250)
        if int(mask.sum()) < 12:
            return None
        b = float(crop[:, :, 0][mask].mean())
        g = float(crop[:, :, 1][mask].mean())
        r = float(crop[:, :, 2][mask].mean())
        tot = r + g + b + 1e-6
        return np.array([r / tot, g / tot], dtype=np.float64)

    def calibrate(self, crops: list[np.ndarray]) -> None:
        feats = [f for f in (self._feature(c) for c in crops if c is not None and c.size > 0)
                 if f is not None]
        if len(feats) < 10:
            print(f"[Teams] Only {len(feats)} usable crops — too few to calibrate. "
                  "Players will be left unclassified.")
            return
        self.kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        self.kmeans.fit(np.array(feats))
        self.calibrated = True
        c0, c1 = self.kmeans.cluster_centers_
        print(f"[Teams] Calibrated on {len(feats)} bib crops (chromaticity). "
              f"Team0 r,g≈({c0[0]:.2f},{c0[1]:.2f}) | Team1 r,g≈({c1[0]:.2f},{c1[1]:.2f})")

    def classify_player(self, tracker_id: int, frame: np.ndarray, bbox: np.ndarray) -> int:
        if not self.calibrated:
            return -1
        votes = self._votes.setdefault(tracker_id, [])

        # Vote only on CLEAR (close/large) frames; tiny distant crops are unreliable.
        clear = (bbox[3] - bbox[1]) >= self.CLEAR_MIN_FRAC * frame.shape[0]
        if clear and len(votes) < self.MAX_VOTES:
            crop = self.crop_player(frame, bbox)
            if crop is not None:
                feat = self._feature(crop)
                if feat is not None:
                    votes.append(int(self.kmeans.predict([feat])[0]))

        if len(votes) >= self.MIN_CONFIDENT_VOTES:
            top = max(set(votes), key=votes.count)
            if votes.count(top) / len(votes) >= self.CONFIDENT_AGREEMENT:
                self.track_team_map[tracker_id] = top

        return self.track_team_map.get(tracker_id, -1)

    def get_color(self, team: int) -> tuple[int, int, int]:
        return self.TEAM_COLORS_BGR.get(team, self.TEAM_COLORS_BGR[-1])
