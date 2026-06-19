import cv2
import numpy as np
from sklearn.cluster import KMeans


class TeamClassifier:
    """
    Separates players into two teams by BIB COLOUR, clustered per match.

    Design choices that measurably beat the alternatives on our footage:
      * **Tight bib-centre crop** (centre of the chest, not the whole torso) — avoids
        arms/skin/shorts/background that otherwise contaminate the colour.
      * **Chromaticity feature** R/(R+G+B), G/(R+G+B) — brightness-independent, so a
        bib in shadow still reads as its true colour. Generalises to ANY two bib
        colours (it clusters whatever two are present), so it works across matches.
      * **Pitch-green rejection** — crops dominated by the green pitch (or too dark /
        unsaturated) don't produce a feature, so background/shadow/blur frames can't
        pollute the cluster centres or a track's vote. (Diagnosed as the main cause
        of contaminated clusters on lower-contrast kits.)
      * **Per-track aggregation** — a track's team is decided from the MEDIAN of its
        clean per-frame features versus the two cluster centres, accepted when one
        centre is clearly nearer (a confidence margin). This is more stable than a
        per-frame majority vote and assigns far more tracks (the old 0.7 vote-
        agreement rule left 16–32% of tracks "unknown").

    Tracks whose colour stays genuinely ambiguous remain "unknown" (-1) so they
    can't create phantom pass/turnover events.

    Public interface (unchanged for analyze.py / replay.py):
        crop_player(frame, bbox)           -> bib crop (np.ndarray) or None
        calibrate(crops)                   -> fit 2-colour KMeans on crops
        classify_player(tid, frame, bbox)  -> team 0/1/-1 (per-track median)
        track_team_map                     -> dict[tracker_id, team]
        get_color(team)
    """

    TEAM_COLORS_BGR = {
        0: (255, 100, 0),    # Blue
        1: (0, 80, 255),     # Red
        -1: (160, 160, 160),  # Unknown (grey)
    }

    MAX_FEATS = 25            # cap features stored per track
    MIN_FEATS = 3             # need at least this many clean features to decide
    CONF_MARGIN = 0.10        # relative gap between the two centre distances to accept
    CLEAR_MIN_FRAC = 0.07     # only sample frames where the player is this tall
    GREEN_H_LO, GREEN_H_HI = 35, 85   # OpenCV hue band treated as pitch green

    def __init__(self, device: str | None = None):   # device kept for call-compat
        self.kmeans: KMeans | None = None
        self.calibrated = False
        self.track_team_map: dict[int, int] = {}
        self._feats: dict[int, list[np.ndarray]] = {}
        # Maps a KMeans cluster index -> canonical team id, chosen so the redder
        # cluster is team 1 and the bluer is team 0 (keeps the blue/red HUD labels
        # meaningful and cuts down on needing the manual team-swap).
        self._cluster_to_team: dict[int, int] = {0: 0, 1: 1}

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

    @classmethod
    def _feature(cls, crop: np.ndarray) -> np.ndarray | None:
        """Brightness-independent chromaticity of the saturated NON-green bib pixels.

        Returns None when the crop is mostly pitch-green, too dark or unsaturated —
        i.e. there isn't a real bib colour to read — so it doesn't vote/calibrate.
        """
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        saturated = (s > 60) & (v > 40) & (v < 250)
        green = (h >= cls.GREEN_H_LO) & (h <= cls.GREEN_H_HI)
        mask = saturated & ~green
        # Require both enough bib pixels AND that they aren't a tiny green-swamped
        # minority (reject crops that are mostly pitch).
        if int(mask.sum()) < 12 or mask.mean() < 0.10:
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
        # Canonicalise: the cluster whose centre is redder (higher r/(r+g+b)) is
        # team 1, the other team 0.
        c0, c1 = self.kmeans.cluster_centers_
        redder = 0 if c0[0] >= c1[0] else 1
        self._cluster_to_team = {redder: 1, 1 - redder: 0}
        print(f"[Teams] Calibrated on {len(feats)} bib crops (chromaticity, green-rejected). "
              f"Team0 r,g~({[c0,c1][1-redder][0]:.2f},{[c0,c1][1-redder][1]:.2f}) | "
              f"Team1 r,g~({[c0,c1][redder][0]:.2f},{[c0,c1][redder][1]:.2f})")

    def classify_player(self, tracker_id: int, frame: np.ndarray, bbox: np.ndarray) -> int:
        if not self.calibrated:
            return -1
        feats = self._feats.setdefault(tracker_id, [])

        # Sample only CLEAR (close/large) frames; tiny distant crops are unreliable.
        clear = (bbox[3] - bbox[1]) >= self.CLEAR_MIN_FRAC * frame.shape[0]
        if clear and len(feats) < self.MAX_FEATS:
            crop = self.crop_player(frame, bbox)
            if crop is not None:
                f = self._feature(crop)
                if f is not None:
                    feats.append(f)

        # Decide from the MEDIAN feature vs the two cluster centres, accepting when
        # one centre is clearly nearer than the other (confidence margin).
        if len(feats) >= self.MIN_FEATS:
            med = np.median(np.array(feats), axis=0)
            d = np.linalg.norm(self.kmeans.cluster_centers_ - med, axis=1)
            near = int(np.argmin(d))
            far = 1 - near
            margin = (d[far] - d[near]) / (d[far] + d[near] + 1e-6)
            if margin >= self.CONF_MARGIN:
                self.track_team_map[tracker_id] = self._cluster_to_team[near]

        return self.track_team_map.get(tracker_id, -1)

    def get_color(self, team: int) -> tuple[int, int, int]:
        return self.TEAM_COLORS_BGR.get(team, self.TEAM_COLORS_BGR[-1])
