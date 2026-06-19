"""
Camera-motion estimation for a panning camera.

Estimates how the camera pans/zooms each frame (using sparse optical flow on the
static background, with players masked out) and accumulates it, so any screen
position can be mapped into a common STABILISED coordinate frame. In that frame a
player's position is continuous even while the camera moves — which is what makes
it possible to stitch a player's fragmented tracks back together across pans.
"""
import cv2
import numpy as np


class CameraMotion:
    def __init__(self, max_corners: int = 400):
        self.max_corners = max_corners
        self.prev_gray: np.ndarray | None = None
        self.prev_pts: np.ndarray | None = None
        self.cum = np.eye(3, dtype=np.float64)   # maps current-frame pixels -> reference frame

    def update(self, frame: np.ndarray, player_boxes) -> np.ndarray:
        """Advance one frame; returns the cumulative 3x3 transform (frame -> reference)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.full(gray.shape, 255, np.uint8)
        for x1, y1, x2, y2 in player_boxes:
            cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 0, -1)

        if self.prev_gray is not None and self.prev_pts is not None and len(self.prev_pts) >= 10:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pts, None)
            if nxt is not None:
                ok = st.flatten() == 1
                old_good = self.prev_pts[ok]
                new_good = nxt[ok]
                if len(old_good) >= 10:
                    M, _ = cv2.estimateAffinePartial2D(
                        old_good, new_good, method=cv2.RANSAC, ransacReprojThreshold=3)
                    if M is not None:
                        Mh = np.vstack([M, [0, 0, 1]])          # prev -> cur
                        try:
                            self.cum = self.cum @ np.linalg.inv(Mh)   # cur -> reference
                        except np.linalg.LinAlgError:
                            pass

        self.prev_gray = gray
        self.prev_pts = cv2.goodFeaturesToTrack(
            gray, maxCorners=self.max_corners, qualityLevel=0.01, minDistance=8, mask=mask)
        return self.cum.copy()


def stabilize(transform_2x3, x: float, y: float) -> tuple[float, float]:
    """Map a screen point (x,y) into the stabilised reference frame using a 2x3 affine."""
    a, b, tx, c, d, ty = transform_2x3
    return a * x + b * y + tx, c * x + d * y + ty
