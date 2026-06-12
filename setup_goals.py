"""
Interactive goal-zone setup tool.

Run once per pitch/camera angle:
    python setup_goals.py "e:/footballVids/match-farhat-fc-2026-05-14.mp4"

Click 4 corners around each goal mouth on the first frame of the video.
The zones are saved next to the video file as <video>_goals.json and will be
loaded automatically by analyze.py.

Controls:
    Left-click  – add point
    R           – reset current zone
    ENTER       – confirm current zone and move to next
    ESC         – skip this zone
    Q           – quit without saving
"""

import cv2
import json
import os
import sys
import numpy as np

_points: list[tuple[int, int]] = []
_scale = 1.0
_frame_display: np.ndarray | None = None


def _mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        _points.append((x, y))


def _draw_zone(img, pts, color, label):
    for i, pt in enumerate(pts):
        cv2.circle(img, pt, 6, color, -1)
        if i > 0:
            cv2.line(img, pts[i - 1], pt, color, 2)
    if len(pts) >= 3:
        arr = np.array(pts, dtype=np.int32)
        cv2.polylines(img, [arr], isClosed=True, color=color, thickness=2)
    if pts:
        cv2.putText(img, label, (pts[0][0] + 8, pts[0][1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def setup_goals(video_path: str) -> str:
    global _points, _scale, _frame_display

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Cannot open: {video_path}")

    # Seek to 5 seconds in so we see actual play, not a black frame
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * 5))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        # Fall back to first frame
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            sys.exit("Cannot read any frame from video.")

    h, w = frame.shape[:2]
    max_dim = 1280
    _scale = min(1.0, max_dim / max(w, h))
    _frame_display = cv2.resize(frame, (int(w * _scale), int(h * _scale)))

    ZONE_COLORS = [(0, 220, 0), (0, 140, 255)]
    ZONE_LABELS = [
        "Zone 0: Team 1 defends (Team 0 scores here)",
        "Zone 1: Team 0 defends (Team 1 scores here)",
    ]

    zones_original: list[list[tuple[int, int]]] = []

    for zone_idx in range(2):
        _points.clear()
        color = ZONE_COLORS[zone_idx]
        win = f"Goal Setup — Zone {zone_idx + 1} of 2"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, _mouse_cb)

        print(f"\n[Setup] {ZONE_LABELS[zone_idx]}")
        print("  Click 4 corners around the goal mouth.")
        print("  ENTER = confirm | R = reset | ESC = skip | Q = quit")

        while True:
            display = _frame_display.copy()
            info = f"{ZONE_LABELS[zone_idx]}  |  Points: {len(_points)}/4  |  ENTER=confirm  R=reset  ESC=skip"
            cv2.putText(display, info, (8, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
            _draw_zone(display, _points, color, f"Zone {zone_idx + 1}")
            cv2.imshow(win, display)

            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10) and len(_points) >= 3:   # ENTER — accept
                original_pts = [
                    (int(x / _scale), int(y / _scale)) for x, y in _points
                ]
                zones_original.append(original_pts)
                print(f"  Zone {zone_idx + 1} confirmed: {original_pts}")
                cv2.destroyWindow(win)
                break
            elif key == ord('r'):
                _points.clear()
            elif key == 27:             # ESC — skip
                print(f"  Zone {zone_idx + 1} skipped.")
                cv2.destroyWindow(win)
                break
            elif key == ord('q'):
                print("Quitting without saving.")
                cv2.destroyAllWindows()
                sys.exit(0)

    cv2.destroyAllWindows()

    if not zones_original:
        print("No zones defined — nothing saved.")
        sys.exit(0)

    out_path = os.path.splitext(video_path)[0] + "_goals.json"
    with open(out_path, "w") as f:
        json.dump({"goal_zones": zones_original}, f, indent=2)

    print(f"\n[Setup] Saved {len(zones_original)} goal zone(s) → {out_path}")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python setup_goals.py <video_path>")
    setup_goals(sys.argv[1])
