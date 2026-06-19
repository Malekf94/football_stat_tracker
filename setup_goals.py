"""
Interactive goal-zone setup tool.

Run once per pitch/camera angle:
    python setup_goals.py "e:/footballVids/match-farhat-fc-2026-05-14.mp4"

Use the slider at the top to scrub to a point where you can clearly see a goal,
then click 4 corners around the goal mouth. Moving the slider clears any
points you've placed, so get the frame right first.

Controls:
    Slider      - scrub through the video
    Left-click  - add point
    R           - reset current zone's points
    ENTER       - confirm current zone and move to next
    ESC         - skip this zone
    Q           - quit without saving
"""

import cv2
import json
import os
import sys
import numpy as np

_points: list[tuple[int, int]] = []


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


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def setup_goals(video_path: str) -> str:
    global _points

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Cannot open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_seconds = max(1, int(total_frames / fps))

    # Compute display scale once from video resolution
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first = cap.read()
    if not ret:
        sys.exit("Cannot read any frame from video.")
    h, w = first.shape[:2]
    scale = min(1.0, 1280 / max(w, h))
    dw, dh = int(w * scale), int(h * scale)

    def read_frame(frame_idx: int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, f = cap.read()
        return cv2.resize(f, (dw, dh)) if ok else None

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

        # Start each zone at 5 seconds in (or wherever the previous zone left off)
        start_sec = min(5, total_seconds)
        cv2.createTrackbar("Time (s)", win, start_sec, total_seconds, lambda _: None)

        print(f"\n[Setup] {ZONE_LABELS[zone_idx]}")
        print("  Drag the slider to the moment you can see the goal clearly.")
        print("  Click 4 corners around the goal mouth.")
        print("  ENTER = confirm | R = reset points | ESC = skip | Q = quit")

        current_frame = read_frame(int(start_sec * fps))
        if current_frame is None:
            current_frame = first.copy()
            current_frame = cv2.resize(current_frame, (dw, dh))

        last_tb = start_sec

        while True:
            tb = cv2.getTrackbarPos("Time (s)", win)

            if tb != last_tb:
                new_frame = read_frame(int(tb * fps))
                if new_frame is not None:
                    current_frame = new_frame
                    _points.clear()   # camera may have panned; old clicks are wrong
                last_tb = tb

            display = current_frame.copy()

            # Time overlay bottom-left
            cv2.putText(display, _fmt_time(last_tb), (8, dh - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2, cv2.LINE_AA)

            # Instruction bar top
            info = (f"{ZONE_LABELS[zone_idx]}  |  "
                    f"Points: {len(_points)}/4  |  "
                    f"ENTER=confirm  R=reset  ESC=skip")
            cv2.putText(display, info, (8, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            _draw_zone(display, _points, color, f"Zone {zone_idx + 1}")
            cv2.imshow(win, display)

            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10) and len(_points) >= 3:   # ENTER
                original_pts = [
                    (int(x / scale), int(y / scale)) for x, y in _points
                ]
                zones_original.append(original_pts)
                print(f"  Zone {zone_idx + 1} confirmed at {_fmt_time(last_tb)}: {original_pts}")
                cv2.destroyWindow(win)
                break
            elif key == ord('r'):
                _points.clear()
            elif key == 27:             # ESC
                print(f"  Zone {zone_idx + 1} skipped.")
                cv2.destroyWindow(win)
                break
            elif key == ord('q'):
                print("Quitting without saving.")
                cap.release()
                cv2.destroyAllWindows()
                sys.exit(0)

    cap.release()
    cv2.destroyAllWindows()

    if not zones_original:
        print("No zones defined — nothing saved.")
        sys.exit(0)

    out_path = os.path.splitext(video_path)[0] + "_goals.json"
    with open(out_path, "w") as f:
        json.dump({"goal_zones": zones_original}, f, indent=2)

    print(f"\n[Setup] Saved {len(zones_original)} goal zone(s) -> {out_path}")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python setup_goals.py <video_path>")
    setup_goals(sys.argv[1])
