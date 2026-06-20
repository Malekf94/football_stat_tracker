"""
label_defense.py — mark DEFENSIVE-CONTRIBUTION moments on a match video, to build
the training set for the Tier-2 video action classifier.

Prereq: run analyze.py on the video first so the tracks log exists:
    .\\venv\\Scripts\\python analyze.py "E:\\footballVids\\<match>.mp4" --no-video --no-goals
That writes output/<base>_tracks.jsonl + _tracks_meta.json, which this tool reads
so every player is already boxed + identified for you to click.

Then label:
    .\\venv\\Scripts\\python label_defense.py "E:\\footballVids\\<match>.mp4"

You only mark POSITIVES (defensive actions). Negative examples (including
"near the ball but did nothing") are auto-sampled later during clip extraction.

Controls
--------
  Trackbar / , . : scrub one logged frame back / forward
  [  ]           : jump 10 logged frames back / forward
  left-click     : select the player under the cursor (the one who made the action)
  t b i c s      : MARK a defensive action by the selected player at this frame,
                   tagged tackle / block / interception / clearance / save. The tag
                   is stored for reference; training groups them as one class.
  m  or  SPACE   : MARK with a generic "defensive" tag (when you don't care which type)
  u              : undo the last mark
  w              : save labels now  ->  output/<base>_defense_labels.json
  q / ESC        : save + quit
"""
import json
import os
import sys

import cv2
import numpy as np

from tracker.teams import TeamClassifier

TYPE_KEYS = {ord('t'): "tackle", ord('b'): "block", ord('i'): "interception",
             ord('c'): "clearance", ord('s'): "save"}


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def main(video_path: str):
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = "output"
    meta_path = os.path.join(out_dir, f"{base}_tracks_meta.json")
    tracks_path = os.path.join(out_dir, f"{base}_tracks.jsonl")
    if not (os.path.exists(meta_path) and os.path.exists(tracks_path)):
        sys.exit(f"No tracks for '{base}'. Run analyze.py on it first "
                 f"(expected {tracks_path}).")

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    fps = meta.get("fps", 30.0)
    display_map = {int(k): v for k, v in meta.get("display_map", {}).items()}
    final_teams = {int(k): int(v) for k, v in meta.get("final_teams", {}).items()}

    # Load logged frames (sorted). Each: frame_no -> (players, ball)
    frames = []
    with open(tracks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            frames.append(rec)
    frames.sort(key=lambda r: r["f"])
    if not frames:
        sys.exit("Tracks log is empty.")

    # Labels live in a TRACKED folder (the hand-work), not the git-ignored output/.
    labels_dir = "defense_labels"
    os.makedirs(labels_dir, exist_ok=True)
    labels_path = os.path.join(labels_dir, f"{base}_defense_labels.json")
    legacy_path = os.path.join(out_dir, f"{base}_defense_labels.json")
    labels: list[dict] = []
    src = labels_path if os.path.exists(labels_path) else (legacy_path if os.path.exists(legacy_path) else None)
    if src:
        with open(src, encoding="utf-8") as f:
            labels = json.load(f).get("labels", [])
        print(f"[Load] {len(labels)} existing labels from {src}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {video_path}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, 1280 / max(W, H))
    dw, dh = int(W * scale), int(H * scale)

    tc = TeamClassifier()
    win = f"Label defensive actions — {base}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    state = {"idx": 0, "selected": None, "type": "defensive", "frame_bgr": None,
             "frame_loaded_for": -1}

    def read_logged(idx: int):
        idx = max(0, min(len(frames) - 1, idx))
        fno = frames[idx]["f"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
        ok, fr = cap.read()
        state["frame_bgr"] = fr if ok else None
        state["frame_loaded_for"] = idx
        return idx

    def players_here():
        return frames[state["idx"]]["players"]

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        ox, oy = x / scale, y / scale       # back to original coords
        best, best_area = None, None
        for p in players_here():
            tid, x1, y1, x2, y2, _team = p
            if x1 <= ox <= x2 and y1 <= oy <= y2:
                area = (x2 - x1) * (y2 - y1)
                if best_area is None or area < best_area:   # smallest box wins (tightest)
                    best, best_area = tid, area
        if best is not None:
            state["selected"] = int(best)

    cv2.setMouseCallback(win, on_mouse)
    cv2.createTrackbar("frame", win, 0, len(frames) - 1, lambda v: state.update(idx=v))

    read_logged(0)
    print(f"[Ready] {len(frames)} logged frames. Click a player, press M to mark a "
          f"defensive action. W saves, Q quits.\n")

    while True:
        # sync trackbar -> idx (and reload frame if changed)
        tb = cv2.getTrackbarPos("frame", win)
        if tb != state["idx"]:
            state["idx"] = tb
        if state["idx"] != state["frame_loaded_for"]:
            read_logged(state["idx"])
            cv2.setTrackbarPos("frame", win, state["idx"])

        frame = state["frame_bgr"]
        if frame is None:
            disp = np.zeros((dh, dw, 3), np.uint8)
        else:
            disp = cv2.resize(frame, (dw, dh))

        fno = frames[state["idx"]]["f"]
        ball = frames[state["idx"]].get("ball")

        # draw players
        for p in players_here():
            tid, x1, y1, x2, y2, fteam = p
            tid = int(tid)
            team = final_teams.get(tid, fteam if fteam in (0, 1) else -1)
            col = tc.get_color(team)
            sx1, sy1, sx2, sy2 = (int(v * scale) for v in (x1, y1, x2, y2))
            sel = (tid == state["selected"])
            cv2.rectangle(disp, (sx1, sy1), (sx2, sy2), (255, 255, 255) if sel else col,
                          3 if sel else 1)
            did = display_map.get(tid, tid)
            cv2.putText(disp, f"#{did}", (sx1, sy1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255) if sel else col, 1, cv2.LINE_AA)

        # draw ball
        if ball is not None:
            bx, by = int(ball[0] * scale), int(ball[1] * scale)
            cv2.circle(disp, (bx, by), 8, (0, 230, 230), 2)

        # marks on this exact frame
        here = [m for m in labels if m["frame"] == fno]

        # HUD
        cv2.rectangle(disp, (0, 0), (dw, 46), (0, 0, 0), -1)
        sel_txt = f"#{display_map.get(state['selected'], state['selected'])}" if state["selected"] else "none"
        hud = (f"{_fmt_time(fno / fps)}  f{fno}  [{state['idx']+1}/{len(frames)}]   "
               f"selected: {sel_txt}   marks: {len(labels)}")
        cv2.putText(disp, hud, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(disp, "click player, then s/b/t/i/c marks it (M=generic)  u=undo  w=save  q=quit",
                    (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        if here:
            cv2.putText(disp, f"MARKED HERE: " + ", ".join(f"#{m['display_id']}({m['type']})" for m in here),
                        (8, dh - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow(win, disp)
        key = cv2.waitKey(20) & 0xFF

        def do_mark(action_type: str):
            """Record a defensive action of action_type by the selected player here."""
            if state["selected"] is None:
                print("[!] Click a player first, then press the type key (or M).")
                return
            tid = state["selected"]
            labels.append({
                "frame": fno, "track_id": tid,
                "display_id": display_map.get(tid, tid),
                "team": final_teams.get(tid, -1), "type": action_type,
                "ball": ball, "time": round(fno / fps, 2),
            })
            state["type"] = action_type
            state["selected"] = None      # deselect so the next action needs a fresh click
            print(f"  + marked {action_type} by #{display_map.get(tid, tid)} "
                  f"at {_fmt_time(fno / fps)} (f{fno})  [{len(labels)} total]")

        if key in (ord('.'), ord('d')):
            state["idx"] = min(len(frames) - 1, state["idx"] + 1)
        elif key in (ord(','), ord('a')):
            state["idx"] = max(0, state["idx"] - 1)
        elif key == ord(']'):
            state["idx"] = min(len(frames) - 1, state["idx"] + 10)
        elif key == ord('['):
            state["idx"] = max(0, state["idx"] - 10)
        elif key in TYPE_KEYS:        # type key marks immediately
            do_mark(TYPE_KEYS[key])
        elif key in (ord('m'), ord(' ')):   # generic mark (no specific type)
            do_mark("defensive")
        elif key == ord('u'):
            if labels:
                m = labels.pop()
                print(f"  - undid mark #{m['display_id']} at f{m['frame']}  [{len(labels)} total]")
        elif key == ord('w'):
            _save(labels_path, base, video_path, meta, labels)
        elif key in (ord('q'), 27):
            _save(labels_path, base, video_path, meta, labels)
            break

        cv2.setTrackbarPos("frame", win, state["idx"])

    cap.release()
    cv2.destroyAllWindows()


def _save(path, base, video_path, meta, labels):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "base": base, "video_path": os.path.abspath(video_path),
            "fps": meta.get("fps"), "skip_frames": meta.get("skip_frames"),
            "frame_w": meta.get("frame_w"), "frame_h": meta.get("frame_h"),
            "labels": labels,
        }, f, indent=2)
    print(f"[Saved] {len(labels)} labels -> {path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: python label_defense.py "<video_path>"')
    main(sys.argv[1])
