"""
make_defense_dataset.py — build the video-clip dataset for the defensive-action
classifier from label_defense.py marks.

For each labelled action (positive) and each auto-sampled non-action (negative) it
cuts a PLAYER-CENTRED clip: a square crop that follows the subject player's box
across ~1.6s, so the camera pan cancels and the player stays centred. Negatives
are half "near the ball but did nothing" (hard) and half random player-moments.

Extraction is SEEK-BASED and one clip at a time (reads only the ~16 frames each
clip needs, low memory) — it does NOT blast through the whole video.

Usage:
    python make_defense_dataset.py "E:/footballVids/elevens 140626.mp4" "E:/footballVids/testing match.mp4"
Outputs:
    dataset_defense/clips/*.npy   (uint8 [T,H,W,3])
    dataset_defense/manifest.csv  (clip, label, match, center_frame, subject)
"""
import csv
import json
import os
import random
import sys

import cv2
import numpy as np

T = 16                 # frames per clip
CROP_SCALE = 2.5       # square crop = this * subject box height (context around player)
OUT = 112              # output clip resolution
EXCLUDE_S = 1.0        # a negative must be at least this far (s) from any positive
DEDUPE_FRAMES = 15     # merge positive marks of same player within this many frames
NEG_STRIDE = 5         # candidates sampled every this many logged frames
NEG_RATIO = 5          # negatives per positive
K_NEAR = 5             # consider the K nearest players to the ball (the defender isn't always #1)
RADIUS_MULT = 3.0      # ...within this * near_px of the ball
POS_WIN_S = 1.0        # a candidate is POSITIVE if within this (s) of a labelled action by the SAME player
OUT_DIR = "dataset_defense"

random.seed(42)


def load_match(video_path):
    base = os.path.splitext(os.path.basename(video_path))[0]
    meta = json.load(open(f"output/{base}_tracks_meta.json", encoding="utf-8"))
    labels_path = f"defense_labels/{base}_defense_labels.json"
    if not os.path.exists(labels_path):
        labels_path = f"output/{base}_defense_labels.json"   # legacy location
    if not os.path.exists(labels_path):
        return None
    labels = json.load(open(labels_path, encoding="utf-8")).get("labels", [])

    # index the tracks log: ordered logged frames, per-frame player boxes + ball
    frame_players, frame_ball, logged = {}, {}, []
    for line in open(f"output/{base}_tracks.jsonl", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        f = rec["f"]
        logged.append(f)
        frame_players[f] = {int(p[0]): (p[1], p[2], p[3], p[4]) for p in rec["players"]}
        frame_ball[f] = rec.get("ball")
    logged.sort()
    return dict(base=base, video=video_path, meta=meta, labels=labels,
                frame_players=frame_players, frame_ball=frame_ball, logged=logged)


def window_frames(logged, center, n=T):
    """The n logged frames centred on `center` (clamped to the ends)."""
    if center not in frame_index_cache(logged):
        # nearest logged frame
        center = min(logged, key=lambda x: abs(x - center))
    i = frame_index_cache(logged)[center]
    lo = max(0, i - n // 2)
    hi = min(len(logged), lo + n)
    lo = max(0, hi - n)
    return logged[lo:hi]


_FIDX = {}
def frame_index_cache(logged):
    key = id(logged)
    if key not in _FIDX:
        _FIDX[key] = {f: i for i, f in enumerate(logged)}
    return _FIDX[key]


def subject_boxes(m, subject, frames):
    """Box for `subject` at each frame, carrying the last known box over gaps."""
    boxes, last = [], None
    for f in frames:
        b = m["frame_players"].get(f, {}).get(subject)
        if b is not None:
            last = b
        boxes.append(last)
    # back-fill leading None with first known
    first = next((b for b in boxes if b is not None), None)
    return [b if b is not None else first for b in boxes], (first is not None)


def gen_candidates(m, stride, k=K_NEAR, radius_mult=RADIUS_MULT):
    """Candidate (frame, player) pairs = the K players nearest the ball (within
    radius_mult * near_px) at every `stride`-th logged frame. The defender who
    makes a tackle/block is often NOT the closest to the ball (the attacker is),
    so we look at several near-ball players. TRAINING and INFERENCE both use this,
    so the model sees the same distribution either way."""
    H = m["meta"].get("frame_h", 1080)
    radius = max(70, int(0.06 * H)) * radius_mult
    logged = m["logged"]
    out = []
    for i in range(T, len(logged) - T, stride):
        f = logged[i]
        ball = m["frame_ball"].get(f)
        if not ball:
            continue
        ds = []
        for tid, (x1, y1, x2, y2) in m["frame_players"].get(f, {}).items():
            cx, cy = (x1 + x2) / 2, y2
            d = ((cx - ball[0]) ** 2 + (cy - ball[1]) ** 2) ** 0.5
            if d < radius:
                ds.append((d, tid))
        ds.sort()
        for _, tid in ds[:k]:
            out.append((f, tid))
    return out


def make_specs(m):
    """Return list of (center_frame, subject_track, label) for one match.

    A candidate is POSITIVE iff it's the SAME player as a labelled action within
    POS_WIN_S; every other candidate is a negative — including the OTHER near-ball
    players at an action moment, which are exactly the hard negatives we want
    (the model must pick the defender out of the near-ball cluster)."""
    W = int(POS_WIN_S * m["meta"]["fps"])
    acts = [(int(l["frame"]), int(l["track_id"])) for l in m["labels"]]

    pos, neg = [], []
    for f, tid in gen_candidates(m, NEG_STRIDE):
        if any(abs(f - af) <= W and tid == at for af, at in acts):
            pos.append((f, tid, 1))
        else:
            neg.append((f, tid, 0))
    random.shuffle(neg)
    return pos + neg[:NEG_RATIO * len(pos)]


def extract_clip(cap, m, center, subject):
    frames = window_frames(m["logged"], center)
    boxes, ok = subject_boxes(m, subject, frames)
    if not ok:
        return None
    need = sorted(set(frames))
    box_at = dict(zip(frames, boxes))
    # read forward from the first needed frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, need[0])
    grabbed, cur = {}, need[0]
    need_set = set(need)
    while cur <= need[-1]:
        okr, fr = cap.read()
        if not okr:
            break
        if cur in need_set:
            grabbed[cur] = fr
        cur += 1
    clip = []
    Hh = m["meta"].get("frame_h", 1080)
    Ww = m["meta"].get("frame_w", 1920)
    for f in frames:
        fr = grabbed.get(f)
        b = box_at.get(f)
        if fr is None or b is None:
            clip.append(np.zeros((OUT, OUT, 3), np.uint8))
            continue
        x1, y1, x2, y2 = b
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half = max(20, CROP_SCALE * (y2 - y1) / 2)
        rx1, ry1 = int(cx - half), int(cy - half)
        rx2, ry2 = int(cx + half), int(cy + half)
        rx1, ry1 = max(0, rx1), max(0, ry1)
        rx2, ry2 = min(Ww, rx2), min(Hh, ry2)
        crop = fr[ry1:ry2, rx1:rx2]
        if crop.size == 0:
            clip.append(np.zeros((OUT, OUT, 3), np.uint8))
        else:
            clip.append(cv2.resize(crop, (OUT, OUT)))
    return np.stack(clip)  # [T, OUT, OUT, 3] uint8


def main(video_paths):
    os.makedirs(f"{OUT_DIR}/clips", exist_ok=True)
    manifest = []
    for vp in video_paths:
        m = load_match(vp)
        if m is None:
            print(f"[skip] no labels for {vp}")
            continue
        specs = make_specs(m)
        npos = sum(1 for s in specs if s[2] == 1)
        print(f"[{m['base']}] {npos} positives + {len(specs)-npos} negatives = {len(specs)} clips")
        cap = cv2.VideoCapture(m["video"])
        if not cap.isOpened():
            print(f"  [!] cannot open {m['video']}"); continue
        specs.sort(key=lambda s: s[0])   # forward-ish seeking
        for k, (center, subject, label) in enumerate(specs):
            clip = extract_clip(cap, m, center, subject)
            if clip is None:
                continue
            name = f"{m['base'].replace(' ', '_')}_{center}_{subject}_{label}.npy"
            np.save(f"{OUT_DIR}/clips/{name}", clip)
            manifest.append({"clip": name, "label": label, "match": m["base"],
                             "center_frame": center, "subject": subject})
            if (k + 1) % 50 == 0:
                print(f"  ...{k+1}/{len(specs)}")
        cap.release()

    with open(f"{OUT_DIR}/manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "label", "match", "center_frame", "subject"])
        w.writeheader()
        w.writerows(manifest)
    npos = sum(1 for r in manifest if r["label"] == 1)
    print(f"\n[Done] {len(manifest)} clips ({npos} pos / {len(manifest)-npos} neg) "
          f"-> {OUT_DIR}/manifest.csv")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: python make_defense_dataset.py "<video1>" ["<video2>" ...]')
    main(sys.argv[1:])
