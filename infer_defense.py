"""
infer_defense.py — run the trained defensive-action classifier over a match and
report detected defensive contributions (grouped) per player.

Slides the model over CANDIDATE moments only — a player near the ball, sampled at a
small temporal stride — extracts the same player-centred clip used in training,
classifies it, thresholds, and de-duplicates consecutive hits of the same player.

    python infer_defense.py "E:/footballVids/testing match.mp4" [--thresh 0.5] [--stride 5]

Prereq: analyze.py has been run on the video (tracks log) and runs_defense/best.pt exists.
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18

from make_defense_dataset import extract_clip, gen_candidates

MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)


def build_index(video_path):
    base = os.path.splitext(os.path.basename(video_path))[0]
    meta = json.load(open(f"output/{base}_tracks_meta.json", encoding="utf-8"))
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
    return dict(base=base, video=video_path, meta=meta,
                frame_players=frame_players, frame_ball=frame_ball, logged=logged)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--model", default="runs_defense/best.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = r3d_18()
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval().to(device)
    mean, std = MEAN.to(device), STD.to(device)

    m = build_index(args.video)
    fps = m["meta"]["fps"]
    display_map = {int(k): v for k, v in m["meta"].get("display_map", {}).items()}
    cands = gen_candidates(m, args.stride)
    print(f"[{m['base']}] {len(cands)} near-ball candidates (K-nearest, stride {args.stride})")

    cap = cv2.VideoCapture(m["video"])
    allp = []  # (frame, track, prob) for every candidate
    with torch.no_grad():
        for k, (f, tid) in enumerate(cands):
            clip = extract_clip(cap, m, f, tid)
            if clip is None:
                continue
            x = torch.from_numpy(clip).float().div(255).permute(3, 0, 1, 2)
            x = ((x - MEAN) / STD).unsqueeze(0).to(device)
            p = torch.softmax(model(x), dim=1)[0, 1].item()
            allp.append((f, tid, p))
            if (k + 1) % 100 == 0:
                print(f"  ...{k+1}/{len(cands)}")
    cap.release()

    win = int(1.5 * fps)

    near_px = max(70, int(0.06 * m["meta"].get("frame_h", 1080)))

    def dedupe(thresh):
        # spatial-temporal NMS: one defensive action per ball-moment. Keep the
        # highest-prob hit, suppress others within `win` frames AND a ball-radius
        # (several near-ball players firing at one action -> a single event).
        hits = sorted([h for h in allp if h[2] >= thresh], key=lambda h: -h[2])
        kept = []
        for f, tid, p in hits:
            ball = m["frame_ball"].get(f)
            dup = False
            for kf, ktid, kp, kball in kept:
                if abs(f - kf) <= win and (
                    not ball or not kball or
                    ((ball[0] - kball[0]) ** 2 + (ball[1] - kball[1]) ** 2) ** 0.5 < 2.5 * near_px):
                    dup = True
                    break
            if not dup:
                kept.append((f, tid, p, ball))
        return sorted((f, tid, p) for f, tid, p, _ in kept)

    # --- ground truth (the user's labels), if present ---
    gt = []
    for cand_path in (f"defense_labels/{m['base']}_defense_labels.json",
                      f"output/{m['base']}_defense_labels.json"):
        if os.path.exists(cand_path):
            gl = json.load(open(cand_path, encoding="utf-8")).get("labels", [])
            # dedupe GT marks within 1.5s (same action marked twice)
            gframes = sorted(set(int(l["frame"]) for l in gl))
            for f in gframes:
                if not gt or f - gt[-1] > win:
                    gt.append(f)
            break

    if gt:
        match_win = int(2.0 * fps)
        print(f"\n=== precision/recall vs {len(gt)} ground-truth actions ===")
        print(" thr  detected   TP  FP  FN   precision  recall")
        for thr in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
            ev = dedupe(thr)
            ev_f = [e[0] for e in ev]
            matched_gt = set()
            tp = 0
            for ef in ev_f:
                hit = next((g for g in gt if g not in matched_gt and abs(g - ef) <= match_win), None)
                if hit is not None:
                    matched_gt.add(hit); tp += 1
            fp = len(ev) - tp
            fn = len(gt) - len(matched_gt)
            prec = tp / len(ev) if ev else 0.0
            rec = tp / len(gt) if gt else 0.0
            print(f" {thr:.2f}   {len(ev):3d}      {tp:3d} {fp:3d} {fn:3d}    {prec:.2f}      {rec:.2f}")

    ev = dedupe(args.thresh)
    per_player = {}
    for f, tid, p in ev:
        d = display_map.get(tid, tid)
        per_player[d] = per_player.get(d, 0) + 1
    print(f"\n=== detected defensive contributions at thresh {args.thresh} ===")
    print(f"TOTAL: {len(ev)}")
    for f, tid, p in ev:
        mm, ss = divmod(int(f / fps), 60)
        print(f"  {mm:02d}:{ss:02d}  #{display_map.get(tid, tid)}  (p={p:.2f})")


if __name__ == "__main__":
    main()
