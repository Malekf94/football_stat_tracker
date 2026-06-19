"""
Build a detector-training dataset from match videos.

Samples frames across each match and PRE-LABELS players + ball with the current
YOLO model, writing a YOLO-format dataset (images + label txt + data.yaml). You
then only have to CORRECT the boxes (fix misses / bad ball detections / add
goalkeeper-referee classes) rather than draw everything from scratch.

Usage:
    .\\venv\\Scripts\\python make_dataset.py "..\\match1.mp4" "..\\match2.mp4" --per 60

Output: dataset/images/*.jpg, dataset/labels/*.txt, dataset/data.yaml
(Upload to Roboflow to correct, or use a local tool — decided next.)
"""
import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import cv2
from ultralytics import YOLO

PERSON, BALL = 0, 32          # COCO class ids in the base model
OUT = "dataset"
# dataset classes (player + ball for the first fine-tune; can add
# goalkeeper/referee later by extending this list and labelling them)
CLASS_NAMES = ["player", "ball"]


def yolo_line(cls, x1, y1, x2, y2, W, H):
    cx = (x1 + x2) / 2 / W
    cy = (y1 + y2) / 2 / H
    w = (x2 - x1) / W
    h = (y2 - y1) / H
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+", help="match video paths")
    ap.add_argument("--per", type=int, default=60, help="frames to sample per video")
    ap.add_argument("--model", default="yolo11x.pt", help="model for pre-labelling (bigger=cleaner)")
    ap.add_argument("--conf", type=float, default=0.30, help="player confidence")
    ap.add_argument("--ball-conf", type=float, default=0.10, help="ball confidence (low: ball is faint)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--min-players", type=int, default=4, help="skip near-empty frames")
    args = ap.parse_args()

    os.makedirs(f"{OUT}/images", exist_ok=True)
    os.makedirs(f"{OUT}/labels", exist_ok=True)
    print(f"[Dataset] pre-labelling with {args.model} …")
    model = YOLO(args.model)

    total_saved = 0
    for v in args.videos:
        cap = cv2.VideoCapture(v)
        if not cap.isOpened():
            print(f"  !! cannot open {v} — skipped")
            continue
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stride = max(1, nframes // max(1, args.per))
        base = os.path.splitext(os.path.basename(v))[0].replace(" ", "_")
        fidx = saved = 0
        while saved < args.per and fidx < nframes:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, frame = cap.read()
            if not ok:
                break
            H, W = frame.shape[:2]
            pr = model.predict(frame, conf=args.conf, imgsz=args.imgsz, classes=[PERSON], verbose=False)[0]
            lines = [yolo_line(0, *b, W, H) for b in pr.boxes.xyxy.cpu().numpy()]
            if len(lines) >= args.min_players:
                br = model.predict(frame, conf=args.ball_conf, imgsz=args.imgsz, classes=[BALL], verbose=False)[0]
                if len(br.boxes) > 0:
                    bi = int(np.argmax(br.boxes.conf.cpu().numpy()))
                    lines.append(yolo_line(1, *br.boxes.xyxy.cpu().numpy()[bi], W, H))
                name = f"{base}_{fidx:06d}"
                cv2.imwrite(f"{OUT}/images/{name}.jpg", frame)
                with open(f"{OUT}/labels/{name}.txt", "w") as f:
                    f.write("\n".join(lines))
                saved += 1
                total_saved += 1
            fidx += stride
        cap.release()
        print(f"  {os.path.basename(v)}: {saved} frames")

    with open(f"{OUT}/data.yaml", "w") as f:
        f.write("path: .\ntrain: images\nval: images\n")
        f.write(f"nc: {len(CLASS_NAMES)}\nnames: {CLASS_NAMES}\n")
    print(f"\n[Dataset] {total_saved} pre-labelled frames in {OUT}/. "
          "Pre-labels are imperfect — correct them next.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: python make_dataset.py "<video1>" ["<video2>" ...] [--per N]')
    main()
